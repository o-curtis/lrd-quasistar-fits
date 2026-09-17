#!/usr/bin/env python3
"""
Broadband fit of The Egg using MODS Red + FIRE jointly.

Model: 1-comp TLUSTY hot atmosphere + AGN PL + (optional) cool TLUSTY.
MODS Red (3900-7400 Å) and FIRE (7400-15500 Å) are treated as separate arms
with independent NNLS amplitude scaling to handle:
  (a) different slit-loss/flux calibration between instruments
  (b) 64-day AGN variability between observations

Both Calzetti and SMC extinction laws tested.

NNLS amplitudes per fit: [A_hot_MODS, A_AGN_MODS, A_hot_FIRE, A_AGN_FIRE, A_cool_FIRE]
(5 amplitudes, all non-negative, all solved analytically at each likelihood call)

Outputs:
  egg_chain_bb_calzetti.npy / .pkl
  egg_chain_bb_smc.npy / .pkl
  egg_bic_broadband.txt
"""
import os, time, pickle
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.interpolate import interp1d
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'egg_data',
                           'Re_ J1025+1402 LBT _ Magellan spectra')
MODS_CACHE = os.path.join(SCRIPT_DIR, 'egg_mods_cache.npz')
FIRE_CACHE = os.path.join(SCRIPT_DIR, 'egg_hires_cache.npz')

Z_EGG     = 0.1007
F_SYS     = 0.05
AGN_PIVOT = 5000.0

# Fit windows per arm
MODSR_LO, MODSR_HI = 3900.0, 7400.0   # Å rest
FIRE_LO,  FIRE_HI  = 7400.0, 15500.0

# ── Load model caches ─────────────────────────────────────────────────────────
print("Loading MODS model cache …", flush=True)
mc       = np.load(MODS_CACHE, allow_pickle=True)
wm       = mc['wave'].astype(float)
mflux    = mc['flux']
mmeta    = list(mc['meta'])
mT    = np.array([m['T']    for m in mmeta], dtype=float)
mlogg = np.array([m['logg'] for m in mmeta], dtype=float)
mMH   = np.array([m['MH']   for m in mmeta], dtype=int)
MH_vals = sorted(set(mMH))

print("Loading FIRE model cache …", flush=True)
fc       = np.load(FIRE_CACHE, allow_pickle=True)
wf       = fc['wave'].astype(float)
fflux    = fc['flux']
fmeta    = list(fc['meta'])
fT    = np.array([m['T']    for m in fmeta], dtype=float)
flogg = np.array([m['logg'] for m in fmeta], dtype=float)
fMH   = np.array([m['MH']   for m in fmeta], dtype=int)
print(f"  MODS: {len(mmeta)} models | FIRE: {len(fmeta)} models", flush=True)

# ── Extinction curves ─────────────────────────────────────────────────────────
def calzetti_ext(wave_aa, a_v):
    w = wave_aa * 1e-4
    k = np.zeros_like(w, dtype=float)
    lo = w < 0.63
    k[lo] = 2.659*(-2.156 + 1.509/w[lo] - 0.198/w[lo]**2 + 0.011/w[lo]**3) + 4.05
    hi = ~lo & (w < 2.2)
    k[hi] = 2.659*(-1.857 + 1.040/w[hi]) + 4.05
    return 10**(-0.4 * np.clip(k,0,None) * a_v / 4.05)

_smc_x   = np.array([0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
                      2.27, 2.70, 3.33, 3.70, 4.55, 5.88])
_smc_rav = np.array([0.112, 0.420, 0.427, 0.453, 0.570, 0.741, 1.000,
                     1.412, 1.785, 2.460, 2.609, 2.770, 3.554])
_smc_fn  = interp1d(_smc_x, _smc_rav, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
def smc_ext(wave_aa, a_v):
    x = 1.0 / (wave_aa * 1e-4)
    return 10**(-0.4 * np.clip(_smc_fn(x), 0, None) * a_v)

# ── Dual-cache model interpolator ─────────────────────────────────────────────
def _interp_cache(T, logg, mh_idx, wave_arr, T_arr, logg_arr, MH_arr, flux_arr, MHvals):
    mh_idx = np.clip(mh_idx, 0.0, len(MHvals)-1.0)
    lo_i = int(np.floor(mh_idx)); hi_i = min(lo_i+1, len(MHvals)-1)
    mf = mh_idx - lo_i

    def _tlogg(T_, logg_, MH_):
        sel = MH_arr == MH_
        if not sel.any(): return None
        Tg = T_arr[sel]; lg = logg_arr[sel]; fl = flux_arr[sel]
        Tu = np.unique(Tg)
        T_ = np.clip(T_, Tu.min(), Tu.max())
        Ti = np.clip(np.searchsorted(Tu, T_, side='right')-1, 0, len(Tu)-2)
        Tlo, Thi = Tu[Ti], Tu[Ti+1]
        tF = (T_-Tlo)/(Thi-Tlo) if Thi > Tlo else 0.0
        result = np.zeros(fl.shape[1])
        for Tv, tf in [(Tlo, 1-tF), (Thi, tF)]:
            if tf == 0: continue
            mT = sel & (T_arr == Tv)
            lv = logg_arr[mT]; fv = flux_arr[mT]
            if not len(lv): return None
            oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
            # Reject if logg is outside this T's local grid by more than 0.5 dex
            if logg_ < ls[0] - 0.5 or logg_ > ls[-1] + 0.5:
                return None
            gi = np.clip(np.searchsorted(ls, logg_, side='right')-1, 0, len(ls)-2)
            glo2, ghi2 = ls[gi], ls[gi+1]
            gF = np.clip((logg_-glo2)/(ghi2-glo2) if ghi2 > glo2 else 0.0, 0.0, 1.0)
            result += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
        return result

    s_lo = _tlogg(T, logg, MHvals[lo_i])
    s_hi = _tlogg(T, logg, MHvals[hi_i])
    if s_lo is None and s_hi is None: return None
    if s_lo is None: return s_hi
    if s_hi is None: return s_lo
    if lo_i == hi_i: return s_lo
    return (1-mf)*s_lo + mf*s_hi

def model_mods(T, logg, mh_idx):
    return _interp_cache(T, logg, mh_idx, wm, mT, mlogg, mMH, mflux, MH_vals)

def model_fire(T, logg, mh_idx):
    return _interp_cache(T, logg, mh_idx, wf, fT, flogg, fMH, fflux, MH_vals)

# ── AGN PL templates (per arm) ────────────────────────────────────────────────
def agn_mods(alpha_pl): return (wm / AGN_PIVOT) ** (-alpha_pl)
def agn_fire(alpha_pl): return (wf / AGN_PIVOT) ** (-alpha_pl)

# ── Velocity broadening (Gaussian in log-λ, separate pixel scales per arm) ───
from scipy.ndimage import gaussian_filter1d
_dloglam_m = np.log(wm[1]/wm[0])  # log-λ pixel scale of MODS cache
_dloglam_f = np.log(wf[1]/wf[0])  # log-λ pixel scale of FIRE cache

def broaden_m(flux, sigma_kms):
    if sigma_kms < 1.0: return flux
    return gaussian_filter1d(flux, sigma=(sigma_kms*1e3/3e8)/_dloglam_m)

def broaden_f(flux, sigma_kms):
    if sigma_kms < 1.0: return flux
    return gaussian_filter1d(flux, sigma=(sigma_kms*1e3/3e8)/_dloglam_f)

# ── Load spectra ──────────────────────────────────────────────────────────────
def load_pypeit(fname):
    with fits.open(os.path.join(DATA_DIR, fname)) as h:
        s = h['SPECTRUM'].data
        w_obs = s['wave'].astype(float)
        flux  = s['flux'].astype(float)
        ivar  = s['ivar'].astype(float)
        msk   = s['mask'].astype(int)
    good = (msk==1) & (ivar>0) & np.isfinite(flux)
    sigma = np.where(good, 1.0/np.sqrt(np.where(ivar>0, ivar, np.inf)), np.nan)
    return w_obs/(1+Z_EGG), flux, sigma, good

print("Loading spectra …", flush=True)
w_mr, f_mr, s_mr, g_mr = load_pypeit('J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
w_fi, f_fi, s_fi, g_fi = load_pypeit('J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

# Interpolate onto model grids
def build_obs_grid(w_src, f_src, s_src, g_src, w_grid, lo, hi):
    ok = g_src & (w_src >= lo) & (w_src <= hi)
    fit_grd = (w_grid >= lo) & (w_grid <= hi)
    f_out = np.full(len(w_grid), np.nan)
    e_out = np.full(len(w_grid), np.nan)
    if ok.sum() > 0:
        f_out[fit_grd] = np.interp(w_grid[fit_grd], w_src[ok], f_src[ok],
                                    left=np.nan, right=np.nan)
        e_out[fit_grd] = np.interp(w_grid[fit_grd], w_src[ok], s_src[ok],
                                    left=np.nan, right=np.nan)
    e_out = np.sqrt(e_out**2 + (F_SYS*np.abs(f_out))**2)
    return f_out, e_out, fit_grd

f_mods, e_mods, grd_mods = build_obs_grid(w_mr, f_mr, s_mr, g_mr, wm, MODSR_LO, MODSR_HI)
f_fire, e_fire, grd_fire = build_obs_grid(w_fi, f_fi, s_fi, g_fi, wf, FIRE_LO,  FIRE_HI)

# ── Emission line masks ───────────────────────────────────────────────────────
def make_line_mask(wave_arr, lines):
    mask = np.ones(len(wave_arr), dtype=bool)
    for cen, hw in lines:
        mask &= ~((wave_arr >= cen-hw) & (wave_arr <= cen+hw))
    return mask

MODS_LINES = [
    (6563, 110), (4861, 100), (4340, 50), (4102, 30),
    (5007, 30), (4959, 20), (6583, 25), (6548, 20),
    (6716, 20), (6731, 20), (3727, 30), (3869, 20),
    (5876, 25), (4686, 25),
]
FIRE_LINES = [
    (10830, 250), (10938, 200), (12820, 250),
    (9015, 150), (9229, 150), (9546, 150), (10049, 200),
    (9069, 40), (9532, 40), (6563, 110),  # Hα in FIRE red edge
]

lmask_mods = make_line_mask(wm, MODS_LINES)
lmask_fire = make_line_mask(wf, FIRE_LINES)

BASE_MODS = grd_mods & lmask_mods
BASE_FIRE = grd_fire & lmask_fire

# Weights: CaT upweighted in both arms (same feature)
CAT_M = (wm >= 8400) & (wm <= 8750)
CAT_F = (wf >= 8400) & (wf <= 8750)
H2O_F = (wf >= 12800) & (wf <= 14500)
FEH_F = (wf >= 9700)  & (wf <= 10100)
wt_mods = np.ones(len(wm)); wt_mods[CAT_M & BASE_MODS] = 5.0
wt_fire = np.ones(len(wf))
wt_fire[CAT_F & BASE_FIRE] = 5.0
# No H2O upweight — The Egg does not show water absorption (unlike z~2.3 LRDs)
wt_fire[FEH_F & BASE_FIRE] = 2.0

n_mods = int((BASE_MODS & np.isfinite(f_mods) & np.isfinite(e_mods) & (e_mods>0)).sum())
n_fire = int((BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)).sum())
print(f"MODS R fit pixels: {n_mods} | FIRE fit pixels: {n_fire}", flush=True)

# ── Log-likelihood ────────────────────────────────────────────────────────────
def make_logl(ext_fn):
    def logl(theta):
        T_hot, logg_hot, T_cool, logg_cool, a_v, mh_h, mh_c, alpha_pl, sigma_v = theta
        if T_hot <= T_cool: return -1e300

        # MODS templates (hot + AGN; cool negligible <7400 Å)
        fm_m = model_mods(T_hot, logg_hot, mh_h)
        if fm_m is None: return -1e300
        ext_m = ext_fn(wm, a_v)
        fhot_m = broaden_m(fm_m * ext_m, sigma_v)
        fagn_m = agn_mods(alpha_pl) * ext_m   # AGN PL not broadened

        # FIRE templates (hot + AGN + cool)
        fm_f = model_fire(T_hot, logg_hot, mh_h)
        fc_f = model_fire(T_cool, logg_cool, mh_c)
        if fm_f is None or fc_f is None: return -1e300
        ext_f = ext_fn(wf, a_v)
        fhot_f  = broaden_f(fm_f * ext_f, sigma_v)
        fagn_f  = agn_fire(alpha_pl) * ext_f
        fcool_f = broaden_f(fc_f * ext_f, sigma_v)

        # Good pixel masks per arm
        ok_m = (BASE_MODS & np.isfinite(f_mods) & np.isfinite(e_mods) & (e_mods>0)
                & np.isfinite(fhot_m) & (fhot_m>0) & np.isfinite(fagn_m) & (fagn_m>0))
        ok_f = (BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)
                & np.isfinite(fhot_f) & (fhot_f>0) & np.isfinite(fagn_f) & (fagn_f>0)
                & np.isfinite(fcool_f) & (fcool_f>0))
        if ok_m.sum() < 10 or ok_f.sum() < 10: return -1e300

        # Shared AGN amplitude: [hot_MODS, hot_FIRE, cool_FIRE, AGN]
        # AGN column has nonzero entries in both MODS and FIRE rows, enforcing
        # one physical AGN normalisation and removing the inter-arm discontinuity.
        nm, nf = ok_m.sum(), ok_f.sum()
        bw_m = wt_mods[ok_m]; bw_f = wt_fire[ok_f]
        inv_e_m = bw_m / e_mods[ok_m]**2
        inv_e_f = bw_f / e_fire[ok_f]**2

        A = np.zeros((nm+nf, 4))
        A[:nm, 0] = fhot_m[ok_m]  * np.sqrt(inv_e_m)   # hot MODS
        A[nm:, 1] = fhot_f[ok_f]  * np.sqrt(inv_e_f)   # hot FIRE
        A[nm:, 2] = fcool_f[ok_f] * np.sqrt(inv_e_f)   # cool FIRE
        A[:nm, 3] = fagn_m[ok_m]  * np.sqrt(inv_e_m)   # AGN MODS (shared)
        A[nm:, 3] = fagn_f[ok_f]  * np.sqrt(inv_e_f)   # AGN FIRE (shared)

        b = np.concatenate([f_mods[ok_m]*np.sqrt(inv_e_m),
                            f_fire[ok_f]*np.sqrt(inv_e_f)])
        amps, _ = nnls(A, b)
        if amps[0] + amps[3] == 0 or amps[1] + amps[3] == 0: return -1e300

        res_m = f_mods[ok_m] - amps[0]*fhot_m[ok_m] - amps[3]*fagn_m[ok_m]
        res_f = f_fire[ok_f] - amps[1]*fhot_f[ok_f] - amps[3]*fagn_f[ok_f] - amps[2]*fcool_f[ok_f]
        return (-0.5 * np.sum(bw_m * (res_m/e_mods[ok_m])**2)
                -0.5 * np.sum(bw_f * (res_f/e_fire[ok_f])**2))
    return logl

def prior_tf(u):
    theta = np.zeros(9)
    theta[0] = 3500 + 4000*u[0]    # T_hot: [3500, 7500] K
    theta[1] = -4.0  + 5.5*u[1]    # logg_hot: [-4.0, 1.5]
    theta[2] = 1500 + 2000*u[2]    # T_cool: [1500, 3500] K
    theta[3] = -4.5  + 5.5*u[3]    # logg_cool: [-4.5, 1.0]
    theta[4] =  8.0*u[4]            # A_V: [0, 8]
    theta[5] = -0.49 + 2.98*u[5]   # mh_hot
    theta[6] = -0.49 + 2.98*u[6]   # mh_cool
    theta[7] =  0.0  + 2.0*u[7]    # alpha_PL: [0, 2]
    theta[8] = 600.0*u[8]           # sigma_v: [0, 600] km/s (Liu+2026 finds ~123 km/s)
    return theta

def run_fit(ext_fn, label, out_stem):
    logl = make_logl(ext_fn)
    print(f'\n{"="*60}', flush=True)
    print(f'Running: {label}  (ndim=9)', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, 9, nlive=600, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=300, nlive_batch=600,
        wt_kwargs={'pfrac': 1.0}, n_effective=8000, print_progress=True)
    print(f'  Elapsed: {time.time()-t0:.0f} s', flush=True)
    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)
    labels = ['T_hot','logg_hot','T_cool','logg_cool','A_V','mh_hot','mh_cool','alpha_PL','sigma_v']
    print(f'  log Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}')
    for i, lbl in enumerate(labels):
        p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
        print(f'  {lbl:12s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}')
    np.save(os.path.join(SCRIPT_DIR, f'{out_stem}.npy'), samp)
    with open(os.path.join(SCRIPT_DIR, f'{out_stem}.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    print(f'  Saved {out_stem}')
    return samp, res

if __name__ == '__main__':
    samp_calz, res_calz = run_fit(calzetti_ext, 'BB MODS R+FIRE — Calzetti', 'egg_chain_bb_calzetti')
    samp_smc,  res_smc  = run_fit(smc_ext,      'BB MODS R+FIRE — SMC',      'egg_chain_bb_smc')

    n = n_mods + n_fire; k = 9
    lines = [
        f"The Egg — Broadband MODS Red + FIRE Fit",
        f"MODS Red: {MODSR_LO:.0f}–{MODSR_HI:.0f} Å ({n_mods} pix)",
        f"FIRE:     {FIRE_LO:.0f}–{FIRE_HI:.0f} Å ({n_fire} pix)",
        f"Model: 1-comp TLUSTY hot + AGN PL (shared NNLS amp) + cool TLUSTY (FIRE only)",
        f"NOTE: MODS (Feb 24) and FIRE (Dec 22) separated by 64 days",
        f"",
    ]
    for samp, res, lbl in [(samp_calz, res_calz, 'Calzetti R_V=4.05'),
                            (samp_smc,  res_smc,  'SMC Gordon+2003')]:
        ll = res.logl.max(); lz = res.logz[-1]; lze = res.logzerr[-1]
        bic = k*np.log(n) - 2*ll
        lines += [
            f"{'─'*50}", f"Extinction: {lbl}",
            f"  log Z = {lz:.2f} ± {lze:.2f}", f"  BIC = {bic:.2f}",
        ]
        lbls = ['T_hot','logg_hot','T_cool','logg_cool','A_V','mh_hot','mh_cool','alpha_PL','sigma_v']
        for i, lb2 in enumerate(lbls):
            p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
            lines.append(f"  {lb2:12s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}")
        lines.append(f"  f(logg_hot < 0): {(samp[:,1]<0).mean():.1%}")
        lines.append(f"  f(logg_hot < -1.0): {(samp[:,1]<-1.0).mean():.1%}")
        lines.append("")
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(SCRIPT_DIR, 'egg_bic_broadband.txt'), 'w') as fh:
        fh.write(report)
    print("Done.")
