#!/usr/bin/env python3
"""
1-component broadband fit of The Egg (J1025+1402) using MODS Red + FIRE.
SMC extinction only (Calzetti already ruled out by ΔBIC=2264 from 2-comp fit).

Model: 1-comp TLUSTY hot atmosphere + AGN power law
NNLS amplitudes: [A_hot_MODS, A_hot_FIRE, A_AGN_shared]  (3 amplitudes, shared AGN)
MCMC params: T_hot, logg_hot, A_V, mh_hot, alpha_PL  (5 params = ndim)

BIC comparison with 2-comp SMC (8 params, BIC=12976) gives Δk=3 extra params,
penalty = 3*ln(7438) = 26.7. If 1-comp BIC < 12976+26.7 = 13003, prefer 1-comp.

Outputs:
  egg_chain_bb_smc_1comp.npy / .pkl
  egg_bic_broadband_1comp.txt
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
MODSR_LO, MODSR_HI = 3900.0, 7400.0
FIRE_LO,  FIRE_HI  = 7400.0, 15500.0

# ── Load model caches ─────────────────────────────────────────────────────────
print("Loading caches …", flush=True)
mc    = np.load(MODS_CACHE, allow_pickle=True)
wm    = mc['wave'].astype(float); mflux = mc['flux']; mmeta = list(mc['meta'])
mT    = np.array([m['T']    for m in mmeta], dtype=float)
mlogg = np.array([m['logg'] for m in mmeta], dtype=float)
mMH   = np.array([m['MH']   for m in mmeta], dtype=int)
MH_vals = sorted(set(mMH))

fc    = np.load(FIRE_CACHE, allow_pickle=True)
wf    = fc['wave'].astype(float); fflux = fc['flux']; fmeta = list(fc['meta'])
fT    = np.array([m['T']    for m in fmeta], dtype=float)
flogg = np.array([m['logg'] for m in fmeta], dtype=float)
fMH   = np.array([m['MH']   for m in fmeta], dtype=int)
print(f"  MODS: {len(mmeta)} models | FIRE: {len(fmeta)} models", flush=True)

# ── SMC extinction ────────────────────────────────────────────────────────────
_smc_x   = np.array([0.29,0.45,0.61,0.80,1.11,1.43,1.82,2.27,2.70,3.33,3.70,4.55,5.88])
_smc_rav = np.array([0.112,0.420,0.427,0.453,0.570,0.741,1.000,1.412,1.785,2.460,2.609,2.770,3.554])
_smc_fn  = interp1d(_smc_x, _smc_rav, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
def smc_ext(wave_aa, a_v):
    return 10**(-0.4 * np.clip(_smc_fn(1.0/(wave_aa*1e-4)), 0, None) * a_v)

# ── Model interpolator ─────────────────────────────────────────────────────────
def _interp_cache(T, logg, mh_idx, T_arr, logg_arr, MH_arr, flux_arr, MHvals):
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
            if logg_ < ls[0] - 0.5 or logg_ > ls[-1] + 0.5: return None
            gi = np.clip(np.searchsorted(ls, logg_, side='right')-1, 0, len(ls)-2)
            g0, g1 = ls[gi], ls[gi+1]
            gF = np.clip((logg_-g0)/(g1-g0) if g1>g0 else 0., 0., 1.)
            result += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
        return result
    sl = _tlogg(T, logg, MHvals[lo_i]); sh = _tlogg(T, logg, MHvals[hi_i])
    if sl is None and sh is None: return None
    if sl is None: return sh
    if sh is None: return sl
    if lo_i == hi_i: return sl
    return (1-mf)*sl + mf*sh

def model_mods(T, logg, mh):
    return _interp_cache(T, logg, mh, mT, mlogg, mMH, mflux, MH_vals)
def model_fire(T, logg, mh):
    return _interp_cache(T, logg, mh, fT, flogg, fMH, fflux, MH_vals)

# ── Velocity broadening ───────────────────────────────────────────────────────
from scipy.ndimage import gaussian_filter1d
_dloglam_m = np.log(wm[1]/wm[0])
_dloglam_f = np.log(wf[1]/wf[0])
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
        w_obs = s['wave'].astype(float); flux = s['flux'].astype(float)
        ivar  = s['ivar'].astype(float); msk  = s['mask'].astype(int)
    good  = (msk==1) & (ivar>0) & np.isfinite(flux)
    sigma = np.where(good, 1./np.sqrt(np.where(ivar>0, ivar, np.inf)), np.nan)
    return w_obs/(1+Z_EGG), flux, sigma, good

print("Loading spectra …", flush=True)
w_mr, f_mr, s_mr, g_mr = load_pypeit('J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
w_fi, f_fi, s_fi, g_fi = load_pypeit('J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

def build_obs_grid(w_src, f_src, s_src, g_src, w_grid, lo, hi):
    ok = g_src & (w_src >= lo) & (w_src <= hi)
    fg = (w_grid >= lo) & (w_grid <= hi)
    f_out = np.full(len(w_grid), np.nan); e_out = np.full(len(w_grid), np.nan)
    if ok.sum() > 0:
        f_out[fg] = np.interp(w_grid[fg], w_src[ok], f_src[ok], left=np.nan, right=np.nan)
        e_out[fg] = np.interp(w_grid[fg], w_src[ok], s_src[ok], left=np.nan, right=np.nan)
    e_out = np.sqrt(e_out**2 + (F_SYS*np.abs(f_out))**2)
    return f_out, e_out, fg

f_mods, e_mods, grd_mods = build_obs_grid(w_mr, f_mr, s_mr, g_mr, wm, MODSR_LO, MODSR_HI)
f_fire, e_fire, grd_fire = build_obs_grid(w_fi, f_fi, s_fi, g_fi, wf, FIRE_LO,  FIRE_HI)

# ── Emission line masks (same as 2-comp fit) ──────────────────────────────────
def make_mask(warr, lines):
    m = np.ones(len(warr), dtype=bool)
    for c, hw in lines: m &= ~((warr >= c-hw) & (warr <= c+hw))
    return m

MODS_LINES = [(6563,110),(4861,100),(4340,50),(4102,30),(5007,30),(4959,20),
              (6583,25),(6548,20),(6716,20),(6731,20),(3727,30),(3869,20),(5876,25),(4686,25)]
FIRE_LINES = [(10830,250),(10938,200),(12820,250),(9015,150),(9229,150),
              (9546,150),(10049,200),(9069,40),(9532,40),(6563,110)]

BASE_MODS = grd_mods & make_mask(wm, MODS_LINES)
BASE_FIRE = grd_fire & make_mask(wf, FIRE_LINES)

# Weights: CaT upweighted 5×, FeH 2×  (no H2O — cool component absent in 1-comp)
CAT_M = (wm >= 8400) & (wm <= 8750); CAT_F = (wf >= 8400) & (wf <= 8750)
FEH_F = (wf >= 9700) & (wf <= 10100)
wt_mods = np.ones(len(wm)); wt_mods[CAT_M & BASE_MODS] = 5.0
wt_fire = np.ones(len(wf)); wt_fire[CAT_F & BASE_FIRE] = 5.0
wt_fire[FEH_F & BASE_FIRE] = 2.0

n_mods = int((BASE_MODS & np.isfinite(f_mods) & np.isfinite(e_mods) & (e_mods>0)).sum())
n_fire = int((BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)).sum())
print(f"MODS R fit pixels: {n_mods} | FIRE fit pixels: {n_fire}", flush=True)

# ── Log-likelihood (1-comp, SMC only) ────────────────────────────────────────
def logl(theta):
    T_hot, logg_hot, a_v, mh_h, alpha_pl, sigma_v = theta

    fm_m = model_mods(T_hot, logg_hot, mh_h)
    if fm_m is None: return -1e300
    ext_m  = smc_ext(wm, a_v)
    fhot_m = broaden_m(fm_m * ext_m, sigma_v)
    fagn_m = (wm / AGN_PIVOT)**(-alpha_pl) * ext_m

    fm_f = model_fire(T_hot, logg_hot, mh_h)
    if fm_f is None: return -1e300
    ext_f  = smc_ext(wf, a_v)
    fhot_f = broaden_f(fm_f * ext_f, sigma_v)
    fagn_f = (wf / AGN_PIVOT)**(-alpha_pl) * ext_f

    ok_m = (BASE_MODS & np.isfinite(f_mods) & np.isfinite(e_mods) & (e_mods>0)
            & np.isfinite(fhot_m) & (fhot_m>0) & np.isfinite(fagn_m) & (fagn_m>0))
    ok_f = (BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)
            & np.isfinite(fhot_f) & (fhot_f>0) & np.isfinite(fagn_f) & (fagn_f>0))
    if ok_m.sum() < 10 or ok_f.sum() < 10: return -1e300

    nm, nf = ok_m.sum(), ok_f.sum()
    bw_m = wt_mods[ok_m]; bw_f = wt_fire[ok_f]
    ie_m = bw_m / e_mods[ok_m]**2; ie_f = bw_f / e_fire[ok_f]**2

    # 3 NNLS amplitudes: [hot_MODS, hot_FIRE, AGN_shared]
    # AGN column has nonzero entries in both arms — enforces one physical AGN
    # normalisation, removing the inter-arm amplitude discontinuity.
    A = np.zeros((nm+nf, 3))
    A[:nm, 0] = fhot_m[ok_m] * np.sqrt(ie_m)   # hot MODS
    A[nm:, 1] = fhot_f[ok_f] * np.sqrt(ie_f)   # hot FIRE
    A[:nm, 2] = fagn_m[ok_m] * np.sqrt(ie_m)   # AGN MODS (shared)
    A[nm:, 2] = fagn_f[ok_f] * np.sqrt(ie_f)   # AGN FIRE (shared)
    b = np.concatenate([f_mods[ok_m]*np.sqrt(ie_m), f_fire[ok_f]*np.sqrt(ie_f)])
    amps, _ = nnls(A, b)
    if amps[0] + amps[2] == 0 or amps[1] + amps[2] == 0: return -1e300

    res_m = f_mods[ok_m] - amps[0]*fhot_m[ok_m] - amps[2]*fagn_m[ok_m]
    res_f = f_fire[ok_f] - amps[1]*fhot_f[ok_f] - amps[2]*fagn_f[ok_f]
    return (-0.5*np.sum(bw_m*(res_m/e_mods[ok_m])**2)
            -0.5*np.sum(bw_f*(res_f/e_fire[ok_f])**2))

def prior_tf(u):
    theta = np.zeros(6)
    theta[0] = 3500 + 4000*u[0]    # T_hot: [3500, 7500] K
    theta[1] = -4.0  + 5.5*u[1]    # logg_hot: [-4.0, 1.5]
    theta[2] =  8.0*u[2]            # A_V: [0, 8]
    theta[3] = -0.49 + 2.98*u[3]   # mh_hot
    theta[4] =  0.0  + 2.0*u[4]    # alpha_PL: [0, 2]
    theta[5] = 600.0*u[5]           # sigma_v: [0, 600] km/s
    return theta

if __name__ == '__main__':
    print(f'\n{"="*60}', flush=True)
    print('Running: 1-comp broadband SMC  (ndim=6)', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, 6, nlive=600, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=300, nlive_batch=600,
        wt_kwargs={'pfrac': 1.0}, n_effective=8000, print_progress=True)
    print(f'  Elapsed: {time.time()-t0:.0f} s', flush=True)

    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)
    out_stem = 'egg_chain_bb_smc_1comp'
    np.save(os.path.join(SCRIPT_DIR, f'{out_stem}.npy'), samp)
    with open(os.path.join(SCRIPT_DIR, f'{out_stem}.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    labels = ['T_hot','logg_hot','A_V','mh_hot','alpha_PL','sigma_v']
    n = n_mods + n_fire; k = 6
    ll_max = res.logl.max(); lz = res.logz[-1]; lze = res.logzerr[-1]
    bic_1comp = k*np.log(n) - 2*ll_max

    # Load 2-comp SMC for comparison
    bic_2comp = None
    txt2 = os.path.join(SCRIPT_DIR, 'egg_bic_broadband.txt')
    if os.path.exists(txt2):
        for line in open(txt2):
            if 'BIC' in line and 'SMC' not in line and bic_2comp is None:
                try: bic_2comp = float(line.split('=')[1].strip().split()[0])
                except: pass
        # more robust: just search for the SMC BIC line
        content = open(txt2).read()
        import re
        m = re.search(r'SMC.*?BIC\s*=\s*([0-9.]+)', content, re.DOTALL)
        if m: bic_2comp = float(m.group(1))

    lines = [
        f"The Egg — Broadband 1-comp SMC Fit",
        f"Fit: MODS Red {MODSR_LO:.0f}–{MODSR_HI:.0f} Å ({n_mods} pix) + "
        f"FIRE {FIRE_LO:.0f}–{FIRE_HI:.0f} Å ({n_fire} pix)",
        f"Model: 1-comp TLUSTY hot + AGN PL | SMC extinction | ndim=6 (incl. sigma_v)",
        f"",
        f"  log Z = {lz:.2f} ± {lze:.2f}",
        f"  BIC (1-comp SMC) = {bic_1comp:.2f}",
    ]
    if bic_2comp is not None:
        dbic = bic_1comp - bic_2comp
        lines.append(f"  BIC (2-comp SMC) = {bic_2comp:.2f}  [from egg_bic_broadband.txt]")
        lines.append(f"  ΔBIC (1-comp − 2-comp) = {dbic:+.2f}  "
                     f"({'1-comp preferred' if dbic < 0 else '2-comp preferred'})")
    lines.append("")
    for i, lbl in enumerate(labels):
        p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
        lines.append(f"  {lbl:12s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}")
    lines += [
        f"  f(logg_hot < 0): {(samp[:,1]<0).mean():.1%}",
        f"  f(logg_hot < -1.0): {(samp[:,1]<-1.0).mean():.1%}",
    ]
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(SCRIPT_DIR, 'egg_bic_broadband_1comp.txt'), 'w') as fh:
        fh.write(report)
    print("Done.")
