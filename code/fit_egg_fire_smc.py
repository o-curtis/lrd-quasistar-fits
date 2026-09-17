#!/usr/bin/env python3
"""
FIRE-only 2-component fit of The Egg (J1025+1402) using SMC extinction.
Uses the same hires cache (egg_hires_cache.npz, R=6001) as the broadband fit.

MCMC params (9): T_hot, logg_hot, T_cool, logg_cool, A_V, mh_hot, mh_cool, alpha_PL, sigma_v
NNLS amps (3): [A_hot_FIRE, A_cool_FIRE, A_AGN_FIRE]

Fit window: 7400–15500 Å rest-frame.
Key diagnostics: Ca II Triplet 8498/8542/8662 (log g), FeH 9700–10100.
No H2O upweight — The Egg does not show water absorption.

Outputs:
  egg_chain_fire_smc.npy / .pkl
  egg_bic_fire_smc.txt
"""
import os, time, pickle
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'egg_data',
                           'Re_ J1025+1402 LBT _ Magellan spectra')
FIRE_CACHE = os.path.join(SCRIPT_DIR, 'egg_hires_cache.npz')

Z_EGG     = 0.1007
F_SYS     = 0.05
AGN_PIVOT = 5000.0
FIRE_LO   = 7400.0
FIRE_HI   = 15500.0

# ── Load FIRE model cache ─────────────────────────────────────────────────────
print("Loading FIRE model cache …", flush=True)
fc    = np.load(FIRE_CACHE, allow_pickle=True)
wf    = fc['wave'].astype(float)
fflux = fc['flux']
fmeta = list(fc['meta'])
fT    = np.array([m['T']    for m in fmeta], dtype=float)
flogg = np.array([m['logg'] for m in fmeta], dtype=float)
fMH   = np.array([m['MH']   for m in fmeta], dtype=int)
MH_vals = sorted(set(fMH))
print(f"  {len(fmeta)} models, grid {wf.min():.0f}–{wf.max():.0f} Å", flush=True)

# ── SMC extinction (Gordon+2003) ──────────────────────────────────────────────
_sx  = np.array([0.29,0.45,0.61,0.80,1.11,1.43,1.82,2.27,2.70,3.33,3.70,4.55,5.88])
_sr  = np.array([0.112,0.420,0.427,0.453,0.570,0.741,1.000,1.412,1.785,2.460,2.609,2.770,3.554])
_sfn = interp1d(_sx, _sr, kind='linear', bounds_error=False, fill_value='extrapolate')
def smc_ext(wa, av):
    return 10**(-0.4 * np.clip(_sfn(1./(wa*1e-4)), 0, None) * av)

# ── TLUSTY interpolator ───────────────────────────────────────────────────────
def _interp(T, logg, mh_idx, Ta, la, Ma, Fa, MHv):
    mh_idx = np.clip(mh_idx, 0., len(MHv)-1.)
    lo_i = int(np.floor(mh_idx)); hi_i = min(lo_i+1, len(MHv)-1)
    mf = mh_idx - lo_i
    def _tl(T_, l_, M_):
        sel = Ma == M_
        if not sel.any(): return None
        Tg = Ta[sel]; lg = la[sel]; fl = Fa[sel]
        Tu = np.unique(Tg); gu = np.unique(lg)
        T_ = np.clip(T_, Tu.min(), Tu.max())
        l_ = np.clip(l_, gu.min(), gu.max())
        Ti = np.clip(np.searchsorted(Tu, T_, side='right')-1, 0, len(Tu)-2)
        Tlo, Thi = Tu[Ti], Tu[Ti+1]
        tF = (T_-Tlo)/(Thi-Tlo) if Thi > Tlo else 0.
        res = np.zeros(fl.shape[1])
        for Tv, tf in [(Tlo, 1-tF), (Thi, tF)]:
            if tf == 0: continue
            mt = sel & (Ta == Tv); lv = la[mt]; fv = Fa[mt]
            if not len(lv): return None
            oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
            gi = np.clip(np.searchsorted(ls, l_, side='right')-1, 0, len(ls)-2)
            g0, g1 = ls[gi], ls[gi+1]
            gF = (l_-g0)/(g1-g0) if g1 > g0 else 0.
            res += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
        return res
    sl = _tl(T, logg, MHv[lo_i]); sh = _tl(T, logg, MHv[hi_i])
    if sl is None and sh is None: return None
    if sl is None: return sh
    if sh is None: return sl
    if lo_i == hi_i: return sl
    return (1-mf)*sl + mf*sh

def model_fire(T, logg, mh):
    return _interp(T, logg, mh, fT, flogg, fMH, fflux, MH_vals)

# ── Velocity broadening ───────────────────────────────────────────────────────
_dloglam_f = np.log(wf[1]/wf[0])
def broaden_f(flux, sigma_kms):
    if sigma_kms < 1.0: return flux
    return gaussian_filter1d(flux, sigma=(sigma_kms*1e3/3e8)/_dloglam_f)

# ── Load FIRE spectrum ────────────────────────────────────────────────────────
print("Loading FIRE spectrum …", flush=True)
fire_fname = 'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits'
with fits.open(os.path.join(DATA_DIR, fire_fname)) as h:
    s = h['SPECTRUM'].data
    w_obs = s['wave'].astype(float)
    flux  = s['flux'].astype(float)
    ivar  = s['ivar'].astype(float)
    msk   = s['mask'].astype(int)
good  = (msk==1) & (ivar>0) & np.isfinite(flux)
sigma = np.where(good, 1./np.sqrt(np.where(ivar>0, ivar, np.inf)), np.nan)
w_rest = w_obs / (1+Z_EGG)
print(f"  {good.sum()} good pixels, rest {w_rest[good].min():.0f}–{w_rest[good].max():.0f} Å",
      flush=True)

# Interpolate onto model grid
fit_grd = (wf >= FIRE_LO) & (wf <= FIRE_HI)
ok_data = good & (w_rest >= FIRE_LO) & (w_rest <= FIRE_HI)
f_fire = np.full(len(wf), np.nan)
e_fire = np.full(len(wf), np.nan)
if ok_data.sum() > 0:
    f_fire[fit_grd] = np.interp(wf[fit_grd], w_rest[ok_data], flux[ok_data],
                                 left=np.nan, right=np.nan)
    e_fire[fit_grd] = np.interp(wf[fit_grd], w_rest[ok_data], sigma[ok_data],
                                 left=np.nan, right=np.nan)
e_fire = np.sqrt(e_fire**2 + (F_SYS*np.abs(f_fire))**2)

# ── Emission line masks ───────────────────────────────────────────────────────
FIRE_LINES = [
    (10830, 250), (10938, 200), (12820, 250),
    (9015, 150), (9229, 150), (9546, 150), (10049, 200),
    (9069, 40), (9532, 40), (6563, 110),
]
lmask = np.ones(len(wf), dtype=bool)
for cen, hw in FIRE_LINES:
    lmask &= ~((wf >= cen-hw) & (wf <= cen+hw))
BASE_FIRE = fit_grd & lmask

# Per-pixel weights — no H2O upweight (The Egg has no water absorption)
CAT_F = (wf >= 8400) & (wf <= 8750)
FEH_F = (wf >= 9700)  & (wf <= 10100)
wt = np.ones(len(wf))
wt[CAT_F & BASE_FIRE] = 5.0
wt[FEH_F & BASE_FIRE] = 2.0

n_fire = int((BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)).sum())
print(f"FIRE fit pixels: {n_fire}", flush=True)

# ── Log-likelihood ────────────────────────────────────────────────────────────
def logl(theta):
    T_hot, logg_hot, T_cool, logg_cool, a_v, mh_h, mh_c, alpha_pl, sigma_v = theta
    if T_hot <= T_cool: return -1e300

    fh = model_fire(T_hot, logg_hot, mh_h)
    fc = model_fire(T_cool, logg_cool, mh_c)
    if fh is None or fc is None: return -1e300

    ext  = smc_ext(wf, a_v)
    fhot  = broaden_f(fh * ext, sigma_v)
    fcool = broaden_f(fc * ext, sigma_v)
    fagn  = (wf / AGN_PIVOT)**(-alpha_pl) * ext

    ok = (BASE_FIRE & np.isfinite(f_fire) & np.isfinite(e_fire) & (e_fire>0)
          & np.isfinite(fhot) & (fhot>0) & np.isfinite(fcool) & (fcool>0)
          & np.isfinite(fagn) & (fagn>0))
    if ok.sum() < 10: return -1e300

    bw = wt[ok]; ie = bw / e_fire[ok]**2
    A = np.zeros((ok.sum(), 3))
    A[:, 0] = fhot[ok]  * np.sqrt(ie)
    A[:, 1] = fcool[ok] * np.sqrt(ie)
    A[:, 2] = fagn[ok]  * np.sqrt(ie)
    b = f_fire[ok] * np.sqrt(ie)
    amps, _ = nnls(A, b)
    if amps.sum() == 0: return -1e300

    res = f_fire[ok] - amps[0]*fhot[ok] - amps[1]*fcool[ok] - amps[2]*fagn[ok]
    return -0.5 * np.sum(bw * (res/e_fire[ok])**2)

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
    theta[8] = 600.0*u[8]           # sigma_v: [0, 600] km/s
    return theta

if __name__ == '__main__':
    print(f'\n{"="*60}', flush=True)
    print('Running: FIRE-only 2-comp SMC  (ndim=9)', flush=True)
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

    out_stem = 'egg_chain_fire_smc'
    np.save(os.path.join(SCRIPT_DIR, f'{out_stem}.npy'), samp)
    with open(os.path.join(SCRIPT_DIR, f'{out_stem}.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    labels = ['T_hot','logg_hot','T_cool','logg_cool','A_V','mh_hot','mh_cool','alpha_PL','sigma_v']
    n = n_fire; k = 9
    ll_max = res.logl.max(); lz = res.logz[-1]; lze = res.logzerr[-1]
    bic = k*np.log(n) - 2*ll_max

    lines = [
        f"The Egg — FIRE-only 2-comp SMC Fit",
        f"FIRE: {FIRE_LO:.0f}–{FIRE_HI:.0f} Å ({n_fire} pix)",
        f"Model: 2-comp TLUSTY hot+cool + AGN PL | SMC extinction | ndim=9",
        f"",
        f"  log Z = {lz:.2f} ± {lze:.2f}",
        f"  BIC = {bic:.2f}",
        f"",
    ]
    for i, lbl in enumerate(labels):
        p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
        lines.append(f"  {lbl:12s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}")
    lines.append(f"  f(logg_hot < 0): {(samp[:,1]<0).mean():.1%}")
    lines.append(f"  f(logg_hot < -1.0): {(samp[:,1]<-1.0).mean():.1%}")

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(SCRIPT_DIR, 'egg_bic_fire_smc.txt'), 'w') as fh:
        fh.write(report)
    print("Done.")
