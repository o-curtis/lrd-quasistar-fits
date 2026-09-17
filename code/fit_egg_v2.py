#!/usr/bin/env python3
"""
Flat-prior dynesty fit for The Egg (J1025+1402, z=0.1007).

Direct adaptation of mcmc_fit_flatprior_dynesty.py used for the two LRDs.
Key differences from the LRD version:
  - Data: MODS Red (7000-7473 Å rest) + FIRE (7473-15500 Å rest) via PypeIt
  - Models: egg_hires_cache.npz (R=6000, Liu+2026 TLUSTY) smoothed to R=2000
    so native FIRE spectral features (TiO, CaT, H2O, FeH) enter the fit
  - Extra line masks: [S III] 9069, 9532 (nebular emission in The Egg)
  - Same FeH upweight (2x) added alongside H2O (3x)

Everything else is IDENTICAL to the LRD flat-prior analysis:
  - Same 2-comp TLUSTY model + NNLS amplitudes
  - Same Calzetti extinction (R_V=4.05)
  - Same H2O band upweight (x3)
  - Same flat priors and bounds
  - Same dynesty settings: nlive=1200, nlive_init=400, nlive_batch=1000, n_eff=8000
  - Same F_SYS=0.10
  - Same 7 free parameters: T_hot, logg_hot, T_cool, logg_cool, A_V, mh_hot, mh_cool

Also runs a 1-component fit (not in the LRD script, but needed for BIC comparison).

Outputs:
  egg_chain_1comp_v2.npy, egg_dynesty_1comp_v2.pkl
  egg_chain_2comp_v2.npy, egg_dynesty_2comp_v2.pkl
  egg_bic_v2.txt
"""
import os, time, pickle
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')

np.random.seed(99)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIRES_PATH = os.path.join(SCRIPT_DIR, 'egg_hires_cache.npz')
DATA_DIR   = os.path.join(SCRIPT_DIR, 'egg_data',
                           'Re_ J1025+1402 LBT _ Magellan spectra')

Z_EGG      = 0.1007
F_SYS      = 0.10     # identical to LRD analysis
FIRE_START = 7473.0   # Å rest: FIRE blue limit in this dataset

# ── 1. Native FIRE resolution grid, 7473-15500 Å (FIRE only, no MODS Red) ────
# R=6000 = native FIRE resolution. Fitting at full resolution maximises logg
# information from CaT pressure broadening and TiO band shapes.
# MODS Red excluded: its native R~2000 would require a separate LSF treatment.
# sigma_broad is a free parameter that absorbs any residual source broadening.
R_OUT  = 6000
FIT_LO = 7473.0   # FIRE blue limit — drop MODS Red segment entirely
FIT_HI = 15500.0
w_edges = np.exp(np.arange(np.log(FIT_LO), np.log(FIT_HI)+1./R_OUT, 1./R_OUT))
w_cen   = 0.5*(w_edges[:-1]+w_edges[1:])
N_PIX   = len(w_cen)
print(f'Output grid: R={R_OUT}, {N_PIX} pixels, {FIT_LO:.0f}-{FIT_HI:.0f} Å', flush=True)

# ── 2. Build R=6000 model array from hires cache (no additional smoothing) ────
print('Loading egg_hires_cache at native R=6000 ...', flush=True)
hires      = np.load(HIRES_PATH, allow_pickle=True)
w_hi       = hires['wave'].astype(float)
f_hi       = hires['flux'].astype(float)
meta_hi    = list(hires['meta'])

# No smoothing: models are already at R=6000, matching native FIRE resolution.
# sigma_broad free parameter handles any residual source/instrument broadening.
N_models   = f_hi.shape[0]
model_flux = np.zeros((N_models, N_PIX), dtype=np.float32)
for i in range(N_models):
    model_flux[i] = np.interp(w_cen, w_hi, f_hi[i], left=np.nan, right=np.nan)

meta_T    = np.array([m['T']    for m in meta_hi], dtype=float)
meta_logg = np.array([m['logg'] for m in meta_hi], dtype=float)
meta_MH   = np.array([m['MH']   for m in meta_hi], dtype=int)
MH_vals   = sorted(set(meta_MH))
print(f'  {N_models} models, T={int(meta_T.min())}-{int(meta_T.max())} K, '
      f'logg={meta_logg.min():.1f}-{meta_logg.max():.1f}, [M/H]={MH_vals}', flush=True)

# ── 3. Calzetti extinction (identical to LRD script) ─────────────────────────
def calzetti_k(wave_aa):
    w = wave_aa * 1e-4
    k = np.zeros_like(w)
    lo = w < 0.63
    k[lo] = 2.659*(-2.156 + 1.509/w[lo] - 0.198/w[lo]**2 + 0.011/w[lo]**3) + 4.05
    hi = ~lo & (w < 2.2)
    k[hi] = 2.659*(-1.857 + 1.040/w[hi]) + 4.05
    return np.clip(k, 0, None)

K  = calzetti_k(w_cen)
RV = 4.05
def ext_curve(av): return 10**(-0.4 * K * av / RV)

# ── 4. Model interpolation (identical logic to LRD script) ───────────────────
def _interp_tlogg(T, logg, MH):
    sel = meta_MH == MH
    if not sel.any(): return None
    Tg = meta_T[sel]; lg = meta_logg[sel]; fl = model_flux[sel]
    T_u = np.unique(Tg); g_u = np.unique(lg)
    T = np.clip(T, T_u.min(), T_u.max())
    Ti = np.clip(np.searchsorted(T_u, T, side='right')-1, 0, len(T_u)-2)
    Tlo, Thi = T_u[Ti], T_u[Ti+1]
    tF = (T-Tlo)/(Thi-Tlo) if Thi > Tlo else 0.0
    result = np.zeros(fl.shape[1])
    weight_sum = 0.0
    for Tv, tf in [(Tlo, 1-tF), (Thi, tF)]:
        if tf == 0: continue
        mT = sel & (meta_T == Tv)
        lv = meta_logg[mT]; fv = model_flux[mT]
        if not len(lv): continue
        # ── Coverage check: skip this T slice if logg is outside its range.
        #    Do NOT clip silently — clipping hid the extrapolation and caused
        #    unconstrained posteriors (every logg below the floor gave the same
        #    spectrum, so the sampler wandered arbitrarily to extreme values).
        if logg < lv.min() or logg > lv.max(): continue
        oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
        gi = np.clip(np.searchsorted(ls, logg, side='right')-1, 0, len(ls)-2)
        glo, ghi = ls[gi], ls[gi+1]
        gF = (logg-glo)/(ghi-glo) if ghi > glo else 0.0
        result += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
        weight_sum += tf
    if weight_sum == 0: return None   # logg out of range at all bracketing T
    return result / weight_sum        # renormalise if one T slice was skipped

def interp_spectrum(T, logg, mh_idx):
    mh_idx = np.clip(mh_idx, 0.0, len(MH_vals)-1.0)
    lo_i = int(np.floor(mh_idx)); hi_i = min(lo_i+1, len(MH_vals)-1)
    mf = mh_idx - lo_i
    s_lo = _interp_tlogg(T, logg, MH_vals[lo_i])
    s_hi = _interp_tlogg(T, logg, MH_vals[hi_i])
    if s_lo is None and s_hi is None: return None
    if s_lo is None: return s_hi
    if s_hi is None: return s_lo
    if lo_i == hi_i: return s_lo
    return (1-mf)*s_lo + mf*s_hi

# ── 5. Load PypeIt spectra ────────────────────────────────────────────────────
def load_pypeit(fname):
    fpath = os.path.join(DATA_DIR, fname)
    with fits.open(fpath) as h:
        s     = h['SPECTRUM'].data
        w_obs = s['wave'].astype(float)
        flux  = s['flux'].astype(float)
        ivar  = s['ivar'].astype(float)
        mask  = s['mask'].astype(int)
    good  = (mask == 1) & (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
    sigma = np.where(good, 1./np.sqrt(np.where(ivar > 0, ivar, np.inf)), np.nan)
    w_rest = w_obs / (1. + Z_EGG)
    return w_rest, flux, sigma, good

print('Loading FIRE ...', flush=True)
w_fire, f_fire, s_fire, g_fire = load_pypeit(
    'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')
print(f'  FIRE: {g_fire.sum()} good, rest '
      f'{w_fire[g_fire].min():.0f}-{w_fire[g_fire].max():.0f} Å', flush=True)

# ── 6. FIRE data onto R=6000 model grid (MODS Red excluded) ──────────────────
f_obs = np.full(N_PIX, np.nan)
e_obs = np.full(N_PIX, np.nan)

ok_f = g_fire & (w_fire >= FIT_LO) & (w_fire <= FIT_HI) \
       & np.isfinite(f_fire) & np.isfinite(s_fire) & (s_fire > 0)
if ok_f.sum() > 0:
    f_obs = np.interp(w_cen, w_fire[ok_f], f_fire[ok_f], left=np.nan, right=np.nan)
    e_obs = np.interp(w_cen, w_fire[ok_f], s_fire[ok_f], left=np.nan, right=np.nan)

# 10% systematic floor — identical to LRD analysis
e_obs = np.sqrt(e_obs**2 + (F_SYS * np.abs(f_obs))**2)
print(f'Merged: {np.isfinite(f_obs).sum()} valid pixels on R={R_OUT} grid', flush=True)

# ── 7. Masks and band weights ─────────────────────────────────────────────────
# Core 3 masks from LRD script + [S III] doublet (strong nebular in The Egg)
LINE_MASKS = [
    (10830, 250),   # He I  — same as LRD
    (12820, 250),   # Pa β  — same as LRD
    (10938, 200),   # Pa γ  — same as LRD
    ( 9069, 150),   # [S III] 9069 — strong in The Egg, not in LRDs at z~2.3
    ( 9532, 150),   # [S III] 9532
]
line_mask = np.ones(N_PIX, dtype=bool)
for cen, hw in LINE_MASKS:
    line_mask &= ~((w_cen >= cen-hw) & (w_cen <= cen+hw))

fit_window = (w_cen >= FIT_LO) & (w_cen <= FIT_HI)
BASE_MASK  = fit_window & line_mask

# Ca II triplet upweight (x3): the key discriminating feature for The Egg.
# LRDs use H2O x3 because they show water absorption; The Egg does not show
# H2O, so that weight biases toward hotter T_eff. CaT (8450-8700 Å rest)
# is the primary log g and T_eff diagnostic here (Liu+2026 fit the CaT directly).
CAT_BAND    = (w_cen >= 8450) & (w_cen <= 8700)
BAND_WEIGHT = np.where(CAT_BAND & fit_window, 6.0, 1.0)

n_pix = int((BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)).sum())
print(f'Fit: {FIT_LO:.0f}-{FIT_HI:.0f} Å | {n_pix} unmasked pixels | '
      f'CaT weight x6 ({CAT_BAND.sum()} pix, 8450-8700 Å)', flush=True)

# ── 8. Log-likelihoods ───────────────────────────────────────────────────────
# Velocity pixel scale on log-lambda R=2000 grid: each pixel = c/R km/s
V_PIX = 3e5 / R_OUT   # 50 km/s per pixel at R=6000

def _broaden(fm, sigma_kms):
    """Apply additional Gaussian broadening in velocity space."""
    if sigma_kms < 1.0:
        return fm
    return gaussian_filter1d(fm, sigma=sigma_kms / V_PIX)

def logl_2comp(theta):
    T_hot, logg_hot, T_cool, logg_cool, av, mh_h, mh_c, sigma_broad = theta
    if T_hot <= T_cool: return -1e300
    fh = interp_spectrum(T_hot,  logg_hot, mh_h)
    fc = interp_spectrum(T_cool, logg_cool, mh_c)
    if fh is None or fc is None: return -1e300
    ex  = ext_curve(av)
    fhe = _broaden(fh*ex, sigma_broad)
    fce = _broaden(fc*ex, sigma_broad)
    ok = (BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)
          & np.isfinite(fhe) & np.isfinite(fce) & (fhe > 0) & (fce > 0))
    if ok.sum() < 15: return -1e300
    bw = BAND_WEIGHT[ok]
    w  = bw / e_obs[ok]**2
    A  = np.column_stack([fhe[ok]*np.sqrt(w), fce[ok]*np.sqrt(w)])
    (a, b), _ = nnls(A, f_obs[ok]*np.sqrt(w))
    if a+b == 0: return -1e300
    res = f_obs[ok] - a*fhe[ok] - b*fce[ok]
    return -0.5 * np.sum(bw * (res/e_obs[ok])**2)

def logl_1comp(theta):
    T, logg, av, mh_idx, sigma_broad = theta
    fm = interp_spectrum(T, logg, mh_idx)
    if fm is None: return -1e300
    fme = _broaden(fm * ext_curve(av), sigma_broad)
    ok = (BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)
          & np.isfinite(fme) & (fme > 0))
    if ok.sum() < 15: return -1e300
    bw = BAND_WEIGHT[ok]
    w  = bw / e_obs[ok]**2
    A  = (fme[ok]*np.sqrt(w))[:, None]
    (a,), _ = nnls(A, f_obs[ok]*np.sqrt(w))
    if a == 0: return -1e300
    res = f_obs[ok] - a*fme[ok]
    return -0.5 * np.sum(bw * (res/e_obs[ok])**2)

# ── 9. Prior transforms ───────────────────────────────────────────────────────
def prior_2comp(u):
    theta = np.zeros(8)
    theta[0] = 4000 + 4000*u[0]    # T_hot:       Uniform [4000, 8000] K
    theta[1] = -2.5 + 4.0*u[1]     # logg_hot:    Uniform [-2.5, 1.5]
    theta[2] = 2000 + 1500*u[2]    # T_cool:      Uniform [2000, 3500] K (library min)
    theta[3] = -4.5 + 5.5*u[3]     # logg_cool:   Uniform [-4.5, 1.0]
    theta[4] = 8.0*u[4]             # A_V:         Uniform [0, 8]
    theta[5] = -0.49 + 2.98*u[5]   # mh_hot:      Uniform [-0.49, 2.49]
    theta[6] = -0.49 + 2.98*u[6]   # mh_cool:     Uniform [-0.49, 2.49]
    theta[7] = 600.0*u[7]           # sigma_broad: Uniform [0, 600] km/s
    return theta

def prior_1comp(u):
    theta = np.zeros(5)
    theta[0] = 2000 + 5500*u[0]    # T:           Uniform [2000, 7500] K
    theta[1] = -4.0 + 5.5*u[1]     # logg:        Uniform [-4.0, 1.5]
    theta[2] = 8.0*u[2]             # A_V:         Uniform [0, 8]
    theta[3] = -0.49 + 2.98*u[3]   # mh:          Uniform [-0.49, 2.49]
    theta[4] = 600.0*u[4]           # sigma_broad: Uniform [0, 600] km/s
    return theta

# ── 10. Run dynesty — IDENTICAL settings to LRD flat-prior script ─────────────
def run_dynesty(logl, prior_tf, ndim, label):
    print(f'\n{"="*62}', flush=True)
    print(f'Fitting: {label}  (ndim={ndim})', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, ndim,
        nlive=1200, bound='multi', sample='rwalk',
    )
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=400, nlive_batch=1000,
        wt_kwargs={'pfrac': 1.0}, n_effective=8000,
        maxbatch=None, maxcall=None, maxcall_init=None, maxiter=None,
        print_progress=True,
    )
    elapsed = time.time() - t0
    res = sampler.results
    print(f'\n  Elapsed: {elapsed:.0f}s', flush=True)
    print(f'  log Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    weights = np.exp(res.logwt - res.logz[-1])
    samples = resample_equal(res.samples, weights)
    ll_max  = res.logl.max()
    return samples, res, ll_max

if __name__ == '__main__':
    # ── 1-component (5 params: +sigma_broad) ─────────────────────────────────
    samp1, res1, ll1 = run_dynesty(logl_1comp, prior_1comp, 5, '1-comp TLUSTY + sigma_broad')
    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_1comp_v5.npy'), samp1)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_1comp_v5.pkl'), 'wb') as fh:
        pickle.dump(res1, fh)
    print('Saved egg_chain_1comp_v5.npy')
    labels1 = ['T_eff', 'logg', 'A_V', 'mh_idx', 'sigma_broad']
    print('\n1-comp posteriors (median +hi -lo):')
    for i, lbl in enumerate(labels1):
        lo, med, hi = np.percentile(samp1[:, i], [16, 50, 84])
        print(f'  {lbl:12s}: {med:8.2f}  +{hi-med:.2f} -{med-lo:.2f}')

    # ── 2-component (8 params: +sigma_broad) ─────────────────────────────────
    samp2, res2, ll2 = run_dynesty(logl_2comp, prior_2comp, 8, '2-comp TLUSTY + sigma_broad')
    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_2comp_v5.npy'), samp2)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_2comp_v5.pkl'), 'wb') as fh:
        pickle.dump(res2, fh)
    print('Saved egg_chain_2comp_v5.npy')
    labels2 = ['T_hot','logg_hot','T_cool','logg_cool','A_V','mh_hot','mh_cool','sigma_broad']
    print('\n2-comp posteriors (median +hi -lo):')
    for i, lbl in enumerate(labels2):
        lo, med, hi = np.percentile(samp2[:, i], [16, 50, 84])
        print(f'  {lbl:12s}: {med:8.2f}  +{hi-med:.2f} -{med-lo:.2f}')
    frac_qs  = np.mean(samp2[:, 1] < 0.0)
    frac_cel = np.mean(samp2[:, 1] > 1.2)
    print(f'  f(logg_hot < 0,   QS regime):     {frac_qs:.1%}')
    print(f'  f(logg_hot > 1.2, lib ceiling):    {frac_cel:.1%}')

    # ── BIC + Bayes evidence ─────────────────────────────────────────────────
    lz1 = res1.logz[-1]; lz1e = res1.logzerr[-1]
    lz2 = res2.logz[-1]; lz2e = res2.logzerr[-1]
    bic1 = 5*np.log(n_pix) - 2*ll1   # k=5: T, logg, A_V, mh, sigma_broad
    bic2 = 8*np.log(n_pix) - 2*ll2   # k=8: +T_cool, logg_cool, mh_cool, sigma_broad

    report = f"""
The Egg (J1025+1402, z={Z_EGG}) — Spectral Fit v5 (CaT x6)
Data: FIRE only ({FIT_LO:.0f}-{FIT_HI:.0f} Å rest, native R=6000)
Models: Liu+2026 TLUSTY at native R={R_OUT} (no additional smoothing)
Fit window: {FIT_LO:.0f}-{FIT_HI:.0f} Å rest | n_pix = {n_pix}
CaT 8450-8700 Å (x6.0) | Calzetti R_V=4.05 | F_SYS=10%
sigma_broad: Uniform [0, 600] km/s (additional Gaussian convolution in velocity)
Same dynesty settings as mcmc_fit_flatprior_dynesty.py (nlive=1200, n_eff=8000)

                 1-comp     2-comp
k (params)  :     5          8
log L_max   : {ll1:9.2f}   {ll2:9.2f}
BIC         : {bic1:9.2f}   {bic2:9.2f}
log Z       : {lz1:6.2f}±{lz1e:.2f}   {lz2:6.2f}±{lz2e:.2f}

ΔBIC (2comp − 1comp)  = {bic2-bic1:+.2f}  (negative → 2-comp preferred)
Δln Z (2comp − 1comp) = {lz2-lz1:+.2f} ± {np.sqrt(lz1e**2+lz2e**2):.2f}

1-comp posteriors:
  T_eff       = {np.median(samp1[:,0]):.0f} K
  logg        = {np.median(samp1[:,1]):.2f}
  A_V         = {np.median(samp1[:,2]):.2f} mag
  [M/H]       ≈ {np.median(samp1[:,3])-2:.1f}
  sigma_broad = {np.median(samp1[:,4]):.0f} km/s

2-comp posteriors:
  T_hot       = {np.median(samp2[:,0]):.0f} K
  logg_hot    = {np.median(samp2[:,1]):.2f}
  T_cool      = {np.median(samp2[:,2]):.0f} K
  logg_cool   = {np.median(samp2[:,3]):.2f}
  A_V         = {np.median(samp2[:,4]):.2f} mag
  sigma_broad = {np.median(samp2[:,7]):.0f} km/s
  f(logg_hot < 0, QS regime)  = {frac_qs:.1%}
  f(logg_hot > 1.2, ceiling)  = {frac_cel:.1%}

Reference: Liu+2026 1-comp: T=4500 K, logg=-2.90, [M/H]=-1, A_V=0.21 (SMC), sigma=123 km/s
"""
    print(report)
    bic_path = os.path.join(SCRIPT_DIR, 'egg_bic_v5.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'Saved {bic_path}')
    print('Done.')
