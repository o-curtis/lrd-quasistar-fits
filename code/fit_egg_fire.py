#!/usr/bin/env python3
"""
Dynesty spectral fits for The Egg (J1025+1402, z=0.0657) using new
LBT/MODS + Magellan/FIRE spectra from Xiaojing.

Three fits (all dynesty, flat priors):
  1. 1-component TLUSTY (4 params)
  2. 2-component TLUSTY flat-prior (7 params) — fiducial
  3. 2-component extended TLUSTY+PHOENIX (7 params)

Data:
  - MODS Red: rest 3909–10500 Å; used in fit window 7700–8000 Å rest
  - FIRE:     rest 7718–23637 Å; used in fit window 8000–15500 Å rest
  - MODS Blue: for plotting only (not fit)

Outputs (egg_analysis dir, _new suffix):
  egg_chain_1comp_new.npy, egg_chain_2comp_new.npy, egg_chain_extended_new.npy
  egg_dynesty_1comp_new.pkl, egg_dynesty_2comp_new.pkl, egg_dynesty_extended_new.pkl
  egg_bic_new.txt
"""
import os, time, pickle, glob, re
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.interpolate import RegularGridInterpolator
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = '/home/omc5226/work/lrdmesa/model_cache.npz'
PHOENIX_DIR = '/home/omc5226/work/lrdmesa/phoenix_grid'
DATA_DIR    = os.path.join(SCRIPT_DIR, 'egg_data',
                           'Re_ J1025+1402 LBT _ Magellan spectra')

Z_EGG = 0.1007   # SDSS spectroscopic z: [OIII] 5007→5512 Å, Hα→7226 Å all give z≈0.1009
F_SYS = 0.05     # 5% systematic flux floor

# ── 1. Wavelength grid (model_cache.npz, R=150) ───────────────────────────────
R_grid = 150
W_LO, W_HI = 2200.0, 16000.0
w_edges = np.exp(np.arange(np.log(W_LO), np.log(W_HI)+1.0/R_grid, 1.0/R_grid))
w_cen   = 0.5*(w_edges[:-1]+w_edges[1:])

FIT_LO, FIT_HI = 7000.0, 15500.0   # match LRD fit window
MODS_CUTOVER   = 7500.0   # Å rest: MODS-R covers 7000-7500, FIRE covers 7500-15500

# ── 2. Calzetti extinction ─────────────────────────────────────────────────────
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

# ── 3. Liu+2026 TLUSTY library ─────────────────────────────────────────────────
print("Loading Liu+2026 model cache …", flush=True)
cache      = np.load(CACHE_PATH, allow_pickle=True)
model_flux = cache['flux']
model_meta = list(cache['meta'])
meta_T    = np.array([m['T']    for m in model_meta], dtype=float)
meta_logg = np.array([m['logg'] for m in model_meta], dtype=float)
meta_MH   = np.array([m['MH']   for m in model_meta], dtype=int)
MH_vals   = sorted(set(meta_MH))
LOGG_LIU_MAX = 1.5
print(f"  {len(model_meta)} models, T={meta_T.min():.0f}–{meta_T.max():.0f} K, "
      f"log g={meta_logg.min():.1f}–{meta_logg.max():.1f}, [M/H]={MH_vals}", flush=True)

def _liu_tlogg(T, logg, MH):
    sel = meta_MH == MH
    if not sel.any(): return None
    Tg = meta_T[sel]; lg = meta_logg[sel]; fl = model_flux[sel]
    T_u = np.unique(Tg); g_u = np.unique(lg)
    T    = np.clip(T,    T_u.min(), T_u.max())
    logg = np.clip(logg, g_u.min(), g_u.max())
    Ti = np.clip(np.searchsorted(T_u, T, side='right')-1, 0, len(T_u)-2)
    Tlo, Thi = T_u[Ti], T_u[Ti+1]
    tF = (T-Tlo)/(Thi-Tlo) if Thi > Tlo else 0.0
    result = np.zeros(fl.shape[1])
    for Tv, tf in [(Tlo, 1-tF), (Thi, tF)]:
        if tf == 0: continue
        mT = sel & (meta_T == Tv)
        lv = meta_logg[mT]; fv = model_flux[mT]
        if not len(lv): return None
        oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
        gi = np.clip(np.searchsorted(ls, logg, side='right')-1, 0, len(ls)-2)
        glo, ghi = ls[gi], ls[gi+1]
        gF = (logg-glo)/(ghi-glo) if ghi > glo else 0.0
        result += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
    return result

def _liu_interp(T, logg, mh_idx):
    mh_idx = np.clip(mh_idx, 0.0, len(MH_vals)-1.0)
    lo_i = int(np.floor(mh_idx)); hi_i = min(lo_i+1, len(MH_vals)-1)
    mf = mh_idx - lo_i
    s_lo = _liu_tlogg(T, logg, MH_vals[lo_i])
    s_hi = _liu_tlogg(T, logg, MH_vals[hi_i])
    if s_lo is None and s_hi is None: return None
    if s_lo is None: return s_hi
    if s_hi is None: return s_lo
    if lo_i == hi_i: return s_lo
    return (1-mf)*s_lo + mf*s_hi

# Alias for 2-comp TLUSTY-only fit
interp_spectrum = _liu_interp

# ── 4. PHOENIX-ACES library (log g 1.5–5.0) ───────────────────────────────────
print("Loading PHOENIX-ACES grid …", flush=True)

def load_phoenix():
    files = glob.glob(os.path.join(PHOENIX_DIR, 'T*_g*_m*.npz'))
    recs  = []
    for fp in files:
        m = re.match(r'T(\d+)_g(\d+\.\d+)_m([+-]\d+\.\d+)\.npz', os.path.basename(fp))
        if m: recs.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), fp))
    T_v    = sorted(set(r[0] for r in recs))
    logg_v = sorted(set(r[1] for r in recs))
    mh_v   = sorted(set(r[2] for r in recs))
    nW     = len(w_cen)
    grid   = np.full((len(T_v), len(logg_v), len(mh_v), nW), np.nan)
    for T, g, mh, fp in recs:
        d = np.load(fp)
        grid[T_v.index(T), logg_v.index(g), mh_v.index(mh)] = \
            np.interp(w_cen, d['wave'], d['flux'], left=np.nan, right=np.nan)
    interp = RegularGridInterpolator(
        (np.array(T_v, float), np.array(logg_v, float), np.array(mh_v, float)),
        grid, method='linear', bounds_error=False, fill_value=None)
    print(f"  PHOENIX: T={T_v[0]}–{T_v[-1]} K, log_g={logg_v[0]:.1f}–{logg_v[-1]:.1f}",
          flush=True)
    return interp, T_v, logg_v, mh_v

phx_interp, phx_T, phx_g, phx_mh = load_phoenix()
MH_PHX = sorted(phx_mh)

def _phx_interp(T, logg, mh_idx):
    T    = np.clip(T,    min(phx_T), max(phx_T))
    logg = np.clip(logg, min(phx_g), max(phx_g))
    mh_idx = np.clip(mh_idx, 0.0, 2.0)
    lo_i = int(np.floor(mh_idx)); hi_i = min(lo_i+1, 2)
    mf = mh_idx - lo_i
    f_lo = phx_interp([[T, logg, MH_PHX[lo_i]]])[0]
    f_hi = phx_interp([[T, logg, MH_PHX[hi_i]]])[0]
    return f_lo if lo_i == hi_i else (1-mf)*f_lo + mf*f_hi

def interp_hot_extended(T, logg, mh_idx):
    if logg <= LOGG_LIU_MAX:
        return _liu_interp(T, logg, mh_idx)
    else:
        return _phx_interp(T, logg, mh_idx)

# ── 5. Load and merge spectra ─────────────────────────────────────────────────
def load_pypeit(fname, z):
    """Load PypeIt coadd1d FITS. mask=1 means good. Returns (w_rest, flux, sigma)."""
    fpath = os.path.join(DATA_DIR, fname)
    with fits.open(fpath) as h:
        s     = h['SPECTRUM'].data
        w_obs = s['wave'].astype(float)
        flux  = s['flux'].astype(float)
        ivar  = s['ivar'].astype(float)
        mask  = s['mask'].astype(int)
    good  = (mask == 1) & (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
    sigma = np.where(good, 1.0 / np.sqrt(np.where(ivar > 0, ivar, np.inf)), np.nan)
    w_rest = w_obs / (1.0 + z)
    return w_rest, flux, sigma, good

print("Loading FIRE spectrum …", flush=True)
w_fire, f_fire, s_fire, g_fire = load_pypeit(
    'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits', Z_EGG)
print(f"  FIRE: {g_fire.sum()} good pixels, rest {w_fire[g_fire].min():.0f}–"
      f"{w_fire[g_fire].max():.0f} Å", flush=True)

print("Loading MODS-Red spectrum …", flush=True)
w_modsr, f_modsr, s_modsr, g_modsr = load_pypeit(
    'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits', Z_EGG)
print(f"  MODS-R: {g_modsr.sum()} good pixels, rest {w_modsr[g_modsr].min():.0f}–"
      f"{w_modsr[g_modsr].max():.0f} Å", flush=True)

def merge_spectra():
    """
    Merge MODS-R (7700–8000 Å rest) and FIRE (8000–15500 Å rest)
    onto the model grid w_cen.
    """
    f_obs = np.full(len(w_cen), np.nan)
    e_obs = np.full(len(w_cen), np.nan)

    # MODS-R segment: 7000–7500 Å rest (below where FIRE starts at z=0.1007)
    ok_m = g_modsr & (w_modsr >= FIT_LO) & (w_modsr < MODS_CUTOVER)
    if ok_m.sum() > 0:
        grid_m = (w_cen >= FIT_LO) & (w_cen < MODS_CUTOVER)
        f_obs[grid_m] = np.interp(w_cen[grid_m], w_modsr[ok_m], f_modsr[ok_m],
                                   left=np.nan, right=np.nan)
        e_obs[grid_m] = np.interp(w_cen[grid_m], w_modsr[ok_m], s_modsr[ok_m],
                                   left=np.nan, right=np.nan)

    # FIRE segment: 7500–15500 Å rest
    ok_f = g_fire & (w_fire >= MODS_CUTOVER) & (w_fire <= FIT_HI)
    if ok_f.sum() > 0:
        grid_f = (w_cen >= MODS_CUTOVER) & (w_cen <= FIT_HI)
        f_obs[grid_f] = np.interp(w_cen[grid_f], w_fire[ok_f], f_fire[ok_f],
                                   left=np.nan, right=np.nan)
        e_obs[grid_f] = np.interp(w_cen[grid_f], w_fire[ok_f], s_fire[ok_f],
                                   left=np.nan, right=np.nan)

    # Add systematic floor
    e_obs = np.sqrt(e_obs**2 + (F_SYS * np.abs(f_obs))**2)
    return f_obs, e_obs

print("Merging spectra onto model grid …", flush=True)
f_obs, e_obs = merge_spectra()

# ── 6. Fit masks and weights ──────────────────────────────────────────────────
fit_window = (w_cen >= FIT_LO) & (w_cen <= FIT_HI)

# Emission line masks (rest-frame centres, half-widths in Å)
LINE_MASKS = [
    (10830, 250),   # He I 1.083 µm
    (10938, 200),   # Pa γ
    (12820, 250),   # Pa β
    ( 9015, 150),   # Pa η
    ( 9229, 150),   # Pa ζ
    ( 9546, 150),   # Pa ε
    (10049, 200),   # Pa δ
    ( 9069, 150),   # [S III] 0.907 µm
    ( 9532, 150),   # [S III] 0.953 µm
]

line_mask = np.ones(len(w_cen), dtype=bool)
for cen, hw in LINE_MASKS:
    line_mask &= ~((w_cen >= cen-hw) & (w_cen <= cen+hw))

BASE_MASK = fit_window & line_mask

# Feature upweights
H2O_BAND   = (w_cen >= 12800) & (w_cen <= 14500)
FEH_BAND   = (w_cen >=  9700) & (w_cen <= 10100)
BAND_WEIGHT = np.ones(len(w_cen))
BAND_WEIGHT[H2O_BAND & fit_window] = 3.0
BAND_WEIGHT[FEH_BAND & fit_window] = 2.0

n_pix = int((BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)).sum())
print(f"Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å rest | {n_pix} unmasked pixels", flush=True)

# ── 7. Log-likelihoods ────────────────────────────────────────────────────────
def make_logl_1comp(f_obs, e_obs):
    def logl(theta):
        T, logg, av, mh_idx = theta
        fm = interp_spectrum(T, logg, mh_idx)
        if fm is None: return -1e300
        ex = ext_curve(av)
        fme = fm * ex
        ok = (BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)
              & np.isfinite(fme) & (fme > 0))
        if ok.sum() < 15: return -1e300
        bw = BAND_WEIGHT[ok]
        w  = bw / e_obs[ok]**2
        A  = (fme[ok] * np.sqrt(w))[:, None]
        (a,), _ = nnls(A, f_obs[ok] * np.sqrt(w))
        if a == 0: return -1e300
        res = f_obs[ok] - a * fme[ok]
        return -0.5 * np.sum(bw * (res / e_obs[ok])**2)
    return logl

def make_logl_2comp(f_obs, e_obs, interp_hot_fn):
    def logl(theta):
        T_hot, logg_hot, T_cool, logg_cool, av, mh_h, mh_c = theta
        if T_hot <= T_cool: return -1e300
        fh = interp_hot_fn(T_hot, logg_hot, mh_h)
        fc = _liu_interp(T_cool, logg_cool, mh_c)
        if fh is None or fc is None: return -1e300
        ex = ext_curve(av)
        fhe = fh*ex; fce = fc*ex
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
    return logl

# ── 8. Prior transforms ───────────────────────────────────────────────────────
def prior_1comp(u):
    theta = np.zeros(4)
    theta[0] = 2000 + 5500*u[0]   # T: flat [2000, 7500] K (full TLUSTY range)
    theta[1] = -4.0 + 5.5*u[1]    # logg: flat [-4.0, 1.5]
    theta[2] = 8.0*u[2]            # A_V: flat [0, 8]
    theta[3] = -0.49 + 2.98*u[3]  # mh_idx: flat [−0.49, 2.49]
    return theta

def prior_2comp_tlusty(u):
    theta = np.zeros(7)
    theta[0] = 4000 + 4000*u[0]   # T_hot: flat [4000, 8000] K
    theta[1] = -2.5 + 4.0*u[1]    # logg_hot: flat [−2.5, 1.5]
    theta[2] = 1500 + 2000*u[2]   # T_cool: flat [1500, 3500] K
    theta[3] = -4.5 + 5.5*u[3]    # logg_cool: flat [−4.5, 1.0]
    theta[4] = 8.0*u[4]            # A_V: flat [0, 8]
    theta[5] = -0.49 + 2.98*u[5]  # mh_hot
    theta[6] = -0.49 + 2.98*u[6]  # mh_cool
    return theta

def prior_2comp_extended(u):
    theta = np.zeros(7)
    theta[0] = 3500 + 4500*u[0]   # T_hot: flat [3500, 8000] K
    theta[1] = -2.5 + 7.5*u[1]    # logg_hot: flat [−2.5, 5.0] (TLUSTY+PHOENIX)
    theta[2] = 1500 + 2000*u[2]   # T_cool: flat [1500, 3500] K
    theta[3] = -4.5 + 5.5*u[3]    # logg_cool: flat [−4.5, 1.0]
    theta[4] = 8.0*u[4]            # A_V: flat [0, 8]
    theta[5] = -0.49 + 2.98*u[5]  # mh_hot
    theta[6] = -0.49 + 2.98*u[6]  # mh_cool
    return theta

# ── 9. Run dynesty ────────────────────────────────────────────────────────────
def run_dynesty(logl, prior_tf, ndim, label, nlive=400, n_eff=4000):
    print(f'\n{"="*62}', flush=True)
    print(f'Fitting: {label}  (ndim={ndim})', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, ndim,
        nlive=nlive, bound='multi', sample='rwalk',
    )
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=nlive//2, nlive_batch=nlive,
        wt_kwargs={'pfrac': 1.0}, n_effective=n_eff,
        print_progress=True,
    )
    print(f'  Elapsed: {time.time()-t0:.0f} s', flush=True)
    res = sampler.results
    print(f'  log Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    weights = np.exp(res.logwt - res.logz[-1])
    samples = resample_equal(res.samples, weights)
    return samples, res

def bic_val(logl_max, k, n):
    return k * np.log(n) - 2 * logl_max

# ── 10. Run fits ──────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # ── 1-component ──────────────────────────────────────────────────────────
    logl_1 = make_logl_1comp(f_obs, e_obs)
    samp1, res1 = run_dynesty(logl_1, prior_1comp, 4, '1-component TLUSTY')

    labels1 = ['T_eff', 'logg', 'A_V', 'mh_idx']
    print('\n1-component medians:')
    for i, lbl in enumerate(labels1):
        lo, med, hi = np.percentile(samp1[:, i], [16, 50, 84])
        print(f'  {lbl:10s}: {med:8.3f}  +{hi-med:.3f} -{med-lo:.3f}')

    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_1comp_new.npy'), samp1)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_1comp_new.pkl'), 'wb') as fh:
        pickle.dump(res1, fh)
    print('Saved egg_chain_1comp_new.npy')

    # ── 2-component TLUSTY flat-prior ────────────────────────────────────────
    logl_2 = make_logl_2comp(f_obs, e_obs, _liu_interp)
    samp2, res2 = run_dynesty(logl_2, prior_2comp_tlusty, 7,
                               '2-component TLUSTY flat-prior')

    labels2 = ['T_hot', 'logg_hot', 'T_cool', 'logg_cool', 'A_V', 'mh_hot', 'mh_cool']
    print('\n2-component medians:')
    for i, lbl in enumerate(labels2):
        lo, med, hi = np.percentile(samp2[:, i], [16, 50, 84])
        print(f'  {lbl:10s}: {med:8.3f}  +{hi-med:.3f} -{med-lo:.3f}')
    frac_qs = np.mean(samp2[:, 1] < 0.0)
    print(f'  log_g_hot < 0 (QS regime): {frac_qs:.1%}')

    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_2comp_new.npy'), samp2)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_2comp_new.pkl'), 'wb') as fh:
        pickle.dump(res2, fh)
    print('Saved egg_chain_2comp_new.npy')

    # ── 2-component extended (TLUSTY+PHOENIX) ────────────────────────────────
    logl_ext = make_logl_2comp(f_obs, e_obs, interp_hot_extended)
    samp_ext, res_ext = run_dynesty(logl_ext, prior_2comp_extended, 7,
                                     '2-component Extended (TLUSTY+PHOENIX)')

    print('\nExtended-library medians:')
    for i, lbl in enumerate(labels2):
        lo, med, hi = np.percentile(samp_ext[:, i], [16, 50, 84])
        print(f'  {lbl:10s}: {med:8.3f}  +{hi-med:.3f} -{med-lo:.3f}')
    frac_qs_ext  = np.mean(samp_ext[:, 1] <= LOGG_LIU_MAX)
    frac_str_ext = np.mean(samp_ext[:, 1] > LOGG_LIU_MAX)
    print(f'  log_g_hot ≤ 1.5 (QS/TLUSTY): {frac_qs_ext:.1%}')
    print(f'  log_g_hot > 1.5 (stellar/PHOENIX): {frac_str_ext:.1%}')

    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_extended_new.npy'), samp_ext)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_extended_new.pkl'), 'wb') as fh:
        pickle.dump(res_ext, fh)
    print('Saved egg_chain_extended_new.npy')

    # ── BIC and evidence comparison ───────────────────────────────────────────
    ll1   = res1.logl.max()
    ll2   = res2.logl.max()
    ll_e  = res_ext.logl.max()
    lz1   = res1.logz[-1]; lz1e = res1.logzerr[-1]
    lz2   = res2.logz[-1]; lz2e = res2.logzerr[-1]
    lz_e  = res_ext.logz[-1]; lz_ee = res_ext.logzerr[-1]

    k1 = 4; k2 = 7; ke = 7
    n  = n_pix

    bic1 = bic_val(ll1, k1, n)
    bic2 = bic_val(ll2, k2, n)
    bice = bic_val(ll_e, ke, n)

    report = f"""
The Egg (J1025+1402, z={Z_EGG}) — Spectral Fit Summary
Data: MODS Red (7700–8000 Å rest) + FIRE (8000–15500 Å rest)
Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å rest  |  n_pix = {n}
Upweights: H2O 12800–14500 Å (×3.0), FeH 9700–10100 Å (×2.0)
Extinction: Calzetti R_V=4.05, no velocity broadening (R=150)

                   1-comp    2-comp     Extended
k (params)    :     {k1}         {k2}          {ke}
log L_max     :  {ll1:9.2f}  {ll2:9.2f}   {ll_e:9.2f}
BIC           :  {bic1:9.2f}  {bic2:9.2f}   {bice:9.2f}
log Z (dynesty):  {lz1:7.2f}±{lz1e:.2f}  {lz2:7.2f}±{lz2e:.2f}   {lz_e:7.2f}±{lz_ee:.2f}

ΔBIC (2comp − 1comp)     = {bic2 - bic1:+.2f}  (>0 → 1-comp preferred)
ΔBIC (extended − 1comp)  = {bice - bic1:+.2f}
Δln Z (2comp − 1comp)    = {lz2 - lz1:+.2f} ± {np.sqrt(lz1e**2+lz2e**2):.2f}  (>0 → 2-comp preferred)
Δln Z (ext − 1comp)      = {lz_e - lz1:+.2f} ± {np.sqrt(lz1e**2+lz_ee**2):.2f}

2-comp TLUSTY posteriors (medians):
  T_hot    = {np.median(samp2[:,0]):.0f} K
  logg_hot = {np.median(samp2[:,1]):.2f}
  T_cool   = {np.median(samp2[:,2]):.0f} K
  logg_cool= {np.median(samp2[:,3]):.2f}
  A_V      = {np.median(samp2[:,4]):.2f} mag
  f(logg_hot < 0, QS regime) = {frac_qs:.1%}

Extended posteriors:
  f(logg_hot <= 1.5, TLUSTY/QS) = {frac_qs_ext:.1%}
  f(logg_hot > 1.5, stellar)    = {frac_str_ext:.1%}

Reference: Liu+2026 1-comp best-fit: T=4500 K, logg=−2.90, [M/H]=−1, A_V=0.21 (SMC)
"""
    print(report)
    bic_path = os.path.join(SCRIPT_DIR, 'egg_bic_new.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'BIC report saved to {bic_path}')
    print('Done.')
