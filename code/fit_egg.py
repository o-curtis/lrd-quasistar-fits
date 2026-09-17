#!/usr/bin/env python3
"""
Spectral fits for The Egg (J1025+1402, z=0.1007) using Liu+2026 TLUSTY library.

Single-component and two-component (hot + cool atmosphere) fits via dynesty.
NO MESA priors — flat priors within Liu library coverage.

Usage:
    python fit_egg.py [spectrum.fits|spectrum.csv]

If no spectrum file is given, generates a mock spectrum based on Liu+2026
best-fit parameters (T=4500 K, log g=-2.90, [M/H]=-1, A_V=0.21).

Outputs (in same directory as script):
    egg_chain_1comp.npy     — equal-weight posterior samples, 1-component
    egg_chain_2comp.npy     — equal-weight posterior samples, 2-component
    egg_dynesty_1comp.pkl   — full dynesty result (log Z)
    egg_dynesty_2comp.pkl   — full dynesty result (log Z)
    egg_bic.txt             — BIC comparison table
"""
import os, sys, time, pickle
import numpy as np
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE      = os.path.join(SCRIPT_DIR, 'egg_cache.npz')

Z_EGG      = 0.1007            # redshift of The Egg
SIGMA_V    = 123.0             # km/s velocity broadening (Liu+2026)
F_SYS      = 0.05              # systematic flux floor

# ── Load cache ─────────────────────────────────────────────────────────────────
print('Loading egg_cache ...', flush=True)
cache      = np.load(CACHE, allow_pickle=True)
W_CEN      = cache['wave']          # rest-frame Å, shape (N_wave,)
model_flux = cache['flux']          # shape (N_models, N_wave)
model_meta = list(cache['meta'])
meta_T     = np.array([m['T']    for m in model_meta], dtype=float)
meta_logg  = np.array([m['logg'] for m in model_meta], dtype=float)
meta_MH    = np.array([m['MH']   for m in model_meta], dtype=int)
MH_VALS    = sorted(set(meta_MH))
print(f'  {len(model_meta)} spectra | {len(W_CEN)} wavelength points', flush=True)

# ── Velocity broadening kernel ─────────────────────────────────────────────────
def apply_broadening(flux, sigma_kms):
    """Gaussian broadening at constant sigma_v in km/s (applied in log-λ)."""
    dloglam = np.diff(np.log(W_CEN)).mean()   # constant for log-spaced grid
    sigma_pix = (sigma_kms * 1e3 / 3e8) / dloglam
    return gaussian_filter1d(flux, sigma_pix, axis=-1)

# Pre-broaden all library spectra (constant σ_v = SIGMA_V)
print(f'Applying σ_v={SIGMA_V} km/s broadening to library ...', flush=True)
model_flux_br = apply_broadening(model_flux, SIGMA_V)

# ── Extinction curve (Calzetti+2000) ──────────────────────────────────────────
def calzetti_k(wave_aa):
    w = wave_aa * 1e-4
    k = np.zeros_like(w)
    lo = w < 0.63
    k[lo] = 2.659 * (-2.156 + 1.509/w[lo] - 0.198/w[lo]**2 + 0.011/w[lo]**3) + 4.05
    hi = ~lo & (w < 2.2)
    k[hi] = 2.659 * (-1.857 + 1.040/w[hi]) + 4.05
    return np.clip(k, 0, None)

K_EXT = calzetti_k(W_CEN)
RV    = 4.05

def ext_curve(av):
    return 10**(-0.4 * K_EXT * av / RV)

# ── Spectral interpolation ─────────────────────────────────────────────────────
def interp_spectrum(T, logg, mh_idx):
    """Bilinear (T,logg) interpolation; linear metallicity blend.
    mh_idx in [0, len(MH_VALS)-1] maps continuously over metallicities.
    Returns broadened, normalised flux array or None if out of range.
    """
    mh_idx = np.clip(float(mh_idx), 0., len(MH_VALS) - 1.)
    lo_i   = int(np.floor(mh_idx))
    hi_i   = min(lo_i + 1, len(MH_VALS) - 1)
    mf     = mh_idx - lo_i

    def _interp_mh(MH):
        sel = meta_MH == MH
        if not sel.any():
            return None
        Tg = meta_T[sel]; lg = meta_logg[sel]; fl = model_flux_br[sel]
        Tu = np.unique(Tg); gu = np.unique(lg)
        Tc = np.clip(T, Tu.min(), Tu.max())
        gc = np.clip(logg, gu.min(), gu.max())
        Ti = np.clip(np.searchsorted(Tu, Tc, 'right') - 1, 0, len(Tu) - 2)
        tF = (Tc - Tu[Ti]) / (Tu[Ti+1] - Tu[Ti]) if Tu[Ti+1] > Tu[Ti] else 0.
        res = np.zeros(fl.shape[1])
        for tv, tf in [(Tu[Ti], 1 - tF), (Tu[Ti+1], tF)]:
            if tf == 0: continue
            mT = sel & (meta_T == tv)
            lv = meta_logg[mT]; fv = model_flux_br[mT]
            if not len(lv): return None
            oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
            gi = np.clip(np.searchsorted(ls, gc, 'right') - 1, 0, len(ls) - 2)
            gF = np.clip((gc - ls[gi]) / (ls[gi+1] - ls[gi]), 0., 1.) if ls[gi+1] > ls[gi] else 0.
            res += tf * ((1 - gF) * fs[gi] + gF * fs[gi+1])
        return res

    sl = _interp_mh(MH_VALS[lo_i])
    sh = _interp_mh(MH_VALS[hi_i])
    if sl is None and sh is None: return None
    if sl is None: return sh
    if sh is None: return sl
    if lo_i == hi_i: return sl
    return (1 - mf) * sl + mf * sh

# ── Observation loading ────────────────────────────────────────────────────────
def load_spectrum_npz(path):
    """Load spectrum saved by our SDSS download script."""
    d = np.load(path)
    w_rest = d['wave_rest'].astype(float)
    flux   = d['flux'].astype(float)
    err    = d['err'].astype(float)
    good   = d['good'].astype(bool)
    flux[~good] = np.nan
    err[~good]  = np.nan
    return w_rest, flux, err

def load_spectrum_fits(path):
    from astropy.io import fits
    with fits.open(path) as hdul:
        for ext in hdul:
            if ext.data is None: continue
            if hasattr(ext.data, 'names'):
                names = [n.upper() for n in ext.data.names]
                wi = next((i for i,n in enumerate(names) if 'WAVE' in n), None)
                fi = next((i for i,n in enumerate(names) if 'FLUX' in n), None)
                ei = next((i for i,n in enumerate(names) if 'ERR' in n or 'SIGMA' in n), None)
                if wi is not None and fi is not None and ei is not None:
                    return (ext.data.field(wi).astype(float),
                            ext.data.field(fi).astype(float),
                            ext.data.field(ei).astype(float))
    raise ValueError(f'Cannot parse FITS file: {path}')

def load_spectrum_csv(path):
    data = np.loadtxt(path, delimiter=',')
    return data[:, 0], data[:, 1], data[:, 2]

def load_spectrum(path):
    if path.endswith('.npz'):
        return load_spectrum_npz(path)
    if path.endswith('.fits') or path.endswith('.fit'):
        w, f, e = load_spectrum_fits(path)
    else:
        w, f, e = load_spectrum_csv(path)
    # convert observed to rest-frame
    w_rest = w / (1 + Z_EGG)
    return w_rest, f, e

def make_mock_spectrum():
    """Simulate The Egg spectrum from Liu+2026 best-fit single-component model.

    Parameters (Liu+2026 Table 1):
        T_eff = 4500 K, log g = -2.90, [M/H] = -1, A_V = 0.21 mag, σ_v = 123 km/s
    SNR ~ 10 in continuum, PRISM-like resolution (already in cache at R~200).
    """
    print('No spectrum file given — generating mock from Liu+2026 best-fit.', flush=True)
    print('  T=4500 K, logg=-2.90, [M/H]=-1, A_V=0.21', flush=True)
    mh_idx = MH_VALS.index(-1)
    f_mod  = interp_spectrum(4500., -2.90, float(mh_idx))
    if f_mod is None:
        raise RuntimeError('Mock generation failed — check cache coverage.')
    ex     = ext_curve(0.21)
    f_true = f_mod * ex
    # Scale to plausible flux units (normalised, arbitrary)
    f_true /= np.nanmedian(f_true[(W_CEN > 8000) & (W_CEN < 9000)])
    # Add noise: SNR~10 in 8000-9000 Å continuum
    noise  = f_true / 10.0
    f_obs  = f_true + np.random.normal(0, noise)
    f_err  = noise * np.ones_like(f_obs)
    return W_CEN.copy(), f_obs, f_err

# ── Fit window and line masks ──────────────────────────────────────────────────
FIT_LO = 3900.   # Å rest, covers Ca H&K
FIT_HI = 8350.   # Å rest, SDSS red limit at z=0.1007

# Emission lines to mask (rest-frame centre, half-width in Å)
LINE_MASKS = [
    (3727, 150),   # [O II]
    (3869, 100),   # [Ne III]
    (4101, 100),   # Hδ
    (4340, 150),   # Hγ
    (4686, 150),   # He II
    (4861, 300),   # Hβ (broad in AGN)
    (4959, 150),   # [O III]
    (5007, 200),   # [O III]
    (5876, 100),   # He I
    (6300, 100),   # [O I]
    (6563, 500),   # Hα (broad AGN line)
    (6716, 100),   # [S II]
    (6731, 100),   # [S II]
]

def build_mask(w):
    mask = (w >= FIT_LO) & (w <= FIT_HI)
    for cen, hw in LINE_MASKS:
        mask &= ~((w >= cen - hw) & (w <= cen + hw))
    return mask

# Feature upweighting: Ca II triplet region + TiO + FeH
def build_weights(w, mask):
    bw = np.ones(len(w))
    bw[(w >= 3900) & (w <= 4050) & mask] = 3.0   # Ca H&K (key gravity diagnostic)
    bw[(w >= 6800) & (w <= 7600) & mask] = 2.5   # TiO + VO bands
    bw[(w >= 8150) & (w <= 8350) & mask] = 3.0   # Ca II 8498 Å (first line, barely in window)
    return bw

# ── Log-likelihood ─────────────────────────────────────────────────────────────
def make_logl_1comp(f_obs, f_err, mask, bw):
    def logl(theta):
        T, logg, av, mh_idx = theta
        f_m = interp_spectrum(T, logg, mh_idx)
        if f_m is None: return -1e300
        ex   = ext_curve(av)
        f_me = f_m * ex
        ok   = mask & np.isfinite(f_obs) & np.isfinite(f_err) & (f_err > 0) & (f_me > 0)
        if ok.sum() < 20: return -1e300
        w_pix = bw[ok] / f_err[ok]**2
        A     = (f_me[ok] * np.sqrt(w_pix))[:, None]
        (a,), _ = nnls(A, f_obs[ok] * np.sqrt(w_pix))
        if a == 0: return -1e300
        res   = f_obs[ok] - a * f_me[ok]
        return -0.5 * np.sum(bw[ok] * (res / f_err[ok])**2)
    return logl

def make_logl_2comp(f_obs, f_err, mask, bw):
    def logl(theta):
        T_h, g_h, T_c, g_c, av, mh_idx = theta
        f_h = interp_spectrum(T_h, g_h, mh_idx)
        f_c = interp_spectrum(T_c, g_c, mh_idx)
        if f_h is None or f_c is None: return -1e300
        ex   = ext_curve(av)
        fhe  = f_h * ex; fce = f_c * ex
        ok   = (mask & np.isfinite(f_obs) & np.isfinite(f_err) & (f_err > 0)
                & (fhe > 0) & (fce > 0))
        if ok.sum() < 20: return -1e300
        w_pix = bw[ok] / f_err[ok]**2
        A     = np.column_stack([fhe[ok] * np.sqrt(w_pix),
                                  fce[ok] * np.sqrt(w_pix)])
        (a, b), _ = nnls(A, f_obs[ok] * np.sqrt(w_pix))
        if a + b == 0: return -1e300
        res = f_obs[ok] - a * fhe[ok] - b * fce[ok]
        return -0.5 * np.sum(bw[ok] * (res / f_err[ok])**2)
    return logl

# ── Prior transforms ───────────────────────────────────────────────────────────
T_LO, T_HI   = 3000., 7500.
G_LO, G_HI   = -4.0,   1.5
AV_LO, AV_HI =  0.0,   3.0
MH_LO, MH_HI = -0.49,  float(len(MH_VALS) - 1) + 0.49

def prior_1comp(u):
    t = np.zeros(4)
    t[0] = T_LO  + (T_HI  - T_LO)  * u[0]   # T_eff
    t[1] = G_LO  + (G_HI  - G_LO)  * u[1]   # logg
    t[2] = AV_LO + (AV_HI - AV_LO) * u[2]   # A_V
    t[3] = MH_LO + (MH_HI - MH_LO) * u[3]   # mh_idx
    return t

def prior_2comp(u):
    t = np.zeros(6)
    t[0] = T_LO  + (T_HI  - T_LO)  * u[0]   # T_hot
    t[1] = G_LO  + (G_HI  - G_LO)  * u[1]   # logg_hot
    t[2] = T_LO  + (T_HI  - T_LO)  * u[2]   # T_cool
    t[3] = G_LO  + (G_HI  - G_LO)  * u[3]   # logg_cool
    t[4] = AV_LO + (AV_HI - AV_LO) * u[4]   # A_V
    t[5] = MH_LO + (MH_HI - MH_LO) * u[5]   # mh_idx (shared)
    return t

# ── Run fits ───────────────────────────────────────────────────────────────────
def run_dynesty(logl, prior_tf, ndim, label):
    print(f'\n{"="*60}', flush=True)
    print(f'Fitting: {label}  (ndim={ndim})', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, ndim,
        nlive=400, bound='multi', sample='rwalk',
    )
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=200, nlive_batch=300,
        wt_kwargs={'pfrac': 1.0}, n_effective=3000,
        print_progress=True,
    )
    print(f'  Elapsed: {time.time()-t0:.0f} s', flush=True)
    res = sampler.results
    print(f'  log Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    weights = np.exp(res.logwt - res.logz[-1])
    samples = resample_equal(res.samples, weights)
    return samples, res

def bic(logz, k, n):
    """BIC = k*ln(n) - 2*ln(L_hat).  For dynesty, use max log-likelihood."""
    return k * np.log(n) - 2 * logz

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Load or mock spectrum ──────────────────────────────────────────────────
    if len(sys.argv) > 1:
        spec_path = sys.argv[1]
        print(f'Loading spectrum from {spec_path}', flush=True)
        w_obs, f_obs_raw, f_err_raw = load_spectrum(spec_path)
        # Interpolate onto cache wave grid
        f_obs = np.interp(W_CEN, w_obs, f_obs_raw, left=np.nan, right=np.nan)
        f_err = np.interp(W_CEN, w_obs, f_err_raw, left=np.nan, right=np.nan)
        f_err = np.sqrt(f_err**2 + (F_SYS * np.abs(f_obs))**2)
    else:
        _, f_obs, f_err = make_mock_spectrum()

    mask = build_mask(W_CEN)
    bw   = build_weights(W_CEN, mask)
    n_pix = mask.sum()
    print(f'Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å rest  | {n_pix} unmasked pixels')

    # ── 1-component fit ────────────────────────────────────────────────────────
    logl_1 = make_logl_1comp(f_obs, f_err, mask, bw)
    samp1, res1 = run_dynesty(logl_1, prior_1comp, 4, '1-component TLUSTY')

    labels1 = ['T_eff', 'logg', 'A_V', 'mh_idx']
    print('\n1-component medians:')
    for i, lbl in enumerate(labels1):
        lo, med, hi = np.percentile(samp1[:, i], [16, 50, 84])
        print(f'  {lbl:8s}: {med:.3f}  +{hi-med:.3f} -{med-lo:.3f}')

    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_1comp.npy'), samp1)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_1comp.pkl'), 'wb') as fh:
        pickle.dump(res1, fh)

    # ── 2-component fit ────────────────────────────────────────────────────────
    logl_2 = make_logl_2comp(f_obs, f_err, mask, bw)
    samp2, res2 = run_dynesty(logl_2, prior_2comp, 6, '2-component TLUSTY (hot + cool)')

    labels2 = ['T_hot', 'logg_hot', 'T_cool', 'logg_cool', 'A_V', 'mh_idx']
    print('\n2-component medians:')
    for i, lbl in enumerate(labels2):
        lo, med, hi = np.percentile(samp2[:, i], [16, 50, 84])
        print(f'  {lbl:10s}: {med:.3f}  +{hi-med:.3f} -{med-lo:.3f}')

    np.save(os.path.join(SCRIPT_DIR, 'egg_chain_2comp.npy'), samp2)
    with open(os.path.join(SCRIPT_DIR, 'egg_dynesty_2comp.pkl'), 'wb') as fh:
        pickle.dump(res2, fh)

    # ── BIC comparison ─────────────────────────────────────────────────────────
    # Max log-likelihood from dynesty results
    logl_max_1 = res1.logl.max()
    logl_max_2 = res2.logl.max()

    # k: MCMC params + NNLS amplitudes (analytic but free)
    k1 = 4 + 1   # (T, logg, A_V, mh) + 1 amplitude
    k2 = 6 + 2   # (T_h, g_h, T_c, g_c, A_V, mh) + 2 amplitudes
    n  = n_pix

    bic1  = bic(logl_max_1, k1, n)
    bic2  = bic(logl_max_2, k2, n)
    dbic  = bic2 - bic1     # positive → 1-comp preferred

    logz1 = res1.logz[-1]; logzerr1 = res1.logzerr[-1]
    logz2 = res2.logz[-1]; logzerr2 = res2.logzerr[-1]
    dlogz = logz2 - logz1   # positive → 2-comp preferred (Bayes factor)

    report = f"""
The Egg (J1025+1402, z={Z_EGG}) — BIC and Evidence Comparison
Liu+2026 TLUSTY library | σ_v={SIGMA_V} km/s | n_pix={n}
Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å rest-frame (Calzetti A_V)

                  1-component    2-component
k (free params) :      {k1}              {k2}
log L_max       :  {logl_max_1:10.2f}     {logl_max_2:10.2f}
BIC             :  {bic1:10.2f}     {bic2:10.2f}
log Z (dynesty) :  {logz1:10.2f} ± {logzerr1:.2f}   {logz2:10.2f} ± {logzerr2:.2f}

ΔBIC (2comp − 1comp) = {dbic:+.2f}
  > 0 → 1-component preferred
  < 0 → 2-component preferred

Δln Z (2comp − 1comp) = {dlogz:+.2f} ± {np.sqrt(logzerr1**2+logzerr2**2):.2f}
  > 0 → 2-component preferred (Bayes evidence)
  < 0 → 1-component preferred

Note: Liu+2026 used a 3-component model (atmosphere + galaxy FSPS + dust BB).
      Fitting only TLUSTY atmospheres here — galaxy host contribution not modelled.
      Best-fit params from Liu+2026: T=4500 K, log g=-2.90, [M/H]=-1, A_V=0.21 (SMC).
      We use Calzetti A_V; expect small systematic offset.
"""
    print(report)
    bic_path = os.path.join(SCRIPT_DIR, 'egg_bic.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'BIC report saved to {bic_path}')
    print('\nDone.')
