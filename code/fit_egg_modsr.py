#!/usr/bin/env python3
"""
Fit The Egg (J1025+1402) using MODS Red spectrum alone.
1-component TLUSTY hot atmosphere + AGN power-law, tested with both
Calzetti (R_V=4.05) and SMC (Gordon+2003) extinction laws.

Fit window: 3900-10100 Å rest-frame.
Key diagnostic: CaT 8450-8700 Å gravity constraint independent of FIRE.

Outputs (egg_analysis/):
  egg_chain_modsr_calzetti.npy / .pkl
  egg_chain_modsr_smc.npy / .pkl
  egg_bic_modsr.txt
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
CACHE_PATH = os.path.join(SCRIPT_DIR, 'egg_mods_cache.npz')

Z_EGG  = 0.1007
F_SYS  = 0.05
FIT_LO = 3900.0
FIT_HI = 10100.0
AGN_PIVOT = 5000.0   # Å rest, pivot for power-law template

# ── Wavelength grid from MODS cache ──────────────────────────────────────────
print("Loading MODS model cache …", flush=True)
cache      = np.load(CACHE_PATH, allow_pickle=True)
w_cen      = cache['wave'].astype(float)
model_flux = cache['flux']
model_meta = list(cache['meta'])
meta_T    = np.array([m['T']    for m in model_meta], dtype=float)
meta_logg = np.array([m['logg'] for m in model_meta], dtype=float)
meta_MH   = np.array([m['MH']   for m in model_meta], dtype=int)
MH_vals   = sorted(set(meta_MH))
print(f"  {len(model_meta)} models, T={meta_T.min():.0f}–{meta_T.max():.0f} K, "
      f"logg={meta_logg.min():.1f}–{meta_logg.max():.1f}", flush=True)

# ── Extinction curves ─────────────────────────────────────────────────────────
def calzetti_ext(wave_aa, a_v):
    """Calzetti+2000, R_V=4.05. Returns transmission 10^(-0.4 A_lambda)."""
    w = wave_aa * 1e-4
    k = np.zeros_like(w, dtype=float)
    lo = w < 0.63
    k[lo] = 2.659*(-2.156 + 1.509/w[lo] - 0.198/w[lo]**2 + 0.011/w[lo]**3) + 4.05
    hi = ~lo & (w < 2.2)
    k[hi] = 2.659*(-1.857 + 1.040/w[hi]) + 4.05
    k = np.clip(k, 0, None)
    return 10**(-0.4 * k * a_v / 4.05)

# Gordon+2003 SMC Bar: tabulated A(λ)/A(V) vs x=1/λ(µm)
_smc_x   = np.array([0.29, 0.45, 0.61, 0.80, 1.11, 1.43, 1.82,
                      2.27, 2.70, 3.33, 3.70, 4.55, 5.88])
_smc_rav = np.array([0.112, 0.420, 0.427, 0.453, 0.570, 0.741, 1.000,
                     1.412, 1.785, 2.460, 2.609, 2.770, 3.554])
_smc_interp = interp1d(_smc_x, _smc_rav, kind='linear',
                        bounds_error=False, fill_value='extrapolate')

def smc_ext(wave_aa, a_v):
    """Gordon+2003 SMC Bar average. Returns transmission 10^(-0.4 A_lambda)."""
    x = 1.0 / (wave_aa * 1e-4)
    rav = np.clip(_smc_interp(x), 0, None)
    return 10**(-0.4 * rav * a_v)

# ── Model interpolator ────────────────────────────────────────────────────────
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

def interp_model(T, logg, mh_idx):
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

# ── AGN power-law template ────────────────────────────────────────────────────
def agn_template(alpha_pl):
    """f_nu ∝ (lambda/pivot)^(-alpha_pl). alpha_pl>0 → rising toward blue."""
    return (w_cen / AGN_PIVOT) ** (-alpha_pl)

# ── Load MODS Red ─────────────────────────────────────────────────────────────
print("Loading MODS Red …", flush=True)
modsr_path = os.path.join(DATA_DIR, 'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
with fits.open(modsr_path) as h:
    s = h['SPECTRUM'].data
    w_obs  = s['wave'].astype(float)
    flux   = s['flux'].astype(float)
    ivar   = s['ivar'].astype(float)
    msk    = s['mask'].astype(int)
good  = (msk == 1) & (ivar > 0) & np.isfinite(flux)
sigma = np.where(good, 1.0/np.sqrt(np.where(ivar>0, ivar, np.inf)), np.nan)
w_rest = w_obs / (1.0 + Z_EGG)
print(f"  {good.sum()} good pixels, rest {w_rest[good].min():.0f}–{w_rest[good].max():.0f} Å",
      flush=True)

# Interpolate onto model grid
fit_grid = (w_cen >= FIT_LO) & (w_cen <= FIT_HI)
ok_data  = good & (w_rest >= FIT_LO) & (w_rest <= FIT_HI)

f_obs = np.full(len(w_cen), np.nan)
e_obs = np.full(len(w_cen), np.nan)
if ok_data.sum() > 0:
    f_obs[fit_grid] = np.interp(w_cen[fit_grid], w_rest[ok_data], flux[ok_data],
                                 left=np.nan, right=np.nan)
    e_obs[fit_grid] = np.interp(w_cen[fit_grid], w_rest[ok_data], sigma[ok_data],
                                 left=np.nan, right=np.nan)
e_obs = np.sqrt(e_obs**2 + (F_SYS * np.abs(f_obs))**2)

# ── Emission line masks ───────────────────────────────────────────────────────
# Broad lines use generous masks (±80 Å ~ ±3000 km/s at 8000 Å)
LINE_MASKS_REST = [
    # Broad Balmer (BLR)
    (6563, 110),   # Hα broad  ±5000 km/s
    (4861,  90),   # Hβ broad
    (4340,  50),   # Hγ (partial BLR)
    (4102,  30),   # Hδ (less contaminated — smaller mask)
    # Narrow forbidden
    (5007,  30),   # [O III]
    (4959,  20),   # [O III]
    (6583,  25),   # [N II]
    (6548,  20),   # [N II]
    (6716,  20),   # [S II]
    (6731,  20),   # [S II]
    (3727,  30),   # [O II]
    (3869,  20),   # [Ne III]
    ( 9069, 40),   # [S III]
    ( 9532, 40),   # [S III]
    # He lines
    ( 5876,  25),  # He I (narrow)
    ( 4686,  25),  # He II
    # Ca H&K — could be in absorption (keep) or emission (watch residuals)
    # Leave unmasked; any emission will show in residuals
]

line_mask = np.ones(len(w_cen), dtype=bool)
for cen, hw in LINE_MASKS_REST:
    line_mask &= ~((w_cen >= cen-hw) & (w_cen <= cen+hw))

BASE_MASK = fit_grid & line_mask

# CaT upweight (primary gravity diagnostic)
CAT_BAND = (w_cen >= 8400) & (w_cen <= 8750)
BAND_WEIGHT = np.ones(len(w_cen))
BAND_WEIGHT[CAT_BAND & fit_grid] = 5.0

n_pix_calz = int((BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)).sum())
print(f"Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å | {n_pix_calz} unmasked pixels", flush=True)

# ── Log-likelihood (parametrised by extinction function) ─────────────────────
def make_logl(ext_fn):
    def logl(theta):
        T, logg, a_v, mh_idx, alpha_pl = theta
        fm = interp_model(T, logg, mh_idx)
        if fm is None: return -1e300
        ext = ext_fn(w_cen, a_v)
        f_star = fm * ext
        f_agn  = agn_template(alpha_pl) * ext   # reddened PL
        ok = (BASE_MASK & np.isfinite(f_obs) & np.isfinite(e_obs) & (e_obs > 0)
              & np.isfinite(f_star) & (f_star > 0) & np.isfinite(f_agn) & (f_agn > 0))
        if ok.sum() < 20: return -1e300
        bw = BAND_WEIGHT[ok]
        w  = bw / e_obs[ok]**2
        A  = np.column_stack([f_star[ok]*np.sqrt(w), f_agn[ok]*np.sqrt(w)])
        amps, _ = nnls(A, f_obs[ok]*np.sqrt(w))
        if amps.sum() == 0: return -1e300
        res = f_obs[ok] - amps[0]*f_star[ok] - amps[1]*f_agn[ok]
        return -0.5 * np.sum(bw * (res/e_obs[ok])**2)
    return logl

def prior_tf(u):
    theta = np.zeros(5)
    theta[0] = 3500 + 4000*u[0]    # T_hot: [3500, 7500] K
    theta[1] = -4.0  + 5.5*u[1]    # logg:  [-4.0, 1.5]
    theta[2] =  8.0*u[2]            # A_V:   [0, 8]
    theta[3] = -0.49 + 2.98*u[3]   # mh_idx → [M/H] in {-2,-1,0}
    theta[4] =  0.0  + 2.0*u[4]    # alpha_PL: [0, 2] — physically motivated for AGN optical PL
    return theta

def run_fit(ext_fn, label, out_stem):
    logl = make_logl(ext_fn)
    print(f'\n{"="*60}', flush=True)
    print(f'Running: {label}  (ndim=5)', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl, prior_tf, 5, nlive=500, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=250, nlive_batch=500,
        wt_kwargs={'pfrac': 1.0}, n_effective=6000, print_progress=True)
    print(f'  Elapsed: {time.time()-t0:.0f} s', flush=True)
    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)
    print(f'  log Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}')
    for i, lbl in enumerate(['T_hot','logg_hot','A_V','mh_idx','alpha_PL']):
        p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
        print(f'  {lbl:10s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}')
    np.save(os.path.join(SCRIPT_DIR, f'{out_stem}.npy'), samp)
    with open(os.path.join(SCRIPT_DIR, f'{out_stem}.pkl'), 'wb') as fh:
        pickle.dump(res, fh)
    print(f'  Saved {out_stem}')
    return samp, res

if __name__ == '__main__':
    samp_calz, res_calz = run_fit(calzetti_ext, 'MODS Red — Calzetti', 'egg_chain_modsr_calzetti')
    samp_smc,  res_smc  = run_fit(smc_ext,      'MODS Red — SMC',      'egg_chain_modsr_smc')

    n = n_pix_calz; k = 5
    lines = [
        f"The Egg (J1025+1402) — MODS Red Only Fit",
        f"Fit window: {FIT_LO:.0f}–{FIT_HI:.0f} Å rest | n_pix = {n}",
        f"Model: 1-comp TLUSTY + AGN PL (amplitude by NNLS) | ndim=5",
        f"CaT 8400-8750 Å upweighted x3",
        f"",
    ]
    for samp, res, lbl in [(samp_calz, res_calz, 'Calzetti R_V=4.05'),
                            (samp_smc,  res_smc,  'SMC Gordon+2003')]:
        ll = res.logl.max(); lz = res.logz[-1]; lze = res.logzerr[-1]
        bic = k*np.log(n) - 2*ll
        lines += [
            f"{'─'*50}",
            f"Extinction: {lbl}",
            f"  log Z = {lz:.2f} ± {lze:.2f}",
            f"  log L_max = {ll:.2f}",
            f"  BIC = {bic:.2f}",
        ]
        for i, lbl2 in enumerate(['T_hot','logg_hot','A_V','mh_idx','alpha_PL']):
            p16,p50,p84 = np.percentile(samp[:,i],[16,50,84])
            lines.append(f"  {lbl2:10s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}")
        lines.append(f"  f(logg < 0): {(samp[:,1]<0).mean():.1%}")
        lines.append(f"  f(logg < -1.0): {(samp[:,1]<-1.0).mean():.1%}")
        lines.append("")

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(SCRIPT_DIR, 'egg_bic_modsr.txt'), 'w') as fh:
        fh.write(report)
    print("Done.")
