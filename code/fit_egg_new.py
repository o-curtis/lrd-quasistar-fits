#!/usr/bin/env python3
"""
Broadband fit of J1025+1402 (The Egg) using the current LRD fitting framework.

72 variants: 3 datasets × 2 ncomp × 2 agn × 3 ext × 2 av_max
  datasets : fire | modsr_fire | mods_fire
  ncomp    : 1 | 2
  agn      : 0 (no AGN PL) | 1 (with AGN PL)
  ext      : calzetti | smc | mw
  av_max   : 0.5 | 8.0

Parameters (all variants include sigma_v):
  1-comp + AGN (6) : T_hot logg_hot A_V MH_hot alpha_PL sigma_v
  1-comp  no AGN(5): T_hot logg_hot A_V MH_hot sigma_v
  2-comp + AGN (9) : T_hot logg_hot T_cool logg_cool A_V MH_hot MH_cool alpha_PL sigma_v
  2-comp  no AGN(8): T_hot logg_hot T_cool logg_cool A_V MH_hot MH_cool sigma_v

NNLS amplitudes (non-negative, solved analytically per likelihood call):
  fire      1c+agn  : [hot_F  agn_F]
  fire      1c      : [hot_F]
  fire      2c+agn  : [hot_F  cool_F  agn_F]
  fire      2c      : [hot_F  cool_F]
  modsr_fire 1c+agn : [hot_M  hot_F  agn]          agn shared across arms
  modsr_fire 1c     : [hot_M  hot_F]
  modsr_fire 2c+agn : [hot_M  hot_F  cool_F  agn]
  modsr_fire 2c     : [hot_M  hot_F  cool_F]
  mods_fire  1c+agn : [hot_MB hot_MR hot_F  agn]
  mods_fire  1c     : [hot_MB hot_MR hot_F]
  mods_fire  2c+agn : [hot_MB hot_MR hot_F  cool_F agn]
  mods_fire  2c     : [hot_MB hot_MR hot_F  cool_F]

Usage:
  python fit_egg_new.py --variant N          # N = 0..71 (single variant)
  python fit_egg_new.py --dataset fire --ncomp 1 --agn 1 --ext smc --av 0.5
  python fit_egg_new.py --list               # print all 72 variants and exit

Output: fits_broadband/Egg/egg_chain_{tag}.npy  +  egg_dynesty_{tag}.pkl
"""
import os, sys, time, pickle, argparse
import numpy as np
from astropy.io import fits
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import dynesty
from dynesty.utils import resample_equal
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'egg_data',
                           'Re_ J1025+1402 LBT _ Magellan spectra')
MODS_CACHE = os.path.join(SCRIPT_DIR, 'egg_mods_cache.npz')
FIRE_CACHE = os.path.join(SCRIPT_DIR, 'egg_hires_cache.npz')
OUT_DIR    = '/home/omc5226/work/lrdmesa/fits_broadband/Egg'
os.makedirs(OUT_DIR, exist_ok=True)

Z_EGG     = 0.1007
F_SYS     = 0.05
AGN_PIVOT = 5000.0   # Å rest

# Fit windows (rest-frame Å)
# When MODS-B is present: non-overlapping split at 5500 Å.
# When MODS-R is used alone (modsr_fire): full 3900-7400 Å window.
MODSB_LO,      MODSB_HI      = 3000.0,  5500.0
MODSR_LO,      MODSR_HI      = 5500.0,  7400.0   # used in mods_fire
MODSR_ONLY_LO, MODSR_ONLY_HI = 3900.0,  7400.0   # used in modsr_fire
FIRE_LO,       FIRE_HI       = 7400.0, 15500.0

# ── Extinction laws ────────────────────────────────────────────────────────────
def calzetti_ext(wave_aa, av):
    w = wave_aa * 1e-4; k = np.zeros_like(w)
    lo = w < 0.63
    k[lo] = 2.659*(-2.156 + 1.509/w[lo] - 0.198/w[lo]**2 + 0.011/w[lo]**3) + 4.05
    hi = ~lo & (w < 2.2)
    k[hi] = 2.659*(-1.857 + 1.040/w[hi]) + 4.05
    return 10**(-0.4 * np.clip(k, 0, None) * av / 4.05)

_smc_x   = np.array([0.29,0.45,0.61,0.80,1.11,1.43,1.82,2.27,2.70,3.33,3.70,4.55,5.88])
_smc_rav = np.array([0.112,0.420,0.427,0.453,0.570,0.741,1.000,1.412,1.785,2.460,2.609,2.770,3.554])
_smc_fn  = interp1d(_smc_x, _smc_rav, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
def smc_ext(wave_aa, av):
    return 10**(-0.4 * np.clip(_smc_fn(1.0/(wave_aa*1e-4)), 0, None) * av)

def mw_ext(wave_aa, av):
    x = 1.0 / (wave_aa * 1e-4)
    a = np.zeros_like(x); b = np.zeros_like(x)
    m = (x >= 0.3) & (x <= 1.1)
    y = x[m]; a[m] = 0.574*y**1.61; b[m] = -0.527*y**1.61
    m = (x > 1.1) & (x <= 3.3)
    y = x[m] - 1.82
    a[m] = (1 + 0.104*y - 0.609*y**2 + 0.701*y**3 - 1.221*y**4
              - 0.908*y**5 + 1.952*y**6 - 0.447*y**7)
    b[m] = (1.952*y + 2.908*y**2 + 1.233*y**3 - 5.385*y**4
              - 0.623*y**5 + 5.303*y**6 - 2.090*y**7)
    m = (x > 3.3) & (x <= 8.0)
    y = x[m]
    Fa = np.where(y > 5.9, -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3, 0.0)
    Fb = np.where(y > 5.9,  0.2130 *(y-5.9)**2 + 0.1207  *(y-5.9)**3, 0.0)
    a[m] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341) + Fa
    b[m] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263) + Fb
    m = (x > 8.0) & (x <= 10.0)
    y = x[m] - 8.0
    a[m] = -1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3
    b[m] =  13.670 + 4.257*y - 0.420*y**2 + 0.374*y**3
    return 10**(-0.4 * np.clip(a + b/3.1, 0, None) * av)

EXT_FNS = {'calzetti': calzetti_ext, 'smc': smc_ext, 'mw': mw_ext}

# ── Load model caches ──────────────────────────────────────────────────────────
print("Loading MODS model cache …", flush=True)
mc    = np.load(MODS_CACHE, allow_pickle=True)
wm    = mc['wave'].astype(float)        # shape (11723,)
mflux = mc['flux']                      # shape (220, 11723)
mmeta = list(mc['meta'])
mT    = np.array([m['T']    for m in mmeta], dtype=float)
mlogg = np.array([m['logg'] for m in mmeta], dtype=float)
mMH   = np.array([m['MH']   for m in mmeta], dtype=int)
MH_vals_m = sorted(set(mMH))

print("Loading FIRE model cache …", flush=True)
fc    = np.load(FIRE_CACHE, allow_pickle=True)
wf    = fc['wave'].astype(float)        # shape (4961,)
fflux = fc['flux']                      # shape (220, 4961)
fmeta = list(fc['meta'])
fT    = np.array([m['T']    for m in fmeta], dtype=float)
flogg = np.array([m['logg'] for m in fmeta], dtype=float)
fMH   = np.array([m['MH']   for m in fmeta], dtype=int)
MH_vals_f = sorted(set(fMH))
print(f"  MODS grid: {len(mmeta)} models × {len(wm)} λ  ({wm[0]:.0f}–{wm[-1]:.0f} Å, R~9000)", flush=True)
print(f"  FIRE grid: {len(fmeta)} models × {len(wf)} λ  ({wf[0]:.0f}–{wf[-1]:.0f} Å, R~6000)", flush=True)

# ── Model interpolation (bilinear in T-logg, linear in MH) ───────────────────
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
            mT2 = sel & (T_arr == Tv)
            lv = logg_arr[mT2]; fv = flux_arr[mT2]
            if not len(lv): return None
            oi = np.argsort(lv); ls = lv[oi]; fs = fv[oi]
            if logg_ < ls[0] - 0.5 or logg_ > ls[-1] + 0.5: return None
            gi = np.clip(np.searchsorted(ls, logg_, side='right')-1, 0, len(ls)-2)
            g0, g1 = ls[gi], ls[gi+1]
            gF = np.clip((logg_-g0)/(g1-g0) if g1>g0 else 0., 0., 1.)
            result += tf * ((1-gF)*fs[gi] + gF*fs[gi+1])
        return result

    sl = _tlogg(T, logg, MHvals[lo_i])
    sh = _tlogg(T, logg, MHvals[hi_i])
    if sl is None and sh is None: return None
    if sl is None: return sh
    if sh is None: return sl
    if lo_i == hi_i: return sl
    return (1-mf)*sl + mf*sh

def model_mods(T, logg, mh):
    return _interp_cache(T, logg, mh, mT, mlogg, mMH, mflux, MH_vals_m)

def model_fire(T, logg, mh):
    return _interp_cache(T, logg, mh, fT, flogg, fMH, fflux, MH_vals_f)

# ── Velocity broadening (Gaussian kernel in log-λ space) ─────────────────────
_dloglam_m = np.log(wm[1]/wm[0])
_dloglam_f = np.log(wf[1]/wf[0])

def broaden_m(flux, sigma_kms):
    if sigma_kms < 1.0: return flux
    return gaussian_filter1d(flux, sigma=(sigma_kms*1e3/3e8)/_dloglam_m)

def broaden_f(flux, sigma_kms):
    if sigma_kms < 1.0: return flux
    return gaussian_filter1d(flux, sigma=(sigma_kms*1e3/3e8)/_dloglam_f)

# ── Load Egg spectra ───────────────────────────────────────────────────────────
def load_pypeit(fname):
    with fits.open(os.path.join(DATA_DIR, fname)) as h:
        s     = h['SPECTRUM'].data
        w_obs = s['wave'].astype(float)
        flux  = s['flux'].astype(float)
        ivar  = s['ivar'].astype(float)
        msk   = s['mask'].astype(int)
    good  = (msk == 1) & (ivar > 0) & np.isfinite(flux)
    sigma = np.where(good, 1.0/np.sqrt(np.where(ivar>0, ivar, np.inf)), np.nan)
    return w_obs / (1+Z_EGG), flux, sigma, good

print("Loading Egg spectra …", flush=True)
w_mb, f_mb_raw, s_mb_raw, g_mb = load_pypeit(
    'J1025+1402_MODSB_coadd1d_tellcorr_slitcorr_dered.fits')
w_mr, f_mr_raw, s_mr_raw, g_mr = load_pypeit(
    'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
w_fi, f_fi_raw, s_fi_raw, g_fi = load_pypeit(
    'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

# Interpolate observed spectra onto model grids
def build_obs_grid(w_src, f_src, s_src, g_src, w_grid, lo, hi):
    ok  = g_src & (w_src >= lo) & (w_src <= hi)
    fg  = (w_grid >= lo) & (w_grid <= hi)
    f_out = np.full(len(w_grid), np.nan)
    e_out = np.full(len(w_grid), np.nan)
    if ok.sum() > 0:
        f_out[fg] = np.interp(w_grid[fg], w_src[ok], f_src[ok],
                              left=np.nan, right=np.nan)
        e_out[fg] = np.interp(w_grid[fg], w_src[ok], s_src[ok],
                              left=np.nan, right=np.nan)
    e_out = np.sqrt(e_out**2 + (F_SYS*np.abs(f_out))**2)
    return f_out, e_out, fg

# MODS-B arm (3000-5500 Å) on MODS model grid
f_modsb, e_modsb, grd_modsb = build_obs_grid(
    w_mb, f_mb_raw, s_mb_raw, g_mb, wm, MODSB_LO, MODSB_HI)
# MODS-R arm (5500-7400 Å) on MODS model grid — used alongside MODS-B
f_modsr, e_modsr, grd_modsr = build_obs_grid(
    w_mr, f_mr_raw, s_mr_raw, g_mr, wm, MODSR_LO, MODSR_HI)
# MODS-R arm (3900-7400 Å) — used when MODS-B is absent
f_modsr_only, e_modsr_only, grd_modsr_only = build_obs_grid(
    w_mr, f_mr_raw, s_mr_raw, g_mr, wm, MODSR_ONLY_LO, MODSR_ONLY_HI)
# FIRE arm
f_fire, e_fire, grd_fire = build_obs_grid(
    w_fi, f_fi_raw, s_fi_raw, g_fi, wf, FIRE_LO, FIRE_HI)

# ── Emission line masks ────────────────────────────────────────────────────────
def make_mask(warr, lines):
    m = np.ones(len(warr), dtype=bool)
    for cen, hw in lines:
        m &= ~((warr >= cen-hw) & (warr <= cen+hw))
    return m

MODS_LINES = [
    (6563, 110), (4861, 100), (4340, 50), (4102, 30),
    (5007, 50),  (4959, 30),  (6583, 30), (6548, 25),
    (6716, 25),  (6731, 25),  (3727, 40), (3869, 25),
    (5876, 30),  (4686, 30),  (3968, 30), (3934, 30),   # Ca H&K
    (6300, 40),  (6364, 25),                            # [O I]
    (6678, 30),                                         # He I
    (7136, 30),                                         # [Ar III]
    (7325, 45),                                         # [O II] 7320+7330
]
FIRE_LINES = [
    (10830, 250), (10938, 200), (12820, 250),
    (9015, 150),  (9229, 150),  (9546, 150), (10049, 200),
    (9069, 40),   (9532, 40),   (6563, 110),
]

lmask_mods = make_mask(wm, MODS_LINES)
lmask_fire = make_mask(wf, FIRE_LINES)

BASE_MODSB      = grd_modsb      & lmask_mods
BASE_MODSR      = grd_modsr      & lmask_mods
BASE_MODSR_ONLY = grd_modsr_only & lmask_mods
BASE_FIRE       = grd_fire       & lmask_fire

# ── Band weights ───────────────────────────────────────────────────────────────
# CaT (8400-8750 Å) ×5: key cool dwarf / giant spectral diagnostic
# FeH Wing-Ford (9700-10100 Å) ×2: cool component indicator
# No H2O upweight: The Egg does not show water absorption (T~5000 K)
CAT_M = (wm >= 8400) & (wm <= 8750)
CAT_F = (wf >= 8400) & (wf <= 8750)
FEH_F = (wf >= 9700) & (wf <= 10100)

wt_mods = np.ones(len(wm))
wt_mods[CAT_M] = 5.0

wt_fire = np.ones(len(wf))
wt_fire[CAT_F] = 5.0

# ── Pixel count helper ─────────────────────────────────────────────────────────
def _ok_base(f, e, base):
    return base & np.isfinite(f) & np.isfinite(e) & (e > 0)

def count_pixels(dataset):
    n_fi = int(_ok_base(f_fire, e_fire, BASE_FIRE).sum())
    if dataset == 'fire':
        return n_fi
    elif dataset == 'modsr_fire':
        n_m = int(_ok_base(f_modsr_only, e_modsr_only, BASE_MODSR_ONLY).sum())
        return n_m + n_fi
    else:  # mods_fire
        n_mb = int(_ok_base(f_modsb, e_modsb, BASE_MODSB).sum())
        n_mr = int(_ok_base(f_modsr, e_modsr, BASE_MODSR).sum())
        return n_mb + n_mr + n_fi

# ── Likelihood factory ─────────────────────────────────────────────────────────
def make_logl(dataset, ncomp, use_agn, ext_fn):
    """Returns a dynesty log-likelihood function for the given configuration."""

    def logl(theta):
        # ── Parse theta ───────────────────────────────────────────────────────
        if ncomp == 1:
            T_hot, logg_hot, av, mh_h = theta[0], theta[1], theta[2], theta[3]
            alpha_pl = theta[4] if use_agn else None
            sigma_v  = theta[4 + int(use_agn)]
        else:
            T_hot, logg_hot = theta[0], theta[1]
            T_cool, logg_cool = theta[2], theta[3]
            av, mh_h, mh_c = theta[4], theta[5], theta[6]
            if T_hot <= T_cool: return -1e300
            alpha_pl = theta[7] if use_agn else None
            sigma_v  = theta[7 + int(use_agn)]

        # ── FIRE templates ────────────────────────────────────────────────────
        fm_f_raw = model_fire(T_hot, logg_hot, mh_h)
        if fm_f_raw is None: return -1e300
        ext_f    = ext_fn(wf, av)
        fhot_f   = broaden_f(fm_f_raw * ext_f, sigma_v)

        if ncomp == 2:
            fc_f_raw = model_fire(T_cool, logg_cool, mh_c)
            if fc_f_raw is None: return -1e300
            fcool_f  = broaden_f(fc_f_raw * ext_f, sigma_v)

        if use_agn:
            fagn_f = (wf / AGN_PIVOT)**(-alpha_pl) * ext_f

        # ── MODS templates (only when MODS arm present) ───────────────────────
        if dataset != 'fire':
            fm_m_raw = model_mods(T_hot, logg_hot, mh_h)
            if fm_m_raw is None: return -1e300
            ext_m  = ext_fn(wm, av)
            fhot_m = broaden_m(fm_m_raw * ext_m, sigma_v)
            if use_agn:
                fagn_m = (wm / AGN_PIVOT)**(-alpha_pl) * ext_m

        # ── Pixel masks ───────────────────────────────────────────────────────
        ok_f = (_ok_base(f_fire, e_fire, BASE_FIRE)
                & np.isfinite(fhot_f) & (fhot_f > 0))
        if ncomp == 2:   ok_f &= np.isfinite(fcool_f) & (fcool_f > 0)
        if use_agn:      ok_f &= np.isfinite(fagn_f)  & (fagn_f  > 0)

        if dataset == 'modsr_fire':
            ok_m = (_ok_base(f_modsr_only, e_modsr_only, BASE_MODSR_ONLY)
                    & np.isfinite(fhot_m) & (fhot_m > 0))
            if use_agn: ok_m &= np.isfinite(fagn_m) & (fagn_m > 0)
        elif dataset == 'mods_fire':
            ok_mb = (_ok_base(f_modsb, e_modsb, BASE_MODSB)
                     & np.isfinite(fhot_m) & (fhot_m > 0))
            ok_mr = (_ok_base(f_modsr, e_modsr, BASE_MODSR)
                     & np.isfinite(fhot_m) & (fhot_m > 0))
            if use_agn:
                ok_mb &= np.isfinite(fagn_m) & (fagn_m > 0)
                ok_mr &= np.isfinite(fagn_m) & (fagn_m > 0)

        # Minimum pixel guard
        if dataset == 'fire':
            if ok_f.sum() < 20: return -1e300
        elif dataset == 'modsr_fire':
            if ok_m.sum() < 20 or ok_f.sum() < 20: return -1e300
        else:
            if ok_mb.sum() < 20 or ok_mr.sum() < 20 or ok_f.sum() < 20: return -1e300

        # ── Build NNLS column-index map ───────────────────────────────────────
        # All assignments are explicit so residual reconstruction is unambiguous.
        n_amps = 0
        col_hot_MB = col_hot_MR = col_hot_M = None
        col_hot_F = col_cool_F = col_agn = None

        if dataset == 'mods_fire':
            col_hot_MB = n_amps; n_amps += 1
            col_hot_MR = n_amps; n_amps += 1
        elif dataset == 'modsr_fire':
            col_hot_M  = n_amps; n_amps += 1
        col_hot_F = n_amps; n_amps += 1
        if ncomp == 2:
            col_cool_F = n_amps; n_amps += 1
        if use_agn:
            col_agn    = n_amps; n_amps += 1

        # ── Build concatenated NNLS matrix ────────────────────────────────────
        nF = ok_f.sum()
        bw_f = wt_fire[ok_f]
        sw_f = np.sqrt(bw_f) / e_fire[ok_f]

        if dataset == 'fire':
            A = np.zeros((nF, n_amps))
            b = f_fire[ok_f] * sw_f
            A[:, col_hot_F] = fhot_f[ok_f] * sw_f
            if ncomp == 2: A[:, col_cool_F] = fcool_f[ok_f] * sw_f
            if use_agn:    A[:, col_agn]    = fagn_f[ok_f]  * sw_f

        elif dataset == 'modsr_fire':
            nM   = ok_m.sum()
            bw_m = wt_mods[ok_m]
            sw_m = np.sqrt(bw_m) / e_modsr_only[ok_m]
            A = np.zeros((nM + nF, n_amps))
            b = np.concatenate([f_modsr_only[ok_m]*sw_m, f_fire[ok_f]*sw_f])
            A[:nM,   col_hot_M] = fhot_m[ok_m] * sw_m
            A[nM:,   col_hot_F] = fhot_f[ok_f] * sw_f
            if ncomp == 2:
                A[nM:, col_cool_F] = fcool_f[ok_f] * sw_f
            if use_agn:
                A[:nM, col_agn]  = fagn_m[ok_m] * sw_m   # shared AGN
                A[nM:, col_agn]  = fagn_f[ok_f] * sw_f

        else:  # mods_fire
            nMB  = ok_mb.sum(); nMR = ok_mr.sum()
            bw_mb = wt_mods[ok_mb]; bw_mr = wt_mods[ok_mr]
            sw_mb = np.sqrt(bw_mb) / e_modsb[ok_mb]
            sw_mr = np.sqrt(bw_mr) / e_modsr[ok_mr]
            rMB = slice(0, nMB)
            rMR = slice(nMB, nMB+nMR)
            rF  = slice(nMB+nMR, nMB+nMR+nF)
            A = np.zeros((nMB+nMR+nF, n_amps))
            b = np.concatenate([f_modsb[ok_mb]*sw_mb,
                                 f_modsr[ok_mr]*sw_mr,
                                 f_fire[ok_f]*sw_f])
            A[rMB, col_hot_MB] = fhot_m[ok_mb] * sw_mb
            A[rMR, col_hot_MR] = fhot_m[ok_mr] * sw_mr
            A[rF,  col_hot_F]  = fhot_f[ok_f]  * sw_f
            if ncomp == 2:
                A[rF, col_cool_F] = fcool_f[ok_f] * sw_f
            if use_agn:
                A[rMB, col_agn] = fagn_m[ok_mb] * sw_mb   # shared AGN
                A[rMR, col_agn] = fagn_m[ok_mr] * sw_mr
                A[rF,  col_agn] = fagn_f[ok_f]  * sw_f

        # ── Solve NNLS ────────────────────────────────────────────────────────
        amps, _ = nnls(A, b)
        if amps.sum() == 0: return -1e300

        # ── Compute chi-squared ───────────────────────────────────────────────
        chi2 = 0.0

        if dataset == 'modsr_fire':
            res_m = f_modsr_only[ok_m] - amps[col_hot_M]*fhot_m[ok_m]
            if use_agn: res_m -= amps[col_agn]*fagn_m[ok_m]
            chi2 += np.sum(bw_m * (res_m/e_modsr_only[ok_m])**2)

        elif dataset == 'mods_fire':
            res_mb = f_modsb[ok_mb] - amps[col_hot_MB]*fhot_m[ok_mb]
            res_mr = f_modsr[ok_mr] - amps[col_hot_MR]*fhot_m[ok_mr]
            if use_agn:
                res_mb -= amps[col_agn]*fagn_m[ok_mb]
                res_mr -= amps[col_agn]*fagn_m[ok_mr]
            chi2 += np.sum(bw_mb * (res_mb/e_modsb[ok_mb])**2)
            chi2 += np.sum(bw_mr * (res_mr/e_modsr[ok_mr])**2)

        model_f = amps[col_hot_F] * fhot_f[ok_f]
        if ncomp == 2: model_f += amps[col_cool_F] * fcool_f[ok_f]
        if use_agn:    model_f += amps[col_agn]    * fagn_f[ok_f]
        res_f = f_fire[ok_f] - model_f
        chi2 += np.sum(bw_f * (res_f/e_fire[ok_f])**2)

        return -0.5 * chi2

    return logl

# ── Prior transform factory ────────────────────────────────────────────────────
def make_prior(ncomp, use_agn, av_max):
    def prior_transform(u):
        p = []
        p.append(3000 + 5000*u[0])         # T_hot:   [3000, 8000] K
        p.append(-4.0  + 5.5*u[1])          # logg_hot: [-4.0, 1.5]
        if ncomp == 2:
            p.append(1500 + 2000*u[2])      # T_cool: [1500, 3500] K
            p.append(-4.5  + 5.5*u[3])      # logg_cool: [-4.5, 1.0]
            p.append(av_max * u[4])          # A_V: [0, av_max]
            p.append(-0.49 + 2.98*u[5])     # MH_hot
            p.append(-0.49 + 2.98*u[6])     # MH_cool
            if use_agn:
                p.append(-1.0 + 5.0*u[7])   # alpha_PL: [-1, 4]
                p.append(600.0*u[8])         # sigma_v: [0, 600] km/s
            else:
                p.append(600.0*u[7])
        else:
            p.append(av_max * u[2])          # A_V
            p.append(-0.49 + 2.98*u[3])     # MH_hot
            if use_agn:
                p.append(-1.0 + 5.0*u[4])   # alpha_PL
                p.append(600.0*u[5])
            else:
                p.append(600.0*u[4])
        return np.array(p)
    return prior_transform

NDIM_MAP = {(1, True): 6, (1, False): 5, (2, True): 9, (2, False): 8}
LABELS_MAP = {
    (1, True):  ['T_hot','logg_hot','A_V','MH_hot','alpha_PL','sigma_v'],
    (1, False): ['T_hot','logg_hot','A_V','MH_hot','sigma_v'],
    (2, True):  ['T_hot','logg_hot','T_cool','logg_cool','A_V',
                  'MH_hot','MH_cool','alpha_PL','sigma_v'],
    (2, False): ['T_hot','logg_hot','T_cool','logg_cool','A_V',
                  'MH_hot','MH_cool','sigma_v'],
}

# ── Variants list (72 total) ───────────────────────────────────────────────────
DATASETS = ['fire', 'modsr_fire', 'mods_fire']
NCOMPS   = [1, 2]
USE_AGNS = [True, False]
EXTS     = ['calzetti', 'smc', 'mw']
AV_MAXS  = [8.0, 0.5]

VARIANTS = []
for ds in DATASETS:
    for nc in NCOMPS:
        for ag in USE_AGNS:
            for ex in EXTS:
                for av in AV_MAXS:
                    av_tag  = 'av8' if av == 8.0 else 'av05'
                    agn_str = 'agn' if ag else 'noagn'
                    tag     = f'{ds}_{nc}comp_{agn_str}_{ex}_{av_tag}'
                    VARIANTS.append(dict(dataset=ds, ncomp=nc, use_agn=ag,
                                         ext=ex, av_max=av, av_tag=av_tag,
                                         agn_str=agn_str, tag=tag))

assert len(VARIANTS) == 72, f"Expected 72 variants, got {len(VARIANTS)}"

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Egg broadband fits — 72 variants')
    ap.add_argument('--variant', type=int, default=None,
                    help='Variant index 0–71')
    ap.add_argument('--dataset',  choices=DATASETS, default=None)
    ap.add_argument('--ncomp',    type=int, choices=[1,2], default=None)
    ap.add_argument('--agn',      type=int, choices=[0,1], default=None)
    ap.add_argument('--ext',      choices=list(EXT_FNS), default=None)
    ap.add_argument('--av',       type=float, default=None,
                    help='A_V upper limit: 0.5 or 8.0')
    ap.add_argument('--list',     action='store_true',
                    help='Print all 72 variants and exit')
    args = ap.parse_args()

    if args.list:
        for i, v in enumerate(VARIANTS):
            done = os.path.exists(os.path.join(OUT_DIR, f"egg_chain_{v['tag']}.npy"))
            status = 'DONE' if done else '    '
            print(f"[{i:2d}] {status}  {v['tag']}")
        sys.exit(0)

    if args.variant is not None:
        v = VARIANTS[args.variant]
    else:
        if None in (args.dataset, args.ncomp, args.agn, args.ext, args.av):
            ap.error('Provide --variant N, or all of --dataset --ncomp --agn --ext --av')
        ds  = args.dataset; nc = args.ncomp; ag = bool(args.agn)
        ex  = args.ext;     av = args.av
        av_tag  = 'av8' if av == 8.0 else 'av05'
        agn_str = 'agn' if ag else 'noagn'
        tag     = f'{ds}_{nc}comp_{agn_str}_{ex}_{av_tag}'
        v = dict(dataset=ds, ncomp=nc, use_agn=ag, ext=ex, av_max=av,
                 av_tag=av_tag, agn_str=agn_str, tag=tag)

    ds, nc, ag, ex, av, tag = (v['dataset'], v['ncomp'], v['use_agn'],
                                v['ext'], v['av_max'], v['tag'])
    ext_fn = EXT_FNS[ex]
    ndim   = NDIM_MAP[(nc, ag)]
    labels = LABELS_MAP[(nc, ag)]
    n_pix  = count_pixels(ds)

    out_chain  = os.path.join(OUT_DIR, f'egg_chain_{tag}.npy')
    out_result = os.path.join(OUT_DIR, f'egg_dynesty_{tag}.pkl')

    print(f'\n{"="*65}', flush=True)
    print(f'The Egg  —  {tag}', flush=True)
    print(f'  dataset={ds}  ncomp={nc}  agn={ag}  ext={ex}  av_max={av}', flush=True)
    print(f'  ndim={ndim}  n_pix={n_pix}', flush=True)

    if os.path.exists(out_chain):
        print(f'  Already exists — skipping.', flush=True)
        sys.exit(0)

    logl  = make_logl(ds, nc, ag, ext_fn)
    prior = make_prior(nc, ag, av)

    sampler = dynesty.DynamicNestedSampler(
        logl, prior, ndim, nlive=600, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=0.01, nlive_init=300, nlive_batch=600,
        wt_kwargs={'pfrac': 1.0}, n_effective=8000, print_progress=True)
    elapsed = time.time() - t0
    print(f'\n  Elapsed: {elapsed:.0f}s', flush=True)

    res     = sampler.results
    weights = np.exp(res.logwt - res.logz[-1])
    samples = resample_equal(res.samples, weights)

    logz     = res.logz[-1]; logz_err = res.logzerr[-1]
    chi2_min = -2.0 * np.max(res.logl)
    bic      = chi2_min + ndim * np.log(n_pix)

    print(f'  log Z = {logz:.2f} ± {logz_err:.2f}', flush=True)
    print(f'  BIC   = {bic:.1f}  (χ²_min={chi2_min:.1f}, k={ndim}, n={n_pix})', flush=True)
    print(f'\n  Posterior medians:', flush=True)
    for i, lbl in enumerate(labels):
        lo, med, hi = np.percentile(samples[:,i], [16,50,84])
        print(f'    {lbl:16s}: {med:8.3f}  +{hi-med:.3f} -{med-lo:.3f}')

    np.save(out_chain, samples)
    with open(out_result, 'wb') as fh:
        pickle.dump(res, fh)
    print(f'\n  Saved {os.path.basename(out_chain)}', flush=True)
