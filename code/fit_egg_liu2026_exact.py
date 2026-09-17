#!/usr/bin/env python3
"""
fit_egg_liu2026_exact.py — Exact Liu+2026 model analog for The Egg.

Replicates Liu+2026 (arXiv:2603.02317) Table 1 as closely as possible in
our Prospector/TLUSTY framework, WITHOUT modifying tlusty_basis.py.

Differences from our standard model finalization runs:
  - Galaxy SFH: constant SFR with free age t_gal (sfh=1, tau=1e10 yr),
    NOT non-parametric (replaces 6 SFH params with 1).
  - Dust: single A_V (SMC) applied to all components simultaneously, NOT
    separate dust2_gal + dust1.
  - Warm BB: additive blackbody at fixed T=1038 K, free log L_dust.
  - log_s: optional noise-inflation term (Uniform[-2,1]), same as Liu's
    "Uncertainty log s" parameter.
  - sigma: fixed at 123 km/s (same as our s123 runs).
  - [M/H]: fixed at -1 (mh_idx=1.0, same as our mhfix runs).
  - logzsol: fixed at -1.0.

Free parameters (k=10 by default; k=9 with --no-logs):
  teff, logg, logL_star, xi_mtb, logmass, tage, AV, gas_logu, logL_dust,
  [log_s]

Usage:
    python fit_egg_liu2026_exact.py
    python fit_egg_liu2026_exact.py --no-logs     # omit log_s term
    python fit_egg_liu2026_exact.py --dry-run

Output (in model_finalization/):
    dynesty_egg_liu2026_exact.pkl
    chain_egg_liu2026_exact.npy
  (or ..._nologs variants with --no-logs)
"""

import os, sys, argparse, time, pickle, functools
import numpy as np
import dynesty
from dynesty.utils import resample_equal
from scipy.ndimage import gaussian_filter1d

os.environ['SPS_HOME'] = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
sys.path.insert(0, '/home/omc5226/prospector_tlusty_dev/prospector')

import fsps
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'model_finalization')

# ── Load fit_egg_prospector_tlusty.py as module F ────────────────────────────
_fspec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(SCRIPT_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(F)

from prospect.models.priors import Uniform
from prospect.models import ProspectorParams
from prospect.sources.tlusty_basis import _smc_transmission

# ── Physical constants ────────────────────────────────────────────────────────
_LSUN   = 3.828e33    # erg/s
_C      = 2.998e10    # cm/s
_C_AA   = _C * 1e8   # Å/s
_H      = 6.626e-27   # erg·s
_KB     = 1.381e-16   # erg/K
_SB     = 5.6704e-5   # Stefan-Boltzmann, erg/cm²/s/K⁴

# ── Fixed Liu+2026 parameters ─────────────────────────────────────────────────
SIGMA_LRD_KMS  = 123.0    # km/s — fixed LOS velocity dispersion
MH_IDX_FIXED   = 1.0      # → [M/H] = −1 on our (−2, −1, 0) index grid
LOGZSOL_FIXED  = -1.0     # log(Z_*/Z_sun) for galaxy component, fixed
T_WARM_BB_K    = 1038.0   # K — warm BB temperature, fixed (Liu+2026 best-fit)
GAL_TAU_YR     = 1e10     # yr — tau for sfh=1 → effectively constant SFR


# =============================================================================
# Warm blackbody component
# =============================================================================

def _warm_bb_lnu(wave_aa, logL_dust):
    """
    Warm blackbody at T_WARM_BB_K, normalised to logL_dust solar luminosities.

    Returns L_ν in L_sun/Hz on the given wavelength grid.
    Uses the Stefan-Boltzmann integral for normalisation so that the result is
    correct even though the grid does not extend to all wavelengths.
    """
    nu   = _C_AA / wave_aa                         # Hz
    x    = np.clip(_H * nu / (_KB * T_WARM_BB_K), 1e-10, 500.0)
    B_nu = 2.0 * _H * nu**3 / _C**2 / np.expm1(x) # erg/s/cm²/Hz/sr
    B_tot = _SB * T_WARM_BB_K**4 / np.pi           # erg/s/cm² (analytic integral)
    L_dust_cgs = 10.0**logL_dust * _LSUN           # erg/s
    return L_dust_cgs / _LSUN * B_nu / B_tot       # L_sun/Hz


# =============================================================================
# Galaxy SPS (constant SFR via tau model)
# =============================================================================

def build_gal_sps():
    """
    Build an FSPS StellarPopulation for the galaxy component.

    sfh=1 (exponential tau) with tau=1e10 yr approximates constant SFR.
    Nebular emission is enabled; dust is handled externally (dust2=0).
    """
    gal_sps = fsps.StellarPopulation(
        zcontinuous=1,
        sfh=1,
        add_dust_emission=False,
        add_neb_emission=True,
        add_neb_continuum=True,
        dust_type=0,    # power-law; set dust2=0 so no internal attenuation
        imf_type=1,     # Chabrier
    )
    gal_sps.params['tau']        = GAL_TAU_YR / 1e9    # FSPS expects Gyr
    gal_sps.params['logzsol']    = LOGZSOL_FIXED
    gal_sps.params['dust2']      = 0.0
    gal_sps.params['gas_logz']   = LOGZSOL_FIXED
    return gal_sps


# =============================================================================
# ProspectorParams model (prior container only — logl computed manually)
# =============================================================================

def build_model_liu2026(include_logs=True):
    """
    Return a ProspectorParams object whose prior_transform and theta_index
    define the Liu+2026 parameter space.

    k=10 with log_s (include_logs=True); k=9 without.
    """
    mp = {
        'teff': {
            'N': 1, 'isfree': True, 'init': 5200.0,
            'prior': Uniform(mini=3500.0, maxi=6750.0),
            'units': 'K',
        },
        'logg': {
            'N': 1, 'isfree': True, 'init': -1.5,
            'prior': Uniform(mini=-3.5, maxi=0.5),
            'units': 'cgs',
        },
        'logL_star': {
            'N': 1, 'isfree': True, 'init': 9.5,
            'prior': Uniform(mini=5.0, maxi=14.0),
            'units': 'log10(L_*/L_sun)',
        },
        'xi_mtb': {
            'N': 1, 'isfree': True, 'init': 4.0,
            'prior': Uniform(mini=2.0, maxi=10.0),
            'units': 'km/s microturbulence',
        },
        'logmass': {
            'N': 1, 'isfree': True, 'init': 9.0,
            'prior': Uniform(mini=6.0, maxi=12.0),
            'units': 'log10(M_formed / M_sun)',
        },
        'tage': {
            'N': 1, 'isfree': True, 'init': 0.5,
            'prior': Uniform(mini=0.01, maxi=3.0),
            'units': 'Gyr — galaxy age (constant SFR)',
        },
        'AV': {
            'N': 1, 'isfree': True, 'init': 2.0,
            'prior': Uniform(mini=0.0, maxi=4.0),
            'units': 'mag — SMC A_V applied to all components',
        },
        'gas_logu': {
            'N': 1, 'isfree': True, 'init': -2.0,
            'prior': Uniform(mini=-4.0, maxi=-1.0),
            'units': 'log ionization parameter',
        },
        'logL_dust': {
            'N': 1, 'isfree': True, 'init': 11.0,
            'prior': Uniform(mini=6.0, maxi=14.0),
            'units': 'log10(L_warm_BB / L_sun), T_bb=1038 K',
        },
    }
    if include_logs:
        mp['log_s'] = {
            'N': 1, 'isfree': True, 'init': -1.0,
            'prior': Uniform(mini=-2.0, maxi=1.0),
            'units': 'noise inflation: sigma_eff = sigma_obs * 10^log_s',
        }

    model = ProspectorParams(mp)
    return model


# =============================================================================
# Log-likelihood
# =============================================================================

def _logl_liu2026(theta, sps, gal_sps, obs_list, model, zred, lumdist):
    """
    Log-likelihood for the Liu+2026 exact analog.

    Components
    ----------
    spec_tl  : TLUSTY stellar atmosphere (TLUSTYBasis.get_galaxy_spectrum),
               no dust, then SMC attenuated, then convolved with sigma=123 km/s.
    spec_gal : Galaxy constant-SFR SED (fsps.StellarPopulation, sfh=1),
               no dust, then SMC attenuated, no extra smoothing.
    spec_bb  : Warm blackbody at T=1038 K, L=logL_dust, SMC attenuated.

    All three components are on separate grids; _smooth_and_interp handles
    flux-normalisation, redshifting, and interpolation to obs frame.

    Noise inflation (log_s)
    -----------------------
    When log_s is free:
        ln L = -0.5 * Σ w_i*(r_i/σ_i)²/f² - N_eff*log_s*ln(10)
    where f = 10^log_s and N_eff = Σ w_i.
    The -N_eff*ln(f) term ensures log_s is not pushed to +∞ by chi² alone.
    """
    idx = model.theta_index

    teff      = float(theta[idx['teff'].start])
    logg      = float(theta[idx['logg'].start])
    logL_star = float(theta[idx['logL_star'].start])
    xi_mtb    = float(theta[idx['xi_mtb'].start])
    logmass   = float(theta[idx['logmass'].start])
    tage      = float(theta[idx['tage'].start])
    AV        = float(theta[idx['AV'].start])
    gas_logu  = float(theta[idx['gas_logu'].start])
    logL_dust = float(theta[idx['logL_dust'].start])
    log_s     = float(theta[idx['log_s'].start]) if 'log_s' in idx else 0.0

    # ── TLUSTY stellar spectrum (no dust) ────────────────────────────────────
    wave_tl, spec_tl, _ = sps.tlusty.get_galaxy_spectrum(
        teff=teff, logg=logg, mh_idx=MH_IDX_FIXED,
        logL_star=logL_star, xi_mtb=xi_mtb, dust2=0.0)

    if np.any(np.isnan(spec_tl)):
        return -1e300

    # ── Galaxy spectrum (no dust) ────────────────────────────────────────────
    gal_sps.params['add_neb_emission'] = True
    gal_sps.params['add_neb_continuum'] = True
    gal_sps.params['gas_logu'] = gas_logu
    wave_gal, spec_gal_per_mass = gal_sps.get_spectrum(tage=tage, peraa=False)
    spec_gal = spec_gal_per_mass * 10.0**logmass   # L_sun/Hz

    # ── Warm BB (no dust) ────────────────────────────────────────────────────
    spec_bb = _warm_bb_lnu(wave_tl, logL_dust)     # on TLUSTY wave grid

    # ── SMC dust attenuation ─────────────────────────────────────────────────
    trans_tl  = _smc_transmission(wave_tl,  AV)
    trans_gal = _smc_transmission(np.array(wave_gal, dtype=float), AV)
    spec_tl   = spec_tl  * trans_tl
    spec_gal  = spec_gal * trans_gal
    spec_bb   = spec_bb  * trans_tl    # warm BB on TLUSTY grid

    # ── Noise-inflation factor ────────────────────────────────────────────────
    # Convention matches Liu+2026: log_s < 0 → inflation (sigma_eff > sigma_obs).
    #   sigma_eff = sigma_obs * 10^(-log_s)
    #   chi2_eff  = chi2_base * 10^(+2*log_s)   (smaller for log_s < 0)
    #   ln L += -0.5 * chi2_base * f2 + N_eff * log_s * ln(10)
    # Best-fit log_s ≈ -1.1 → 12.6× inflation of sigma_obs.
    f2 = 10.0 ** (2.0 * log_s)        # chi² multiplier < 1 for log_s < 0

    lnp = 0.0
    for obs in obs_list:
        sigma_inst = obs['sigma_inst']

        # Photosphere: sigma=123, library resolution SIGMA_LIB
        pred_tl = F._smooth_and_interp(
            wave_tl, spec_tl, obs['wave_obs'],
            SIGMA_LRD_KMS, sigma_inst, F.SIGMA_LIB, zred, lumdist)

        # Galaxy: no extra smoothing, library resolution SIGMA_C3K
        pred_gal = F._smooth_and_interp(
            np.array(wave_gal, dtype=float), spec_gal, obs['wave_obs'],
            0.0, sigma_inst, F.SIGMA_C3K, zred, lumdist)

        # Warm BB: smooth blackbody, treat as zero library resolution
        pred_bb = F._smooth_and_interp(
            wave_tl, spec_bb, obs['wave_obs'],
            0.0, sigma_inst, 0.0, zred, lumdist)

        pred = pred_tl + pred_gal + pred_bb
        mask = obs['mask'] & np.isfinite(pred) & (pred > 0)
        if mask.sum() < 10:
            return -1e300

        res  = obs['flux'][mask] - pred[mask]
        unc  = obs['unc'][mask]

        # CaT weighting (same as F._logl for a fair comparison)
        cat_sel = ((obs['wave_rest'][mask] >= 8400.) &
                   (obs['wave_rest'][mask] <= 8750.))
        weights = np.where(cat_sel, 15.0, 1.0)

        chi2_nofactor = np.sum(weights * (res / unc)**2)
        n_eff         = float(weights.sum())

        lnp += -0.5 * chi2_nofactor * f2 + n_eff * log_s * np.log(10.0)

    return float(lnp)


# =============================================================================
# Cached obs list (build_obs is slow; only call once)
# =============================================================================
_obs_cache = None

def _get_obs():
    global _obs_cache
    if _obs_cache is None:
        _obs_cache = F.build_obs()
    return _obs_cache


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Liu+2026 exact analog dynesty run')
    p.add_argument('--no-logs', action='store_true',
                   help='Omit log_s noise-inflation term (k=9 instead of k=10)')
    p.add_argument('--dry-run', action='store_true',
                   help='Build model, fire mid-prior logl, then exit')
    return p.parse_args()


def main():
    args    = parse_args()
    include_logs = not args.no_logs
    os.makedirs(OUT_DIR, exist_ok=True)

    STEM = 'egg_liu2026_exact' + ('' if include_logs else '_nologs')

    print(f'\n{"="*60}', flush=True)
    print(f'The Egg — Liu+2026 exact analog', flush=True)
    print(f'  include_logs={include_logs}  STEM={STEM}', flush=True)
    print(f'  Fixed: sigma={SIGMA_LRD_KMS} km/s  [M/H]=-1  logzsol=-1', flush=True)
    print(f'  Galaxy: sfh=1 (constant SFR, tau={GAL_TAU_YR:.0e} yr)', flush=True)
    print(f'  Warm BB: T={T_WARM_BB_K} K', flush=True)
    print(f'{"="*60}\n', flush=True)

    print('Building TLUSTY SPS (include_xi10=True) …', flush=True)
    sps = F.build_sps(include_xi10=True, verbose=True)

    print('Building galaxy SPS (fsps sfh=1) …', flush=True)
    gal_sps = build_gal_sps()

    print('Loading observations …', flush=True)
    obs_list = _get_obs()

    print('Building Liu+2026 model …', flush=True)
    model = build_model_liu2026(include_logs=include_logs)

    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)

    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    from astropy.cosmology import FlatLambdaCDM
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    zred    = F.Z_SOURCE
    lumdist = cosmo.luminosity_distance(zred).to('Mpc').value

    ll_mid = _logl_liu2026(th_mid, sps, gal_sps, obs_list, model, zred, lumdist)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    if args.dry_run:
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

    logl_fn = functools.partial(
        _logl_liu2026, sps=sps, gal_sps=gal_sps,
        obs_list=obs_list, model=model, zred=zred, lumdist=lumdist)

    print('\nStarting DynamicNestedSampler …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl_fn, model.prior_transform, NDIM,
        nlive=F.NLIVE, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=F.DLOGZ_INIT,
        nlive_init=F.NLIVE_INIT,
        nlive_batch=F.NLIVE_BATCH,
        wt_kwargs={'pfrac': 1.0},
        n_effective=F.N_EFFECTIVE,
        print_progress=True)
    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f} s', flush=True)

    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)

    pkl_path = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
