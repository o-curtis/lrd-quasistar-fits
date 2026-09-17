#!/usr/bin/env python3
"""
fit_egg_liu2026_corrected.py — Corrected Liu+2026 model for The Egg.

Fixes all bugs found in fit_egg_liu2026_exact.py vs. Liu+2026 (arXiv:2603.02317):

  1. Likelihood (Eq. C2): model-proportional noise floor, not uniform inflation.
       σ_eff,i² = σ_data,i² + s² · f_model,i²
       ln P = -½ Σ_i [(f_data,i - f_model,i)² / σ_eff,i² + ln(σ_eff,i²)]

  2. Data representation: 200 log-spaced bins in rest-frame 3200–22300 Å,
     not 21,983 raw correlated pixels.

  3. No CaT 15× weighting — all bins equal weight.

  4. Prior ranges matching Liu+2026 Table 1:
       teff     ∈ [4000, 5000] K           (was [3500, 6750])
       tage     ∈ [0.001, 0.012] Gyr       (was [0.01, 3.0])
       logmass  ∈ [4, 9]                   (was [6, 12])
       AV       ∈ [0, 3] mag               (was [0, 4])
       logL_star ∈ [7.42, 11.42] log L_sun (was [5, 14])
       logL_dust ∈ [7.42, 11.42] log L_sun (was [6, 14])
       log_s    ∈ [−2, 1]                  (matches Liu+2026 Table 1; was erroneously [−2, 0])

Still uses TLUSTY basis (no changes to tlusty_basis.py) for direct comparison
with our standard M2/M3 finalization runs.

Usage:
    python fit_egg_liu2026_corrected.py
    python fit_egg_liu2026_corrected.py --dry-run
    python fit_egg_liu2026_corrected.py --no-logs
"""

import os, sys, argparse, time, pickle, functools
import numpy as np
import dynesty
from dynesty.utils import resample_equal

os.environ['SPS_HOME'] = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
sys.path.insert(0, '/home/omc5226/prospector_tlusty_dev/prospector')

import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'model_finalization')

# ── Load fit_egg_prospector_tlusty.py as F ────────────────────────────────────
_fspec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(SCRIPT_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(F)

# ── Load fit_egg_liu2026_exact.py as L (for shared helpers) ──────────────────
_lspec = importlib.util.spec_from_file_location(
    'fit_egg_liu', os.path.join(SCRIPT_DIR, 'fit_egg_liu2026_exact.py'))
L = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(L)

from prospect.models.priors import Uniform
from prospect.models import ProspectorParams
from prospect.sources.tlusty_basis import _smc_transmission

# ── Liu+2026 rebinning parameters ─────────────────────────────────────────────
WAVE_REST_MIN = 3200.0    # Å rest-frame
WAVE_REST_MAX = 22300.0   # Å rest-frame
N_BINS        = 200


# =============================================================================
# Precompute 200-bin data structure
# =============================================================================

def build_rebinned_obs(obs_list, z):
    """
    Pool all masked pixels from the three arms and rebin to N_BINS log-spaced
    bins in rest-frame [WAVE_REST_MIN, WAVE_REST_MAX] Å.

    Returns a dict with:
      wave_rest  (N_BINS,) — rest-frame bin centres (geometric mean of edges)
      wave_obs   (N_BINS,) — observed-frame bin centres
      flux       (N_BINS,) — mean flux per bin [maggies]
      unc        (N_BINS,) — propagated uncertainty = sqrt(Σσ²)/n [maggies]
      mask       (N_BINS,) — True if ≥1 valid pixel falls in the bin
      npix       (N_BINS,) — number of valid pixels per bin
      arm_info   list of (wave_obs_min, wave_obs_max, sigma_inst) per arm
    """
    edges_rest = np.logspace(np.log10(WAVE_REST_MIN), np.log10(WAVE_REST_MAX),
                             N_BINS + 1)
    ctr_rest   = np.sqrt(edges_rest[:-1] * edges_rest[1:])   # geometric mean
    ctr_obs    = ctr_rest * (1.0 + z)

    # Pool all pixels
    all_wrest = np.concatenate([o['wave_rest'] for o in obs_list])
    all_flux  = np.concatenate([o['flux']      for o in obs_list])
    all_unc   = np.concatenate([o['unc']        for o in obs_list])
    all_valid = np.concatenate([o['mask']       for o in obs_list]).astype(bool)

    bin_flux = np.full(N_BINS, np.nan)
    bin_unc  = np.full(N_BINS, np.nan)
    bin_mask = np.zeros(N_BINS, dtype=bool)
    bin_npix = np.zeros(N_BINS, dtype=int)

    for i in range(N_BINS):
        sel = (all_wrest >= edges_rest[i]) & (all_wrest < edges_rest[i + 1]) & all_valid
        n = int(sel.sum())
        if n >= 1:
            bin_flux[i] = np.mean(all_flux[sel])
            bin_unc[i]  = np.sqrt(np.sum(all_unc[sel] ** 2)) / n
            bin_mask[i] = True
            bin_npix[i] = n

    arm_info = [(o['wave_obs'].min(), o['wave_obs'].max(), float(o['sigma_inst']))
                for o in obs_list]

    return dict(wave_rest=ctr_rest, wave_obs=ctr_obs,
                flux=bin_flux, unc=bin_unc, mask=bin_mask, npix=bin_npix,
                arm_info=arm_info)


# =============================================================================
# Model (prior container)
# =============================================================================

def build_model_corrected(include_logs=True):
    """
    ProspectorParams with Liu+2026 corrected priors.
    k=10 with log_s; k=9 without.
    """
    mp = {
        'teff': {
            'N': 1, 'isfree': True, 'init': 4500.0,
            'prior': Uniform(mini=4000.0, maxi=5000.0),
            'units': 'K',
        },
        'logg': {
            'N': 1, 'isfree': True, 'init': -1.5,
            'prior': Uniform(mini=-3.5, maxi=0.5),
            'units': 'cgs',
        },
        'logL_star': {
            'N': 1, 'isfree': True, 'init': 9.5,
            'prior': Uniform(mini=7.42, maxi=11.42),
            'units': 'log10(L_*/L_sun)  [41–45 erg/s]',
        },
        'xi_mtb': {
            'N': 1, 'isfree': True, 'init': 4.0,
            'prior': Uniform(mini=2.0, maxi=10.0),
            'units': 'km/s microturbulence velocity',
        },
        'logmass': {
            'N': 1, 'isfree': True, 'init': 7.0,
            'prior': Uniform(mini=4.0, maxi=9.0),
            'units': 'log10(M_formed / M_sun)',
        },
        'tage': {
            'N': 1, 'isfree': True, 'init': 0.006,
            'prior': Uniform(mini=0.001, maxi=0.012),
            'units': 'Gyr — galaxy age (1–12 Myr; sfh=1 constant SFR)',
        },
        'AV': {
            'N': 1, 'isfree': True, 'init': 1.5,
            'prior': Uniform(mini=0.0, maxi=3.0),
            'units': 'mag — SMC A_V applied to all components',
        },
        'gas_logu': {
            'N': 1, 'isfree': True, 'init': -2.0,
            'prior': Uniform(mini=-4.0, maxi=-1.0),
            'units': 'log10 ionization parameter',
        },
        'logL_dust': {
            'N': 1, 'isfree': True, 'init': 10.0,
            'prior': Uniform(mini=7.42, maxi=11.42),
            'units': 'log10(L_warm_BB / L_sun), T_bb=1038 K  [41–45 erg/s]',
        },
    }
    if include_logs:
        mp['log_s'] = {
            'N': 1, 'isfree': True, 'init': -1.0,
            'prior': Uniform(mini=-2.0, maxi=1.0),   # Liu+2026 Table 1: [-2, 1], best-fit -1.1
            'units': 'log10 fractional noise floor; s=10^log_s in (0.01, 10.0)',
        }

    return ProspectorParams(mp)


# =============================================================================
# Log-likelihood
# =============================================================================

def _logl_corrected(theta, sps, gal_sps, binned_obs, model, zred, lumdist):
    """
    Liu+2026 Eq. C2 likelihood on 200-bin rebinned data.

    Noise model: σ_eff,i² = σ_data,i² + s² · f_model,i²
                 s = 10^log_s  (fractional floor, s ∈ [0.01, 1])

    ln P = -½ Σ_i [ (f_data,i - f_model,i)² / σ_eff,i² + ln(σ_eff,i²) ]

    The +ln(σ_eff²) normalisation prevents s from growing without bound.
    No CaT weighting — all bins equal weight (Liu+2026 equal-weight bins).
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
    log_s     = float(theta[idx['log_s'].start]) if 'log_s' in idx else -6.0
    s_lin     = 10.0 ** log_s    # linear fractional noise floor

    # ── TLUSTY stellar spectrum ───────────────────────────────────────────────
    wave_tl, spec_tl, _ = sps.tlusty.get_galaxy_spectrum(
        teff=teff, logg=logg, mh_idx=L.MH_IDX_FIXED,
        logL_star=logL_star, xi_mtb=xi_mtb, dust2=0.0)
    if np.any(np.isnan(spec_tl)):
        return -1e300

    # ── Galaxy SPS (constant SFR, sfh=1) ─────────────────────────────────────
    gal_sps.params['add_neb_emission']  = True
    gal_sps.params['add_neb_continuum'] = True
    gal_sps.params['gas_logu']          = gas_logu
    wave_gal, spec_gal_pm = gal_sps.get_spectrum(tage=tage, peraa=False)
    wave_gal = np.array(wave_gal, dtype=float)
    spec_gal = spec_gal_pm * 10.0 ** logmass

    # ── Warm blackbody at 1038 K ──────────────────────────────────────────────
    spec_bb = L._warm_bb_lnu(wave_tl, logL_dust)

    # ── SMC dust attenuation ──────────────────────────────────────────────────
    # Liu+2026: "we redden the spectrum by an SMC-averaged dust law at AV"
    # referring explicitly to the atmosphere only.
    # "We assume no dust extinction for the galaxy as implied by the very
    #  blue UV slope" — galaxy gets NO attenuation.
    # Warm BB: paper is silent; we omit dust for consistency with Liu's intent
    # (the warm BB IS dust emission; further attenuation would require a separate
    # cold foreground screen not described in the paper).
    trans_tl  = _smc_transmission(wave_tl, AV)
    spec_tl   = spec_tl * trans_tl
    # spec_gal: no dust (Liu explicit)
    # spec_bb:  no dust (Liu silent; omitting for consistency)

    # ── Model evaluated at each bin centre (per-arm sigma_inst) ──────────────
    bin_wave_obs = binned_obs['wave_obs']
    arm_info     = binned_obs['arm_info']

    pred = np.full(N_BINS, np.nan)
    for arm_lo, arm_hi, sigma_inst in arm_info:
        in_arm = (bin_wave_obs >= arm_lo) & (bin_wave_obs <= arm_hi)
        if not in_arm.any():
            continue
        bw = bin_wave_obs[in_arm]

        p_tl  = F._smooth_and_interp(wave_tl,  spec_tl,  bw,
                                      L.SIGMA_LRD_KMS, sigma_inst, F.SIGMA_LIB,
                                      zred, lumdist)
        p_gal = F._smooth_and_interp(wave_gal, spec_gal, bw,
                                      0.0, sigma_inst, F.SIGMA_C3K,
                                      zred, lumdist)
        p_bb  = F._smooth_and_interp(wave_tl,  spec_bb,  bw,
                                      0.0, sigma_inst, 0.0,
                                      zred, lumdist)
        pred[in_arm] = p_tl + p_gal + p_bb

    # ── Liu+2026 Eq. C2 likelihood ────────────────────────────────────────────
    valid = binned_obs['mask'] & np.isfinite(pred)
    if valid.sum() < 20:
        return -1e300

    fd      = binned_obs['flux'][valid]
    fm      = pred[valid]
    sigma_d = binned_obs['unc'][valid]

    # σ_eff² = σ_data² + s²·f_model²  (model-proportional noise floor)
    sigma_eff2 = sigma_d ** 2 + s_lin ** 2 * fm ** 2
    if np.any(sigma_eff2 <= 0):
        return -1e300

    residuals = fd - fm
    lnp = -0.5 * float(np.sum(residuals ** 2 / sigma_eff2 + np.log(sigma_eff2)))

    if not np.isfinite(lnp):
        return -1e300
    return lnp


# =============================================================================
# Cached obs
# =============================================================================

_obs_cache     = None
_binned_cache  = None

def _get_obs():
    global _obs_cache
    if _obs_cache is None:
        _obs_cache = F.build_obs()
    return _obs_cache


def _get_binned_obs():
    global _binned_cache
    if _binned_cache is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
        zred    = F.Z_SOURCE
        obs_list = _get_obs()
        _binned_cache = build_rebinned_obs(obs_list, zred)
    return _binned_cache


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-logs',  action='store_true',
                   help='Omit log_s (k=9 instead of k=10)')
    p.add_argument('--dry-run',  action='store_true',
                   help='Build, fire mid-prior logl, report, then exit')
    return p.parse_args()


def main():
    args = parse_args()
    include_logs = not args.no_logs
    os.makedirs(OUT_DIR, exist_ok=True)

    STEM = 'egg_liu2026_corrected' + ('' if include_logs else '_nologs')

    print(f'\n{"=" * 60}', flush=True)
    print(f'The Egg — Liu+2026 corrected model', flush=True)
    print(f'  Fixes: Eq.C2 likelihood + 200-bin rebin + correct priors', flush=True)
    print(f'  include_logs={include_logs}  STEM={STEM}', flush=True)
    print(f'{"=" * 60}\n', flush=True)

    print('Building TLUSTY SPS (include_xi10=True) …', flush=True)
    sps = F.build_sps(include_xi10=True, verbose=True)

    print('Building galaxy SPS (fsps sfh=1) …', flush=True)
    gal_sps = L.build_gal_sps()

    print('Loading observations …', flush=True)
    obs_list = _get_obs()

    print('Prebinning to 200 log-spaced rest-frame bins …', flush=True)
    binned_obs = _get_binned_obs()
    n_valid_bins = int(binned_obs['mask'].sum())
    n_pix_total  = int(binned_obs['npix'][binned_obs['mask']].sum())
    print(f'  Valid bins: {n_valid_bins}/{N_BINS}  '
          f'(from {n_pix_total} raw pixels)', flush=True)

    print('Building corrected Liu+2026 model …', flush=True)
    model  = build_model_corrected(include_logs=include_logs)
    NDIM   = model.ndim
    LABELS = model.theta_labels()

    from astropy.cosmology import FlatLambdaCDM
    cosmo   = FlatLambdaCDM(H0=70, Om0=0.3)
    zred    = F.Z_SOURCE
    lumdist = cosmo.luminosity_distance(zred).to('Mpc').value

    print(f'\nNDIM={NDIM}  |  valid bins={n_valid_bins}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = _logl_corrected(th_mid, sps, gal_sps, binned_obs, model, zred, lumdist)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    if args.dry_run:
        # Report binning diagnostics
        print('\n--- Binning summary ---', flush=True)
        print(f'  Bin edges: {WAVE_REST_MIN:.0f}–{WAVE_REST_MAX:.0f} Å rest, '
              f'log-spaced, Δλ/λ = {np.log(WAVE_REST_MAX/WAVE_REST_MIN)/N_BINS:.4f}', flush=True)
        print(f'  Mid-prior params:', flush=True)
        for lbl, val in zip(LABELS, th_mid):
            print(f'    {lbl:<14s} = {val:.5g}', flush=True)
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

    logl_fn = functools.partial(
        _logl_corrected, sps=sps, gal_sps=gal_sps,
        binned_obs=binned_obs, model=model, zred=zred, lumdist=lumdist)

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
    print(f'\nElapsed: {elapsed:.0f} s ({elapsed/3600:.1f} h)', flush=True)

    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)

    pkl_path = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)

    map_idx = np.argmax(res.logl)
    map_th  = res.samples[map_idx]
    print(f'\nMAP ln(L) = {res.logl[map_idx]:.2f}', flush=True)
    print(f'ln(Z)     = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    for lbl, val in zip(LABELS, map_th):
        print(f'  {lbl:<14s} = {val:.5g}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
