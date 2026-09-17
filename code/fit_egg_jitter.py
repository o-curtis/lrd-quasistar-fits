#!/usr/bin/env python3
"""
fit_egg_jitter.py — M3 σ-free z-free model for The Egg + PER-INSTRUMENT jitter.

Takes the BIC/lnZ-winning model behind spectra_egg_duste_sfree_zfree.png
(M3 = TLUSTY photosphere + continuity-SFH galaxy + Draine&Li dust emission;
sigma_smooth free, logzsol free, mh free, ndim=18) and adds one noise-rescaling
("jitter") parameter PER spectrograph arm (MODS_B, MODS_R, FIRE), so each arm
carries its own systematic floor instead of the single global F_SYS=0.05.

Motivation
----------
In the fiducial run sigma_smooth (→135) and xi_mtb (→10) rail against their
upper limits because _logl treats 20,559 correlated pixels as independent with
near-raw errors — the likelihood is overconfident and the sampler exploits the
broadening knobs to shave χ². The Liu+2026-corrected run (which adds a noise
floor) unpins xi_mtb 10 → 8.7 with believable errors. Per-instrument jitter
brings that fix into the main M3 model while letting MODS-B (Balmer break) and
FIRE (tellurics) self-calibrate independently.

Two noise models (flag --noise-model):
  mult  : σ_eff,i = σ_obs,i · 10^(−log_s_arm)              (Liu+2026 "exact" port)
            lnL += −0.5·χ²_arm·10^(2·log_s_arm) + n_eff_arm·log_s_arm·ln10
          A per-arm scalar inflation. Fixes error bars + re-balances arms; may
          only partially relax the σ/ξ railing (preserves within-arm χ² shape).
  floor : σ_eff,i² = σ_obs,i² + (s_arm·f_model,i)², s_arm = 10^log_s_arm
            lnL += −0.5·Σ w_i·[ r_i²/σ_eff,i² + ln(σ_eff,i²) ]   (Liu "corrected" port)
          Model-proportional floor — de-weights line cores/continuum peaks;
          this is the form that demonstrably unpinned ξ. RECOMMENDED.

The 15× CaT weighting (8400–8750 Å rest) is preserved by default to match the
fiducial model exactly; disable with --no-cat-weight.

Everything else (priors, SPS, obs, sigma_smooth Uniform(115,135), masks, FIRE
flux scaling) is identical to the fiducial M3 σ-free z-free run.

Usage:
    python fit_egg_jitter.py --noise-model floor
    python fit_egg_jitter.py --noise-model mult
    python fit_egg_jitter.py --noise-model floor --dry-run
    python fit_egg_jitter.py --noise-model floor --no-cat-weight

Output (model_finalization/):
    dynesty_egg_duste_sfree_zfree_jitter_{form}.pkl
    chain_egg_duste_sfree_zfree_jitter_{form}.npy
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

# ── Load fit_egg_prospector_tlusty.py as module F ────────────────────────────
_fspec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(SCRIPT_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(F)

from prospect.models.priors import Uniform
from prospect.models import ProspectorParams
from prospect.models.transforms import logsfr_ratios_to_masses

_LN10 = np.log(10.0)

# arm name (obs['name']) → jitter parameter name
JITTER_NAME = {'MODS_B': 'log_s_modsb', 'MODS_R': 'log_s_modsr', 'FIRE': 'log_s_fire'}


# =============================================================================
# Model: M3 σ-free z-free + 3 per-arm jitter params
# =============================================================================

def build_model_jitter():
    """M3 σ-free z-free (ndim=18) plus log_s_modsb/modsr/fire ⇒ ndim=21."""
    m0 = F.build_model_variant('M3', sigma_mode='free', zsol_mode='free',
                               mh_mode='free', neb_mode='off')
    mp = m0.config_dict
    for pname in ('log_s_modsb', 'log_s_modsr', 'log_s_fire'):
        mp[pname] = {
            'N': 1, 'isfree': True, 'init': -0.5,
            'prior': Uniform(mini=-2.0, maxi=1.0),   # Liu+2026 Table 1 range
            'units': 'per-arm log10 noise rescale (s=10^log_s)',
        }
    m = ProspectorParams(mp)
    m.configure(reset=True)
    return m


# =============================================================================
# Log-likelihood with per-instrument jitter
#   Forward model replicates F._logl exactly; only the noise term differs.
# =============================================================================

def _logl_jitter(theta, sps, obs_list, model, noise_model='floor', cat_weight=True):
    idx = model.theta_index

    # ── Photosphere ──────────────────────────────────────────────────────────
    teff         = float(theta[idx['teff']])
    logg         = float(theta[idx['logg']])
    logL_star    = float(theta[idx['logL_star']])
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    mh_idx       = (float(theta[idx['mh_idx']]) if 'mh_idx' in idx
                    else float(model.params['mh_idx']))
    xi_mtb       = (float(theta[idx['xi_mtb']]) if 'xi_mtb' in idx
                    else float(model.params.get('xi_mtb', 2.0)))

    # ── Galaxy ───────────────────────────────────────────────────────────────
    logmass       = float(theta[idx['logmass']])
    logzsol       = (float(theta[idx['logzsol']]) if 'logzsol' in idx
                     else mh_idx - 2.0)
    dust2_gal     = float(theta[idx['dust2_gal']])
    dust1         = float(theta[idx['dust1']])
    logsfr_ratios = theta[idx['logsfr_ratios']]

    agebins = np.array(model.params['agebins'])
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])

    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)

    add_dust_emission = 'duste_umin' in idx
    add_neb_emission  = 'gas_logu' in idx
    sps_params = dict(
        teff=teff, logg=logg, mh_idx=mh_idx, xi_mtb=xi_mtb,
        logL_star=logL_star, logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1,
        logsfr_ratios=logsfr_ratios, mass=mass, agebins=agebins,
        sfh=int(model.params['sfh']),
        imf_type=int(model.params['imf_type']),
        dust_type=int(model.params['dust_type']),
        dust1_index=float(model.params['dust1_index']),
        add_neb_emission=add_neb_emission,
        add_neb_continuum=add_neb_emission,
        add_dust_emission=add_dust_emission, zred=zred,
    )
    if add_neb_emission:
        sps_params['gas_logu'] = float(theta[idx['gas_logu']])
        sps_params['gas_logz'] = float(model.params.get('gas_logz', -1.0))
    if add_dust_emission:
        sps_params['duste_umin']  = float(theta[idx['duste_umin']])
        sps_params['duste_qpah']  = float(theta[idx['duste_qpah']])
        sps_params['duste_gamma'] = float(theta[idx['duste_gamma']])

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)
    if np.any(np.isnan(spec_tl)):
        return -1e300

    lnp = 0.0
    for obs in obs_list:
        sigma_inst = obs['sigma_inst']
        log_s = float(theta[idx[JITTER_NAME[obs['name']]]])

        pred_tl  = F._smooth_and_interp(wave, spec_tl,  obs['wave_obs'],
                                        sigma_smooth, sigma_inst, F.SIGMA_LIB,
                                        zred, lumdist)
        pred_gal = F._smooth_and_interp(wave, spec_gal, obs['wave_obs'],
                                        0.0, sigma_inst, F.SIGMA_C3K,
                                        zred, lumdist)
        pred = pred_tl + pred_gal
        mask = obs['mask'] & np.isfinite(pred) & (pred > 0)
        if mask.sum() < 10:
            return -1e300

        res = obs['flux'][mask] - pred[mask]
        unc = obs['unc'][mask]
        if cat_weight:
            cat_sel = ((obs['wave_rest'][mask] >= 8400.) &
                       (obs['wave_rest'][mask] <= 8750.))
            w = np.where(cat_sel, 15.0, 1.0)
        else:
            w = np.ones_like(res)

        if noise_model == 'mult':
            # σ_eff = σ_obs·10^(−log_s); χ² scaled by 10^(2 log_s); + n_eff·log_s·ln10
            chi2  = np.sum(w * (res / unc) ** 2)
            n_eff = float(w.sum())
            f2    = 10.0 ** (2.0 * log_s)
            lnp  += -0.5 * chi2 * f2 + n_eff * log_s * _LN10
        elif noise_model == 'floor':
            # σ_eff² = σ_obs² + (s·f_model)²,  s = 10^log_s  (model-proportional)
            s_lin     = 10.0 ** log_s
            sigma_eff2 = unc ** 2 + (s_lin * pred[mask]) ** 2
            if np.any(sigma_eff2 <= 0):
                return -1e300
            lnp += -0.5 * np.sum(w * (res ** 2 / sigma_eff2 + np.log(sigma_eff2)))
        else:
            raise ValueError(f"noise_model must be mult/floor, got {noise_model!r}")

    return float(lnp) if np.isfinite(lnp) else -1e300


# =============================================================================
# Cached obs
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
    p = argparse.ArgumentParser(description='Egg M3 σ-free z-free + per-instrument jitter')
    p.add_argument('--noise-model', default='floor', choices=['floor', 'mult'],
                   help='floor=model-proportional (recommended) | mult=per-arm scalar inflation')
    p.add_argument('--no-cat-weight', action='store_true',
                   help='disable the 15× CaT (8400–8750 Å) weighting')
    p.add_argument('--dry-run', action='store_true',
                   help='build, fire mid-prior logl, then exit')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    cat_weight = not args.no_cat_weight

    STEM = f'egg_duste_sfree_zfree_jitter_{args.noise_model}'
    if not cat_weight:
        STEM += '_nocat'

    print(f'\n{"="*64}', flush=True)
    print(f'The Egg — M3 σ-free z-free + per-instrument jitter', flush=True)
    print(f'  noise_model={args.noise_model}  cat_weight={cat_weight}', flush=True)
    print(f'  STEM={STEM}', flush=True)
    print(f'{"="*64}\n', flush=True)

    print('Building SPS (include_xi10=True) …', flush=True)
    sps = F.build_sps(include_xi10=True, verbose=True)
    print('Loading observations …', flush=True)
    obs_list = _get_obs()
    print('Building M3 + jitter model …', flush=True)
    model = build_model_jitter()

    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)
    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = _logl_jitter(th_mid, sps, obs_list, model,
                          noise_model=args.noise_model, cat_weight=cat_weight)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    if args.dry_run:
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

    logl_fn = functools.partial(_logl_jitter, sps=sps, obs_list=obs_list, model=model,
                                noise_model=args.noise_model, cat_weight=cat_weight)

    print('\nStarting DynamicNestedSampler …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl_fn, model.prior_transform, NDIM,
        nlive=F.NLIVE, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=F.DLOGZ_INIT, nlive_init=F.NLIVE_INIT,
        nlive_batch=F.NLIVE_BATCH, wt_kwargs={'pfrac': 1.0},
        n_effective=F.N_EFFECTIVE, print_progress=True)
    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f} s ({elapsed/3600:.2f} h)', flush=True)

    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)

    pkl_path = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)

    map_idx = int(np.argmax(res.logl))
    print(f'\nln Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    print('Posterior medians:', flush=True)
    for i, lbl in enumerate(LABELS):
        q = np.percentile(samp[:, i], [16, 50, 84])
        print(f'  {lbl:<16} = {q[1]:.4g}  (-{q[1]-q[0]:.3g}/+{q[2]-q[1]:.3g})', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
