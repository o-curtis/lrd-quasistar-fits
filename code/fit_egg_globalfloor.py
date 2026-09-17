#!/usr/bin/env python3
"""
fit_egg_globalfloor.py — original M3 σ-free z-free model + TWO changes only:
  (1) DROP the 15× CaT weighting (all pixels equal weight, like Liu+2026).
  (2) Add ONE GLOBAL Eq. C2 noise floor: σ_eff² = σ_obs² + s²·f_model², a single
      free log_s for ALL arms (Liu+2026 Eq. C2 likelihood, global s — NOT per-arm).

Goal: keep T≈4500 and logg≈−3 (the original gets these from the continuum shape —
nothing here touches that), while letting σ_smooth and ξ_mtb come off their rails
(the CaT weighting + raw-pixel overconfidence were what pinned them).

EXPLICITLY NOT changed (vs the original `egg_duste_sfree_zfree` run):
  - M3 model (TLUSTY photosphere + continuity-SFH galaxy + Draine&Li dust), shared SMC dust.
  - σ_smooth prior stays Uniform(115,135)   [change (3) "widen σ prior" NOT applied].
  - ξ_mtb free [2,10], [M/H] free, logzsol free.
  - NO speccal polynomial, NO per-arm jitter.
  - Forward model identical to F._logl; only the noise term + weights differ.

Runs on the CLAMPED tlusty_basis (logg cannot extrapolate below the grid floor).
Imports the core at runtime → R_FIRE = 6000.

Likelihood:
  lnL = −½ Σ_arms Σ_pix [ (d−m)²/σ_eff² + ln(σ_eff²) ],  σ_eff² = σ_obs² + (s·m)²,  s = 10^log_s
  (no weights; single global s; sum over all unmasked pixels in all three arms.)

ndim = 19 (the 18 of M3 σ-free z-free + log_s).

Usage:
    python fit_egg_globalfloor.py --pool 8
    python fit_egg_globalfloor.py --pool 8 --dry-run

Output (model_finalization/):
    dynesty_egg_duste_sfree_zfree_globalfloor.pkl
    chain_egg_duste_sfree_zfree_globalfloor.npy
"""

import os, sys, argparse, time, pickle
import numpy as np
import dynesty
from dynesty.utils import resample_equal

os.environ['SPS_HOME'] = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
sys.path.insert(0, '/home/omc5226/prospector_tlusty_dev/prospector')

import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'model_finalization')


def _load(name, fname):
    sp = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    m  = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


F = _load('fit_egg', 'fit_egg_prospector_tlusty.py')

from prospect.models.priors import Uniform
from prospect.models import ProspectorParams
from prospect.models.transforms import logsfr_ratios_to_masses

# ── module globals for the fork pool ─────────────────────────────────────────
_SPS = None
_OBS = None
_MODEL = None


def build_model_globalfloor(sigma_mode='free', zsol_mode='free'):
    """M3 + one global log_s (Eq. C2 noise floor). sigma_mode/zsol_mode are 'free' or
    'fixed'; 'fixed' sets sigma_smooth=123 km/s and logzsol=mh_idx-2 (~-1) — the Liu+2026
    values. ndim = 19 (both free) down to 17 (both fixed)."""
    m0 = F.build_model_variant('M3', sigma_mode=sigma_mode, zsol_mode=zsol_mode,
                               mh_mode='free', neb_mode='off')
    mp = m0.config_dict
    mp['log_s'] = {
        'N': 1, 'isfree': True, 'init': -1.0,
        'prior': Uniform(mini=-2.0, maxi=1.0),     # Liu+2026 Table 1 range
        'units': 'global Eq.C2 noise floor: sigma_eff^2 = sigma_obs^2 + (10^log_s * f_model)^2',
    }
    m = ProspectorParams(mp)
    m.configure(reset=True)
    return m


def _logl_globalfloor(theta, sps, obs_list, model):
    """Forward model identical to F._logl; global Eq.C2 floor; NO CaT weighting."""
    idx = model.theta_index

    teff         = float(theta[idx['teff']])
    logg         = float(theta[idx['logg']])
    logL_star    = float(theta[idx['logL_star']])
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    mh_idx       = (float(theta[idx['mh_idx']]) if 'mh_idx' in idx
                    else float(model.params['mh_idx']))
    xi_mtb       = (float(theta[idx['xi_mtb']]) if 'xi_mtb' in idx
                    else float(model.params.get('xi_mtb', 2.0)))

    logmass       = float(theta[idx['logmass']])
    logzsol       = (float(theta[idx['logzsol']]) if 'logzsol' in idx else mh_idx - 2.0)
    dust2_gal     = float(theta[idx['dust2_gal']])
    dust1         = float(theta[idx['dust1']])
    logsfr_ratios = theta[idx['logsfr_ratios']]
    log_s         = float(theta[idx['log_s']])

    agebins = np.array(model.params['agebins'])
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])

    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)

    add_dust_emission = 'duste_umin' in idx
    sps_params = dict(
        teff=teff, logg=logg, mh_idx=mh_idx, xi_mtb=xi_mtb,
        logL_star=logL_star, logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1,
        logsfr_ratios=logsfr_ratios, mass=mass, agebins=agebins,
        sfh=int(model.params['sfh']), imf_type=int(model.params['imf_type']),
        dust_type=int(model.params['dust_type']),
        dust1_index=float(model.params['dust1_index']),
        add_neb_emission=False, add_neb_continuum=False,
        add_dust_emission=add_dust_emission, zred=zred,
    )
    if add_dust_emission:
        sps_params['duste_umin']  = float(theta[idx['duste_umin']])
        sps_params['duste_qpah']  = float(theta[idx['duste_qpah']])
        sps_params['duste_gamma'] = float(theta[idx['duste_gamma']])

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)
    if np.any(np.isnan(spec_tl)):
        return -1e300

    s_lin = 10.0 ** log_s
    lnp = 0.0
    for obs in obs_list:
        sigma_inst = obs['sigma_inst']
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
        sigma_eff2 = unc ** 2 + (s_lin * pred[mask]) ** 2          # global s; Eq. C2
        if np.any(sigma_eff2 <= 0):
            return -1e300
        lnp += -0.5 * np.sum(res ** 2 / sigma_eff2 + np.log(sigma_eff2))  # NO CaT weight

    return float(lnp) if np.isfinite(lnp) else -1e300


def _logl_global(theta):
    return _logl_globalfloor(theta, _SPS, _OBS, _MODEL)


def _ptform_global(u):
    return _MODEL.prior_transform(u)


def parse_args():
    p = argparse.ArgumentParser(description='Egg M3 + global Eq.C2 floor + no CaT weight')
    p.add_argument('--pool', type=int, default=8)
    p.add_argument('--sigma', default='free', choices=['free', 'fixed'],
                   help="fixed = sigma_smooth=123 km/s (Liu+2026)")
    p.add_argument('--zsol', default='free', choices=['free', 'fixed'],
                   help="fixed = logzsol=mh_idx-2 (~-1, Liu+2026 galaxy [M/H])")
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    global _SPS, _OBS, _MODEL
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    stag = {'free': 'sfree', 'fixed': 's123'}[args.sigma]
    ztag = {'free': 'zfree', 'fixed': 'zfix'}[args.zsol]
    STEM = f'egg_duste_{stag}_{ztag}_globalfloor'

    print(f'\n{"="*70}', flush=True)
    print(f'The Egg — M3 + GLOBAL Eq.C2 floor + NO CaT weight (sigma={args.sigma}, zsol={args.zsol})', flush=True)
    print(f'  pool={args.pool}  R_FIRE={F.R_FIRE}  STEM={STEM}', flush=True)
    print(f'  (no speccal; no per-arm jitter; sigma fixed=123 / zsol fixed=mh_idx-2 when "fixed")', flush=True)
    print(f'{"="*70}\n', flush=True)

    print('Building SPS (shared dust, include_xi10) …', flush=True)
    _SPS = F.build_sps(include_xi10=True, verbose=True)
    print('Loading observations …', flush=True)
    _OBS = F.build_obs()
    print(f'Building M3 + global-floor model (sigma={args.sigma}, zsol={args.zsol}) …', flush=True)
    _MODEL = build_model_globalfloor(sigma_mode=args.sigma, zsol_mode=args.zsol)

    NDIM   = _MODEL.ndim
    LABELS = _MODEL.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in _OBS)
    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = _ptform_global(u_mid)
    print(f'Mid-prior ln L = {_logl_global(th_mid):.1f}', flush=True)

    if args.dry_run:
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(args.pool)
    print(f'\nStarting DynamicNestedSampler over {args.pool}-worker fork pool …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        _logl_global, _ptform_global, NDIM,
        bound='multi', sample='rwalk', pool=pool, queue_size=args.pool)
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=F.DLOGZ_INIT, nlive_init=F.NLIVE_INIT,
        nlive_batch=F.NLIVE_BATCH, wt_kwargs={'pfrac': 1.0},
        n_effective=F.N_EFFECTIVE, print_progress=True)
    elapsed = time.time() - t0
    pool.close(); pool.join()
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

    print(f'\nln Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    print('Posterior medians:', flush=True)
    for i, lbl in enumerate(LABELS):
        q = np.percentile(samp[:, i], [16, 50, 84])
        print(f'  {lbl:<16} = {q[1]:.4g}  (-{q[1]-q[0]:.3g}/+{q[2]-q[1]:.3g})', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
