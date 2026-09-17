#!/usr/bin/env python3
"""
fit_egg_jitter_dustsep.py — floor-jitter Egg run with DECOUPLED photosphere/galaxy dust.

Identical to fit_egg_jitter.py --noise-model floor (M3 σ-free z-free + per-arm
model-proportional jitter), with exactly ONE model change:

  phot_dust_mode='independent' in the TLUSTY basis  →  the photosphere gets its
  own free A_V (av_phot, SMC law) instead of sharing the galaxy's dust2_gal.
  dust2_gal now attenuates the GALAXY only.  ndim 21 → 22.

Everything else is unchanged: priors, continuity SFH, Draine&Li dust emission,
sigma_smooth Uniform(115,135), [M/H] free, xi_mtb, the three per-arm jitter
floors (log_s_modsb/modsr/fire), CaT 15× weighting, dynesty settings.

Speed: runs dynesty over a multiprocessing (fork) Pool so likelihood calls
parallelize across cores.  SPS/obs/model are module-level globals initialized in
the parent and inherited by forked workers (FSPS is not picklable, so we never
pass it as an argument — fork + COW shares it).

NOTE: imports the core at runtime, so it picks up R_FIRE = 6000 (now canonical).

Usage:
    python fit_egg_jitter_dustsep.py --pool 8
    python fit_egg_jitter_dustsep.py --pool 8 --dry-run
    python fit_egg_jitter_dustsep.py --no-cat-weight --pool 8

Output (model_finalization/):
    dynesty_egg_duste_sfree_zfree_jitter_floor_dustsep.pkl
    chain_egg_duste_sfree_zfree_jitter_floor_dustsep.npy
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


def _load(name, fname):
    sp = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fname))
    m  = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


F = _load('fit_egg', 'fit_egg_prospector_tlusty.py')
J = _load('fit_egg_jitter', 'fit_egg_jitter.py')

from prospect.models.priors import Uniform
from prospect.models import ProspectorParams
from prospect.models.transforms import logsfr_ratios_to_masses

_LN10 = np.log(10.0)

# ── Module-level globals for fork-based pool workers ──────────────────────────
_SPS = None
_OBS = None
_MODEL = None
_NOISE_MODEL = 'floor'
_CAT_WEIGHT = True


def build_sps_independent(verbose=True):
    """TLUSTYPlusGalaxyBasis with phot_dust_mode='independent' (decoupled dust)."""
    mod = F._import_tlusty_module()
    return mod.TLUSTYPlusGalaxyBasis(
        spec_dir=F.SPEC_DIR, target_R=F.TARGET_R,
        cache_file=F.TLUSTY_CACHE, xi10_cache_file=F.XI10_CACHE,
        verbose=verbose, phot_dust_mode='independent', phot_dust_law='smc')


def build_model_dustsep():
    """J.build_model_jitter() + av_phot (photosphere A_V, independent of galaxy)."""
    m0 = J.build_model_jitter()
    mp = m0.config_dict
    mp['av_phot'] = {
        'N': 1, 'isfree': True, 'init': 1.5,
        'prior': Uniform(mini=0.0, maxi=4.0),
        'units': 'mag — photosphere A_V (SMC), independent of galaxy dust2_gal',
    }
    m = ProspectorParams(mp)
    m.configure(reset=True)
    return m


def _logl_dustsep(theta, sps, obs_list, model, noise_model='floor', cat_weight=True):
    """Same as J._logl_jitter, but passes av_phot to the photosphere (independent dust)."""
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
    av_phot      = float(theta[idx['av_phot']])

    logmass       = float(theta[idx['logmass']])
    logzsol       = (float(theta[idx['logzsol']]) if 'logzsol' in idx else mh_idx - 2.0)
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
        av_phot=av_phot,                 # ← photosphere dust (independent mode)
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
        log_s = float(theta[idx[J.JITTER_NAME[obs['name']]]])

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
            chi2  = np.sum(w * (res / unc) ** 2)
            n_eff = float(w.sum())
            f2    = 10.0 ** (2.0 * log_s)
            lnp  += -0.5 * chi2 * f2 + n_eff * log_s * _LN10
        elif noise_model == 'floor':
            s_lin      = 10.0 ** log_s
            sigma_eff2 = unc ** 2 + (s_lin * pred[mask]) ** 2
            if np.any(sigma_eff2 <= 0):
                return -1e300
            lnp += -0.5 * np.sum(w * (res ** 2 / sigma_eff2 + np.log(sigma_eff2)))
        else:
            raise ValueError(f"noise_model must be mult/floor, got {noise_model!r}")

    return float(lnp) if np.isfinite(lnp) else -1e300


# ── Top-level wrappers for the fork pool (reference module globals) ───────────
def _logl_global(theta):
    return _logl_dustsep(theta, _SPS, _OBS, _MODEL,
                         noise_model=_NOISE_MODEL, cat_weight=_CAT_WEIGHT)


def _ptform_global(u):
    return _MODEL.prior_transform(u)


def parse_args():
    p = argparse.ArgumentParser(description='Egg floor-jitter run with decoupled phot/gal dust')
    p.add_argument('--pool', type=int, default=8, help='number of pool workers (cores)')
    p.add_argument('--no-cat-weight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    global _SPS, _OBS, _MODEL, _NOISE_MODEL, _CAT_WEIGHT
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    _CAT_WEIGHT = not args.no_cat_weight
    _NOISE_MODEL = 'floor'

    STEM = 'egg_duste_sfree_zfree_jitter_floor_dustsep'
    if not _CAT_WEIGHT:
        STEM += '_nocat'

    print(f'\n{"="*68}', flush=True)
    print(f'The Egg — floor jitter + DECOUPLED phot/gal dust (independent av_phot)', flush=True)
    print(f'  pool={args.pool}  cat_weight={_CAT_WEIGHT}  R_FIRE={F.R_FIRE}', flush=True)
    print(f'  STEM={STEM}', flush=True)
    print(f'{"="*68}\n', flush=True)

    print('Building SPS (independent dust, include_xi10) …', flush=True)
    _SPS = build_sps_independent(verbose=True)
    print('Loading observations …', flush=True)
    _OBS = F.build_obs()
    print('Building M3 + jitter + av_phot model …', flush=True)
    _MODEL = build_model_dustsep()

    NDIM   = _MODEL.ndim
    LABELS = _MODEL.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in _OBS)
    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    # Warm up FSPS in the parent so forked workers inherit a built SPS (COW).
    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = _ptform_global(u_mid)
    ll_mid = _logl_global(th_mid)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    if args.dry_run:
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

    import multiprocessing as mp
    ctx  = mp.get_context('fork')
    pool = ctx.Pool(args.pool)
    print(f'\nStarting DynamicNestedSampler over {args.pool}-worker fork pool …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        _logl_global, _ptform_global, NDIM,
        bound='multi', sample='rwalk',
        pool=pool, queue_size=args.pool)
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
