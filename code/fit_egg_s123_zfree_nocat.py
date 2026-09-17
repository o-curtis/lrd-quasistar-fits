#!/usr/bin/env python3
"""
fit_egg_s123_zfree_nocat.py — VALIDATION TASK 3: "Reproduce-Liu" test fit of The Egg.

Question: does removing the 15x CaT upweight (and fixing sigma at Liu+2026's
pre-determined 123 km/s) move the fitted (T_eff, log g) from our adopted M3
solution toward Liu+2026's published (4500 K, -2.90)?

Config = ADOPTED M3 sigma-free z-free (`egg_duste_sfree_zfree`) EXCEPT:
  - sigma_smooth  FIXED = 123 km/s   (Liu+2026 Section V.1: pre-determined from
                                      the CaT region and held fixed in their fit)
  - CaT upweight REMOVED             (weight 1 everywhere; Liu weights all equally)
Everything else IDENTICAL to the adopted run:
  - M3 model: TLUSTY photosphere + FSPS continuity-SFH galaxy + Draine&Li duste
  - xi_mtb FREE Uniform(2,10)  (as in the adopted config; NOT fixed like the
    earlier s126_xi4_nocat probe)
  - zsol free, mh_idx free, shared SMC dust, dust1 free
  - plain per-pixel chi^2 (no jitter/floor), same masks, same F_SYS=0.05
  - identical dynesty settings (NLIVE=600 family)

Free params (17): teff, logg, logL_star, mh_idx, logmass, logzsol, dust2_gal,
                  dust1, logsfr_ratios_1..5, xi_mtb, duste_umin, duste_qpah,
                  duste_gamma

Usage:
    python fit_egg_s123_zfree_nocat.py --pool 6
    python fit_egg_s123_zfree_nocat.py --dry-run

Output (fit_validation/):
    dynesty_egg_duste_s123_zfree_nocat.pkl
    chain_egg_duste_s123_zfree_nocat.npy
"""

import os, sys, argparse, time, pickle
sys.dont_write_bytecode = True          # never write __pycache__ into read-only dirs
import numpy as np
import dynesty
from dynesty.utils import resample_equal

os.environ['SPS_HOME'] = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
sys.path.insert(0, '/home/omc5226/prospector_tlusty_dev/prospector')

import importlib.util

EGG_DIR = '/home/omc5226/work/lrdmesa/egg_analysis'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STEM    = 'egg_duste_s123_zfree_nocat'


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m  = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


F = _load('fit_egg', os.path.join(EGG_DIR, 'fit_egg_prospector_tlusty.py'))


def _logl_nocat(theta, sps, obs_list, model):
    """Forward model identical to F._logl; plain chi^2, NO CaT upweight."""
    idx = model.theta_index

    teff         = float(theta[idx['teff']])
    logg         = float(theta[idx['logg']])
    logL_star    = float(theta[idx['logL_star']])
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))      # fixed -> 123
    mh_idx       = (float(theta[idx['mh_idx']]) if 'mh_idx' in idx
                    else float(model.params['mh_idx']))
    xi_mtb       = (float(theta[idx['xi_mtb']]) if 'xi_mtb' in idx
                    else float(model.params.get('xi_mtb', 2.0)))   # FREE here

    logmass       = float(theta[idx['logmass']])
    logzsol       = (float(theta[idx['logzsol']]) if 'logzsol' in idx
                     else mh_idx - 2.0)
    dust2_gal     = float(theta[idx['dust2_gal']])
    dust1         = float(theta[idx['dust1']])
    logsfr_ratios = theta[idx['logsfr_ratios']]

    agebins = np.array(model.params['agebins'])
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])
    from prospect.models.transforms import logsfr_ratios_to_masses
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
        lnp += -0.5 * np.sum((res / unc) ** 2)     # weight 1 everywhere: NO CaT x15

    return float(lnp) if np.isfinite(lnp) else -1e300


_SPS = _OBS = _MODEL = None
def _logl_global(theta): return _logl_nocat(theta, _SPS, _OBS, _MODEL)
def _ptform_global(u):   return _MODEL.prior_transform(u)


def main():
    global _SPS, _OBS, _MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print(f'\n{"="*70}', flush=True)
    print(f'The Egg — VALIDATION: adopted M3, sigma=123 FIXED (Liu), NO CaT x15', flush=True)
    print(f'  xi_mtb FREE (adopted config) | pool={args.pool} | R_FIRE={F.R_FIRE} | STEM={STEM}', flush=True)
    print(f'{"="*70}\n', flush=True)

    print('Building SPS (shared SMC dust, include_xi10) …', flush=True)
    _SPS = F.build_sps(include_xi10=True, verbose=True)
    print('Loading observations …', flush=True)
    _OBS = F.build_obs()
    print('Building model M3 sigma=fixed(123) zsol=free mh=free …', flush=True)
    _MODEL = F.build_model_variant('M3', 'fixed', 'free', mh_mode='free', neb_mode='off')

    NDIM   = _MODEL.ndim
    LABELS = _MODEL.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in _OBS)
    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)
    print(f'Fixed: sigma_smooth={float(_MODEL.params["sigma_smooth"]):.1f} km/s', flush=True)

    th_mid = _ptform_global(0.5 * np.ones(NDIM))
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
    samp = resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
    pkl_path = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)

    print(f'\nln Z = {res.logz[-1]:.2f} ± {res.logzerr[-1]:.2f}', flush=True)
    map_th = res.samples[np.argmax(res.logl)]
    print('MAP (argmax logl):', flush=True)
    for lbl, val in zip(LABELS, map_th):
        print(f'  {lbl:<16} = {val:.5g}', flush=True)
    print('Posterior medians:', flush=True)
    for i, lbl in enumerate(LABELS):
        q = np.percentile(samp[:, i], [16, 50, 84])
        print(f'  {lbl:<16} = {q[1]:.4g}  (-{q[1]-q[0]:.3g}/+{q[2]-q[1]:.3g})', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
