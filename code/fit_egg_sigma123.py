#!/usr/bin/env python3
"""
fit_egg_sigma123.py — Same as fit_egg_prospector_tlusty.py but with
sigma_smooth fixed at 123 km/s (not a free parameter).
Output files use tag 'egg_sigma123' instead of 'egg_mhfree'.
"""
import os, sys, time, pickle
import numpy as np
import dynesty
from dynesty.utils import resample_equal

os.environ['SPS_HOME'] = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
sys.path.insert(0, '/home/omc5226/prospector_tlusty_dev/prospector')

import importlib.util
_spec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEM       = 'egg_sigma123_prospector_tlusty'

def build_model_fixed_sigma():
    model = F.build_model()
    model.config_dict['sigma_smooth']['isfree'] = False
    model.config_dict['sigma_smooth']['init']   = 123.0
    model.configure(reset=True)
    return model


if __name__ == '__main__':
    import functools

    print(f'\n{"="*60}', flush=True)
    print('The Egg — TLUSTY photosphere + continuity SFH', flush=True)
    print('sigma_smooth FIXED at 123 km/s', flush=True)
    print(f'{"="*60}\n', flush=True)

    print('Building SPS …', flush=True)
    sps = F.build_sps(verbose=True)
    print('Loading observations …', flush=True)
    obs_list = F.build_obs()
    print('Building model (sigma fixed=123) …', flush=True)
    model = build_model_fixed_sigma()

    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)

    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = F._logl(th_mid, sps, obs_list, model)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    logl_fn = functools.partial(F._logl, sps=sps, obs_list=obs_list, model=model)

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

    pkl_path = os.path.join(SCRIPT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(SCRIPT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)
    print('Done.', flush=True)
