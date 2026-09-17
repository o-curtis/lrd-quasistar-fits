#!/usr/bin/env python3
"""
fit_egg_model_finalization.py  —  12-run model comparison for The Egg.

Launches a single dynesty run for one (model, sigma, zsol) combination.
All output is written to egg_analysis/model_finalization/.

Usage:
    python fit_egg_model_finalization.py --model M1 --sigma free  --zsol free
    python fit_egg_model_finalization.py --model M2 --sigma fixed --zsol free
    python fit_egg_model_finalization.py --model M3 --sigma free  --zsol fixed

Arguments:
    --model  {M1, M2, M3}   M1=base, M2=+xi_mtb, M3=+xi_mtb+dust_emission
    --sigma  {free, fixed}  free=Uniform(115,135) km/s; fixed=123.0 km/s
    --zsol   {free, fixed}  free=Uniform(-2,0.19); fixed=mh_idx-2 (derived)
    --dry-run               Build model, fire mid-prior logl, then exit.

Output files (in model_finalization/):
    dynesty_egg_{mtag}_{stag}_{ztag}.pkl
    chain_egg_{mtag}_{stag}_{ztag}.npy
"""

import os, sys, argparse, time, pickle, functools
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
OUT_DIR    = os.path.join(SCRIPT_DIR, 'model_finalization')

MODEL_TAG = {'M1': 'base', 'M2': 'xitb', 'M3': 'duste'}
SIGMA_TAG = {'free': 'sfree', 'fixed': 's123'}
ZSOL_TAG  = {'free': 'zfree', 'fixed': 'zfix'}
MH_TAG    = {'free': '', 'fixed': '_mhfix'}
NEB_TAG   = {'off': '', 'on': '_neb'}


def parse_args():
    p = argparse.ArgumentParser(description='Egg model-finalization dynesty run')
    p.add_argument('--model',   required=True, choices=['M1', 'M2', 'M3'])
    p.add_argument('--sigma',   required=True, choices=['free', 'fixed'])
    p.add_argument('--zsol',    required=True, choices=['free', 'fixed'])
    p.add_argument('--mh',      default='free', choices=['free', 'fixed'],
                   help='mh_idx: free=Uniform(0,2) | fixed=1.0 ([M/H]=−1, Liu+2026)')
    p.add_argument('--neb',     default='off',  choices=['off', 'on'],
                   help='Nebular emission: off=disabled | on=add_neb_emission=True, '
                        'gas_logu free Uniform(-4,-1), gas_logz fixed=-1')
    p.add_argument('--dry-run', action='store_true',
                   help='Build and fire one logl call, then exit without sampling')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    STEM = ('BMKP_' + f'egg_{MODEL_TAG[args.model]}_{SIGMA_TAG[args.sigma]}_'
            f'{ZSOL_TAG[args.zsol]}{MH_TAG[args.mh]}{NEB_TAG[args.neb]}' + '_nmask')

    print(f'\n{"="*60}', flush=True)
    print(f'The Egg — model finalization run', flush=True)
    print(f'  model={args.model}  sigma={args.sigma}  zsol={args.zsol}  '
          f'mh={args.mh}  neb={args.neb}', flush=True)
    print(f'  STEM  = {STEM}', flush=True)
    print(f'  OUT   = {OUT_DIR}', flush=True)
    print(f'{"="*60}\n', flush=True)

    print('Building SPS …', flush=True)
    sps = F.build_sps(include_xi10=(args.model in ('M2', 'M3')))
    print('Loading observations …', flush=True)
    obs_list = F.build_obs()
    _n = 0
    for _o in obs_list:
        _bad = (_o['wave_rest'] >= 3900.0) & (_o['wave_rest'] <= 3980.0)
        _n += int((_o['mask'] & _bad).sum())
        _o['mask'] = _o['mask'] & ~_bad
    print('BMKP break mask removed %d fit pixels' % _n, flush=True)
    print(f'Building model (type={args.model}, sigma={args.sigma}, zsol={args.zsol}, '
          f'mh={args.mh}, neb={args.neb}) …', flush=True)
    model = F.build_model_variant(args.model, args.sigma, args.zsol,
                                  mh_mode=args.mh, neb_mode=args.neb)
    _pt0 = model.prior_transform
    def _pt_nmask(u, _f=_pt0):
        th = _f(u)
        pass
        return th
    model.prior_transform = _pt_nmask
    print('NMASK: only CaII H+K cores 3900-3980 masked; no clamps', flush=True)

    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)

    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = F._logl(th_mid, sps, obs_list, model)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    if args.dry_run:
        print('\n[dry-run] Exiting before dynesty.', flush=True)
        return

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

    pkl_path = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
    npy_path = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
