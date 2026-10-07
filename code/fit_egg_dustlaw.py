#!/usr/bin/env python3
"""
fit_egg_dustlaw.py — dust-law battery for The Egg.

Same 18-parameter M3 sigma-free zsol-free model as the adopted fiducial;
the ONLY change is the attenuation law applied to both components
(fiducial = SMC).  The law is swapped after build_sps() by setting
sps.phot_dust_law, which TLUSTYPlusGalaxyBasis reads live in
get_spectra_components():
  calzetti — photosphere via _calzetti_transmission (delta=0);
             galaxy diffuse dust handled INSIDE FSPS (dust_type=4),
             with the FSPS slope modifier pinned to 0 here so both
             components see pure Calzetti.
  mw       — CCM+O'Donnell R_V=3.1 applied externally to both
             components (FSPS receives dust2=0), same path as SMC.
Birth-cloud dust (dust1, C&F 2000) stays inside FSPS in every variant.

--law      {calzetti, mw}       (smc = fiducial, already run)
--maskmode {full, nmask, none}  full  = 3400-4100 A break window (BMKP)
                                nmask = 3900-3980 A CaII H+K cores only
                                none  = full spectrum (adopted pixel set)

STEM  : DLAW_egg_duste_sfree_zfree_{law}_{masktag}
Output: model_finalization/dynesty_{STEM}.pkl + chain_{STEM}.npy

Runs locally and on ROAR unchanged: all paths (SPS_HOME, prospector,
Liu library) are taken from fit_egg_prospector_tlusty.py in this same
directory, which carries the machine-specific substitutions.
"""

import os, sys, argparse, time, pickle, functools
import numpy as np
import dynesty
from dynesty.utils import resample_equal
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(SCRIPT_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

OUT_DIR = os.path.join(SCRIPT_DIR, 'model_finalization')

MASK_WINDOWS = {'full': (3400.0, 4100.0), 'nmask': (3900.0, 3980.0), 'none': None}
MASK_TAG     = {'full': 'bmkp', 'nmask': 'nmask', 'none': 'fullspec'}
LAW_TAG      = {'calzetti': 'calz', 'mw': 'mw'}


def parse_args():
    p = argparse.ArgumentParser(description='Egg dust-law variant dynesty run')
    p.add_argument('--law',      required=True, choices=['calzetti', 'mw'])
    p.add_argument('--maskmode', required=True, choices=['full', 'nmask', 'none'])
    # accepted for command-line compatibility with the battery sub scripts
    p.add_argument('--model', default='M3',   choices=['M1', 'M2', 'M3'])
    p.add_argument('--sigma', default='free', choices=['free', 'fixed'])
    p.add_argument('--zsol',  default='free', choices=['free', 'fixed'])
    p.add_argument('--dry-run', action='store_true',
                   help='Build and fire one logl call, then exit without sampling')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    STEM = (f'DLAW_egg_duste_sfree_zfree_'
            f'{LAW_TAG[args.law]}_{MASK_TAG[args.maskmode]}')

    print(f'\n{"="*60}', flush=True)
    print('The Egg — dust-law variant run', flush=True)
    print(f'  law={args.law}  maskmode={args.maskmode}  '
          f'model={args.model} sigma={args.sigma} zsol={args.zsol}', flush=True)
    print(f'  STEM  = {STEM}', flush=True)
    print(f'  OUT   = {OUT_DIR}', flush=True)
    print(f'{"="*60}\n', flush=True)

    print('Building SPS …', flush=True)
    sps = F.build_sps(include_xi10=(args.model in ('M2', 'M3')))

    # ── swap the attenuation law (read live by get_spectra_components) ──
    assert sps.phot_dust_mode == 'shared', 'expected shared dust mode'
    print(f'Swapping phot_dust_law: {sps.phot_dust_law} -> {args.law}', flush=True)
    sps.phot_dust_law = args.law
    if args.law == 'calzetti':
        # Galaxy diffuse dust is handled inside FSPS for calzetti mode
        # (dust_type=4 = Noll+09).  Pin the FSPS slope modifier to 0 so the
        # galaxy sees pure Calzetti, matching the photosphere side, whose
        # _calzetti_transmission defaults to dust_index=0.
        try:
            sps.galaxy.ssp.params['dust_index'] = 0.0
            print('FSPS dust_index pinned to 0.0 (pure Calzetti)', flush=True)
        except Exception as e:
            print(f'WARNING: could not pin FSPS dust_index=0: {e}', flush=True)

    print('Loading observations …', flush=True)
    obs_list = F.build_obs()
    win = MASK_WINDOWS[args.maskmode]
    if win is not None:
        n = 0
        for o in obs_list:
            bad = (o['wave_rest'] >= win[0]) & (o['wave_rest'] <= win[1])
            n += int((o['mask'] & bad).sum())
            o['mask'] = o['mask'] & ~bad
        print(f'Mask {args.maskmode}: removed {n} fit pixels in '
              f'{win[0]:.0f}-{win[1]:.0f} A', flush=True)
    else:
        print('Mask none: full adopted-convention pixel set', flush=True)

    print(f'Building model ({args.model}, sigma={args.sigma}, '
          f'zsol={args.zsol}) …', flush=True)
    model = F.build_model_variant(args.model, args.sigma, args.zsol)

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
