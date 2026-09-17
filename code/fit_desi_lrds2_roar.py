#!/usr/bin/env python3
"""
LRDS2 campaign: Egg-style TLUSTY+host fit of one (LRDs)^2 DESI DR1 LRD.

Reuses the Egg machinery (egg_analysis/fit_egg_prospector_tlusty.py) verbatim:
build_sps, build_model_variant (ADOPTED config: M3 sigma-free z-free), _logl,
and the same dynesty settings. Only two source-specific changes:
  1. build_obs_desi: DESI coadd split into three pseudo-arms (B/R/Z cameras,
     fixed effective R = 2500/3500/4500), f_lambda(1e-17) -> maggies,
     NJC-informed emission-line mask at the source redshift.
  2. sigma_smooth prior widened to Uniform(20, 400) km/s — the Egg's (115,135)
     encodes ITS narrow-line width; Lin's sample spans sigma_n ~ 20-55 km/s.

Usage: python fit_desi_lrds2.py NAME [--dry-run]
"""
import os, sys, argparse, time, pickle, functools
import numpy as np

import importlib.util
EGG_DIR = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/egg_analysis'
_spec = importlib.util.spec_from_file_location(
    'fit_egg', os.path.join(EGG_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

HERE = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/lrds2'
_SPS = _OBS = _MODEL = None
def _logl_g(theta):
    return F._logl(theta, _SPS, _OBS, _MODEL)
def _ptf_g(u):
    return _MODEL.prior_transform(u)
OUT = os.path.join(HERE, 'fits_out')
_CKMS = 2.998e5
_jansky_cgs = 1e-23

LINES = [(6564.6, 900), (4862.7, 500), (5008.2, 350), (4960.3, 300), (4341.7, 300),
         (4102.9, 250), (3970.1, 200), (3889.1, 200), (3868.8, 250), (3727.4, 300),
         (4686.7, 250), (5877.2, 250), (6302.0, 200), (6549.9, 400), (6585.3, 400),
         (6718.3, 300), (6732.7, 300), (7067.1, 200), (7137.8, 200), (9071.1, 250),
         (9533.2, 250), (3426.8, 250), (2798.0, 400)]
ARMS = [('DESI-B', 3600., 5800., 2500.), ('DESI-R', 5800., 7600., 3500.),
        ('DESI-Z', 7600., 9830., 4500.)]

def line_mask(w_rest, lines):
    m = np.ones(len(w_rest), bool)
    for l0, half in lines:
        m &= np.abs(_CKMS*(w_rest - l0)/l0) > half
    return m

def build_obs_desi(name, z, fit_lo=2400., fit_hi=8000.):
    d = np.load(os.path.join(HERE, 'spectra', f'{name}.npz'))
    w, fl, er = d['wave'].astype(float), d['flux'].astype(float), d['err'].astype(float)
    f_nu = fl*1e-17 * w**2 / 2.998e18
    e_nu = er*1e-17 * w**2 / 2.998e18
    mag = f_nu / (3631.0*_jansky_cgs); emag = e_nu / (3631.0*_jansky_cgs)
    w_rest = w/(1.+z)
    good = np.isfinite(mag) & np.isfinite(emag) & (emag > 0)
    obs = []
    for aname, lo, hi, R in ARMS:
        sel = good & (w >= lo) & (w < hi)
        if sel.sum() < 50: continue
        mask = line_mask(w_rest[sel], LINES) & (w_rest[sel] >= fit_lo) & (w_rest[sel] <= fit_hi)
        obs.append(dict(name=aname, wave_obs=w[sel], wave_rest=w_rest[sel],
                        flux=mag[sel], unc=emag[sel], mask=mask,
                        sigma_inst=_CKMS/(R*2.355)))
        print(f'  {aname}: {sel.sum()} pix, {int(mask.sum())} fit pix, '
              f'sigma_inst={_CKMS/(R*2.355):.1f} km/s', flush=True)
    return obs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--pool', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    import json
    man = {r['name']: r for r in json.load(open(os.path.join(HERE, 'lrds2_manifest.json')))}
    z = float(man[a.name]['z'])
    STEM = 'lrds2_' + a.name.split('.')[0].replace('+', 'p').replace('-', 'm')
    print(f'=== LRDS2 {a.name}  z={z}  STEM={STEM} ===', flush=True)
    obs_list = build_obs_desi(a.name, z)

    print('Building SPS ...', flush=True)
    sps = F.build_sps(include_xi10=True)                 # M3 uses xi grid
    print('Building model (M3 sfree zfree, zred=%.4f) ...' % z, flush=True)
    model = F.build_model_variant('M3', 'free', 'free', mh_mode='free',
                                  neb_mode='off', zred=z)
    # widen sigma_smooth prior for this sample (see header)
    from prospect.models import priors as _priors
    model.config_dict['sigma_smooth']['prior'] = _priors.TopHat(mini=20.0, maxi=400.0)
    model.config_dict['sigma_smooth']['init'] = 100.0

    NDIM = model.ndim
    n_pix = sum(int(o['mask'].sum()) for o in obs_list)
    print(f'ndim={NDIM} | fit pixels={n_pix}', flush=True)
    print('free:', model.theta_labels(), flush=True)
    th_mid = model.prior_transform(0.5*np.ones(NDIM))
    ll_mid = F._logl(th_mid, sps, obs_list, model)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)
    if a.dry_run:
        print('[dry-run] exit before dynesty.', flush=True)
        sys.exit(0)

    import dynesty
    from dynesty.utils import resample_equal
    # fork-pool pattern (as in m3port): globals inherited by workers, functions
    # referenced by name -> picklable. Statistically identical to serial.
    _SPS, _OBS, _MODEL = sps, obs_list, model
    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(a.pool) if a.pool > 1 else None
    kw = dict(pool=pool, queue_size=a.pool) if pool else {}
    print(f'Starting DynamicNestedSampler ({a.pool}-worker fork pool) ...', flush=True)
    sampler = dynesty.DynamicNestedSampler(_logl_g, _ptf_g, NDIM,
                                           nlive=F.NLIVE, bound='multi', sample='rwalk', **kw)
    t0 = time.time()
    sampler.run_nested(dlogz_init=F.DLOGZ_INIT, nlive_init=F.NLIVE_INIT,
                       nlive_batch=F.NLIVE_BATCH, wt_kwargs={'pfrac': 1.0},
                       n_effective=F.N_EFFECTIVE, print_progress=True)
    if pool: pool.close(); pool.join()
    print(f'elapsed {time.time()-t0:.0f} s', flush=True)
    res = sampler.results
    wts = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)
    pickle.dump(res, open(os.path.join(OUT, f'dynesty_{STEM}.pkl'), 'wb'))
    np.save(os.path.join(OUT, f'chain_{STEM}.npy'), samp)
    print(f'saved chain_{STEM}.npy', flush=True)
