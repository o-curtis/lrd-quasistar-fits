#!/usr/bin/env python3
"""GLIMPSE-17775 (Kokorev+2026 cocoon LRD, z=3.5022, mu=2.04): Egg-machinery fit
of the coadded deep G395M continuum (rest 6400-11300 A; Ca II triplet IN range
and unmasked — the constraining feature). Single medium-res arm, R~1000.
Note: phi = kappa sigma T^4/(g c) is magnification-independent; logL (and hence
v_esc via R) carries a mu = 2.04 +- 0.21 correction applied downstream.
Usage: python fit_glimpse17775.py [--dry-run] [--pool N]"""
import os, sys, argparse, time, pickle, functools
import numpy as np
import importlib.util
EGG_DIR = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/egg_analysis'
_spec = importlib.util.spec_from_file_location('fit_egg', os.path.join(EGG_DIR, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(F)

HERE = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/newsrc'
OUT = os.path.join(HERE, 'fits_out')
_CKMS = 2.998e5; _j = 1e-23
Z = 3.5022
STEM = 'glimpse17775_m3'
LINES = [(6564.6, 900), (6549.9, 400), (6585.3, 400), (6718.3, 300), (6732.7, 300),
         (7067.1, 250), (7137.8, 250), (7321.0, 250), (7332.0, 250), (9071.1, 300),
         (9533.2, 300), (9548.6, 300), (10052.6, 300), (10833.3, 400), (10941.1, 300),
         (8449.6, 200)]           # CaT 8498/8542/8662 deliberately NOT masked

_SPS = _OBS = _MODEL = None
def _logl_g(theta): return F._logl(theta, _SPS, _OBS, _MODEL)
def _ptf_g(u): return _MODEL.prior_transform(u)

def build_obs():
    d = np.load(os.path.join(HERE, 'glimpse17775_coadd.npz'))
    w = d['wave'].astype(float)*1e4          # um -> A
    fl = d['flux'].astype(float)*1e-6        # uJy -> Jy
    er = d['err'].astype(float)*1e-6
    mag = fl/3631.; emag = er/3631.
    w_rest = w/(1.+Z)
    good = np.isfinite(mag) & np.isfinite(emag) & (emag > 0)
    mask = good & (w_rest >= 6400.) & (w_rest <= 11300.)
    for l0, half in LINES:
        mask &= np.abs(_CKMS*(w_rest-l0)/l0) > half
    print('  G395M: %d pix, %d fit pix' % (int(good.sum()), int(mask.sum())), flush=True)
    return [dict(name='G395M', wave_obs=w, wave_rest=w_rest, flux=mag, unc=emag,
                 mask=mask, sigma_inst=_CKMS/(1000.*2.355))]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true'); ap.add_argument('--pool', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    obs_list = build_obs()
    sps = F.build_sps(include_xi10=True)
    model = F.build_model_variant('M3', 'free', 'free', mh_mode='free', neb_mode='off', zred=Z)
    from prospect.models import priors as _priors
    model.config_dict['sigma_smooth']['prior'] = _priors.TopHat(mini=20.0, maxi=400.0)
    model.config_dict['sigma_smooth']['init'] = 100.0
    NDIM = model.ndim
    print('ndim=%d | fit pixels=%d' % (NDIM, sum(int(o['mask'].sum()) for o in obs_list)), flush=True)
    th_mid = model.prior_transform(0.5*np.ones(NDIM))
    ll = F._logl(th_mid, sps, obs_list, model)
    print('Mid-prior ln L = %.1f' % ll, flush=True)
    if a.dry_run: sys.exit(0)
    import dynesty
    from dynesty.utils import resample_equal
    _SPS, _OBS, _MODEL = sps, obs_list, model
    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(a.pool) if a.pool > 1 else None
    kw = dict(pool=pool, queue_size=a.pool) if pool else {}
    sampler = dynesty.DynamicNestedSampler(_logl_g, _ptf_g, NDIM, nlive=F.NLIVE,
                                           bound='multi', sample='rwalk', **kw)
    t0 = time.time()
    sampler.run_nested(dlogz_init=F.DLOGZ_INIT, nlive_init=F.NLIVE_INIT,
                       nlive_batch=F.NLIVE_BATCH, wt_kwargs={'pfrac': 1.0},
                       n_effective=F.N_EFFECTIVE, print_progress=True)
    if pool: pool.close(); pool.join()
    res = sampler.results
    samp = resample_equal(res.samples, np.exp(res.logwt-res.logz[-1]))
    pickle.dump(res, open(os.path.join(OUT, f'dynesty_{STEM}.pkl'), 'wb'))
    np.save(os.path.join(OUT, f'chain_{STEM}.npy'), samp)
    print('elapsed %.0f s; saved chain_%s.npy' % (time.time()-t0, STEM), flush=True)
