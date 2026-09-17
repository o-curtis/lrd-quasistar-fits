#!/usr/bin/env python3
"""
Injection-recovery test for The Egg — M3 (duste, sigma free, zsol free, mh free).

Procedure
---------
1. Load chain_egg_duste_sfree_zfree.npy (must exist — run after the rerun completes).
2. Find the MAP sample (highest dynesty weight from the pkl).
3. Evaluate forward model at MAP → predicted flux per arm.
4. Build synthetic obs: f_syn = f_model + N(0, unc)  using REAL unc arrays.
   Wavelength grids, masks, and uncertainties are unchanged from the real data.
5. Run DynamicNestedSampler with identical settings to the original M3 run.
6. Save outputs to model_finalization/:
     chain_egg_duste_sfree_zfree_injrec.npy
     dynesty_egg_duste_sfree_zfree_injrec.pkl
     chain_egg_duste_sfree_zfree_injrec_truth.npy   ← MAP theta used as truth

The truth file allows direct posterior-vs-truth comparison after the run.
"""

import os, sys, pickle, time, functools
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
STEM       = 'egg_duste_sfree_zfree'
STEM_INJ   = 'egg_duste_sfree_zfree_injrec'
RNG_SEED   = 42

CHAIN_PATH  = os.path.join(OUT_DIR, f'chain_{STEM}.npy')
PKL_PATH    = os.path.join(OUT_DIR, f'dynesty_{STEM}.pkl')
OUT_PKL     = os.path.join(OUT_DIR, f'dynesty_{STEM_INJ}.pkl')
OUT_NPY     = os.path.join(OUT_DIR, f'chain_{STEM_INJ}.npy')
OUT_TRUTH   = os.path.join(OUT_DIR, f'chain_{STEM_INJ}_truth.npy')


def find_map(chain_path, pkl_path):
    """Return the highest-weight sample from the dynesty results as the MAP."""
    # Try pkl first (has logwt)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as fh:
            res = pickle.load(fh)
        map_idx = np.argmax(res.logwt)
        print(f'  MAP from dynesty logwt: idx={map_idx}  logwt={res.logwt[map_idx]:.2f}',
              flush=True)
        return res.samples[map_idx]
    # Fallback: last sample in chain (reasonable approximation)
    chain = np.load(chain_path)
    print(f'  MAP fallback: last sample in chain (shape {chain.shape})', flush=True)
    return chain[-1]


def build_synthetic_obs(theta_map, sps, obs_list, model, rng_seed=42):
    """
    Evaluate forward model at theta_map and inject Gaussian noise.

    Returns a new obs_list with synthetic fluxes; all other fields
    (wavelengths, masks, uncertainties) are identical to the real data.
    """
    rng = np.random.default_rng(rng_seed)
    sps_p  = F._make_sps_params(theta_map, model)
    result = F._eval_model(sps_p, sps, obs_list, model)
    if result is None:
        raise ValueError('Forward model returned None at MAP — bad parameter point.')
    _, _, _, preds = result

    syn_obs_list = []
    for obs, (ptl, pgal) in zip(obs_list, preds):
        f_model = ptl + pgal
        # Replace NaN model pixels with zero (outside model range; will be masked anyway)
        f_model = np.where(np.isfinite(f_model), f_model, 0.0)
        noise   = rng.normal(0.0, obs['unc'])
        f_syn   = f_model + noise

        syn_obs = dict(obs)          # shallow copy — shares wavelength/mask/unc arrays
        syn_obs['flux'] = f_syn
        syn_obs_list.append(syn_obs)

    return syn_obs_list


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'\n{"="*60}', flush=True)
    print('The Egg — injection-recovery test', flush=True)
    print(f'  source stem : {STEM}', flush=True)
    print(f'  output stem : {STEM_INJ}', flush=True)
    print(f'{"="*60}\n', flush=True)

    # ── Build SPS and model ───────────────────────────────────────────────────
    print('Building SPS (include_xi10=True) …', flush=True)
    sps = F.build_sps(include_xi10=True)

    print('Loading real observations …', flush=True)
    obs_list = F.build_obs()

    print('Building M3 model (sigma free, zsol free, mh free) …', flush=True)
    model = F.build_model_variant('M3', 'free', 'free', mh_mode='free')
    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)
    print(f'  NDIM={NDIM}  fit pixels={n_pix}', flush=True)
    print(f'  Free params: {LABELS}', flush=True)

    # ── Find MAP from existing chain ──────────────────────────────────────────
    print(f'\nFinding MAP from {CHAIN_PATH} …', flush=True)
    theta_map = find_map(CHAIN_PATH, PKL_PATH)
    np.save(OUT_TRUTH, theta_map)
    print(f'  Truth saved: {OUT_TRUTH}', flush=True)

    # Print MAP parameter values
    for lbl, val in zip(LABELS, theta_map):
        print(f'    {lbl:20s} = {val:.4f}', flush=True)

    # ── Build synthetic obs ───────────────────────────────────────────────────
    print(f'\nBuilding synthetic observations (seed={RNG_SEED}) …', flush=True)
    syn_obs_list = build_synthetic_obs(theta_map, sps, obs_list, model, RNG_SEED)
    for syn_obs in syn_obs_list:
        print(f'  {syn_obs["name"]:8s}: fit pixels={syn_obs["mask"].sum()}  '
              f'SNR_median={np.median(np.abs(syn_obs["flux"][syn_obs["mask"]] / syn_obs["unc"][syn_obs["mask"]])):.1f}',
              flush=True)

    # Mid-prior sanity check on synthetic data
    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = F._logl(th_mid, sps, syn_obs_list, model)
    print(f'\nMid-prior ln L (synthetic) = {ll_mid:.1f}', flush=True)
    ll_map = F._logl(theta_map, sps, syn_obs_list, model)
    print(f'MAP       ln L (synthetic) = {ll_map:.1f}', flush=True)

    # ── Run dynesty ───────────────────────────────────────────────────────────
    logl_fn = functools.partial(F._logl, sps=sps, obs_list=syn_obs_list, model=model)

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
    print(f'\nElapsed: {elapsed:.0f} s ({elapsed/3600:.1f} hr)', flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)

    with open(OUT_PKL, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(OUT_NPY, samp)

    print(f'\nSaved:')
    print(f'  {OUT_PKL}')
    print(f'  {OUT_NPY}')
    print(f'  {OUT_TRUTH}')
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
