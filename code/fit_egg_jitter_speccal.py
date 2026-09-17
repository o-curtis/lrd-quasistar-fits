#!/usr/bin/env python3
"""
fit_egg_jitter_speccal.py — Egg floor-jitter run, RECOUPLED (shared SMC) dust,
plus a per-arm order-1 spectrophotometric calibration polynomial.

Relative to fit_egg_jitter.py --noise-model floor (M3 σ-free z-free + per-arm
floor jitter, shared single SMC A_V), the ONLY addition is:

  Per arm, the total model (photosphere+galaxy) is multiplied by a low-order
  polynomial P(x) = c0 + c1·x  (x = wavelength normalised to [-1,1] across the
  arm).  The coefficients are solved ANALYTICALLY by weighted least squares at
  each likelihood call (linear in c0,c1 — like the old broadband NNLS per-arm
  amplitudes, extended to amplitude+tilt), so they add NO sampled dimensions.

  Order 1 is deliberate: it captures the smooth flux-calibration/slit-loss tilt
  that dominates the MODS-B misfit (data/model runs ~1.2→0.7 blue→red; the MODS-B
  header even carries a LINEAR slit-loss model) but is too gentle to fabricate
  the sharp Balmer break at 3646 Å — so the headline break signal is preserved.
  Do NOT raise the order without checking the break residual survives.

Dust is RECOUPLED: phot_dust_mode='shared' (single SMC A_V = dust2_gal for both
components).  Everything else identical: priors, continuity SFH, Draine&Li dust
emission, sigma_smooth Uniform(115,135), [M/H] free, xi_mtb, the three per-arm
floors log_s_modsb/modsr/fire, CaT 15× weighting, dynesty settings.  ndim = 21.

Noise model is the model-proportional floor: σ_eff² = σ_obs² + (s·m)², with the
floor on the RAW model m so the polynomial solve stays exactly linear.

Speed: dynesty over a multiprocessing fork pool (--pool N).

Imports the core at runtime → R_FIRE = 6000 (canonical).

Usage:
    python fit_egg_jitter_speccal.py --pool 8
    python fit_egg_jitter_speccal.py --pool 8 --order 1 --dry-run

Output (model_finalization/):
    dynesty_egg_duste_sfree_zfree_jitter_floor_speccal{order}.pkl
    chain_egg_duste_sfree_zfree_jitter_floor_speccal{order}.npy
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
J = _load('fit_egg_jitter', 'fit_egg_jitter.py')

from prospect.models.transforms import logsfr_ratios_to_masses

_LN10 = np.log(10.0)

# ── globals for fork pool ─────────────────────────────────────────────────────
_SPS = None
_OBS = None
_MODEL = None
_CAT_WEIGHT = True
_ORDER = 1


def _add_xcal(obs_list):
    """Add obs['xcal'] = rest wavelength normalised to [-1,1] over each arm's
    masked fit range (fixed array; basis for the calibration polynomial)."""
    for o in obs_list:
        wr = o['wave_rest']
        m  = o['mask']
        lo, hi = wr[m].min(), wr[m].max()
        o['xcal'] = 2.0 * (wr - lo) / (hi - lo) - 1.0
    return obs_list


def _solve_poly(d, m, V, w, x, order):
    """Weighted least-squares polynomial P(x)=Σ c_k x^k applied as f = P·m.
    Returns (f, coeffs). Linear in c because V does not depend on c."""
    W = w / V
    # design columns: x^k * m
    cols = [m] if order >= 0 else []
    for k in range(1, order + 1):
        cols.append((x ** k) * m)
    A = np.vstack(cols).T                      # (N, order+1)
    AtW = A.T * W                              # (order+1, N)
    M = AtW @ A                                # (order+1, order+1)
    b = AtW @ d
    try:
        c = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return m.copy(), np.array([1.0] + [0.0] * order)
    f = A @ c
    return f, c


def _logl_speccal(theta, sps, obs_list, model, cat_weight=True, order=1,
                  return_coeffs=False):
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
    dust2_gal     = float(theta[idx['dust2_gal']])     # shared A_V (recoupled)
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
        sfh=int(model.params['sfh']), imf_type=int(model.params['imf_type']),
        dust_type=int(model.params['dust_type']),
        dust1_index=float(model.params['dust1_index']),
        add_neb_emission=add_neb_emission, add_neb_continuum=add_neb_emission,
        add_dust_emission=add_dust_emission, zred=zred,
    )
    if add_dust_emission:
        sps_params['duste_umin']  = float(theta[idx['duste_umin']])
        sps_params['duste_qpah']  = float(theta[idx['duste_qpah']])
        sps_params['duste_gamma'] = float(theta[idx['duste_gamma']])

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)
    if np.any(np.isnan(spec_tl)):
        return (-1e300, {}) if return_coeffs else -1e300

    lnp = 0.0
    coeffs = {}
    for obs in obs_list:
        sigma_inst = obs['sigma_inst']
        log_s = float(theta[idx[J.JITTER_NAME[obs['name']]]])

        pred_tl  = F._smooth_and_interp(wave, spec_tl,  obs['wave_obs'],
                                        sigma_smooth, sigma_inst, F.SIGMA_LIB,
                                        zred, lumdist)
        pred_gal = F._smooth_and_interp(wave, spec_gal, obs['wave_obs'],
                                        0.0, sigma_inst, F.SIGMA_C3K,
                                        zred, lumdist)
        m_all = pred_tl + pred_gal
        mask = obs['mask'] & np.isfinite(m_all) & (m_all > 0)
        if mask.sum() < 10:
            return (-1e300, {}) if return_coeffs else -1e300

        m   = m_all[mask]
        d   = obs['flux'][mask]
        sig = obs['unc'][mask]
        x   = obs['xcal'][mask]
        if cat_weight:
            cat_sel = ((obs['wave_rest'][mask] >= 8400.) &
                       (obs['wave_rest'][mask] <= 8750.))
            w = np.where(cat_sel, 15.0, 1.0)
        else:
            w = np.ones_like(d)

        s_lin = 10.0 ** log_s
        V = sig ** 2 + (s_lin * m) ** 2        # floor on RAW model → poly solve linear

        f, c = _solve_poly(d, m, V, w, x, order)
        good = f > 0
        if good.sum() < 10:
            return (-1e300, {}) if return_coeffs else -1e300

        lnp += -0.5 * np.sum(w[good] * ((d[good] - f[good]) ** 2 / V[good]
                                        + np.log(V[good])))
        coeffs[obs['name']] = c

    if not np.isfinite(lnp):
        return (-1e300, {}) if return_coeffs else -1e300
    return (float(lnp), coeffs) if return_coeffs else float(lnp)


def _logl_global(theta):
    return _logl_speccal(theta, _SPS, _OBS, _MODEL, cat_weight=_CAT_WEIGHT, order=_ORDER)


def _ptform_global(u):
    return _MODEL.prior_transform(u)


def parse_args():
    p = argparse.ArgumentParser(description='Egg floor-jitter + per-arm speccal polynomial')
    p.add_argument('--pool', type=int, default=8)
    p.add_argument('--order', type=int, default=1, help='calibration polynomial order (default 1)')
    p.add_argument('--no-cat-weight', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    global _SPS, _OBS, _MODEL, _CAT_WEIGHT, _ORDER
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    _CAT_WEIGHT = not args.no_cat_weight
    _ORDER = args.order

    STEM = f'egg_duste_sfree_zfree_jitter_floor_speccal{args.order}'
    if not _CAT_WEIGHT:
        STEM += '_nocat'

    print(f'\n{"="*70}', flush=True)
    print(f'The Egg — floor jitter + RECOUPLED shared SMC dust + per-arm speccal poly', flush=True)
    print(f'  poly_order={args.order}  pool={args.pool}  cat_weight={_CAT_WEIGHT}  R_FIRE={F.R_FIRE}', flush=True)
    print(f'  STEM={STEM}', flush=True)
    print(f'{"="*70}\n', flush=True)

    print('Building SPS (shared dust, include_xi10) …', flush=True)
    _SPS = F.build_sps(include_xi10=True, verbose=True)   # shared mode (recoupled)
    print('Loading observations …', flush=True)
    _OBS = _add_xcal(F.build_obs())
    print('Building M3 + jitter model (shared dust) …', flush=True)
    _MODEL = J.build_model_jitter()

    NDIM   = _MODEL.ndim
    LABELS = _MODEL.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in _OBS)
    print(f'\nndim={NDIM}  |  fit pixels={n_pix}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = _ptform_global(u_mid)
    ll_mid, c_mid = _logl_speccal(th_mid, _SPS, _OBS, _MODEL,
                                  cat_weight=_CAT_WEIGHT, order=_ORDER,
                                  return_coeffs=True)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)
    print(f'Mid-prior poly coeffs: ' +
          '; '.join(f'{k}={np.array2string(v, precision=3)}' for k, v in c_mid.items()), flush=True)

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
    # report calibration polynomial at the posterior median
    med = np.median(samp, axis=0)
    _, c_med = _logl_speccal(med, _SPS, _OBS, _MODEL, cat_weight=_CAT_WEIGHT,
                             order=_ORDER, return_coeffs=True)
    print('Calibration polynomial coeffs at median (P(x)=c0+c1 x, x in [-1,1]):', flush=True)
    for k, v in c_med.items():
        print(f'  {k:<8} = {np.array2string(v, precision=4)}', flush=True)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
