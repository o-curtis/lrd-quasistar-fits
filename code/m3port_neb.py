#!/usr/bin/env python3
"""
m3port.py — shared engine that ports the ADOPTED Egg M3 fiducial to a new source.

The adopted Egg fiducial is run STEM `egg_duste_sfree_zfree` (M3: TLUSTY photosphere +
FSPS continuity-SFH galaxy + Draine&Li dust emission, shared SMC dust). This module
reproduces that model + likelihood VERBATIM, exposing only the per-source knobs we agreed
to vary (user spec 2026-06-12):

  - sigma_smooth (LOSVD) : FIXED per source  (water dots 0 km/s; RS 155 km/s from CaT2)
  - xi_mtb              : FIXED = 2 km/s     (all three sources)
  - weight region(s)    : H2O 12800-14500 A x5 (water dots) | CaT 8400-8750 A x15 (RS)
  - F_SYS               : 0.10 systematic floor (in each source's build_obs)
  - source redshift

Everything else — components, priors, dust treatment, FastStepBasis SFH, smoothing,
flux normalization, the chi^2 form — is identical to the Egg M3.  Free params = 16
(M3's 18 minus sigma_smooth and xi_mtb, which are now fixed).

The 3 driver scripts (fit_uncover_m3.py / fit_wide_m3.py / fit_rs_m3.py) supply a
build_obs(), the source redshift, the fixed LOSVD, and the weight region(s), then call
m3port.fit(...).  Nothing in this module is auto-run.
"""
import os, sys, time, pickle
import numpy as np

os.environ.setdefault('SPS_HOME', '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/sps-data/fsps-master')
sys.path.insert(0, '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/prospector')

import importlib.util
EGG = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/egg_analysis'
_fs = importlib.util.spec_from_file_location('fit_egg', os.path.join(EGG, 'fit_egg_prospector_tlusty.py'))
F = importlib.util.module_from_spec(_fs); _fs.loader.exec_module(F)

from prospect.models import ProspectorParams
from prospect.models.priors import Uniform
from prospect.models.transforms import logsfr_ratios_to_masses

# ── flux-unit constants ───────────────────────────────────────────────────────
_C_AA       = 2.998e18      # Å/s
_JANSKY_CGS = 1e-23
F_SYS       = 0.10          # 10% systematic floor (user 2026-06-12)


def flam_to_maggies(w_obs_aa, flam):
    """erg/s/cm^2/Å (observed frame) -> maggies (f_nu / 3631 Jy)."""
    return (flam * w_obs_aa**2 / _C_AA) / (3631.0 * _JANSKY_CGS)


def line_mask(w_rest, lines):
    """Boolean keep-mask; lines = list of (cen,hw) or (cen,hw_lo,hw_hi) in rest Å."""
    m = np.ones(len(w_rest), dtype=bool)
    for e in lines:
        cen, hwl = e[0], e[1]; hwh = e[2] if len(e) == 3 else e[1]
        m &= ~((w_rest >= cen-hwl) & (w_rest <= cen+hwh))
    return m


def build_sps(dust_law='smc', include_xi10=True, verbose=True):
    """TLUSTYPlusGalaxyBasis with a selectable attenuation law (smc/calzetti/mw).
    F.build_sps hardcodes 'smc'; this wrapper lets us switch the curve without
    touching the shared Egg module."""
    mod = F._import_tlusty_module()
    return mod.TLUSTYPlusGalaxyBasis(
        spec_dir=F.SPEC_DIR, target_R=F.TARGET_R, cache_file=F.TLUSTY_CACHE,
        xi10_cache_file=(F.XI10_CACHE if include_xi10 else None), verbose=verbose,
        phot_dust_mode='shared', phot_dust_law=dust_law)


def build_model(z, sigma_fix=123.0, sigma_prior=None, teff_lo=3500.0, teff_hi=6750.0,
                mh_fix=None, av_cap=4.0, dust1_fix=None, dust1_cap=4.0):
    """Adopted Egg M3 with per-source knobs (defaults reproduce the original Egg M3).

    sigma_fix   : LOSVD fixed value (used when sigma_prior is None)
    sigma_prior : (lo,hi) -> sigma_smooth FREE with Uniform(lo,hi)
    teff_lo/hi  : T_eff prior bounds (Plan A water dots/RS: lo=2000 to reach the cool grid)
    mh_fix      : if set, mh_idx FIXED to this (Plan A water dots: 1.0 => [M/H]=-1, the
                  only metallicity with cool-T<4000K library coverage); else free
    av_cap      : dust2_gal (shared SMC A_V) prior upper bound (Plan A: 0.5)
    dust1_fix   : if set, dust1 (birth-cloud) FIXED to this (Plan A water dots: 0.0); else free
    dust1_cap   : dust1 prior upper bound when free
    xi_mtb is always FIXED = 2.
    """
    m0 = F.build_model_variant('M3',
                               sigma_mode=('free' if sigma_prior is not None else 'fixed'),
                               zsol_mode='free',
                               mh_mode=('fixed' if mh_fix is not None else 'free'),
                               neb_mode='off', zred=z)
    mp = m0.config_dict

    # T_eff prior
    mp['teff']['prior'] = Uniform(mini=float(teff_lo), maxi=float(teff_hi))
    mp['teff']['init']  = float(min(max(mp['teff']['init'], teff_lo+10), teff_hi-10))

    # LOSVD
    if sigma_prior is not None:
        mp['sigma_smooth']['isfree'] = True
        mp['sigma_smooth']['prior']  = Uniform(mini=float(sigma_prior[0]), maxi=float(sigma_prior[1]))
        mp['sigma_smooth']['init']   = float(0.5*(sigma_prior[0]+sigma_prior[1]))
    else:
        mp['sigma_smooth']['isfree'] = False
        mp['sigma_smooth']['init']   = float(sigma_fix)

    # microturbulence fixed
    mp['xi_mtb']['isfree'] = False
    mp['xi_mtb']['init']   = 2.0

    # photosphere [M/H]
    if mh_fix is not None:
        mp['mh_idx']['isfree'] = False
        mp['mh_idx']['init']   = float(mh_fix)

    # shared SMC A_V cap
    mp['dust2_gal']['prior'] = Uniform(mini=0.0, maxi=float(av_cap))
    mp['dust2_gal']['init']  = float(min(mp['dust2_gal']['init'], av_cap*0.5))

    # birth-cloud dust
    if dust1_fix is not None:
        mp['dust1']['isfree'] = False
        mp['dust1']['init']   = float(dust1_fix)
    else:
        mp['dust1']['prior'] = Uniform(mini=0.0, maxi=float(dust1_cap))

    m = ProspectorParams(mp)
    m.configure(reset=True)
    return m


def build_model_2c(z, av_cap=0.5, teff_hot_lo=3500.0, teff_hot_hi=6750.0,
                   cold_lo=2000.0, cold_hi=3500.0, tie_cold=None):
    """Two-TLUSTY-component model: HOT photosphere (existing teff/logg/logL_star,
    restricted to T>=teff_hot_lo) + COLD photosphere (teff_cold/logg_cold/logL_cold,
    restricted to T<=cold_hi), both [M/H]=-1, sigma=0, xi=2. Plus the FSPS galaxy
    (kept; set m3port._USE_GALAXY=False to drop it). dust1=0, A_V<=av_cap, Calzetti.
    Free params = water-dot m3b (14) + teff_cold/logg_cold/logL_cold = 17.

    tie_cold (2026-07-03, ROAR water-dot-tests):
      None     -> logg_cold free (the 2c_freeav baseline; ndim 17)
      'isophi' -> logg_cold = logg + 4*log10(teff_cold/teff); imposes
                  phi_cold == phi_hot, the spherical-wind geometry (same star,
                  L conserved through the wind); ndim 16
      'spot'   -> logg_cold = logg; cold patches on the same surface; ndim 16
    The tie is evaluated in _cold_logg() at likelihood time via the module
    global _TIE_COLD (set here; inherited by fork-pool workers, same pattern
    as _USE_GALAXY)."""
    global _TIE_COLD
    if tie_cold not in (None, 'isophi', 'spot'):
        raise ValueError(f'unknown tie_cold={tie_cold!r}')
    _TIE_COLD = tie_cold
    m = build_model(z, sigma_fix=0.0, teff_lo=teff_hot_lo, teff_hi=teff_hot_hi,
                    mh_fix=1.0, av_cap=av_cap, dust1_fix=0.0)
    mp = m.config_dict
    mp['teff_cold'] = {'N': 1, 'isfree': True, 'init': 2600.0,
                       'prior': Uniform(mini=cold_lo, maxi=cold_hi),
                       'units': 'K — cold TLUSTY photosphere'}
    mp['logg_cold'] = {'N': 1, 'isfree': tie_cold is None, 'init': -2.5,
                       'prior': Uniform(mini=-3.5, maxi=0.5)}
    mp['logL_cold'] = {'N': 1, 'isfree': True, 'init': 9.5,
                       'prior': Uniform(mini=5.0, maxi=14.0)}
    m2 = ProspectorParams(mp)
    m2.configure(reset=True)
    return m2


def _sps_params(theta, model):
    """Build the get_spectra_components kwargs from theta (shared by _logl & predict)."""
    idx = model.theta_index
    teff      = float(theta[idx['teff']]);  logg = float(theta[idx['logg']])
    logL_star = float(theta[idx['logL_star']])
    mh_idx  = float(theta[idx['mh_idx']]) if 'mh_idx' in idx else float(model.params['mh_idx'])
    logmass = float(theta[idx['logmass']])
    logzsol = float(theta[idx['logzsol']]) if 'logzsol' in idx else mh_idx - 2.0
    dust2_gal = float(theta[idx['dust2_gal']])
    dust1 = float(theta[idx['dust1']]) if 'dust1' in idx else float(model.params['dust1'])
    logsfr_ratios = theta[idx['logsfr_ratios']]
    agebins = np.array(model.params['agebins'])
    zred = float(model.params['zred'])
    mass = logsfr_ratios_to_masses(logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)
    add_dust = 'duste_umin' in idx
    sp = dict(
        teff=teff, logg=logg, mh_idx=mh_idx, xi_mtb=float(model.params['xi_mtb']),
        logL_star=logL_star, logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1, logsfr_ratios=logsfr_ratios, mass=mass,
        agebins=agebins, sfh=int(model.params['sfh']), imf_type=int(model.params['imf_type']),
        dust_type=int(model.params['dust_type']), dust1_index=float(model.params['dust1_index']),
        add_neb_emission=True, add_neb_continuum=True, add_dust_emission=add_dust, zred=zred)
    if add_dust:
        sp['duste_umin']  = float(theta[idx['duste_umin']])
        sp['duste_qpah']  = float(theta[idx['duste_qpah']])
        sp['duste_gamma'] = float(theta[idx['duste_gamma']])
    return sp


_USE_GALAXY = True   # set False by fit() for hot+cold-TLUSTY-only (no FSPS galaxy)
_TIE_COLD   = None   # None | 'isophi' | 'spot' — set by build_model_2c(tie_cold=...)


def _cold_logg(theta, idx):
    """logg of the cold component: free parameter, or computed from the tie.

    'isophi': logg_cold = logg_hot + 4 log10(Tc/Th)  <=>  phi_cold = phi_hot
              (kappa_es sigma T^4 / g c equal for both components — the
              covering-fraction-independent spherical-wind constraint).
    'spot':   logg_cold = logg_hot (same surface).
    Ties can push logg_cold below the TLUSTY grid floor; the basis clamps
    there, same behaviour as the free-parameter prior edge."""
    if 'logg_cold' in idx:
        return float(theta[idx['logg_cold']])
    gh = float(theta[idx['logg']])
    if _TIE_COLD == 'spot':
        return gh
    if _TIE_COLD == 'isophi':
        th = float(theta[idx['teff']]); tc = float(theta[idx['teff_cold']])
        return gh + 4.0*np.log10(tc/th)
    raise RuntimeError('logg_cold is not free but _TIE_COLD is not set')


def _components(theta, sps, model):
    """Assemble (wave, spec_tl, spec_gal) in Lsun/Hz. spec_tl is the hot photosphere
    plus, if the model has a cold component ('teff_cold' free), a second dusted TLUSTY
    photosphere. spec_gal is zeroed when _USE_GALAXY is False."""
    sp = _sps_params(theta, model)
    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sp)
    idx = model.theta_index
    if 'teff_cold' in idx:
        tc = float(theta[idx['teff_cold']]); gc = _cold_logg(theta, idx)
        Lc = float(theta[idx['logL_cold']])
        # undusted cold photosphere (no 'dust2' key -> get_galaxy_spectrum applies none)
        _, cold, _ = sps.tlusty.get_galaxy_spectrum(
            teff=tc, logg=gc, mh_idx=1.0, xi_mtb=2.0, logL_star=Lc)
        a_v = float(sp['dust2_gal'])
        if a_v > 0.0:
            cold = cold * sps._dust_trans(a_v, sp)   # same law/A_V as the hot photosphere
        spec_tl = spec_tl + cold
    if not _USE_GALAXY:
        spec_gal = np.zeros_like(spec_gal)
    return wave, spec_tl, spec_gal


def predict_split(theta, sps, obs_list, model):
    """Return per-obs (p_hot, p_cold, p_gal) in maggies, with hot/cold photospheres
    separated (p_cold is zeros for a 1-component model). For plotting the breakdown."""
    idx = model.theta_index
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    zred = float(model.params['zred']); lumdist = float(model.params['lumdist'])
    sp = _sps_params(theta, model)
    wave, spec_hot, spec_gal, _ = sps.get_spectra_components(**sp)
    if not _USE_GALAXY:
        spec_gal = np.zeros_like(spec_gal)
    if 'teff_cold' in idx:
        tc = float(theta[idx['teff_cold']]); gc = _cold_logg(theta, idx)
        Lc = float(theta[idx['logL_cold']])
        _, cold, _ = sps.tlusty.get_galaxy_spectrum(teff=tc, logg=gc, mh_idx=1.0, xi_mtb=2.0, logL_star=Lc)
        a_v = float(sp['dust2_gal'])
        if a_v > 0.0:
            cold = cold * sps._dust_trans(a_v, sp)
    else:
        cold = np.zeros_like(spec_hot)
    out = []
    for obs in obs_list:
        si = obs['sigma_inst']
        ph = F._smooth_and_interp(wave, spec_hot, obs['wave_obs'], sigma_smooth, si, F.SIGMA_LIB, zred, lumdist)
        pc = F._smooth_and_interp(wave, cold,     obs['wave_obs'], sigma_smooth, si, F.SIGMA_LIB, zred, lumdist)
        pg = F._smooth_and_interp(wave, spec_gal, obs['wave_obs'], 0.0,          si, F.SIGMA_C3K, zred, lumdist)
        out.append((ph, pc, pg))
    return out


def predict_components(theta, sps, obs_list, model):
    """Return per-obs (pred_tl, pred_gal) in maggies on each arm's wave_obs grid.
    Same forward model as _logl; used for plotting the best-fit spectrum."""
    idx = model.theta_index
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    zred = float(model.params['zred']); lumdist = float(model.params['lumdist'])
    wave, spec_tl, spec_gal = _components(theta, sps, model)
    out = []
    for obs in obs_list:
        si = obs['sigma_inst']
        ptl  = F._smooth_and_interp(wave, spec_tl,  obs['wave_obs'], sigma_smooth, si, F.SIGMA_LIB, zred, lumdist)
        pgal = F._smooth_and_interp(wave, spec_gal, obs['wave_obs'], 0.0,          si, F.SIGMA_C3K, zred, lumdist)
        out.append((ptl, pgal))
    return out


def _logl(theta, sps, obs_list, model, weights_spec):
    """Egg M3 forward model + plain chi^2 with configurable upweighted region(s).

    weights_spec : list of (lo_rest, hi_rest, weight) applied to obs['wave_rest'].
    Identical to F._logl except sigma_smooth/xi_mtb are read from model.params (fixed)
    and the upweighted region is parameterised instead of hard-coded CaT x15.
    """
    idx = model.theta_index
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    zred = float(model.params['zred']); lumdist = float(model.params['lumdist'])
    wave, spec_tl, spec_gal = _components(theta, sps, model)
    if np.any(np.isnan(spec_tl)):
        return -1e300

    lnp = 0.0
    for obs in obs_list:
        si = obs['sigma_inst']
        pred_tl  = F._smooth_and_interp(wave, spec_tl,  obs['wave_obs'], sigma_smooth, si, F.SIGMA_LIB, zred, lumdist)
        pred_gal = F._smooth_and_interp(wave, spec_gal, obs['wave_obs'], 0.0,          si, F.SIGMA_C3K, zred, lumdist)
        pred = pred_tl + pred_gal
        mk = obs['mask'] & np.isfinite(pred) & (pred > 0)
        if mk.sum() < 10:
            return -1e300
        res = obs['flux'][mk] - pred[mk]; unc = obs['unc'][mk]
        wr  = obs['wave_rest'][mk]
        w = np.ones(mk.sum())
        for lo, hi, wt in weights_spec:
            w[(wr >= lo) & (wr <= hi)] = wt
        lnp += -0.5 * np.sum(w * (res/unc)**2)
    return float(lnp) if np.isfinite(lnp) else -1e300


# ── fork-pool globals (inherited by workers at fork time) ─────────────────────
_SPS = _OBS = _MODEL = _WEIGHTS = None
def _logl_g(theta): return _logl(theta, _SPS, _OBS, _MODEL, _WEIGHTS)
def _ptf_g(u):      return _MODEL.prior_transform(u)


def fit(stem, z, model_kwargs, build_obs_fn, weights_spec, out_dir,
        dust_law='smc', two_comp=False, use_galaxy=True,
        pool_n=1, dry_run=False, source_label=''):
    """Build SPS/obs/model and run the adopted-M3 dynesty fit (or dry-run).
    two_comp=True -> build_model_2c (hot+cold TLUSTY); use_galaxy=False drops the galaxy."""
    global _SPS, _OBS, _MODEL, _WEIGHTS, _USE_GALAXY
    import dynesty
    from dynesty.utils import resample_equal

    _USE_GALAXY = use_galaxy
    print(f"\n{'='*68}\n{source_label or stem}  —  adopted M3 port (z={z})", flush=True)
    print(f"  model_kwargs = {model_kwargs}", flush=True)
    print(f"  two_comp={two_comp} use_galaxy={use_galaxy} | dust_law={dust_law} | "
          f"xi=2 FIXED | F_SYS={F_SYS}", flush=True)
    print(f"  upweighted: " + (", ".join(f"{lo:.0f}-{hi:.0f}A x{wt:g}" for lo,hi,wt in weights_spec)
                               or "none (uniform)"), flush=True)
    print(f"{'='*68}\n", flush=True)

    _WEIGHTS = weights_spec
    print('Building SPS …', flush=True);       _SPS   = build_sps(dust_law=dust_law, verbose=True)
    print('Loading obs …', flush=True);        _OBS   = build_obs_fn()
    print('Building model …', flush=True)
    _MODEL = build_model_2c(z, **model_kwargs) if two_comp else build_model(z, **model_kwargs)
    NDIM = _MODEL.ndim; LAB = _MODEL.theta_labels()
    print(f"\nndim={NDIM} (expect 16)\nfree = {LAB}", flush=True)
    th = _ptf_g(0.5*np.ones(NDIM))
    print(f"Mid-prior ln L = {_logl_g(th):.1f}", flush=True)
    if dry_run:
        print('\n[dry-run] setup OK — not sampling.', flush=True); return

    import multiprocessing as mp
    pool = mp.get_context('fork').Pool(pool_n)
    print(f"\nStarting DynamicNestedSampler ({pool_n}-worker fork pool) …", flush=True)
    smp = dynesty.DynamicNestedSampler(_logl_g, _ptf_g, NDIM, bound='multi', sample='rwalk',
                                       pool=pool, queue_size=pool_n)
    t0 = time.time()
    import os as _os
    _nlive = int(_os.environ.get('WDT_NLIVE', F.NLIVE_INIT))
    smp.run_nested(dlogz_init=F.DLOGZ_INIT, nlive_init=_nlive, nlive_batch=F.NLIVE_BATCH,
                   wt_kwargs={'pfrac': 1.0}, n_effective=F.N_EFFECTIVE, print_progress=True)
    pool.close(); pool.join()
    res = smp.results
    samp = resample_equal(res.samples, np.exp(res.logwt - res.logz[-1]))
    os.makedirs(out_dir, exist_ok=True)
    pickle.dump(res, open(os.path.join(out_dir, f'dynesty_{stem}.pkl'), 'wb'))
    np.save(os.path.join(out_dir, f'chain_{stem}.npy'), samp)
    print(f"\nElapsed {time.time()-t0:.0f}s; lnZ={res.logz[-1]:.2f}; saved chain_{stem}.npy", flush=True)
    for i, l in enumerate(LAB):
        q = np.percentile(samp[:, i], [16, 50, 84])
        print(f"  {l:<16}= {q[1]:.4g} (-{q[1]-q[0]:.3g}/+{q[2]-q[1]:.3g})")
