"""Generic robustness-variant driver. Inherits EVERYTHING from the adopted fit
and changes exactly one thing, so nothing is confounded with the dust law."""
import os, sys, argparse, importlib, numpy as np
import m3port as M

def run(adopted_mod, variant, pool=1):
    ad = importlib.import_module(adopted_mod)
    base = getattr(ad, 'base', ad)          # self-contained drivers are their own base
    stem = ad.STEM + '_' + variant
    kw   = dict(ad.MODEL_KWARGS)
    two  = getattr(ad, 'TWO_COMP', adopted_mod.startswith('fit2c_'))
    gal  = getattr(ad, 'USE_GALAXY', True)
    bobs = ad.build_obs
    # --- variant: emission-line masks OFF -------------------------------
    if 'nomask' in variant:
        _o = bobs
        def bobs():
            obs = _o()
            for o in obs:
                w = o['wave_rest']
                good = np.isfinite(o['flux']) & np.isfinite(o['unc']) & (o['unc'] > 0)
                _lo = getattr(base, 'FIT_LO', -np.inf); _hi = getattr(base, 'FIT_HI', np.inf)
                o['mask'] = good & (w >= _lo) & (w <= _hi)
                print('  NOMASK pixels:', int(o['mask'].sum()), flush=True)
            return obs
    # --- variant: host forced YOUNG (no Balmer break) --------------------
    if 'youngal' in variant:
        _sp = M._sps_params
        def _young(theta, model):
            sp = _sp(theta, model)
            n = len(np.atleast_1d(sp['logsfr_ratios']))
            from prospect.models.transforms import logsfr_ratios_to_masses
            r = np.full(n, 3.0)                      # youngest bin dominates
            sp['logsfr_ratios'] = r
            sp['mass'] = logsfr_ratios_to_masses(
                logmass=float(theta[model.theta_index['logmass']]),
                logsfr_ratios=r, agebins=np.array(model.params['agebins']))
            return sp
        M._sps_params = _young
    # --- variant: NEBULAR emission + continuum ON -------------------------
    if 'neb' in variant:
        _spn = M._sps_params
        def _nebp(theta, model, _f=_spn):
            sp = _f(theta, model)
            sp['add_neb_emission'] = True; sp['add_neb_continuum'] = True
            sp['gas_logu'] = -2.0; sp['gas_logz'] = -1.0
            return sp
        M._sps_params = _nebp
    # --- variant: [M/H] FREED (build_model_2c hardcodes mh_fix=1.0 = [M/H]=-1) ---
    if 'mhfree' in variant:
        _bm = M.build_model
        def _bm_free(z, **kw):
            kw['mh_fix'] = None
            return _bm(z, **kw)
        M.build_model = _bm_free
    # --- variant: Ca II TRIPLET MASKED --------------------------------------
    if 'catmask' in variant:
        _oc = bobs
        def bobs(_f=_oc):
            obs = _f()
            for o in obs:
                w = o['wave_rest']
                n0 = int(o['mask'].sum())
                o['mask'] = o['mask'] & ~((w > 8400) & (w < 8760))
                print(f'  CaT masked: {n0} -> {int(o["mask"].sum())} pixels', flush=True)
            return obs
    # --- variant: host DELETED -------------------------------------------
    if 'nogal' in variant:
        gal = False
    # --- variant: HOT component at [M/H]=-2 ON GRID ------------------------
    # The [M/H]=-2 TLUSTY slice exists only for T >= 4000 K, so a metal-poor hot
    # photosphere is only representable above that floor. Fix mh_idx=0 ([M/H]=-2)
    # and raise the hot T floor to 4000 K so nothing is served by the clamp.
    # The COLD component keeps [M/H]=-1 (hardcoded in _components) and its own
    # low-T range, which is the only slice with sub-3000 K water coverage.
    if 'mh2hot' in variant:
        _b0 = M.build_model
        def _b_mh2(z, _f=_b0, **kwx):
            kwx['mh_fix'] = 0.0      # mh_idx 0 == [M/H] = -2
            kwx['teff_lo'] = 4000.0
            return _f(z, **kwx)
        M.build_model = _b_mh2
        kw = dict(kw); kw['teff_hot_lo'] = 4000.0
        print('  MH2HOT: hot [M/H]=-2 fixed, hot T floor 4000 K (on grid)', flush=True)
    # --- variant: host logmass capped at the Sun+2026 empirical host mass ---
    # Sun+2026 [OIII]-anchored host stack: log M* = 8.3 +0.2/-0.4; clustering 7.5-8.5.
    # Cap = 8.5. Same pixels + weights as adopted, so lnZ comparisons are VALID.
    if 'hostm85' in variant:
        from prospect.models.priors import Uniform as _U
        for _bname in ('build_model', 'build_model_2c'):
            _b0 = getattr(M, _bname)
            def _b_cap(z, _f=_b0, **kw):
                m = _f(z, **kw)
                m.config_dict['logmass']['prior'] = _U(mini=7.0, maxi=8.5)
                m.config_dict['logmass']['init'] = float(min(float(m.config_dict['logmass'].get('init', 8.0)), 8.3))
                m.configure(reset=True)
                print('  HOSTM85: logmass prior -> Uniform[7.0, 8.5]', flush=True)
                return m
            setattr(M, _bname, _b_cap)
    ws = list(getattr(base, 'WEIGHTS', []) or [])
    # --- variant: CaT band upweighted x15 (the adopted RS m3c convention) ---
    if 'catw' in variant:
        ws = ws + [(8420.0, 8740.0, 15.0)]
        print('  CaT band 8420-8740 A upweighted x15 (RS convention)', flush=True)
    M.fit(stem, base.Z, kw, bobs, ws, out_dir=getattr(ad, 'CAMP', os.path.dirname(os.path.abspath(__file__))),
          dust_law='smc', two_comp=two, use_galaxy=gal, pool_n=pool,
          source_label=base.LABEL + ' [' + variant + ']')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('adopted'); ap.add_argument('variant')
    ap.add_argument('--pool', type=int, default=3)
    a = ap.parse_args(); run(a.adopted, a.variant, a.pool)
