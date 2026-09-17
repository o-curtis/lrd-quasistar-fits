#!/usr/bin/env python3
"""MATCHED nebular-on control. Identical to the ADOPTED fit (dust_law='smc')
except the FSPS galaxy carries nebular emission + continuum. The pre-existing
*_neb chains used dust_law='calzetti', which confounds nebular with the dust
law; this driver removes that confound."""
import os, argparse
import m3port as M
import fit_wide_m3 as base
_orig = M._sps_params
def _neb(theta, model):
    sp = _orig(theta, model)
    sp['add_neb_emission'] = True; sp['add_neb_continuum'] = True
    sp['gas_logu'] = -2.0; sp['gas_logz'] = -1.0
    return sp
M._sps_params = _neb
OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'WIDE_2c_smc_neb_ujy'
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', type=int, default=1)
    a = ap.parse_args()
    M.fit(STEM, base.Z, dict(av_cap=8.0, cold_hi=3000.0), base.build_obs, base.WEIGHTS, out_dir=OUT,
          dust_law='smc', two_comp=True, use_galaxy=True, pool_n=a.pool,
          source_label=base.LABEL + ' (matched neb ON, smc)')
