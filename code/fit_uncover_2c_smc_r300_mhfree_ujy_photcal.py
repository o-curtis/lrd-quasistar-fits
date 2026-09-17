#!/usr/bin/env python3
"""UNCOVER 2c SMC R=300 with the HOT photosphere metallicity free (mh_idx U[0,2]); cold fixed at [M/H]=-1."""
import os, argparse
import m3port as M
import fit_uncover_m3_r300_photcal as base
from prospect.models import ProspectorParams
from prospect.models.priors import Uniform
OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'UNCOVER_2c_smc_r300_mhfree_ujy_photcal'
DUST_LAW = 'smc'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
WEIGHTS = base.WEIGHTS
MODEL_KWARGS = dict(av_cap=8.0)
_orig_b2c = M.build_model_2c
def _b2c_mhfree(z, **kw):
    m = _orig_b2c(z, **kw)
    mp = m.config_dict
    mp['mh_idx'] = {'N': 1, 'isfree': True, 'init': 1.0, 'prior': Uniform(mini=0.0, maxi=2.0),
                    'units': 'hot-photosphere metallicity index (0=-2, 1=-1, 2=0); cold stays at [M/H]=-1'}
    m2 = ProspectorParams(mp); m2.configure(reset=True)
    return m2
M.build_model_2c = _b2c_mhfree
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS, out_dir=OUT, dust_law=DUST_LAW, two_comp=True,
          use_galaxy=USE_GALAXY, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c SMC R300, hot [M/H] free)')
