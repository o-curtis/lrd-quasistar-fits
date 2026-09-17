#!/usr/bin/env python3
"""WIDE 2c SMC with the second photosphere allowed up to 4500 K and NO water-band upweight."""
import os, argparse
import m3port as M
import fit_capers_m3 as base
from prospect.models import ProspectorParams
from prospect.models.priors import Uniform
OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'CAPERS_2c_smc_warm_ujy'
DUST_LAW = 'smc'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
WEIGHTS = [(12800.0, 14500.0, 1.0)]
MODEL_KWARGS = dict(av_cap=8.0, cold_hi=4500.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS, out_dir=OUT, dust_law=DUST_LAW, two_comp=True,
          use_galaxy=USE_GALAXY, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c SMC, second photosphere up to 4500 K, H2O x1)')
