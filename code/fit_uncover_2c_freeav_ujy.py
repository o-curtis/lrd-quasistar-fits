#!/usr/bin/env python3
"""UNCOVER — 2-component (hot+cold) with A_V FREE.

Exact copy of fit_uncover_2c.py (hot+cold TLUSTY + galaxy, cold T in [2000,3500], [M/H]=-1,
sigma=0, xi=2, dust1=0, Calzetti, H2O x5) EXCEPT the shared A_V (dust2_gal) is uncapped
(av_cap 0.5 -> 8). Tests whether the cool component survives when dust is unconstrained."""
import os, argparse
import m3port as M
import fit_uncover_m3 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'UNCOVER_2c_freeav_ujy'
DUST_LAW = 'calzetti'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
MODEL_KWARGS = dict(av_cap=8.0)   # A_V FREE (cold T<=3500 default, as UNCOVER_2c)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, base.Z, MODEL_KWARGS, base.build_obs, base.WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=True, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=base.LABEL + ' (2c, A_V free)')
