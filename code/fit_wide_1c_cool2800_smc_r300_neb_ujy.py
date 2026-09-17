#!/usr/bin/env python3
"""WIDE — 2-component (hot+cold) with A_V FREE.

Exact copy of fit_wide_2c_v2.py (hot+cold TLUSTY + galaxy, cold T FORCED [2000,3000], [M/H]=-1,
sigma=0, xi=2, dust1=0, Calzetti, H2O x5) EXCEPT the shared A_V (dust2_gal) is uncapped
(av_cap 0.5 -> 8)."""
import os, argparse
import m3port_neb as M
import fit_wide_m3_r300 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'WIDE_1c_cool2800_smc_r300_neb_ujy'
DUST_LAW = 'smc'
USE_GALAXY = True
TWO_COMP = False
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
MODEL_KWARGS = dict(av_cap=8.0, teff_lo=2000.0, teff_hi=2800.0, mh_fix=1.0, sigma_fix=0.0, dust1_fix=0.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, base.Z, MODEL_KWARGS, base.build_obs, base.WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=False, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=base.LABEL + ' (2c, A_V free)')
