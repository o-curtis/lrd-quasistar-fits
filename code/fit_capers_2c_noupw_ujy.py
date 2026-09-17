#!/usr/bin/env python3
"""WIDE — robustness variant 3: NO H2O UPWEIGHT (band weight 1.0).

Exact copy of fit_capers_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3000], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the H2O 12800-14500 A band weight is 1.0 instead of x5 — a plain
uniform-chi^2 fit. Tests whether the two-photosphere geometry conclusions
depend on the x5 upweight. ndim 17.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_capers_m3 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'CAPERS_2c_noupw_ujy'
TEST = 'H2O band weight 1.0 (no x5 upweight)'
DUST_LAW = 'calzetti'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
WEIGHTS = [(12800.0, 14500.0, 1.0)]
MODEL_KWARGS = dict(av_cap=8.0, cold_hi=3000.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=TWO_COMP, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c, H2O x1)')
