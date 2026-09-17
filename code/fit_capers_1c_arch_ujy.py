#!/usr/bin/env python3
"""WIDE — robustness variant 4: SINGLE PHOTOSPHERE, exact 2c likelihood.

Exact copy of fit_capers_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3000], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the cold TLUSTY component is removed — ONE photosphere only, with its
T_eff floor lowered from 3500 K to 2000 K so the sampler can choose a cool
single star instead of the hot+cold pair. Everything else (galaxy + duste,
shared free A_V<=8, Calzetti, dust1=0, [M/H]=-1, sigma=0, xi=2, H2O x5, mask,
F_SYS) is identical — the same m3port._logl evaluates it, so lnZ is directly
comparable to the 2c runs. This is the architecture test. ndim 14.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_capers_m3 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'CAPERS_1c_arch_ujy'
TEST = 'single photosphere (2000K floor) under exact 2c likelihood'
DUST_LAW = 'calzetti'
USE_GALAXY = True
TWO_COMP = False
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
WEIGHTS = base.WEIGHTS
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, teff_hi=6750.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=TWO_COMP, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (1c architecture test)')
