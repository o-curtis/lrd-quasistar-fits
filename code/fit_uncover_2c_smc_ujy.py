#!/usr/bin/env python3
"""UNCOVER — robustness variant 6: ATTENUATION SWAP Calzetti -> SMC.

Exact copy of fit_uncover_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3500], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the shared attenuation law is SMC instead of the baseline Calzetti
(the Egg M3 original was SMC; the water-dot baseline switched to Calzetti,
user 2026-06-13). build_sps(dust_law='smc') applies SMC to both photospheres
AND (externally, FSPS dust2=0) to the galaxy — same shared A_V, same free
prior. ndim 17.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_uncover_m3 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'UNCOVER_2c_smc_ujy'
TEST = 'attenuation law SMC instead of Calzetti'
DUST_LAW = 'smc'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
WEIGHTS = base.WEIGHTS
MODEL_KWARGS = dict(av_cap=8.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=TWO_COMP, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c, SMC dust)')
