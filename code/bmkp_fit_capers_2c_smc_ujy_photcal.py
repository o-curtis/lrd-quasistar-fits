#!/usr/bin/env python3
"""Masked clone: identical config to fit_capers_2c_smc, with rest 3400-4100 A removed
from the likelihood on every arm. Original untouched."""
import argparse
import m3port as M
import fit_capers_2c_smc as drv
import fit_capers_m3_photcal as base

STEM = 'BMKP_CAPERS_2c_smc_ujy_photcal'
MODEL_KWARGS = drv.MODEL_KWARGS
DUST_LAW = drv.DUST_LAW
TWO_COMP = True
USE_GALAXY = True

def build_obs():
    arms = base.build_obs()
    n = 0
    for arm in arms:
        bad = (arm['wave_rest'] >= 3400.0) & (arm['wave_rest'] <= 4100.0)
        n += int((arm['mask'] & bad).sum())
        arm['mask'] = arm['mask'] & ~bad
    print('BMKP break mask removed %d fit pixels' % n, flush=True)
    return arms

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, base.Z, drv.MODEL_KWARGS, build_obs, base.WEIGHTS,
          out_dir=drv.OUT, dust_law=drv.DUST_LAW, two_comp=True, use_galaxy=True, pool_n=a.pool,
          dry_run=a.dry_run, source_label=base.LABEL + ' (BMKP break-masked)')
