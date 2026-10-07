#!/usr/bin/env python3
"""Masked clone of fit_cliff_m3.py (MAST pilot spectrum, identical config)
MODEL_KWARGS = drv.MODEL_KWARGS
DUST_LAW = drv.DUST_LAW
Z = drv.Z
WEIGHTS = drv.WEIGHTS
LABEL = drv.LABEL
CAMP = drv.CAMP
with rest 3400-4100 A removed from the likelihood. Original untouched."""
import argparse
import m3port as M
import fit_cliff_m3 as drv

STEM = 'BMKA_CLIFF_m3_pilot'

def build_obs():
    arms = drv.build_obs()
    n = 0
    for arm in arms:
        bad = (arm['wave_rest'] >= 3400.0) & (arm['wave_rest'] <= 4100.0)
        n += int((arm['mask'] & bad).sum())
        arm['mask'] = arm['mask'] & ~bad
    print('BMKA break mask removed %d fit pixels' % n, flush=True)
    return arms

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, drv.Z, drv.MODEL_KWARGS, build_obs, drv.WEIGHTS,
          out_dir=drv.CAMP, dust_law=drv.DUST_LAW, pool_n=a.pool,
          dry_run=a.dry_run, source_label=drv.LABEL + ' (BMKA break-masked)')
