#!/usr/bin/env python3
"""MASK-OFF control. Identical to the ADOPTED fit except the emission-line
masks are removed, so every line pixel enters chi-squared. Directly answers
"why is it OK to ignore the emission lines" by measuring how far the fit moves
when we do not."""
import os, argparse, numpy as np
import m3port as M
import fit_wide_m3 as base
OUT = os.path.dirname(os.path.abspath(__file__))
STEM = 'WIDE_2c_smc_nomask_ujy'
def build_obs_nomask():
    obs = base.build_obs()
    for o in obs:
        w = o['wave_rest']
        good = np.isfinite(o['flux']) & np.isfinite(o['unc']) & (o['unc'] > 0)
        o['mask'] = good & (w >= base.FIT_LO) & (w <= base.FIT_HI)
        print('  NOMASK fit pixels:', int(o['mask'].sum()), flush=True)
    return obs
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pool', type=int, default=1)
    a = ap.parse_args()
    M.fit(STEM, base.Z, dict(av_cap=8.0, cold_hi=3000.0), build_obs_nomask, base.WEIGHTS, out_dir=OUT,
          dust_law='smc', two_comp=True, use_galaxy=True, pool_n=a.pool,
          source_label=base.LABEL + ' (masks OFF, smc)')

# added 2026-09-15 so _variant.run(youngal) can read the adopted config

# module-level copies of the inline config, so _variant.run() (youngal) can read them (added 2026-09-15)
MODEL_KWARGS = dict(av_cap=8.0, cold_hi=3000.0)
DUST_LAW = 'smc'
build_obs = build_obs_nomask
WEIGHTS = base.WEIGHTS
Z = base.Z
LABEL = base.LABEL
USE_GALAXY = True
TWO_COMP = True
