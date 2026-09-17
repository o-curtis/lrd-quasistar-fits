#!/usr/bin/env python3
"""Rosetta Stone — RS_m3c: fix the over-deep CaT by freeing metallicity.

Same as RS_m3b (single quasi-star photosphere + galaxy, Calzetti, A_V<0.5, CaT x15, sigma free)
EXCEPT:
  - [M/H] FREE (was fixed -1) — the CaT-depth test shows data (11.9%) sits between [M/H]=-1
    (22%) and -2 (8%), so a free [M/H]~-1.5 lands the CaT at ~12%.
  - teff_lo = 4000 K — keeps T>4000 so the free [M/H] interpolation is always valid
    (no cool-T artifact); RS sits at ~4205 K anyway.
  - sigma prior widened to U(50,800) (was railed at the 500 cap in m3b).
Runs as a fresh queued job.
"""
import os, argparse
import m3port as M
import fit_rs_m3 as base

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'RS_m3c_smc'
DUST_LAW = 'smc'
Z = base.Z
LABEL = base.LABEL
build_obs = base.build_obs
# [M/H] FREE (no mh_fix), T floor 4000 (artifact-safe), sigma free & wider
MODEL_KWARGS = dict(sigma_prior=(50.0, 800.0), teff_lo=4000.0, av_cap=0.5)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, base.Z, MODEL_KWARGS, base.build_obs, base.WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run,
          source_label=base.LABEL + ' ([M/H] free, CaT fix)')
