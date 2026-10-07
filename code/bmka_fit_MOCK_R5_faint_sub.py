#!/usr/bin/env python3
"""Masked clone of fit_MOCK_R5_faint_sub: identical mock, rest 3400-4100 A removed."""
import argparse
import m3port as M
import fit_MOCK_R5_faint_sub as drv

STEM = "BMKA_" + drv.STEM

def build_obs():
    arms = drv.build_obs()
    n = 0
    for arm in arms:
        bad = (arm["wave_rest"] >= 3400.0) & (arm["wave_rest"] <= 4100.0)
        n += int((arm["mask"] & bad).sum())
        arm["mask"] = arm["mask"] & ~bad
    print("BMKA break mask removed %d fit pixels" % n, flush=True)
    return arms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    M.fit(STEM, drv.Z, drv.MODEL_KWARGS, build_obs, getattr(drv, "WEIGHTS", []),
          out_dir=drv.CAMP, dust_law=drv.DUST_LAW, pool_n=a.pool,
          dry_run=a.dry_run, source_label="BMKA " + drv.STEM)
