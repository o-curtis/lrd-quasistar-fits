#!/usr/bin/env python3
"""UNCOVER — robustness variant 7: H2O BAND MASKED (mask-swap cross-validation).

Exact copy of fit_uncover_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3500], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the H2O 12800-14500 A (rest) band is MASKED entirely — the fit sees
only the continuum outside the band (weights uniform; the x5 is moot on masked
pixels). Tests whether the cold component and its logg survive on
continuum-shape information alone. ndim 17.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_uncover_m3 as base

H2O_LO, H2O_HI = 12800.0, 14500.0

def build_obs():
    obs = base.build_obs()
    for o in obs:
        band = (o['wave_rest'] >= H2O_LO) & (o['wave_rest'] <= H2O_HI)
        o['mask'] &= ~band
        print(f"  H2O band masked ({int(band.sum())} px removed) -> "
              f"{int(o['mask'].sum())} fit pixels", flush=True)
    return obs

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'UNCOVER_2c_h2omask_ujy'
TEST = 'H2O band masked entirely (continuum-only)'
DUST_LAW = 'calzetti'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
WEIGHTS = []
MODEL_KWARGS = dict(av_cap=8.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=TWO_COMP, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c, H2O masked)')
