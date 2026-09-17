#!/usr/bin/env python3
"""UNCOVER — robustness variant 1: NEBULAR CONTINUUM ON.

Exact copy of fit_uncover_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3500], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the FSPS galaxy component now carries nebular emission + nebular
continuum. m3port._sps_params hardcodes add_neb_emission=False /
add_neb_continuum=False (the one-line switch); we flip both True via an
in-process wrapper of m3port._sps_params — the m3port.py FILE is untouched, so
concurrently running jobs that import m3port are unaffected. gas_logu=-2.0
FIXED (Liu+2026 best fit; the egg-engine neb_mode='on' init) and gas_logz=-1.0
FIXED (matches the [M/H]=-1 photosphere; egg neb_mode='on' convention), so
ndim stays 17. FSPS also puts nebular LINES in the model spectrum
(add_neb_continuum requires add_neb_emission), but all strong lines are masked
in build_obs, so the constraint is the nebular CONTINUUM (Balmer/Paschen jumps
+ two-photon). Tests whether the cold photosphere survives when nebular
continuum can fill the Balmer-jump/NIR region.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_uncover_m3 as base

# in-process nebular switch (fork-pool workers inherit the patched module)
_orig_sps_params = M._sps_params
def _sps_params_neb(theta, model):
    sp = _orig_sps_params(theta, model)
    sp['add_neb_emission']  = True
    sp['add_neb_continuum'] = True
    sp['gas_logu'] = -2.0
    sp['gas_logz'] = -1.0
    return sp
M._sps_params = _sps_params_neb

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'UNCOVER_2c_neb_ujy'
TEST = 'nebular continuum ON (gas_logu=-2, gas_logz=-1 fixed)'
DUST_LAW = 'calzetti'
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
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c, neb ON)')
