#!/usr/bin/env python3
"""WIDE — robustness variant 2: TRUE PRISM WAVELENGTH-DEPENDENT RESOLUTION.

Exact copy of fit_capers_2c_freeav.py (hot+cold TLUSTY + FSPS galaxy + duste,
shared A_V free to 8, cold T in [2000,3000], hot T in [3500,6750] unless stated,
[M/H]=-1, sigma=0, xi=2, dust1=0, Calzetti, H2O 12800-14500 A x5 unless stated)
EXCEPT: the fixed R=100 instrumental resolution is replaced by the true
NIRSpec PRISM R(lambda) from jwst_nirspec_prism_disp.fits (the STScI PRISM
dispersion product, msaexp copy; R = 30.1-421.6 over 0.5-6.0 um; R(3um)=101,
so the baseline R=100 is only right near 3 um — the rest-frame H2O band sits
at ~4.4-4.9 um observed where the true R is 200-280). obs['sigma_inst']
becomes the per-pixel array c/(R(lambda_obs)*2.355) and F._smooth_and_interp
is wrapped in-process to apply it as NSEG=24 piecewise-constant-sigma segments
(geometric binning in sigma; within-segment sigma error <~5 per cent, far below
the LSF systematic). Each segment call reuses the ORIGINAL _smooth_and_interp
(smooth full library grid at that sigma, interp at that segment's pixels), so
per-pixel values are exactly the original pipeline evaluated at the segment
sigma. m3port.py and the egg-engine FILES are untouched. ndim 17.
Definitive robustness suite 2026-07-04; nlive=2000 via WDT_NLIVE in the sbatch."""
import os, argparse
import m3port as M
import fit_capers_m3 as base
import numpy as np
from astropy.io import fits as pyfits

DISP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jwst_nirspec_prism_disp.fits')
NSEG = 24

with pyfits.open(DISP) as _h:
    _d = _h[1].data
    _DISP_W = _d['WAVELENGTH'].astype(float)      # micron
    _DISP_R = _d['R'].astype(float)               # resolving power

def _r_of_lambda(w_obs_aa):
    """True PRISM R at observed wavelength in Angstrom (endpoint-clamped interp)."""
    return np.interp(np.asarray(w_obs_aa, dtype=float) / 1e4, _DISP_W, _DISP_R)

_orig_smooth = M.F._smooth_and_interp
def _smooth_and_interp_var(wave_rest, spec, obs_wave, sig_smooth, sig_inst, sig_lib, zred, lumdist):
    """Variable-sigma wrapper: scalar sigma_inst -> original; array -> NSEG
    piecewise-constant-sigma segments (geometric bins; representative sigma =
    median of the pixels in the bin)."""
    if np.ndim(sig_inst) == 0:
        return _orig_smooth(wave_rest, spec, obs_wave, sig_smooth, sig_inst, sig_lib, zred, lumdist)
    si = np.asarray(sig_inst, dtype=float)
    out = np.full(len(obs_wave), np.nan)
    edges = np.geomspace(si.min() * (1 - 1e-9), si.max() * (1 + 1e-9), NSEG + 1)
    seg = np.clip(np.digitize(si, edges) - 1, 0, NSEG - 1)
    for k in range(NSEG):
        m = seg == k
        if m.any():
            out[m] = _orig_smooth(wave_rest, spec, obs_wave[m], sig_smooth,
                                  float(np.median(si[m])), sig_lib, zred, lumdist)
    return out
M.F._smooth_and_interp = _smooth_and_interp_var

def build_obs():
    obs = base.build_obs()
    for o in obs:
        R = _r_of_lambda(o['wave_obs'])
        o['sigma_inst'] = M.F._CKMS / (R * 2.355)
        print(f"  PRISM R(lambda): R {R.min():.0f}-{R.max():.0f}  "
              f"sigma_inst {o['sigma_inst'].min():.0f}-{o['sigma_inst'].max():.0f} km/s  "
              f"({NSEG} segments)", flush=True)
    return obs

OUT  = os.path.dirname(os.path.abspath(__file__))
STEM = 'CAPERS_2c_prismR_ujy'
TEST = 'true PRISM R(lambda) instead of fixed R=100'
DUST_LAW = 'calzetti'
USE_GALAXY = True
TWO_COMP = True
Z = base.Z
LABEL = base.LABEL
WEIGHTS = base.WEIGHTS
MODEL_KWARGS = dict(av_cap=8.0, cold_hi=3000.0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, two_comp=TWO_COMP, use_galaxy=USE_GALAXY,
          pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL + ' (2c, PRISM R(lambda))')
    if a.dry_run:
        import time as _t
        th = M._ptf_g(0.5 * np.ones(M._MODEL.ndim))
        t0 = _t.time(); n = 3
        for _ in range(n): M._logl_g(th)
        print(f"[prismR] mean lnL call time: {(_t.time()-t0)/n:.2f}s", flush=True)
