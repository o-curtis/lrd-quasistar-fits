#!/usr/bin/env python3
"""
fit_wide_m3.py — water dot CAPERS-UDS-23216 (z=2.2957), adopted Egg M3 port.

Identical to fit_uncover_m3.py except for the source (z, file) and one extra mask line:
the ~0.31 um (3101 A rest) feature seen only in WIDE. NIRSpec PRISM native pixels -> maggies,
sigma_smooth FIXED=0, xi_mtb FIXED=2, H2O 12800-14500 A x5, F_SYS=0.10.

READY TO SUBMIT — do NOT auto-run.
    python fit_wide_m3.py --pool N
    python fit_wide_m3.py --dry-run
"""
import os, argparse
import numpy as np
from astropy.io import fits
import m3port as M

ROOT  = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack'
OUT   = os.path.dirname(os.path.abspath(__file__))
FITS  = os.path.join(ROOT, 'lrd_spectra/capers-udsp1-v4_prism-clear_6368_23216.spec.fits')
Z     = 2.2957
LABEL = 'CAPERS-UDS-23216'
STEM  = 'CAPERS_m3b_photcal3'             # Plan A + Calzetti dust ([M/H]=-1 fixed, A_V<0.5, dust1=0)
DUST_LAW = 'calzetti'          # was SMC in m3a; matches old WIDE fiducial (Calzetti)

# model knobs: cool-T water-bearing photosphere, [M/H] fixed -1, capped dust
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=0.5, dust1_fix=0.0)

FIT_LO, FIT_HI = 2200.0, 15500.0
R_PRISM    = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)     # ~1273 km/s

WATER_LINES = [(10830,250),(12820,250),(10938,200),(10049,200),(9546,150),
               (6563,294),(4861,200),(5007,150,220),(4959,120),(4686,120),
               (4340,100),(4102,80),(3869,80),(2798,250),(2326,100)]
WIDE_EXTRA  = []            # no extra mask for CAPERS
WEIGHTS     = [(12800.0, 14500.0, 5.0)]


def build_obs():
    with fits.open(FITS) as h:
        d = h['SPEC1D'].data
        w_obs = d['wave'].astype(float) * 1e4
        # DJA v4 flux/err are uJy (TUNIT uJy, BUNIT microJansky): F_nu -> f_lambda.
        flam  = d['flux'].astype(float) * 1e-29 * 2.998e18 / w_obs**2
        elam  = d['err'].astype(float)  * 1e-29 * 2.998e18 / w_obs**2
    # PHOTCAL3 (2026-09-16): anchor the DJA v4 PRISM spectrum to DJA v7 NIRCam photometry (exact band match)
    _lam_um = w_obs * 1e-4
    _c = np.interp(_lam_um, [2.76, 3.5484, 4.0787, 4.3787], [1.0, 0.8139, 0.8643, 0.8643])
    flam = flam * _c; elam = elam * _c
    w_rest = w_obs / (1.0 + Z)
    elam = np.sqrt(elam**2 + (M.F_SYS * np.abs(flam))**2)
    flux = M.flam_to_maggies(w_obs, flam)
    unc  = M.flam_to_maggies(w_obs, elam)
    good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
    mask = (M.line_mask(w_rest, WATER_LINES + WIDE_EXTRA)
            & (w_rest >= FIT_LO) & (w_rest <= FIT_HI) & good)
    print(f"  PRISM fit pixels: {int(mask.sum())}", flush=True)
    return [dict(name='PRISM', wave_obs=w_obs, wave_rest=w_rest,
                 flux=flux, unc=unc, mask=mask, sigma_inst=SIGMA_INST)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL)
