#!/usr/bin/env python3
"""
fit_uncover_m3.py — water dot UNCOVER-A2744-20698 (z=2.4173), adopted Egg M3 port.

NIRSpec PRISM, native pixels -> maggies. sigma_smooth FIXED=0 (PRISM R~100 washes out the
LOSVD), xi_mtb FIXED=2. H2O band 12800-14500 A (rest) weighted x5 (NO CaT, NO FeH).
F_SYS=0.10. Model/likelihood otherwise identical to the Egg M3 (see m3port.py).

READY TO SUBMIT — do NOT auto-run.
    python fit_uncover_m3.py --pool N            # submit
    python fit_uncover_m3.py --dry-run           # verify setup only
"""
import os, argparse
import numpy as np
from astropy.io import fits
import m3port as M

ROOT  = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack'
OUT   = os.path.dirname(os.path.abspath(__file__))
FITS  = os.path.join(ROOT, 'lrd_spectra/uncover_a2744_20698_prism.fits')
Z     = 2.4173
LABEL = 'UNCOVER-A2744-20698'
STEM  = 'UNCOVER_m3b'           # Plan A + Calzetti dust ([M/H]=-1 fixed, A_V<0.5, dust1=0)
DUST_LAW = 'calzetti'           # was SMC in m3a; user switch 2026-06-13

# model knobs: cool-T water-bearing photosphere, [M/H] fixed -1, capped dust
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=0.5, dust1_fix=0.0)

FIT_LO, FIT_HI = 2200.0, 15500.0    # rest-frame Å fit window
R_PRISM    = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)     # ~1273 km/s

# finalized water-dot mask (rest Å). (cen,hw) or (cen,hw_lo,hw_hi).
WATER_LINES = [(10830,250),(12820,250),(10938,200),(10049,200),(9546,150),
               (6563,294),(4861,200),(5007,150,220),(4959,120),(4686,120),
               (4340,100),(4102,80),(3869,80),(2798,250),(2326,100)]
WEIGHTS = [(12800.0, 14500.0, 5.0)]     # H2O band x5


def build_obs():
    with fits.open(FITS) as h:
        d = h['SPEC1D'].data
        w_obs = d['wave'].astype(float) * 1e4          # um -> Å (observed)
        # DJA v4 flux/err are uJy (TUNIT uJy, BUNIT microJansky): F_nu -> f_lambda.
        flam  = d['flux'].astype(float) * 1e-29 * 2.998e18 / w_obs**2
        elam  = d['err'].astype(float)  * 1e-29 * 2.998e18 / w_obs**2
    w_rest = w_obs / (1.0 + Z)
    elam = np.sqrt(elam**2 + (M.F_SYS * np.abs(flam))**2)
    flux = M.flam_to_maggies(w_obs, flam)
    unc  = M.flam_to_maggies(w_obs, elam)
    good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
    mask = (M.line_mask(w_rest, WATER_LINES)
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
