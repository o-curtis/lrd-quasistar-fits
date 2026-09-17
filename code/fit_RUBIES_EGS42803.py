#!/usr/bin/env python3
"""DJA-spectrum M3 phi fit (RUBIES-36 go-big). SPEC1D ext, wave(um)/flux(uJy)/err."""
import os, argparse
import numpy as np
from astropy.io import fits
import m3port as M

CAMP  = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
FITS  = os.path.join(CAMP, "dja_RUBIES_EGS42803.fits")
Z     = 7.152200
LABEL = "RUBIES-EGS-42803"
STEM  = "RUBIES_EGS42803_m3"
DUST_LAW = "smc"
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
FIT_LO, FIT_HI = 1400.0, 11500.0
R_PRISM = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),(6563,294),(4861,200),
         (5007,150,220),(4959,120),(4686,120),(4340,100),(4102,80),(3970,80),
         (3869,80),(3727,100),(2798,250),(2326,100)]
WEIGHTS = []

def build_obs():
    with fits.open(FITS) as h:
        d = h["SPEC1D"].data
        w_obs = d["wave"].astype(float) * 1e4
        fnu   = d["flux"].astype(float) * 1e-6   # uJy -> Jy
        enu   = d["err"].astype(float)  * 1e-6
    w_rest = w_obs / (1.0 + Z)
    enu  = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
    flux = fnu / 3631.0
    unc  = enu / 3631.0
    good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
    mask = (M.line_mask(w_rest, LINES) & (w_rest >= FIT_LO) & (w_rest <= FIT_HI) & good)
    print("  PRISM fit pixels: %d" % int(mask.sum()), flush=True)
    return [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest,
                 flux=flux, unc=unc, mask=mask, sigma_inst=SIGMA_INST)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS, out_dir=CAMP,
          dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL)
