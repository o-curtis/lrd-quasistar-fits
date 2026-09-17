#!/usr/bin/env python3
"""fit_cliff_m3.py -- The Cliff (RUBIES-UDS-154183, z=3.55), campaign pilot.
MAST per-source x1d (EXTRACT1D: WAVELENGTH um, FLUX Jy). M3 1-component,
SMC, free A_V (cap 8), [M/H] fixed -1 (cool grid rule), sigma fixed 0 (PRISM),
xi_mtb fixed 2. F_SYS=0.10. No band upweights: the Balmer break carries the fit.
"""
import os, argparse
import numpy as np
from astropy.io import fits
import m3port as M

CAMP  = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
FITS  = os.path.join(CAMP, "jw04233-o006_s000055604_nirspec_clear-prism_x1d.fits")
Z     = 6.992
LABEL = "RUBIES-EGS55604"
STEM  = "EGS55604_m3_2c"
DUST_LAW = "smc"
MODEL_KWARGS = dict(av_cap=8.0)   # build_model_2c signature (water-dot conventions baked in)
FIT_LO, FIT_HI = 1400.0, 11500.0
R_PRISM = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),
         (6563,294),(4861,200),(5007,150,220),(4959,120),(4686,120),
         (4340,100),(4102,80),(3970,80),(3869,80),(3727,100),(2798,250),(2326,100)]
WEIGHTS = []

def build_obs():
    with fits.open(FITS) as h:
        d = h["EXTRACT1D"].data
        w_obs = d["WAVELENGTH"].astype(float) * 1e4
        fnu   = d["FLUX"].astype(float)
        enu   = d["FLUX_ERROR"].astype(float)
    w_rest = w_obs / (1.0 + Z)
    enu  = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
    flux = fnu / 3631.0
    unc  = enu / 3631.0
    good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
    mask = (M.line_mask(w_rest, LINES)
            & (w_rest >= FIT_LO) & (w_rest <= FIT_HI) & good)
    print("  PRISM fit pixels: %d" % int(mask.sum()), flush=True)
    return [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest,
                 flux=flux, unc=unc, mask=mask, sigma_inst=SIGMA_INST)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=CAMP, dust_law=DUST_LAW, two_comp=True, pool_n=a.pool, dry_run=a.dry_run,
          source_label=LABEL)
