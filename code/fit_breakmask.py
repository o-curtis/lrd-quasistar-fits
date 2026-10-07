#!/usr/bin/env python3
"""M3 phi fits for the 15 strongest-Balmer-break LRDs of de Graaff et al. (2025).

Identical model, priors, line mask, systematic floor and sampler settings to the
paper's archival PRISM fits (cf. fit_RUBIES_UDS40579.py, fit_MoMBH1.py):
1-component TLUSTY photosphere + FSPS host under a shared SMC screen,
sigma fixed 0, xi_mtb fixed 2, [M/H] fixed -1, A_V free to 8, F_SYS = 0.10.
All 15 use the DJA v4 PRISM spectrum listed in the de Graaff release, so the
sample is homogeneous in both data source and pipeline.

Usage: python -u fit_breakmask.py <srcid> [--pool N]
"""
import os, sys, json, argparse
import numpy as np
from astropy.io import fits
import m3port as M

CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
BDIR = os.path.join(CAMP, "break15")
SPECDIR = os.path.join(BDIR, "spectra")

DUST_LAW = "smc"
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
FIT_LO, FIT_HI = 1400.0, 11500.0
R_PRISM = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),(6563,294),(4861,200),
         (5007,150,220),(4959,120),(4686,120),(4340,100),(4102,80),(3970,80),
         (3869,80),(3727,100),(3750,350),(2798,250),(2326,100)]
WEIGHTS = []

SEL = {str(s['file']).split('_')[-1].split('.')[0]: s
       for s in json.load(open(os.path.join(BDIR, 'top15.json')))}


def make_build_obs(path, z):
    def build_obs():
        with fits.open(path) as h:
            d = h["SPEC1D"].data
            w_obs = d["wave"].astype(float) * 1e4
            fnu   = d["flux"].astype(float) * 1e-6      # uJy -> Jy
            enu   = d["err"].astype(float)  * 1e-6
        w_rest = w_obs / (1.0 + z)
        enu  = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
        flux = fnu / 3631.0
        unc  = enu / 3631.0
        good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
        mask = (M.line_mask(w_rest, LINES) & (w_rest >= FIT_LO)
                & (w_rest <= FIT_HI) & good)
        print("  PRISM fit pixels: %d" % int(mask.sum()), flush=True)
        return [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest,
                     flux=flux, unc=unc, mask=mask, sigma_inst=SIGMA_INST)]
    return build_obs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("srcid")
    ap.add_argument("--pool", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    s = SEL[a.srcid]
    path = os.path.join(SPECDIR, s['file'])
    M.fit("BMK_%s" % a.srcid, s['z'], MODEL_KWARGS,
          make_build_obs(path, s['z']), WEIGHTS, out_dir=BDIR,
          dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run,
          source_label="%s (dG25 break %.2f, z=%.3f)"
                       % (a.srcid, s['break_p'][2], s['z']))
