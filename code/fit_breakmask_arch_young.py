#!/usr/bin/env python3
"""Masked-break refit of an archival PRISM m3 fit: identical config to the
original driver except rest 3400-4100 A is added to the line mask.
Usage: python -u fit_breakmask_arch.py <stem> [--pool N]"""
import os, sys, re, glob, argparse
import numpy as np
from astropy.io import fits
import m3port as M

# --- host forced YOUNG (mirrors _variant.py youngal exactly) ---
_sp0 = M._sps_params
def _young_sps(theta, model):
    sp = _sp0(theta, model)
    from prospect.models.transforms import logsfr_ratios_to_masses
    n = len(np.atleast_1d(sp["logsfr_ratios"]))
    r = np.full(n, 3.0)
    sp["logsfr_ratios"] = r
    sp["mass"] = logsfr_ratios_to_masses(
        logmass=float(theta[model.theta_index["logmass"]]),
        logsfr_ratios=r, agebins=np.array(model.params["agebins"]))
    return sp
M._sps_params = _young_sps

CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
RFF = os.path.dirname(os.path.abspath(__file__))
MODEL_KWARGS = dict(sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
FIT_LO, FIT_HI = 1400.0, 11500.0
R_PRISM = 100.0
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),(6563,294),(4861,200),
         (5007,150,220),(4959,120),(4686,120),(4340,100),(4102,80),(3970,80),
         (3869,80),(3727,100),(3750,350),(2798,250),(2326,100)]

def z_of(stem):
    for cand in glob.glob(os.path.join(RFF, "fit_*%s*.py" % stem)):
        m = re.search(r"^Z\s*=\s*([0-9.]+)", open(cand).read(), re.M)
        if m:
            return float(m.group(1))
    raise SystemExit("no z for %s" % stem)

ap = argparse.ArgumentParser()
ap.add_argument("stem"); ap.add_argument("--pool", type=int, default=3)
a = ap.parse_args()
z = z_of(a.stem)
path = os.path.join(CAMP, "dja_%s.fits" % a.stem)

def make_build_obs(path, z):
    def build_obs():
        with fits.open(path) as h:
            d = h["SPEC1D"].data
            w_obs = d["wave"].astype(float) * 1e4
            fnu = d["flux"].astype(float) * 1e-6
            enu = d["err"].astype(float) * 1e-6
        w_rest = w_obs / (1.0 + z)
        enu = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
        flux, unc = fnu / 3631.0, enu / 3631.0
        good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0)
        mask = (M.line_mask(w_rest, LINES) & (w_rest >= FIT_LO)
                & (w_rest <= FIT_HI) & good)
        return [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest, flux=flux,
                     unc=unc, mask=mask, sigma_inst=SIGMA_INST)]
    return build_obs

M.fit("BMKY_%s" % a.stem, z, MODEL_KWARGS, make_build_obs(path, z), [],
      out_dir=CAMP, dust_law="smc", pool_n=a.pool,
      source_label="%s break-masked (z=%.3f)" % (a.stem, z))
