#!/usr/bin/env python3
"""R5: generate an injection-recovery mock on the Cliff wavelength+noise grid.
Injects a known low-gravity photosphere (teff, logg) + host at the REAL Cliff
per-pixel uncertainties, saves mock_<tag>.npz. The refit (fit_mock_<tag>.py)
then tests whether logg -- and hence phi -- is recoverable from an
absorption-light LRD continuum.
"""
import os, sys, numpy as np
from astropy.io import fits
import m3port as M

CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
TRUTH = dict(teff=5000.0, logg=-1.5, dust2_gal=1.0, logmass=9.0)  # injected
Z = 3.55
TAG = "cliff"
FITS = os.path.join(CAMP, "jw04233-o003_s000154183_nirspec_clear-prism_x1d.fits")
FIT_LO, FIT_HI = 2200.0, 11500.0
R_PRISM = 100.0

sps = M.build_sps(dust_law="smc", verbose=False)
model = M.build_model(Z, sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
idx = model.theta_index
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)

with fits.open(FITS) as h:
    d = h["EXTRACT1D"].data
    w_obs = d["WAVELENGTH"].astype(float) * 1e4
    fnu = d["FLUX"].astype(float); enu = d["FLUX_ERROR"].astype(float)
w_rest = w_obs / (1.0 + Z)
enu = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
real_flux = fnu / 3631.0; unc = enu / 3631.0
good = np.isfinite(real_flux) & np.isfinite(unc) & (unc > 0)
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),(6563,294),(4861,200),
         (5007,150,220),(4959,120),(4686,120),(4340,100),(4102,80),(3970,80),
         (3869,80),(3727,100),(2798,250),(2326,100)]
mask = (M.line_mask(w_rest, LINES) & (w_rest >= FIT_LO) & (w_rest <= FIT_HI) & good)
obs = [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest, flux=real_flux,
            unc=unc, mask=mask, sigma_inst=SIGMA_INST)]

# ground-truth theta from prior mid, override the injected params
th = model.prior_transform(0.5*np.ones(model.ndim))
th[idx["teff"]] = TRUTH["teff"]; th[idx["logg"]] = TRUTH["logg"]
th[idx["dust2_gal"]] = TRUTH["dust2_gal"]; th[idx["logmass"]] = TRUTH["logmass"]
th[idx["logzsol"]] = -1.0
# match L to the real continuum: flux ~ 10^logL, solve offset on line-free median
th[idx["logL_star"]] = 9.0
pt = M.predict_components(th, sps, obs, model)[0]
pred0 = pt[0] + pt[1]
lf = mask & (np.abs(M.F._CKMS*(w_rest/6563-1))>4000)
scale = np.nanmedian(real_flux[lf]) / np.nanmedian(pred0[lf])
th[idx["logL_star"]] += np.log10(scale)
th[idx["logmass"]] += np.log10(scale)   # keep host/photosphere ratio
pt = M.predict_components(th, sps, obs, model)[0]
truth_flux = pt[0] + pt[1]

# inject noise
rng = np.random.default_rng(20260714)
mock_flux = truth_flux + rng.normal(size=len(truth_flux)) * unc

# recover truth phi: phi = kes*sigma*Teff^4/(g c), kes=0.34
kes, sig_sb, cc = 0.34, 5.6704e-5, 2.998e10
phi_true = kes*sig_sb*TRUTH["teff"]**4/(10**TRUTH["logg"]*cc)
np.savez(os.path.join(CAMP, "mock_%s.npz"%TAG),
         wave_obs=w_obs, wave_rest=w_rest, flux=mock_flux, truth_flux=truth_flux,
         unc=unc, mask=mask, sigma_inst=SIGMA_INST, z=Z,
         teff_true=TRUTH["teff"], logg_true=TRUTH["logg"], phi_true=phi_true,
         logL_true=th[idx["logL_star"]], av_true=TRUTH["dust2_gal"])
print("mock_%s.npz written | truth teff=%.0f logg=%.2f phi=%.1f | fit pixels=%d"
      % (TAG, TRUTH["teff"], TRUTH["logg"], phi_true, mask.sum()))
