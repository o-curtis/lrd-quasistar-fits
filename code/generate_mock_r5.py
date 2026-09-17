#!/usr/bin/env python3
"""R5 injection-recovery mock generator for the RUBIES-36 sample (DJA SPEC1D input).
Clone of generate_mock.py adapted to uJy SPEC1D files, parameterized:
  python generate_mock_r5.py TAG DJA_FITS Z TEFF LOGG AV
Writes mock_<TAG>.npz in the campaign dir (same schema as mock_cliff.npz)."""
import os, sys, numpy as np
from astropy.io import fits
import m3port as M

CAMP = "/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/blueshift-phi-campaign"
TAG, FITSFILE, Z, TEFF, LOGG, AV = sys.argv[1], sys.argv[2], float(sys.argv[3]), \
                                   float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
FIT_LO, FIT_HI = 1400.0, 11500.0
R_PRISM = 100.0
LINES = [(10830,250),(10938,200),(10049,200),(9546,150),(6563,294),(4861,200),
         (5007,150,220),(4959,120),(4686,120),(4340,100),(4102,80),(3970,80),
         (3869,80),(3727,100),(2798,250),(2326,100)]

sps = M.build_sps(dust_law="smc", verbose=False)
model = M.build_model(Z, sigma_fix=0.0, teff_lo=2000.0, mh_fix=1.0, av_cap=8.0, dust1_fix=0.0)
idx = model.theta_index
SIGMA_INST = M.F._CKMS / (R_PRISM * 2.355)

with fits.open(os.path.join(CAMP, FITSFILE)) as h:
    d = h["SPEC1D"].data
    w_obs = d["wave"].astype(float) * 1e4
    fnu   = d["flux"].astype(float) * 1e-6          # uJy -> Jy
    enu   = d["err"].astype(float)  * 1e-6
w_rest = w_obs / (1.0 + Z)
enu = np.sqrt(enu**2 + (M.F_SYS * np.abs(fnu))**2)
real_flux = fnu / 3631.0; unc = enu / 3631.0
good = np.isfinite(real_flux) & np.isfinite(unc) & (unc > 0)
mask = (M.line_mask(w_rest, LINES) & (w_rest >= FIT_LO) & (w_rest <= FIT_HI) & good)
obs = [dict(name="PRISM", wave_obs=w_obs, wave_rest=w_rest, flux=real_flux,
            unc=unc, mask=mask, sigma_inst=SIGMA_INST)]

th = model.prior_transform(0.5*np.ones(model.ndim))
th[idx["teff"]] = TEFF; th[idx["logg"]] = LOGG
th[idx["dust2_gal"]] = AV; th[idx["logmass"]] = 9.0
th[idx["logzsol"]] = -1.0
th[idx["logL_star"]] = 9.0
pt = M.predict_components(th, sps, obs, model)[0]
pred0 = pt[0] + pt[1]
lf = mask & (np.abs(M.F._CKMS*(w_rest/6563.0-1.0)) > 4000.0)
scale = np.nanmedian(real_flux[lf]) / np.nanmedian(pred0[lf])
th[idx["logL_star"]] += np.log10(scale)
th[idx["logmass"]]  += np.log10(scale)
pt = M.predict_components(th, sps, obs, model)[0]
truth_flux = pt[0] + pt[1]

rng = np.random.default_rng(abs(hash(TAG)) % 2**31)
mock_flux = truth_flux + rng.normal(size=len(truth_flux)) * unc

kes, sig_sb, cc = 0.34, 5.6704e-5, 2.998e10
phi_true = kes*sig_sb*TEFF**4/(10**LOGG*cc)
np.savez(os.path.join(CAMP, "mock_%s.npz" % TAG),
         wave_obs=w_obs, wave_rest=w_rest, flux=mock_flux, truth_flux=truth_flux,
         unc=unc, mask=mask, sigma_inst=SIGMA_INST, z=Z,
         teff_true=TEFF, logg_true=LOGG, phi_true=phi_true,
         logL_true=th[idx["logL_star"]], av_true=AV)
print("mock_%s.npz | truth teff=%.0f logg=%.2f phi=%.2f | pixels=%d"
      % (TAG, TEFF, LOGG, phi_true, mask.sum()), flush=True)
