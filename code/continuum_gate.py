#!/usr/bin/env python3
"""Continuum-detection gate + line-based redshifts for campaign targets.
For each PRISM x1d: (1) find the strongest emission line, hypothesize it is
Halpha, confirm with [O III]5007+Hbeta at the implied z; (2) mask +-3000 km/s
around the standard strong lines; (3) report line-free continuum S/N per pixel
and per 5-pixel bin in three rest windows. Verdict: fittable if binned S/N >= 3
in at least two windows."""
import numpy as np
from astropy.io import fits

C = 2.998e5
FILES = [
    ("The Cliff", "jw04233-o003_s000154183_nirspec_clear-prism_x1d.fits", 3.55),
    ("EGS-42046", "jw04233-o005_s000042046_nirspec_clear-prism_x1d.fits", None),
    ("EGS-55604", "jw04233-o006_s000055604_nirspec_clear-prism_x1d.fits", None),
    ("EGS-49140", "jw04233-o006_s000049140_nirspec_clear-prism_x1d.fits", None),
    ("UDS-182791", "jw04233-o002_s000182791_nirspec_clear-prism_x1d.fits", None),
]
LINES = [1216, 1909, 2798, 3727, 3869, 4102, 4340, 4861, 4959, 5007, 5876,
         6563, 6716, 9069, 9531, 10830, 10938, 12820, 18750]
WINDOWS = [(2000, 3600, "blue-of-break"), (4000, 5600, "optical"),
           (6800, 9500, "red")]

for name, f, zfix in FILES:
    with fits.open(f) as h:
        d = h["EXTRACT1D"].data
        w = d["WAVELENGTH"].astype(float) * 1e4
        fl = d["FLUX"].astype(float) * 1e6
        er = d["FLUX_ERROR"].astype(float) * 1e6
    ok = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    w, fl, er = w[ok], fl[ok], er[ok]
    if zfix is None:
        # strongest peak = Halpha hypothesis; confirm with [O III]
        i = np.argmax(fl)
        z = w[i] / 6564.6 - 1
        o3 = 5008.2 * (1 + z)
        sel = np.abs(w - o3) < 0.02 * o3
        conf = "O III peak %.1f sigma" % ((fl[sel].max() - np.median(fl)) /
                np.median(er[sel])) if sel.sum() else "no O III coverage"
        print("%s: strongest line %.3f um -> z(Ha)=%.3f | %s" % (name, w[i]/1e4, z, conf))
    else:
        z = zfix
        print("%s: z fixed %.3f" % (name, z))
    rest = w / (1 + z)
    linefree = np.ones(len(w), bool)
    for l0 in LINES:
        linefree &= np.abs(C * (rest / l0 - 1)) > 3000
    verdicts = []
    for lo, hi, lab in WINDOWS:
        s = linefree & (rest > lo) & (rest < hi)
        if s.sum() < 8:
            print("   %-14s: no coverage" % lab); verdicts.append(False); continue
        med_f = np.median(fl[s]); med_e = np.median(er[s])
        snpix = med_f / med_e
        n = s.sum() // 5 * 5
        idx = np.where(s)[0][:n]
        fb = fl[idx].reshape(-1, 5).mean(1)
        eb = er[idx].reshape(-1, 5).mean(1) / np.sqrt(5)
        snbin = np.median(fb / eb)
        verdicts.append(snbin >= 3)
        print("   %-14s: F=%6.3f uJy  S/N_pix=%5.2f  S/N_bin5=%5.2f  (N=%d)"
              % (lab, med_f, snpix, snbin, s.sum()))
    print("   VERDICT: %s" % ("FITTABLE (continuum detected)" if sum(verdicts) >= 2
          else "EMISSION-LINE DOMINATED — do not fit photosphere"))
