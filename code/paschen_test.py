#!/usr/bin/env python3
"""Paschen discriminant for the Egg's stratified absorption (run on ROAR).
If the slow Balmer notch traces dense gas at the wind base (n=2 needs
n_H >~ 1e8 via Lyman trapping) while metastable He I 2^3S survives into the
fast outer wind, then the Paschen lines (n=3, denser still) must sit at or
below the Balmer velocity, never at the He I velocity. An eruptive fast
shell instead absorbs all species at the shell speed.
Same emission-model machinery as e6; windows tightened for Pa-gamma, whose
blue sideband at -3000 km/s contains the He I complex.
"""
import sys
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit

Z, C = 0.1007, 2.998e5
with fits.open(sys.argv[1]) as h:
    s = h['SPECTRUM'].data
    w = s['wave'].astype(float) / (1 + Z)
    f = s['flux'].astype(float)
    iv = s['ivar'].astype(float)
    m = s['mask'].astype(int)
g = (m == 1) & (iv > 0) & np.isfinite(f)
w, f, sig = w[g], f[g], 1 / np.sqrt(iv[g])

def emis(v, c0, an, sn_, ab, vb, sb_):
    return (c0 + an * np.exp(-0.5 * (v / sn_) ** 2)
            + ab * np.exp(-0.5 * ((v - vb) / sb_) ** 2))

def measure(lab, l0, win, vmax, cont):
    v = C * (w / l0 - 1)
    sel = np.abs(v) < vmax
    vv, ff, ss = v[sel], f[sel], sig[sel]
    cc = (np.abs(vv) > cont[0]) & (np.abs(vv) < cont[1])
    p = np.polyfit(vv[cc], ff[cc], 1)
    norm = ff / np.polyval(p, vv)
    nerr = ss / np.polyval(p, vv)
    fitsel = ~((vv > win[0]) & (vv < win[1]))
    insel = (vv > win[0]) & (vv < win[1])
    p0 = [1.0, max(norm.max() - 1, 0.3), 150.0, 0.5, 0.0, 800.0]
    lo = [0.5, 0.0, 50.0, 0.0, -300.0, 300.0]
    hi = [1.5, np.inf, 500.0, np.inf, 300.0, 3000.0]
    rng = np.random.default_rng(3)

    def one(y):
        try:
            popt, _ = curve_fit(emis, vv[fitsel], y[fitsel], p0=p0,
                                sigma=np.maximum(nerr[fitsel], 1e-3),
                                bounds=(lo, hi), maxfev=20000)
        except Exception:
            return np.nan, np.nan
        depth = emis(vv[insel], *popt) - y[insel]
        pos = np.clip(depth, 0, None)
        if pos.sum() <= 0 or np.max(depth / np.maximum(nerr[insel], 1e-3)) < 3:
            return np.nan, np.nan
        return (np.sum(vv[insel] * pos) / pos.sum(),
                np.trapz(pos, vv[insel]))
    cen0, ew0 = one(norm)
    boot = np.array([one(norm + rng.normal(0, 1, len(norm)) * nerr)
                     for _ in range(200)])
    ndet = np.isfinite(boot[:, 0]).sum()
    ec = np.nanstd(boot[:, 0])
    if np.isfinite(cen0):
        print('%-14s centroid %6.0f +- %3.0f km/s | EW %5.0f km/s | det frac %.2f'
              % (lab, cen0, ec, ew0, ndet / 200))
    else:
        print('%-14s no absorption above 3 sigma (det frac %.2f)' % (lab, ndet / 200))

measure('He I 10833', 10833.2, (-950, -40), 4000, (1500, 4000))
measure('Pa-gamma 10941', 10941.1, (-700, -40), 2000, (1200, 2000))
measure('Pa-beta 12822', 12821.6, (-700, -40), 4000, (1500, 4000))
measure('Pa-delta 10053', 10052.6, (-700, -40), 2000, (1200, 2000))
