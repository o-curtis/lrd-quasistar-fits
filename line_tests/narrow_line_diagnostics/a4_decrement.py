#!/usr/bin/env python3
"""A4: broad Balmer decrement (broad Ha / broad Hb) for the broad-HeI sources.
Refits the Ha region storing the TOTAL broad flux (1-2 broad Gaussians +
narrow Ha + [NII] doublet), then decrement vs case-B 2.86 -> wind reddening /
collisional enhancement. Egg uses MODS-R."""
import json, glob, os, numpy as np
from scipy.optimize import curve_fit
C = 2.99792458e5
L_HA, L_N2A, L_N2B = 6564.6, 6549.86, 6585.27
v2 = {r['name']: r for r in json.load(open('lrds2/ladder_desi_v2.json')) if 'error' not in r}
bh = {r['name']: r for r in json.load(open('lrds2/broad_helium.json')) if 'error' not in r}

def gauss(x, l, s, A): return A*np.exp(-0.5*((x-l)/(s/C*l))**2)

def fit_ha_broad(w, fn, en, z, sig_n, sig_free=False):
    lam0 = L_HA*(1+z)
    v = C*(w-lam0)/lam0
    sel = (np.abs(v) < 6000.) & ~((w > 7590.) & (w < 7680.))
    if sel.sum() < 40: return None
    x, y, e = w[sel], fn[sel], en[sel]
    def ncplx(x, dv, sn, Aha, An2):
        m = gauss(x, L_HA*(1+z)*(1+dv/C), sn, Aha)
        m += gauss(x, L_N2B*(1+z)*(1+dv/C), sn, An2)
        m += gauss(x, L_N2A*(1+z)*(1+dv/C), sn, An2/2.96)
        return m
    def m2(x, dv, Aha, An2, dvb, sb, Ab, dv2, s2, A2, c0, c1):
        sn = sig_n
        return (c0+c1*(x-lam0) + ncplx(x, dv, sn, Aha, An2)
                + gauss(x, lam0*(1+dvb/C), sb, Ab) + gauss(x, lam0*(1+dv2/C), s2, A2))
    med = np.median(y); a0 = max(np.max(y)-med, np.std(y))
    try:
        p, cov = curve_fit(m2, x, y,
            p0=[0., a0*.7, a0*.05, 0., 900., a0*.25, 0., 2500., a0*.05, med, 0.],
            sigma=e, absolute_sigma=True, maxfev=200000,
            bounds=([-120., 0., 0., -800., 400., 0., -1200., 800., 0., -np.inf, -np.inf],
                    [120., np.inf, np.inf, 800., 4000., np.inf, 1200., 8000., np.inf, np.inf, np.inf]))
    except Exception:
        return None
    perr = np.sqrt(np.diag(cov))
    F1 = p[5]*(p[4]/C*lam0)*np.sqrt(2*np.pi)
    F2 = p[8]*(p[7]/C*lam0)*np.sqrt(2*np.pi)
    E1 = perr[5]*(p[4]/C*lam0)*np.sqrt(2*np.pi)
    E2 = perr[8]*(p[7]/C*lam0)*np.sqrt(2*np.pi)
    return dict(F_bHa=float(F1+F2), E_bHa=float(np.hypot(E1, E2)))

SRC = ['J012930.87+062843.32', 'J082921.37+131237.44', 'J094411.31-024908.65',
       'J101742.79+311459.07', 'J165450.36+033741.74', 'J171741.74+380752.47']
out = {}
print(f"{'source':<24}{'F_bHa':>10}{'F_bHb':>9}{'decrement':>11}")
for n in SRC:
    d = np.load(os.path.join('lrds2/spectra', n+'.npz'))
    w, fl, er = d['wave'].astype(float), d['flux'].astype(float), d['err'].astype(float)
    ok = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    w, fl, er = w[ok], fl[ok], er[ok]
    z, sig = v2[n]['z'], v2[n]['sig_kms']
    r = fit_ha_broad(w, fl, er, z, sig)
    if r is None:
        print(n[:22], 'Ha fit fail/out of band'); continue
    FbHb = bh[n]['F_bHb']
    dec = r['F_bHa']/FbHb if FbHb > 0 else np.nan
    out[n] = dict(**r, F_bHb=FbHb, decrement=float(dec))
    print(f"{n[:22]:<24}{r['F_bHa']:>10.1f}{FbHb:>9.1f}{dec:>11.2f}")
# Egg from MODS-R
from astropy.io import fits as pf
p = glob.glob('/home/omc5226/work/lrdmesa/egg_analysis/egg_data/*/J1025+1402_MODSR*.fits')[0]
dd = pf.open(p)[1].data
w = np.asarray(dd['wave'], float); fl = np.asarray(dd['flux'], float)
iv = np.asarray(dd['ivar'], float); mk = np.asarray(dd['mask'], int)
ok = (mk != 0) & np.isfinite(fl) & (iv > 0)
w, fl, er = w[ok], fl[ok], 1/np.sqrt(iv[ok])
z = v2['Egg_J1025+1402_MODSB']['z']
r = fit_ha_broad(w, fl, er, z, 71.)
FbHb = bh['Egg_J1025+1402']['F_bHb']
dec = r['F_bHa']/FbHb
out['Egg_J1025+1402'] = dict(**r, F_bHb=FbHb, decrement=float(dec))
print(f"{'Egg':<24}{r['F_bHa']:>10.1f}{FbHb:>9.1f}{dec:>11.2f}  (case B = 2.86)")
json.dump(out, open('a4_decrement.json', 'w'), indent=1)
print('wrote a4_decrement.json')
