#!/usr/bin/env python3
"""A2 finale: (1) absorption-fair exponential test for the Egg; (2) joint
Hb+[OIII] fit with SHARED intermediate kinematics for the 4 sources that
require a third component -> the intermediate's [OIII]/Hb ratio discriminates
host outflow (~narrow ratio, ~5) from atmosphere gas (<<1, forbidden-line
suppressed) from scattering-profile artifact (exponential wins instead)."""
import json, glob, os, numpy as np
from scipy.optimize import curve_fit
C = 2.99792458e5
L_HB, L_O3, L_O3B = 4862.7, 5008.24, 4960.3
v2 = {r['name']: r for r in json.load(open('lrds2/ladder_desi_v2.json')) if 'error' not in r}

def load(name):
    if name.startswith('Egg'):
        from astropy.io import fits
        p = glob.glob('/home/omc5226/work/lrdmesa/egg_analysis/egg_data/*/J1025+1402_MODSB*.fits')[0]
        d = fits.open(p)[1].data
        w = np.asarray(d['wave'], float); fl = np.asarray(d['flux'], float)
        iv = np.asarray(d['ivar'], float); mk = np.asarray(d['mask'], int)
        ok = (mk != 0) & np.isfinite(fl) & (iv > 0)
        return w[ok], fl[ok], 1/np.sqrt(iv[ok])
    d = np.load(os.path.join('lrds2/spectra', name+'.npz'))
    w, fl, er = d['wave'].astype(float), d['flux'].astype(float), d['err'].astype(float)
    ok = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    return w[ok], fl[ok], er[ok]

def gauss(x, l, s, A): return A*np.exp(-0.5*((x-l)/(s/C*l))**2)

# ---------- (1) Egg absorption-fair profile contest
name = 'Egg_J1025+1402_MODSB'
w, fl, er = load(name)
z, sig = v2[name]['z'], v2[name]['sig_kms']
lam = L_HB*(1+z)
sel = np.abs(C*(w-lam)/lam) < 4500.
x, y, e = w[sel], fl[sel], er[sel]
med = np.median(y); a0 = max(np.max(y)-med, np.std(y))
ABS = lambda x, dva, sa, Aa: gauss(x, lam*(1+dva/C), sa, Aa)
def m2(x, dvn, An, dvb, sb, Ab, dva, sa, Aa, c0, c1, c2):
    u = (x-lam)/100.
    return (c0+c1*u+c2*u*u + gauss(x, lam*(1+dvn/C), sig, An)
            + gauss(x, lam*(1+dvb/C), sb, Ab) + ABS(x, dva, sa, Aa))
def m3(x, dvn, An, dvb, sb, Ab, dvi, si, Ai, dva, sa, Aa, c0, c1, c2):
    return m2(x, dvn, An, dvb, sb, Ab, dva, sa, Aa, c0, c1, c2) + gauss(x, lam*(1+dvi/C), si, Ai)
def mexp(x, dvn, An, dve, W, Ae, dva, sa, Aa, c0, c1, c2):
    u = (x-lam)/100.
    v = C*(x-lam*(1+dve/C))/lam
    return (c0+c1*u+c2*u*u + gauss(x, lam*(1+dvn/C), sig, An)
            + Ae*np.exp(-np.abs(v)/W) + ABS(x, dva, sa, Aa))
fits = {}
p2, _ = curve_fit(m2, x, y, p0=[0., a0*.8, 0., 700., a0*.15, -150., 150., -a0*.05, med, 0., 0.],
                  sigma=e, absolute_sigma=True, maxfev=200000,
                  bounds=([-120., 0., -800., 400., 0., -600., 40., -np.inf, -np.inf, -np.inf, -np.inf],
                          [120., np.inf, 800., 4000., np.inf, 150., 500., 0., np.inf, np.inf, np.inf]))
fits['2G+abs'] = (np.sum(((y-m2(x, *p2))/e)**2), 11)
p3, _ = curve_fit(m3, x, y, p0=list(p2[:5])+[0., 300., a0*.05]+list(p2[5:]),
                  sigma=e, absolute_sigma=True, maxfev=300000,
                  bounds=([-120., 0., -800., 400., 0., -300., 150., 0., -600., 40., -np.inf, -np.inf, -np.inf, -np.inf],
                          [120., np.inf, 800., 4000., np.inf, 300., 480., np.inf, 150., 500., 0., np.inf, np.inf, np.inf]))
fits['3G+abs'] = (np.sum(((y-m3(x, *p3))/e)**2), 14)
pe, _ = curve_fit(mexp, x, y, p0=[0., a0*.8, 0., 600., a0*.2, -150., 150., -a0*.05, med, 0., 0.],
                  sigma=e, absolute_sigma=True, maxfev=300000,
                  bounds=([-120., 0., -500., 100., 0., -600., 40., -np.inf, -np.inf, -np.inf, -np.inf],
                          [120., np.inf, 500., 3000., np.inf, 150., 500., 0., np.inf, np.inf, np.inf]))
fits['exp+abs'] = (np.sum(((y-mexp(x, *pe))/e)**2), 11)
N = len(x)
print('EGG absorption-fair contest (BIC):')
for k, (chi, kk) in fits.items():
    print('  %-8s chi2=%8.1f k=%2d BIC=%8.1f' % (k, chi, kk, chi+kk*np.log(N)))
print('  exp decay W = %.0f km/s' % pe[3])

# ---------- (2) joint Hb+[OIII] shared-intermediate fits
print('\nJoint Hb+[OIII] shared-intermediate discriminator:')
print(f"{'source':<24}{'sig_i':>7}{'dv_i':>7}{'[OIII]/Hb(int)':>15}{'[OIII]/Hb(narrow)':>18}  verdict")
res = {}
for name in ('J012930.87+062843.32', 'J094411.31-024908.65', 'J171741.74+380752.47', 'Egg_J1025+1402_MODSB'):
    w, fl, er = load(name)
    z, sig = v2[name]['z'], v2[name]['sig_kms']
    lamB, lamO = L_HB*(1+z), L_O3*(1+z)
    s1 = np.abs(C*(w-lamB)/lamB) < 4500.
    s2 = np.abs(C*(w-lamO)/lamO) < 2600.
    s1 &= ~s2
    x1, y1, e1 = w[s1], fl[s1], er[s1]
    x2, y2, e2 = w[s2], fl[s2], er[s2]
    X = np.concatenate([x1, x2]); Y = np.concatenate([y1, y2]); E = np.concatenate([e1, e2])
    F = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    def dbl(x, dv, s_kms, A):
        return (gauss(x, L_O3*(1+z)*(1+dv/C), s_kms, A)
                + gauss(x, L_O3B*(1+z)*(1+dv/C), s_kms, A/2.98))
    def model(xf, dvn, AnB, dvb, sb, Ab, dvi, si, AiB, AnO, AiO,
              dva, sa, Aa, c0, c1, d0, d1):
        x = xf[0]; f = xf[1]
        mB = (c0 + c1*(x-lamB)/100. + gauss(x, lamB*(1+dvn/C), sig, AnB)
              + gauss(x, lamB*(1+dvb/C), sb, Ab) + gauss(x, lamB*(1+dvi/C), si, AiB)
              + ABS_l(x, lamB, dva, sa, Aa))
        mO = (d0 + d1*(x-lamO)/100. + dbl(x, dvn, sig, AnO) + dbl(x, dvi, si, AiO))
        return np.where(f < 0.5, mB, mO)
    def ABS_l(x, l, dva, sa, Aa): return gauss(x, l*(1+dva/C), sa, Aa)
    allow_abs = name.startswith('Egg')
    med1, med2 = np.median(y1), np.median(y2)
    a1 = max(np.max(y1)-med1, np.std(y1)); a2 = max(np.max(y2)-med2, np.std(y2))
    p0 = [0., a1*.8, 0., 800., a1*.15, 0., 300., a1*.05, a2*.9, a2*.1,
          -150., 150., (-a1*.05 if allow_abs else 0.), med1, 0., med2, 0.]
    lo = [-120., 0., -800., 400., 0., -300., 150., 0., 0., 0.,
          -600., 40., (-np.inf if allow_abs else -1e-12), -np.inf, -np.inf, -np.inf, -np.inf]
    hi = [120., np.inf, 800., 4000., np.inf, 300., 480., np.inf, np.inf, np.inf,
          150., 500., 1e-12, np.inf, np.inf, np.inf, np.inf]
    try:
        pp, cov = curve_fit(model, (X, F), Y, p0=p0, sigma=E, absolute_sigma=True,
                            maxfev=400000, bounds=(lo, hi))
        perr = np.sqrt(np.diag(cov))
        si_ = pp[6]
        FiB = pp[7]*(si_/C*lamB)*np.sqrt(2*np.pi)
        FiO = pp[9]*(si_/C*lamO)*np.sqrt(2*np.pi)*(1+1/2.98)
        FnB = pp[1]*(sig/C*lamB)*np.sqrt(2*np.pi)
        FnO = pp[8]*(sig/C*lamO)*np.sqrt(2*np.pi)*(1+1/2.98)
        Rint = FiO/FiB if FiB > 0 else np.nan
        Rnar = FnO/FnB if FnB > 0 else np.nan
        snr_iO = pp[9]/max(perr[9], 1e-30)
        if not np.isfinite(Rint): v = 'no intermediate Hb flux'
        elif snr_iO < 2: v = f'[OIII]int <2sig -> ATMOSPHERE/artifact (R<{2*perr[9]*(si_/C*lamO)*np.sqrt(2*np.pi)*(1+1/2.98)/FiB:.1f})'
        elif Rint > 0.5*Rnar: v = 'HOST OUTFLOW (int ratio ~ narrow ratio)'
        elif Rint < 0.15*Rnar: v = 'ATMOSPHERE (forbidden-suppressed)'
        else: v = 'MIXED'
        print(f"{name[:22]:<24}{si_:>7.0f}{pp[5]:>7.0f}{Rint:>15.2f}{Rnar:>18.2f}  {v}")
        res[name] = dict(sig_int=float(si_), dv_int=float(pp[5]), R_int=float(Rint),
                         R_narrow=float(Rnar), snr_intOIII=float(snr_iO), verdict=v)
    except Exception as ex:
        print(name[:22], 'FAIL', str(ex)[:60])
json.dump(res, open('a2_discriminator.json', 'w'), indent=1)
print('\nwrote a2_discriminator.json')
