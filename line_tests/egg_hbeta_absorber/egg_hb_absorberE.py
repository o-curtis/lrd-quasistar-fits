#!/usr/bin/env python3
"""Does The Egg's Hbeta complex support a redshifted absorber?

definitive_egg_fit.py pins its single absorber to dva in (-260,-100), so a
+73 km/s component (Lin+2025) is excluded by construction. Here we relax that,
using the SAME data prep, gauss broad family, and joint Hb+[OIII] anchoring:

  A  baseline   : 1 absorber, dva in (-260,-100)      k=19   (published)
  B  free-blue  : 1 absorber, dva in (-400, +250)     k=19
  C  two-absorb : blue in (-400,-40) + red in (-40,+250) k=22

Reports chi2 and BIC. BIC penalty for C over A is 3*lnN.
"""
import json, glob
import numpy as np
from scipy.optimize import differential_evolution

C = 2.99792458e5
L_HB, L_O3, L_O3B = 4862.7, 5008.24, 4960.3
z0 = 0.10066

from astropy.io import fits
p = glob.glob('/home/omc5226/work/lrdmesa/egg_analysis/egg_data/*/J1025+1402_MODSB*.fits')[0]
d = fits.open(p)[1].data
w = np.asarray(d['wave'], float); fl = np.asarray(d['flux'], float)
iv = np.asarray(d['ivar'], float); mk = np.asarray(d['mask'], int)
ok = (mk != 0) & np.isfinite(fl) & (iv > 0)
w, fl, er = w[ok], fl[ok], 1/np.sqrt(iv[ok])

lamB, lamO = L_HB*(1+z0), L_O3*(1+z0)
selB = np.abs(C*(w-lamB)/lamB) < 4500.
for m, h in ((4659.35, 250.), (4712.58, 250.), (4741.45, 250.)):
    ml = m*(1+z0); selB &= np.abs(C*(w-ml)/ml) > h
selO = np.abs(C*(w-lamO)/lamO) < 2600.
selB &= ~selO
xB, yB, eB = w[selB], fl[selB], er[selB]
xO, yO, eO = w[selO], fl[selO], er[selO]
N = len(xB) + len(xO); lnN = np.log(N)

def g(x, l, s, A): return A*np.exp(-0.5*((x-l)/(s/C*l))**2)
def dbl(x, dv, s, A):
    return (g(x, L_O3*(1+z0)*(1+dv/C), s, A) + g(x, L_O3B*(1+z0)*(1+dv/C), s, A/2.98))

def model(P, two):
    if two:
        (dvn, sn, b1, b2, Ab, dvi, si, Ai, dva, sa, Aa,
         dva2, sa2, Aa2, An, Ao, Aoi, c0, c1, c2, d0, d1) = P
    else:
        (dvn, sn, b1, b2, Ab, dvi, si, Ai, dva, sa, Aa,
         An, Ao, Aoi, c0, c1, c2, d0, d1) = P
    u = (xB-lamB)/100.
    mB = c0 + c1*u + c2*u*u
    mB += g(xB, lamB*(1+dvn/C), sn, An)
    mB += g(xB, lamB*(1+b1/C), b2, Ab)          # gauss broad (canon)
    mB += g(xB, lamB*(1+dvi/C), si, Ai)
    mB += g(xB, lamB*(1+dva/C), sa, Aa)
    if two:
        mB += g(xB, lamB*(1+dva2/C), sa2, Aa2)
    mO = d0 + d1*(xO-lamO)/100.
    mO += dbl(xO, dvn, sn, Ao)
    mO += dbl(xO, dvi, si, Aoi)
    return mB, mO

def chi2(P, two):
    mB, mO = model(P, two)
    return float(np.sum(((yB-mB)/eB)**2) + np.sum(((yO-mO)/eO)**2))

medB, medO = np.median(yB), np.median(yO)
aB = max(np.max(yB)-medB, np.std(yB)); aO = max(np.max(yO)-medO, np.std(yO))
head = [(-120, 120), (30, 140), (-800, 800), (400, 4000), (0, 4*aB),
        (-300, 300), (150, 480), (0, 2*aB)]
tail = [(0, 6*aB), (0, 6*aO), (0, 2*aO),
        (medB-2*aB, medB+2*aB), (-3, 3), (-3, 3), (medO-2*aO, medO+2*aO), (-3, 3)]

PUB = np.load('/home/omc5226/work/lrdmesa/gobig_work/definitive_egg_gauss_params.npy')
BV, BS, BA = PUB[8], PUB[9], PUB[10]
eps = 1e-6
CASES = {
    'A published (blue pinned)  ': (head + [(-260, -100), (40, 160), (-2*aB, 0)] + tail, False),
    'E blue FROZEN + red free   ': (head + [(BV-eps, BV+eps), (BS-eps, BS+eps), (BA-eps, BA+eps)]
                                         + [(0, 250), (40, 200), (-2*aB, 0)] + tail, True),
}

out = {}
for lab, (bnds, two) in CASES.items():
    res = differential_evolution(lambda P: chi2(P, two), bnds, seed=42, maxiter=500,
                                 popsize=22, tol=1e-8, polish=True, workers=1,
                                 mutation=(0.4, 1.2), recombination=0.8)
    k = len(bnds) - (3 if 'FROZEN' in lab else 0); bic = res.fun + k*lnN
    P = res.x
    out[lab] = dict(chi2=float(res.fun), k=k, bic=float(bic), params=P.tolist())
    print('%s  chi2=%9.1f  k=%2d  BIC=%9.1f' % (lab, res.fun, k, bic))
    print('        blue absorber: v=%+7.1f sigma=%5.1f amp=%+7.2f' % (P[8], P[9], P[10]))
    if two:
        print('        RED  absorber: v=%+7.1f sigma=%5.1f amp=%+7.2f' % (P[11], P[12], P[13]))
    print()

a, c = out['A published (blue pinned)  '], out['E blue FROZEN + red free   ']
b = a
print('N points = %d, lnN = %.3f' % (N, lnN))

print('E - A : dchi2 = %+8.1f (3 extra)  dBIC = %+8.1f' % (c['chi2']-a['chi2'], c['bic']-a['bic']))
json.dump(out, open('/home/omc5226/work/lrdmesa/gobig_work/egg_hb_absorberE.json', 'w'), indent=1)
print('\nwrote egg_hb_absorberE.json')
