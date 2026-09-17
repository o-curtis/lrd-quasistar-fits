#!/usr/bin/env python3
"""Joint Egg Hbeta solution: blue absorber, red absorber, and broad family all
optimized together, with the intermediate component tested for necessity.

Motivation. definitive_egg_fit.py pinned the absorber to the blue notch and
found GAUSSIAN by dchi2 ~ 100. Freezing that absorber and adding a redshifted
one (case E) gave dchi2 = -631, but then flipped the family to EXPONENTIAL by
108, so neither the parameters nor the family were settled. Cases B/C/D showed
why a naive free fit cannot arbitrate: a degenerate emission/absorption
canceling pair opens, driving the blue absorber to amplitudes of -63 to -74
against a published -2.1.

The guard here is physical rather than a prior. An absorber can only remove
flux that exists, so the total model must stay positive. The canceling pair
violates that badly (an absorber of depth 74 at -124 km/s sits where the
emission model supplies only ~20), so a positivity penalty closes the
degeneracy without pinning anything by hand.

Grid: {gauss, exp} x {with intermediate, without} x 2 absorbers, plus the
1-absorber controls. Blue absorber velocity stays inside the independently
observed notch (-260,-100); everything else is free.
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

def unpack(P, inter, nabs):
    i = 0
    dvn, sn = P[i], P[i+1]; i += 2
    b1, b2, Ab = P[i], P[i+1], P[i+2]; i += 3
    if inter: dvi, si, Ai = P[i], P[i+1], P[i+2]; i += 3
    else:     dvi, si, Ai = 0., 200., 0.
    absorb = []
    for _ in range(nabs):
        absorb.append((P[i], P[i+1], P[i+2])); i += 3
    An, Ao = P[i], P[i+1]; i += 2
    Aoi = P[i] if inter else 0.; i += 1 if inter else 0
    c0, c1, c2, d0, d1 = P[i], P[i+1], P[i+2], P[i+3], P[i+4]
    return dvn, sn, b1, b2, Ab, dvi, si, Ai, absorb, An, Ao, Aoi, c0, c1, c2, d0, d1

def model(P, family, inter, nabs):
    (dvn, sn, b1, b2, Ab, dvi, si, Ai, absorb,
     An, Ao, Aoi, c0, c1, c2, d0, d1) = unpack(P, inter, nabs)
    u = (xB-lamB)/100.
    emisB = c0 + c1*u + c2*u*u
    emisB = emisB + g(xB, lamB*(1+dvn/C), sn, An)
    if family == 'gauss':
        emisB = emisB + g(xB, lamB*(1+b1/C), b2, Ab)
    else:
        v = C*(xB-lamB*(1+b1/C))/lamB
        emisB = emisB + Ab*np.exp(-np.abs(v)/b2)
    if inter:
        emisB = emisB + g(xB, lamB*(1+dvi/C), si, Ai)
    absB = np.zeros_like(xB)
    for (dv, s, A) in absorb:
        absB = absB + g(xB, lamB*(1+dv/C), s, A)
    mB = emisB + absB
    mO = d0 + d1*(xO-lamO)/100. + dbl(xO, dvn, sn, Ao)
    if inter:
        mO = mO + dbl(xO, dvi, si, Aoi)
    return mB, mO, emisB

def chi2(P, family, inter, nabs):
    mB, mO, emisB = model(P, family, inter, nabs)
    c = float(np.sum(((yB-mB)/eB)**2) + np.sum(((yO-mO)/eO)**2))
    # physical guard: absorption cannot remove more flux than exists
    viol = np.clip(-mB, 0, None)
    if viol.any():
        c += 1e6*float(np.sum(viol**2))
    return c

medB, medO = np.median(yB), np.median(yO)
aB = max(np.max(yB)-medB, np.std(yB)); aO = max(np.max(yO)-medO, np.std(yO))

def bounds(family, inter, nabs):
    b = [(-120, 120), (30, 140)]
    b += [(-800, 800), (400, 4000)] if family == 'gauss' else [(-500, 500), (100, 3000)]
    b += [(0, 4*aB)]
    if inter: b += [(-300, 300), (150, 480), (0, 2*aB)]
    b += [(-260, -100), (40, 160), (-2*aB, 0)]            # blue absorber
    if nabs == 2: b += [(0, 250), (40, 200), (-2*aB, 0)]  # red absorber
    b += [(0, 6*aB), (0, 6*aO)]
    if inter: b += [(0, 2*aO)]
    b += [(medB-2*aB, medB+2*aB), (-3, 3), (-3, 3), (medO-2*aO, medO+2*aO), (-3, 3)]
    return b

CASES = []
for nabs in (1, 2):
    for inter in (True, False):
        for family in ('gauss', 'exp'):
            CASES.append((family, inter, nabs))

out = {}
best = None
for family, inter, nabs in CASES:
    bn = bounds(family, inter, nabs)
    r = differential_evolution(lambda P: chi2(P, family, inter, nabs), bn, seed=42,
                               maxiter=600, popsize=20, tol=1e-8, polish=True,
                               workers=1, mutation=(0.4, 1.2), recombination=0.8)
    k = len(bn); bic = r.fun + k*lnN
    lab = '%-5s inter=%-5s nabs=%d' % (family, inter, nabs)
    up = unpack(r.x, inter, nabs)
    out[lab] = dict(chi2=float(r.fun), k=k, bic=float(bic), params=r.x.tolist())
    print('%s  chi2=%9.1f  k=%2d  BIC=%9.1f' % (lab, r.fun, k, bic))
    for j, (dv, s, A) in enumerate(up[8]):
        print('      absorber %d: v=%+7.1f sigma=%5.1f amp=%+7.2f' % (j+1, dv, s, A))
    if inter:
        print('      intermediate: v=%+7.1f sigma=%5.1f amp=%6.3f' % (up[5], up[6], up[7]))
    if best is None or bic < best[1]:
        best = (lab, bic)
    print()

print('BEST BY BIC: %s (BIC=%.1f)' % best)
json.dump(out, open('/home/omc5226/work/lrdmesa/gobig_work/egg_hb_joint.json', 'w'), indent=1)
print('wrote egg_hb_joint.json')
