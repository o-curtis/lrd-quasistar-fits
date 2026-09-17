#!/usr/bin/env python3
"""Re-decide the Gaussian vs exponential broad family WITH the redshifted
absorber present.

The published contest (definitive_egg_ladder_fixedabs.json) gave GAUSSIAN by
dchi2 = 102 at equal k, with the absorber pinned to the Ha-notch kinematics and
no red component. Case E showed the Hbeta complex needs a second absorber at
+171 km/s. The broad component reshaped when it was added (26% narrower, 41%
taller), so the family verdict has to be re-run.

Setup: blue absorber FROZEN at the published solution (so no canceling pair can
open), red absorber free, both families at equal complexity.
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

PUB = np.load('/home/omc5226/work/lrdmesa/gobig_work/definitive_egg_gauss_params.npy')
BV, BS, BA = PUB[8], PUB[9], PUB[10]

def model(P, family, red):
    if red:
        (dvn, sn, b1, b2, Ab, dvi, si, Ai, dva2, sa2, Aa2,
         An, Ao, Aoi, c0, c1, c2, d0, d1) = P
    else:
        (dvn, sn, b1, b2, Ab, dvi, si, Ai,
         An, Ao, Aoi, c0, c1, c2, d0, d1) = P
    u = (xB-lamB)/100.
    mB = c0 + c1*u + c2*u*u
    mB += g(xB, lamB*(1+dvn/C), sn, An)
    if family == 'gauss':
        mB += g(xB, lamB*(1+b1/C), b2, Ab)
    else:
        v = C*(xB-lamB*(1+b1/C))/lamB
        mB += Ab*np.exp(-np.abs(v)/b2)
    mB += g(xB, lamB*(1+dvi/C), si, Ai)
    mB += g(xB, lamB*(1+BV/C), BS, BA)              # blue absorber, FROZEN
    if red:
        mB += g(xB, lamB*(1+dva2/C), sa2, Aa2)
    mO = d0 + d1*(xO-lamO)/100.
    mO += dbl(xO, dvn, sn, Ao)
    mO += dbl(xO, dvi, si, Aoi)
    return mB, mO

def chi2(P, family, red):
    mB, mO = model(P, family, red)
    return float(np.sum(((yB-mB)/eB)**2) + np.sum(((yO-mO)/eO)**2))

medB, medO = np.median(yB), np.median(yO)
aB = max(np.max(yB)-medB, np.std(yB)); aO = max(np.max(yO)-medO, np.std(yO))
def bounds(family, red):
    b = [(-120, 120), (30, 140)]
    b += [(-800, 800), (400, 4000)] if family == 'gauss' else [(-500, 500), (100, 3000)]
    b += [(0, 4*aB), (-300, 300), (150, 480), (0, 2*aB)]
    if red: b += [(0, 250), (40, 200), (-2*aB, 0)]
    b += [(0, 6*aB), (0, 6*aO), (0, 2*aO),
          (medB-2*aB, medB+2*aB), (-3, 3), (-3, 3), (medO-2*aO, medO+2*aO), (-3, 3)]
    return b

out = {}
for red in (False, True):
    tag = 'WITH red absorber' if red else 'no red absorber (published setup)'
    print('=== %s ===' % tag)
    res = {}
    for family in ('gauss', 'exp'):
        bn = bounds(family, red)
        r = differential_evolution(lambda P: chi2(P, family, red), bn, seed=42,
                                   maxiter=500, popsize=22, tol=1e-8, polish=True,
                                   workers=1, mutation=(0.4, 1.2), recombination=0.8)
        res[family] = (r.fun, r.x.tolist(), len(bn))
        print('  %-6s chi2=%9.1f  k=%2d' % (family, r.fun, len(bn)))
    dc = res['exp'][0] - res['gauss'][0]
    verdict = 'GAUSSIAN' if dc > 0 else 'EXPONENTIAL'
    print('  dchi2 (exp - gauss) = %+8.1f at equal k  ->  %s preferred\n' % (dc, verdict))
    out[tag] = dict(gauss_chi2=res['gauss'][0], exp_chi2=res['exp'][0],
                    k=res['gauss'][2], dchi2_exp_minus_gauss=dc, verdict=verdict,
                    gauss_params=res['gauss'][1], exp_params=res['exp'][1])
json.dump(out, open('/home/omc5226/work/lrdmesa/gobig_work/egg_hb_family_redabs.json', 'w'), indent=1)
print('wrote egg_hb_family_redabs.json')
