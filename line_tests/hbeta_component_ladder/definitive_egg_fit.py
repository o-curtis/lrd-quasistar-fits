#!/usr/bin/env python3
"""Definitive Egg Hbeta decomposition, ghat-style: differential-evolution global
optimization (polished), joint Hbeta+[OIII] anchoring, nested-floor checks, and
the Gaussian-vs-exponential (electron-scattering) broad-family contest at equal k.

Anchors: narrow sigma shared with [OIII]; intermediate (dv, sigma) shared with
the [OIII] wings; absorber pinned to the independently observed Balmer notch
(dva in [-260,-100], sigma_a in [40,160]).
"""
import json, glob
import numpy as np
from scipy.optimize import differential_evolution, least_squares
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
N = len(xB) + len(xO)
lnN = np.log(N)

def g(x, l, s, A): return A*np.exp(-0.5*((x-l)/(s/C*l))**2)
def dbl(x, dv, s, A):
    return (g(x, L_O3*(1+z0)*(1+dv/C), s, A) + g(x, L_O3B*(1+z0)*(1+dv/C), s, A/2.98))

# parameter vector (19): 0 dvn, 1 sn, 2 b1, 3 b2, 4 Ab(broad amp),
# 5 dvi, 6 si, 7 Ai_hb, 8 dva, 9 sa, 10 Aa, 11 An_hb, 12 A_o3n, 13 A_o3i,
# 14-16 contB(c0,c1,c2), 17-18 contO(d0,d1)
def model_joint(P, family):
    dvn, sn, b1, b2, Ab, dvi, si, Ai, dva, sa, Aa, An, Ao, Aoi, c0, c1, c2, d0, d1 = P
    u = (xB-lamB)/100.
    mB = c0 + c1*u + c2*u*u
    mB += g(xB, lamB*(1+dvn/C), sn, An)
    if family == 'gauss':
        mB += g(xB, lamB*(1+b1/C), b2, Ab)
    else:
        v = C*(xB-lamB*(1+b1/C))/lamB
        mB += Ab*np.exp(-np.abs(v)/b2)
    mB += g(xB, lamB*(1+dvi/C), si, Ai)
    mB += g(xB, lamB*(1+dva/C), sa, Aa)
    mO = d0 + d1*(xO-lamO)/100.
    mO += dbl(xO, dvn, sn, Ao)
    mO += dbl(xO, dvi, si, Aoi)
    return mB, mO

def chi2(P, family):
    mB, mO = model_joint(P, family)
    return float(np.sum(((yB-mB)/eB)**2) + np.sum(((yO-mO)/eO)**2))

medB, medO = np.median(yB), np.median(yO)
aB = max(np.max(yB)-medB, np.std(yB)); aO = max(np.max(yO)-medO, np.std(yO))
bounds_common = [(-120, 120), (30, 140), None, None, (0, 4*aB),
                 (-300, 300), (150, 480), (0, 2*aB), (-260, -100), (40, 160), (-2*aB, 0),
                 (0, 6*aB), (0, 6*aO), (0, 2*aO),
                 (medB-2*aB, medB+2*aB), (-3, 3), (-3, 3), (medO-2*aO, medO+2*aO), (-3, 3)]
def bounds_for(family):
    b = list(bounds_common)
    if family == 'gauss': b[2] = (-800, 800); b[3] = (400, 4000)
    else:                 b[2] = (-500, 500); b[3] = (100, 3000)
    return b

results = {}
for family in ('gauss', 'exp'):
    bnds = bounds_for(family)
    res = differential_evolution(lambda P: chi2(P, family), bnds, seed=42, maxiter=500,
                                 popsize=22, tol=1e-8, polish=True, workers=1,
                                 mutation=(0.4, 1.2), recombination=0.8)
    k = 19
    results[family] = (res.fun, res.x)
    print(f"{family:>6}: chi2={res.fun:9.1f}  k={k}  BIC={res.fun + k*lnN:9.1f}")
    P = res.x
    lab = ('dvb', 'sb') if family == 'gauss' else ('dve', 'W')
    print(f"        narrow dv={P[0]:+.0f} sn={P[1]:.0f} | {lab[0]}={P[2]:+.0f} {lab[1]}={P[3]:.0f} Ab={P[4]:.2f}")
    print(f"        outflow dvi={P[5]:+.0f} si={P[6]:.0f} Ai={P[7]:.2f} | abs dva={P[8]:+.0f} sa={P[9]:.0f} Aa={P[10]:.2f}")

# nested floor: 2-component (no outflow, no abs? floor vs recorded 2-comp) sanity
def chi2_sub(P, family):
    Q = np.array(P); Q[7] = 0.; Q[10] = 0.; Q[13] = 0.
    return chi2(Q, family)
for family in ('gauss', 'exp'):
    c_full = results[family][0]
    c_sub = chi2_sub(results[family][1], family)
    print(f"nested floor [{family}]: chi2(full)={c_full:.1f} <= chi2(sub-at-optimum)={c_sub:.1f}: {'OK' if c_full <= c_sub else 'VIOLATED'}")
dBIC = (results['exp'][0] - results['gauss'][0])
print(f"\nDelta chi2 (exp - gauss) = {dBIC:+.1f} at equal k -> {'EXPONENTIAL preferred' if dBIC < 0 else 'GAUSSIAN preferred'}")
np.save('definitive_egg_exp_params.npy', results['exp'][1])
np.save('definitive_egg_gauss_params.npy', results['gauss'][1])
json.dump({f: {'chi2': float(results[f][0]), 'BIC': float(results[f][0] + 19*lnN)} for f in results},
          open('definitive_egg_ladder.json', 'w'), indent=1)
print('saved params + ladder')
