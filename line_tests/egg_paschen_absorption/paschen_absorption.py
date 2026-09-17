#!/usr/bin/env python3
"""Paschen companion to fig:eggwind, in the identical style.

Analysis configuration is paschen_test.py VERBATIM (recovered from the
2026-07-14 ROAR run), so the figure displays exactly the fit behind the
quoted centroids:
    measure('Pa-gamma 10941', 10941.1, win=(-700,-40), vmax=2000, cont=(1200,2000))
    measure('Pa-delta 10053', 10052.6, win=(-700,-40), vmax=2000, cont=(1200,2000))
Reproduces Pa gamma -428+-26 and Pa delta -355+-11 km/s (He I -381+-4).
The bounded continuum annulus 1200 < |v| < 2000 km/s is what keeps the
He I 10830 complex, which sits at -2957 km/s in the Pa gamma frame, out of
the continuum fit; an unbounded |v|>1500 window pulls Pa gamma to -514.
"""
import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

Z = 0.1007
C = 2.998e5
EGG = '/home/omc5226/work/lrdmesa/egg_analysis'
HERE = os.path.dirname(os.path.abspath(__file__))
WIN, VMAX, CONT = (-700., -40.), 2000., (1200., 2000.)

def find(fname):
    for root, _, files in os.walk(EGG):
        if fname in files:
            return os.path.join(root, fname)
    raise FileNotFoundError(fname)

with fits.open(find('J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')) as h:
    sp = h['SPECTRUM'].data
    w, f, iv, m = (sp['wave'].astype(float), sp['flux'].astype(float),
                   sp['ivar'].astype(float), sp['mask'].astype(int))
g = (m == 1) & (iv > 0) & np.isfinite(f)
w, f, sig = w[g]/(1 + Z), f[g], 1/np.sqrt(iv[g])

def emis(v, c0, an, sn_, ab, vb, sb_):
    return (c0 + an*np.exp(-0.5*(v/sn_)**2)
            + ab*np.exp(-0.5*((v - vb)/sb_)**2))

LINES = [(r'Pa$\gamma$ 10938 (FIRE)', 10941.1),
         (r'Pa$\delta$ 10049 (FIRE)', 10052.6)]

fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.4), sharex=True,
                         gridspec_kw={'hspace': 0.08})
for k, (ax, (lab, l0)) in enumerate(zip(axes, LINES)):
    v = C*(w/l0 - 1)
    sel = np.abs(v) < VMAX
    vv, ff, ss = v[sel], f[sel], sig[sel]
    cc = (np.abs(vv) > CONT[0]) & (np.abs(vv) < CONT[1])
    p = np.polyfit(vv[cc], ff[cc], 1)
    norm = ff/np.polyval(p, vv)
    nerr = ss/np.polyval(p, vv)
    fitsel = ~((vv > WIN[0]) & (vv < WIN[1]))
    insel = (vv > WIN[0]) & (vv < WIN[1])
    p0 = [1.0, max(norm.max() - 1, 0.3), 150.0, 0.5, 0.0, 800.0]
    lo = [0.5, 0.0, 50.0, 0.0, -300.0, 300.0]
    hi = [1.5, np.inf, 500.0, np.inf, 300.0, 3000.0]
    popt, _ = curve_fit(emis, vv[fitsel], norm[fitsel], p0=p0,
                        sigma=np.maximum(nerr[fitsel], 1e-3),
                        bounds=(lo, hi), maxfev=20000)
    mod = emis(vv, *popt)
    dep = np.clip(mod[insel] - norm[insel], 0, None)
    cen = np.sum(vv[insel]*dep)/dep.sum()
    print('%s: centroid %.0f km/s, peak depth/noise %.1f'
          % (lab, cen, np.max((mod[insel] - norm[insel])
                              / np.maximum(nerr[insel], 1e-3))))

    ax.step(vv, norm, where='mid', color='0.25', lw=0.8)
    ax.fill_between(vv, norm - nerr, norm + nerr, step='mid', color='0.25',
                    alpha=0.18, lw=0)
    ax.fill_between(vv[insel], norm[insel], mod[insel],
                    where=mod[insel] > norm[insel], step='mid',
                    color='crimson', alpha=0.4, lw=0, zorder=3,
                    label='absorbed flux')
    vg = np.linspace(-VMAX, VMAX, 2000)
    ax.plot(vg, emis(vg, *popt), color='#228833', lw=1.2, ls='--', alpha=0.9,
            zorder=6, label='emission model')
    ax.axhline(1.0, color='0.6', lw=0.7, ls=(0, (4, 3)))
    ax.axvline(0, color='0.6', lw=0.7)
    ax.axvspan(-130, -44, color='#0173B2', alpha=0.18, lw=0)
    ax.axvline(-190, color='#D55E00', lw=1.0, ls=':')
    ax.axvline(-381, color='#009E73', lw=1.0, ls='--')
    ax.text(0.03, 0.86, lab, transform=ax.transAxes, fontsize=9)
    ax.set_xlim(-VMAX, VMAX)
    lo_, hi_ = (np.nanpercentile(norm, 1), np.nanpercentile(norm, 99.7))
    ax.set_ylim(min(0.55, lo_ - 0.1), hi_*1.06)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=10)
    if k == 0:
        ax.legend(fontsize=5.2, loc='upper right', frameon=False,
                  handlelength=1.5, bbox_to_anchor=(0.962, 0.93))
axes[-1].set_xlabel(r'velocity [km s$^{-1}$]', fontsize=10)
fig.supylabel('normalized flux', fontsize=10, x=0.02)
for ext in ('png', 'pdf'):
    out = os.path.join(HERE, f'paschen_absorption.{ext}')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)
