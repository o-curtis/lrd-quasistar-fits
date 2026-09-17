#!/usr/bin/env python3
"""E4: search for blueshifted wind absorption in The Egg (Table 3 row 2).

Data: our own MODS-R (Halpha, Hbeta) and FIRE (He I 10830) PyPeIt coadds
(tellcorr, slitcorr, dereddened), the same files the paper fits. z=0.1007.
For each line: normalize by a local continuum fit (linear, from windows at
+-(1500-4000) km/s excluding the line core), plot normalized flux vs velocity.
Markers: v=0, -150 km/s (the de Graaff communication), and the escape bracket
-(44-130) km/s. No decomposition, just the profile as observed.
1-column figure, three stacked panels.
"""
import os
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

Z = 0.1007
C = 2.998e5
EGG = '/home/omc5226/work/lrdmesa/egg_analysis'

def find(fname):
    for root, _, files in os.walk(EGG):
        if fname in files:
            return os.path.join(root, fname)
    raise FileNotFoundError(fname)

def load(fname):
    with fits.open(find(fname)) as h:
        s = h['SPECTRUM'].data
        w, f, iv, m = (s['wave'].astype(float), s['flux'].astype(float),
                       s['ivar'].astype(float), s['mask'].astype(int))
    good = (m == 1) & (iv > 0) & np.isfinite(f)
    sig = np.where(good, 1 / np.sqrt(np.where(iv > 0, iv, np.inf)), np.nan)
    return w[good] / (1 + Z), f[good], sig[good]

mr = load('J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
fi = load('J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

LINES = [(r'H$\alpha$ 6563', 6564.61, mr), (r'H$\beta$ 4861', 4862.68, mr),
         (r'He I 10830', 10833.2, fi)]

fig, axes = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True,
                         gridspec_kw={'hspace': 0.08})
for ax, (lab, l0, (w, f, s)) in zip(axes, LINES):
    v = C * (w / l0 - 1)
    sel = np.abs(v) < 4000
    if sel.sum() < 20:
        ax.text(0.5, 0.5, 'no coverage', transform=ax.transAxes, ha='center')
        continue
    vv, ff, ss = v[sel], f[sel], s[sel]
    cont = (np.abs(vv) > 1500)
    p = np.polyfit(vv[cont], ff[cont], 1)
    norm = ff / np.polyval(p, vv)
    nerr = ss / np.polyval(p, vv)
    ax.step(vv, norm, where='mid', color='0.25', lw=0.8)
    ax.fill_between(vv, norm - nerr, norm + nerr, step='mid', color='0.25',
                    alpha=0.18, lw=0)
    ax.axhline(1.0, color='0.6', lw=0.7, ls=(0, (4, 3)))
    ax.axvline(0, color='0.6', lw=0.7)
    ax.axvspan(-130, -44, color='#0173B2', alpha=0.18, lw=0)
    ax.axvline(-150, color='#D55E00', lw=1.0, ls=':')
    ax.text(0.03, 0.86, lab, transform=ax.transAxes, fontsize=9)
    ax.set_xlim(-2400, 2400)
    lo = np.nanpercentile(norm[np.abs(vv) < 2400], 1)
    hi = np.nanpercentile(norm[np.abs(vv) < 2400], 99.7)
    ax.set_ylim(min(0.55, lo - 0.1), hi * 1.06)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    # deepest pixel in the blue escape-to-300 window
    win = (vv > -400) & (vv < 0)
    if win.sum() > 3:
        i = np.nanargmin(norm[win])
        print('%s: deepest blue pixel at v=%.0f km/s, depth=%.3f +- %.3f'
              % (lab, vv[win][i], 1 - norm[win][i], nerr[win][i]))
axes[-1].set_xlabel(r'velocity [km s$^{-1}$]', fontsize=10)
axes[1].set_ylabel('normalized flux', fontsize=10)
fig.savefig('e4_egg_absorption.png', dpi=200, bbox_inches='tight')
fig.savefig('e4_egg_absorption.pdf', bbox_inches='tight')
print('saved e4_egg_absorption.png/.pdf')
