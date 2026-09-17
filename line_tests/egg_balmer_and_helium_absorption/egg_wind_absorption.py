#!/usr/bin/env python3
"""Paper Figure fig:eggwind (egg_wind_absorption.pdf), adopted 2026-08-05.

Successor of e4_egg_absorption.py with three changes: the Hbeta panel shows
MODS-B (the arm the paper fits Hbeta with; 5352 A observed is MODS-R's bad
blue edge below the dichroic), each panel labels its instrument, and the
Hbeta panel overlays the anchored joint Hb+[OIII] decomposition (gauss canon,
definitive_egg_gauss_params.npy) self-normalized by its own fitted continuum,
as the total model and the same model with the absorber zeroed. The gap
between the two curves is the -206 km/s absorber.
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
HERE = os.path.dirname(os.path.abspath(__file__))

# ── anchored decomposition (gauss canon), observed-frame model ───────────────
P = np.load('/home/omc5226/work/lrdmesa/gobig_work/definitive_egg_gauss_params.npy')
(dvn, sn, b1, b2, Ab, dvi, si, Ai, dva, sa, Aa,
 An, Ao, Aoi, c0, c1, c2, d0, d1) = P
Z_FIT = 0.10066
LAMB = 4862.7*(1 + Z_FIT)

def _g(x, l, s, A):
    return A*np.exp(-0.5*((x - l)/(s/C*l))**2)

def hb_model_norm(v_panel, l0):
    """(total/cont, no-absorber/cont) at panel velocities about rest l0."""
    x = l0*(1 + v_panel/C)*(1 + Z)          # observed-frame wavelength
    u = (x - LAMB)/100.
    cont = c0 + c1*u + c2*u*u
    emis = (_g(x, LAMB*(1+dvn/C), sn, An) + _g(x, LAMB*(1+b1/C), b2, Ab)
            + _g(x, LAMB*(1+dvi/C), si, Ai))
    absr = _g(x, LAMB*(1+dva/C), sa, Aa)
    return (cont + emis + absr)/cont, (cont + emis)/cont

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
    sig = np.where(good, 1/np.sqrt(np.where(iv > 0, iv, np.inf)), np.nan)
    return w[good]/(1 + Z), f[good], sig[good]

mb = load('J1025+1402_MODSB_coadd1d_tellcorr_slitcorr_dered.fits')
mr = load('J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
fi = load('J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

LINES = [(r'H$\alpha$ 6563 (MODS-R)', 6564.61, mr),
         (r'H$\beta$ 4861 (MODS-B)', 4862.68, mb),
         (r'He I 10830 (FIRE)', 10833.2, fi)]

fig, axes = plt.subplots(3, 1, figsize=(3.6, 6.4), sharex=True,
                         gridspec_kw={'hspace': 0.08})
for k, (ax, (lab, l0, (w, f, s))) in enumerate(zip(axes, LINES)):
    v = C*(w/l0 - 1)
    sel = np.abs(v) < 4000
    vv, ff, ss = v[sel], f[sel], s[sel]
    cont = (np.abs(vv) > 1500)
    p = np.polyfit(vv[cont], ff[cont], 1)
    norm = ff/np.polyval(p, vv)
    nerr = ss/np.polyval(p, vv)
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
    ax.set_ylim(min(0.55, lo - 0.1), hi*1.06)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(which='both', direction='in', top=True, right=True,
                   labelsize=10)
    if k == 1:   # Hbeta panel: anchored-model overlay
        vg = np.linspace(-2400, 2400, 3000)
        tot, noabs = hb_model_norm(vg, l0)
        ax.plot(vg, tot, color='crimson', lw=1.4, alpha=0.9, zorder=5,
                label='anchored model')
        ax.plot(vg, noabs, color='#228833', lw=1.2, ls='--', alpha=0.9,
                zorder=6, label='model, absorber removed')
        h, lg = ax.get_legend_handles_labels()
        ax.legend(h[::-1], lg[::-1], fontsize=5.2, loc='upper right',
                  frameon=False, handlelength=1.5,
                  bbox_to_anchor=(0.962, 0.93))
        # thin zoom inset on the blue flank, left-center, indented with the
        # panel label
        axi = ax.inset_axes([0.08642, 0.37, 0.36, 0.26])
        axi.step(vv, norm, where='mid', color='0.25', lw=0.8)
        axi.axvspan(-130, -44, color='#0173B2', alpha=0.18, lw=0)
        axi.axvline(-150, color='#D55E00', lw=0.8, ls=':')
        axi.plot(vg, tot, color='crimson', lw=1.3, alpha=0.9, zorder=5)
        axi.plot(vg, noabs, color='#228833', lw=1.1, ls='--', alpha=0.9,
                 zorder=6)
        axi.set_xlim(-500, 0)
        axi.set_ylim(0.7, 3.4)
        axi.tick_params(which='both', direction='in', labelsize=7.0,
                        length=2.5)
axes[-1].set_xlabel(r'velocity [km s$^{-1}$]', fontsize=10)
axes[1].set_ylabel('normalized flux', fontsize=10)
for ext in ('png', 'pdf'):
    out = os.path.join(HERE, f'egg_wind_absorption.{ext}')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('saved', out)
