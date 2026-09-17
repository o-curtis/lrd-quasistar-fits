#!/usr/bin/env python3
"""RS (GN-28074) per-line absorption + Paschen tiebreaker, mirroring the Egg
Figure 2 / e6 analysis. JADES medium gratings (R~1000): Halpha in G235M,
He I 10830 + Paschen in G395M. z=2.26. Tests whether Paschen sides with He I
(fast shell) or Halpha (slow base), as for the Egg."""
import os, numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

Z, C = 2.26, 2.998e5
RS = '/home/omc5226/work/lrdmesa/rosetta_stone'
def load(tag):
    f = os.path.join(RS, 'hlsp_jades_jwst_nirspec_goods-n-mediumhst-00028074_%s_v1.0_x1d.fits' % tag)
    d = fits.open(f)[1].data
    w = d['WAVELENGTH'].astype(float)*1e4/(1+Z)   # rest A
    fl = d['FLUX'].astype(float); er = d['FLUX_ERR'].astype(float)
    ok = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    return w[ok], fl[ok], er[ok]
g235 = load('f170lp-g235m'); g395 = load('f290lp-g395m')

def emis(v, c0, an, sn, ab, vb, sb):
    return c0 + an*np.exp(-0.5*(v/sn)**2) + ab*np.exp(-0.5*((v-vb)/sb)**2)

# (label, rest wl, dataset, absorption window, emission halfwidth)
LINES = [('H$\\alpha$', 6564.61, g235, (-800,-40), 3000),
         ('He I 10830', 10833.2, g395, (-900,-40), 2600),
         ('Pa$\\gamma$', 10941.1, g395, (-800,-40), 1800),
         ('Pa$\\delta$', 10052.6, g395, (-800,-40), 1800)]

rng = np.random.default_rng(5)
results = {}
fig, axes = plt.subplots(4, 1, figsize=(3.6, 6.9), gridspec_kw={'hspace': 0.42})
for (lab, l0, (w,f,s), win, ehw), ax in zip(LINES, axes):
    v = C*(w/l0 - 1); sel = np.abs(v) < ehw
    vv, ff, ss = v[sel], f[sel], s[sel]
    cont = np.abs(vv) > 0.75*ehw
    p = np.polyfit(vv[cont], ff[cont], 1); norm = ff/np.polyval(p, vv); ner = ss/np.polyval(p, vv)
    fitsel = ~((vv>win[0])&(vv<win[1])); ins = (vv>win[0])&(vv<win[1])
    def one(y):
        try:
            po,_ = curve_fit(emis, vv[fitsel], y[fitsel], p0=[1,max(y.max()-1,.3),200,.4,-300,500],
                             sigma=np.maximum(ner[fitsel],1e-3),
                             bounds=([.5,0,60,0,-600,150],[1.5,np.inf,900,np.inf,100,2500]), maxfev=20000)
            base = emis(vv[ins],*po)
        except Exception: return np.nan
        depth = base - y[ins]; pos = np.clip(depth,0,None)
        if pos.sum()<=0 or np.max(depth/np.maximum(ner[ins],1e-3))<2: return np.nan
        return np.sum(vv[ins]*pos)/pos.sum()
    cen = one(norm)
    boot = np.array([one(norm+rng.normal(0,1,len(norm))*ner) for _ in range(200)])
    ec = np.nanstd(boot); ndet = np.isfinite(boot).sum()
    results[lab] = (cen, ec, ndet/200)
    # plot
    po,_ = curve_fit(emis, vv[fitsel], norm[fitsel], p0=[1,max(norm.max()-1,.3),200,.4,-300,500],
                     sigma=np.maximum(ner[fitsel],1e-3),
                     bounds=([.5,0,60,0,-600,150],[1.5,np.inf,900,np.inf,100,2500]), maxfev=20000)
    ax.step(vv, norm, where='mid', color='0.25', lw=0.8)
    ax.fill_between(vv, norm-ner, norm+ner, step='mid', color='0.25', alpha=0.18, lw=0)
    vg = np.linspace(-ehw, ehw, 400); ax.plot(vg, emis(vg,*po), color='#EE7733', lw=1.0, alpha=0.8)
    ax.axhline(1, color='0.6', lw=0.6, ls=(0,(4,3))); ax.axvline(0, color='0.6', lw=0.5)
    ax.axvspan(-506, -351, color='#EE7733', alpha=0.15, lw=0)  # Juodzbalis range
    if np.isfinite(cen): ax.axvline(cen, color='#CC3311', lw=1.0, ls=':')
    ax.text(0.03, 0.10, lab, transform=ax.transAxes, fontsize=9, fontweight='bold')
    ax.set_xlim(-ehw*0.55, ehw*0.35); ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    print('%-12s centroid = %s km/s  (det frac %.2f)' %
          (lab, '%6.0f +- %3.0f'%(cen,ec) if np.isfinite(cen) else 'no absorption', ndet/200))
axes[-1].set_xlabel(r'velocity [km s$^{-1}$]', fontsize=10); axes[1].set_ylabel('normalized flux', fontsize=10)
axes[0].text(0.97, 0.08, 'shaded: Juodzbalis+2024 351-506', transform=axes[0].transAxes,
             fontsize=6, ha='right', color='#EE7733')
fig.savefig('rs_absorption.png', dpi=170, bbox_inches='tight')
fig.savefig('rs_absorption.pdf', bbox_inches='tight')
print('saved rs_absorption')
ha = results.get('H$\\alpha$',(np.nan,))[0]; hei = results.get('He I 10830',(np.nan,))[0]
print('TIEBREAKER: Halpha=%.0f, HeI=%.0f, Pag=%.0f, Pad=%.0f' %
      (ha, hei, results.get('Pa$\\gamma$',(np.nan,))[0], results.get('Pa$\\delta$',(np.nan,))[0]))
