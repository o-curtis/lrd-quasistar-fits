#!/usr/bin/env python3
"""A5: velocity-space stack of the He II 4686 region across all DESI sources
with significant broad Hb (bHb_snr >= 5), each normalized by its broad-Hb
flux, inverse-variance weighted -> one deep population limit on
<broad HeII / broad Hb>. Continuum removed per-source with a quadratic fit
(He II +-2500 km/s excluded)."""
import json, os, numpy as np
from scipy.optimize import curve_fit
C = 2.99792458e5
L_HEII = 4687.02
bh = {r['name']: r for r in json.load(open('lrds2/broad_helium.json')) if 'error' not in r}
v2 = {r['name']: r for r in json.load(open('lrds2/ladder_desi_v2.json')) if 'error' not in r}
vgrid = np.arange(-4000., 4001., 100.)
num = np.zeros_like(vgrid); den = np.zeros_like(vgrid)
used, sigbs = [], []
for n, r in bh.items():
    if n.startswith('Egg') or r.get('bHb_snr', 0) < 5: continue
    d = np.load(os.path.join('lrds2/spectra', n+'.npz'))
    w, fl, er = d['wave'].astype(float), d['flux'].astype(float), d['err'].astype(float)
    ok = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    w, fl, er = w[ok], fl[ok], er[ok]
    z = v2[n]['z']
    lam = L_HEII*(1+z)
    v = C*(w-lam)/lam
    sel = np.abs(v) < 9000.
    if sel.sum() < 60: continue
    x, y, e, vv = w[sel], fl[sel], er[sel], v[sel]
    side = np.abs(vv) > 2500.
    # mask narrow lines in sidebands ([ArIV], HeI 4923, Hb blue wing edge)
    for mrest in (4712.58, 4741.45, 4923.3):
        ml = mrest*(1+z)
        side &= np.abs(C*(x-ml)/ml) > 400.
    if side.sum() < 20: continue
    pc = np.polyfit(vv[side], y[side], 2)
    resid = (y - np.polyval(pc, vv))/r['F_bHb']     # units: per (km/s-pixel), normalized
    ivar = (r['F_bHb']/e)**2
    ri = np.interp(vgrid, vv, resid, left=np.nan, right=np.nan)
    wi = np.interp(vgrid, vv, ivar, left=0, right=0)
    good = np.isfinite(ri)
    num[good] += ri[good]*wi[good]; den[good] += wi[good]
    used.append(n); sigbs.append(r['sig_b'])
stack = np.where(den > 0, num/np.maximum(den, 1e-30), np.nan)
estack = np.where(den > 0, 1/np.sqrt(np.maximum(den, 1e-30)), np.nan)
sb = float(np.median(sigbs))
# fit broad Gaussian amplitude at fixed sigma=median sig_b, center 0; also narrow
def m(v, An, Ab, c0):
    return c0 + An*np.exp(-0.5*(v/90.)**2) + Ab*np.exp(-0.5*(v/sb)**2)
gg = np.isfinite(stack)
p, cov = curve_fit(m, vgrid[gg], stack[gg], p0=[0.001, 0.0005, 0.], sigma=estack[gg],
                   absolute_sigma=True, maxfev=100000)
pe = np.sqrt(np.diag(cov))
# convert amplitude (normalized flux density per A... careful: resid is flux/F_bHb per A)
# integrate Gaussian in wavelength: need dlambda per km/s at mean lam -> use ratio form:
# stacked broad HeII/broad Hb = Ab * sqrt(2pi)*sb (km/s) * <lam/c> ... resid units are per-A.
lam_mean = np.mean([L_HEII*(1+v2[n]['z']) for n in used])
amp_to_flux = np.sqrt(2*np.pi)*(sb/C*lam_mean)
ratio = p[1]*amp_to_flux
eratio = pe[1]*amp_to_flux
print('stacked %d sources (median sig_b=%.0f km/s)' % (len(used), sb))
print('stacked broad HeII/broad Hb = %.4f +- %.4f -> %s' % (
    ratio, eratio, ('DETECTION %.1f sig' % (ratio/eratio)) if ratio/eratio >= 3 else
    ('2-sigma limit < %.4f' % (max(ratio, 0)+2*eratio))))
json.dump(dict(nsrc=len(used), used=used, sig_b=sb, ratio=float(ratio), err=float(eratio)),
          open('a5_stack.json', 'w'), indent=1)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ax.step(vgrid, stack*1e3, where='mid', color='0.3', lw=0.9)
ax.fill_between(vgrid, (stack-estack)*1e3, (stack+estack)*1e3, color='0.7', alpha=0.7, lw=0)
ax.plot(vgrid, m(vgrid, *p)*1e3, color='crimson', lw=1.2, alpha=0.9)
ax.axvline(0, color='0.6', lw=0.6, ls=':')
ax.set_xlabel('velocity relative to He II 4686 [km s$^{-1}$]', fontsize=9)
ax.set_ylabel(r'stacked flux / $F_{\rm bH\beta}$ [$10^{-3}\,\mathrm{\AA}^{-1}$]', fontsize=9)
ax.set_title('Population stack: broad He II search (%d DESI sources)' % len(used), fontsize=9)
ax.tick_params(direction='in', labelsize=8)
fig.tight_layout(); fig.savefig('a5_heii_stack.png', dpi=150, bbox_inches='tight')
print('saved a5_heii_stack.png')
