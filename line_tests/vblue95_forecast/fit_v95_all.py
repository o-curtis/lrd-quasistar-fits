#!/usr/bin/env python3
"""Uniform v_blue,95 (Naidu+2026 definition, Matthee+2026 Eq. 1 model, validated
on his anchors) applied to the remaining Figure-7 sources:
DESI (LRDs)^2, the Egg (MODS-R), GN-28074 (G235M), GLIMPSE-17775 (G395M coadd),
UDS-40579 (He I mode), MoM-BH*-1 (attempt).
Lines: 'ha' (with [N II]), 'hb' (no [N II], [O III]4960 masked), 'hei' (10833)."""
import os, sys, json, glob
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_matthee_eq1 as M
from scipy.optimize import least_squares
C_KMS = M.C_KMS
LINES_CEN = {'ha': 6564.632, 'hb': 4862.7, 'hei': 10833.2}
CONT_MASKS = [(6564.632, 6000.), (4862.7, 4000.), (4341.7, 2500.), (5008.24, 2500.),
              (4960.295, 2500.), (6725.0, 2500.), (5877.25, 2500.), (6302.0, 2000.),
              (7067.1, 2000.), (7325.0, 2000.), (9071.1, 2000.), (9533.2, 2000.),
              (10833.2, 4000.), (10052.1, 2000.), (10941.1, 2000.), (3971.2, 2000.),
              (3890.2, 2000.), (4102.9, 2000.), (5891.6, 2500.)]

def desi_R(w_A):
    R = np.where(w_A < 5800., 2500., np.where(w_A < 7600., 3500., 4500.))
    return R

def load_desi(name):
    d = np.load(f'/home/omc5226/work/lrdmesa/gobig_work/lrds2/spectra/{name}.npz')
    w, f, e = d['wave'].astype(float), d['flux'].astype(float), d['err'].astype(float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    return w[ok]/1e4, f[ok], e[ok], ('desi', None), 1.0

def load_mods(_):
    p = glob.glob('/home/omc5226/work/lrdmesa/egg_analysis/egg_data/*/J1025+1402_MODSR*.fits')[0]
    d = fits.open(p)[1].data
    w = np.asarray(d['wave'], float); f = np.asarray(d['flux'], float)
    iv = np.asarray(d['ivar'], float); mk = np.asarray(d['mask'], int)
    ok = (mk != 0) & np.isfinite(f) & (iv > 0)
    e = 1/np.sqrt(iv[ok])
    e = np.sqrt(e**2 + (0.05*np.abs(f[ok]))**2)   # paper's 5% Egg floor
    return w[ok]/1e4, f[ok], e, ('const', 2250.), 1.0

def load_mods_b(_):
    p = glob.glob('/home/omc5226/work/lrdmesa/egg_analysis/egg_data/*/J1025+1402_MODSB*.fits')[0]
    d = fits.open(p)[1].data
    w = np.asarray(d['wave'], float); f = np.asarray(d['flux'], float)
    iv = np.asarray(d['ivar'], float); mk = np.asarray(d['mask'], int)
    ok = (mk != 0) & np.isfinite(f) & (iv > 0)
    e = 1/np.sqrt(iv[ok]); e = np.sqrt(e**2 + (0.05*np.abs(f[ok]))**2)
    return w[ok]/1e4, f[ok], e, ('const', 1850.), 1.0

def load_dja(files, grating):
    lo, hi = {'G395M': (2.87, 5.30), 'G235M': (1.66, 3.07)}[grating]
    ws, fs, es = [], [], []
    for fn in files:
        d = fits.open(fn)['SPEC1D'].data
        ws.append(np.asarray(d['wave'], float)); fs.append(np.asarray(d['flux'], float))
        es.append(np.asarray(d['err'], float))
    w0 = ws[0]
    F, W = np.zeros_like(w0), np.zeros_like(w0)
    for w, f, e in zip(ws, fs, es):
        fi = np.interp(w0, w, f, left=np.nan, right=np.nan)
        ei = np.interp(w0, w, e, left=np.nan, right=np.nan)
        ok = np.isfinite(fi) & np.isfinite(ei) & (ei > 0)
        wt = np.zeros_like(w0); wt[ok] = 1/ei[ok]**2
        F += np.where(ok, fi, 0)*wt; W += wt
    f = np.where(W > 0, F/np.maximum(W, 1e-30), np.nan); e = np.where(W > 0, 1/np.sqrt(np.maximum(W, 1e-30)), np.nan)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (w0 >= lo) & (w0 <= hi)
    e = np.sqrt(e**2 + (0.10*np.abs(f))**2)      # paper's 10% JWST floor
    g = grating.lower()
    dd = fits.open(M.DISP.format(g))[1].data
    return w0[ok], f[ok], e[ok], ('curve', (np.asarray(dd['WAVELENGTH'], float), np.asarray(dd['R'], float))), 1.8

def R_at(rspec, lam_um):
    kind, val = rspec
    if kind == 'const': return val
    if kind == 'desi': return float(desi_R(np.array([lam_um*1e4]))[0])
    return float(np.interp(lam_um, val[0], val[1]))

def fit_source(name, z0, loader, line='ha', oiii='fit', plotdir=None, no_abs=False, seed=11):
    w, f, e, rspec, lsf_fac = loader(name)
    sigN, z_sys, oflag = 150., z0, 'fixed'
    if oiii == 'fit':
        try:
            z_sys, sigN = M.fit_oiii(w, f, e, z0); oflag = 'fit'
        except Exception as ex:
            oflag = f'fallback'
    lam_c = LINES_CEN[line]*(1+z_sys)
    v_all = C_KMS*(w*1e4 - lam_c)/lam_c
    keep = np.abs(v_all) < 9000.
    if keep.sum() < 40: raise RuntimeError(f'{name}: line not covered')
    # local masked continuum, deg 2, |v|<25000
    cw = np.abs(v_all) < 25000.
    cm = cw.copy()
    for lam0, hw in CONT_MASKS:
        lc = lam0*(1+z_sys)
        cm &= np.abs(C_KMS*(w*1e4-lc)/lc) > hw
    deg = 2 if cm.sum() > 40 else 1
    cp = np.polyfit(w[cm], f[cm], deg, w=1./np.maximum(e[cm], 1e-30))
    fsub = f - np.polyval(cp, w)
    v_pix = v_all[keep]; ff = fsub[keep]; ee = e[keep]
    # mask the [O III] doublet inside an Hb window (velocity offsets from the Hb center)
    if line == 'hb':
        for lam0 in (4960.295, 5008.24):
            voff = C_KMS*(lam0*(1+z_sys) - lam_c)/lam_c
            bad = np.abs(v_pix - voff) < 1200.
            v_pix, ff, ee = v_pix[~bad], ff[~bad], ee[~bad]
    dv = np.median(np.diff(v_pix))
    R = R_at(rspec, lam_c/1e4)
    sig_lsf = C_KMS/(lsf_fac*R)/2.3548
    step = max(abs(dv)/10., sig_lsf/4.)
    step = min(step, abs(dv))
    vg = np.arange(v_pix.min()-3*abs(dv), v_pix.max()+3*abs(dv), step)
    pk = np.nanmax(ff[np.abs(v_pix) < 1500.])
    if not np.isfinite(pk) or pk <= 0: raise RuntimeError(f'{name}: no line peak')
    ff, ee = ff/pk, ee/pk
    with_nii = (line == 'ha')
    def full(t): return np.concatenate([t, [sigN, 0.]])
    def res(t):
        mod, _ = M.model_on_pixels(full(t), vg, v_pix, sig_lsf, abs(dv), with_nii=with_nii)
        return (mod - ff)/ee
    lo = [0.05, -1000., 100.,  400., 0.0, -0.5, 0.0, 0.0, 0.05, -1000.,  40., 0.1, -8.]
    hi = [5.00,  1000., 2500., 3000., 6.0,  0.5, 3.0, 2.0, 8.0,  1000., 500., 0.3, 8.]
    if no_abs: lo[8], hi[8] = 1e-6, 1e-4
    starts = [(-60.,3.,130.,0.), (-200.,3.,130.,0.), (-400.,3.,130.,0.), (-100.,1.5,200.,2.),
              (-30.,5.,90.,-2.), (-600.,3.,130.,0.), (-20.,5.,260.,-1.), (-100.,2.,300.,0.),
              (-25.,6.,300.,-4.), (60.,3.,130.,0.), (150.,3.,200.,0.), (-150.,4.,180.,-1.)]
    rng = np.random.default_rng(seed)
    best = None
    for va0, t00, s0, sk0 in starts:
        p0 = np.clip(np.array([1., 50., 500., 1400., 1.0, 0.0, 0.3, 0.05, t00, va0, s0, 0.2, sk0])
                     *(1+0.05*rng.standard_normal(13)), lo, hi)
        try: r = least_squares(res, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    if best is None: raise RuntimeError(f'{name}: all fits failed')
    t = best.x
    npix = len(ff)
    chi2 = 2*best.cost/max(npix-13, 1)
    v95 = M.v95_of(full(t), vg, mode='full')
    tauf = M.tau_of(full(t), vg)
    v_trough = float(vg[int(np.argmax(tauf))])
    ew = float(np.sum(1.-np.exp(-np.clip(tauf, 0, 50)))*(vg[1]-vg[0])/C_KMS*LINES_CEN[line])
    # absorber significance: EW error via residual scatter proxy
    depth = float(1-np.exp(-t[8]))
    out = dict(name=name, line=line, z_sys=float(z_sys), oiii=oflag, sig_narrow=float(sigN),
               chi2r=float(chi2), v95=v95, v_abs=float(t[9]), v_trough=v_trough, sig_abs=float(t[10]),
               skew=float(t[12]), tau0=float(t[8]), ew_rest=ew, depth=depth,
               fw_exp=float(t[3]), R=float(R), cost=float(best.cost), npix=int(npix))
    if plotdir:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        mod, _ = M.model_on_pixels(full(t), vg, v_pix, sig_lsf, abs(dv), with_nii=with_nii)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.2, 1]))
        a1.step(v_pix, ff, where='mid', color='0.3', lw=0.8)
        a1.fill_between(v_pix, ff-ee, ff+ee, step='mid', color='0.3', alpha=0.15, lw=0)
        a1.plot(v_pix, mod, color='#CC3311', lw=1.3)
        if v95 is not None:
            a1.axvline(v95, color='#CC3311', ls='--', lw=1.1, label=r'$v_{95}=%.0f$' % v95)
        a1.axvline(v_trough, color='#0077BB', ls='-', lw=1.0, alpha=0.8, label=r'trough $=%.0f$' % v_trough)
        a1.axvline(t[9], color='0.4', ls=':', lw=1.0, label=r'centroid $\lambda_0=%.0f$' % t[9])
        a1.legend(fontsize=7.5, loc='upper right')
        a1.set_ylabel('normalized flux')
        a1.set_title(f"{name} {line}  z={z_sys:.4f} ({oflag})  chi2r={chi2:.2f}  v95={v95 and round(v95)}",
                     fontsize=10, loc='left')
        U, N = M.components_of(full(t), vg)
        T = (U*np.exp(-np.clip(tauf, 0, 50)) + N)/np.maximum(U+N, 1e-9)
        a2.plot(vg, T, color='#0077BB', lw=1.2); a2.axhline(0.95, color='k', ls=':', lw=0.9)
        if v95 is not None: a2.axvline(v95, color='#CC3311', ls='--', lw=1.1)
        a2.axvline(v_trough, color='#0077BB', ls='-', lw=1.0, alpha=0.8)
        a2.axvline(t[9], color='0.4', ls=':', lw=1.0)
        a2.set_xlim(-3000, 3000); a2.set_ylim(-0.05, 1.35)
        a2.set_xlabel('velocity [km/s]'); a2.set_ylabel('transmission')
        for a in (a1, a2): a.tick_params(direction='in', top=True, right=True)
        fig.tight_layout(); fig.savefig(os.path.join(plotdir, f'v95_{name}_{line}.png'), dpi=140)
        plt.close('all')
    return out
