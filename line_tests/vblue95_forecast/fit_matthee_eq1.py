#!/usr/bin/env python3
"""Exact reimplementation of the Matthee+2026 Eq. (1) Balmer-line model, as used
by Naidu+2026 to define v_blue,95%.

  f(lam) = [ e^-tau_e * I + (1-e^-tau_e) * (I conv E) + C ] * e^-tau(lam) + N + NII
  tau(lam) = tau0 * Voigt(v - v_abs; sig, gamL)/Voigt(0) * (1 + erf[g0 (v-v_abs)/(sqrt2 sig)])
  N, [N II] (ratio 3.049): width+z tied to the [O III] doublet fit.
  Full model convolved with the LSF at 1.8 x R(lambda) (msaexp disp curves),
  built on a 10x oversampled grid and rebinned to the data pixels.
  v95: bluest v where e^-tau = 0.95, on the fitted absorber alone.

Usage: python fit_matthee_eq1.py NAME z_guess file.spec.fits [--plot DIR]
"""
import sys, os, json
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares, curve_fit
from scipy.special import voigt_profile, erf
from scipy.signal import fftconvolve
C_KMS = 2.99792458e5
L_HA, L_NII_B, L_NII_R = 6564.632, 6549.86, 6585.27      # vacuum
L_O3B, L_O3R = 4960.295, 5008.24
BANDPASS = {'G395M': (2.87, 5.30), 'G235M': (1.66, 3.17)}
DISP = '/home/omc5226/.local/lib/python3.12/site-packages/msaexp/data/jwst_nirspec_{}_disp.fits'

def load(fname, clamp=True):
    g = 'g395m' if 'g395m' in fname.lower() else 'g235m'
    with fits.open(fname) as h:
        d = h['SPEC1D'].data
        w, f, e = [np.asarray(d[k], float) for k in ('wave', 'flux', 'err')]
    lo, hi = BANDPASS[g.upper()]
    if not clamp: lo = lo - 0.03
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (w >= lo) & (w <= hi)
    with fits.open(DISP.format(g)) as h:
        dd = h[1].data
        Rw, Rr = np.asarray(dd['WAVELENGTH'], float), np.asarray(dd['R'], float)
    return w[ok], f[ok], e[ok], (Rw, Rr), g.upper()

def fit_oiii(w, f, e, z0):
    """[O III] doublet, ratio 2.98, common z+width + linear continuum -> z_sys, sig_narrow(km/s)."""
    lam0 = L_O3R*(1+z0)/1e4
    v = C_KMS*(w-lam0)/lam0
    m = np.abs(v) < 3500.
    if m.sum() < 15: raise RuntimeError('no OIII coverage')
    x, y, s = w[m]*1e4, f[m], e[m]
    def mod(x, z, sig, A, c0, c1):
        out = c0 + c1*(x/1e4 - lam0)
        for lam, amp in ((L_O3B, A/2.98), (L_O3R, A)):
            mu = lam*(1+z); sw = sig/C_KMS*mu
            out = out + amp*np.exp(-0.5*((x-mu)/sw)**2)
        return out
    p0 = [z0, 150., max(np.max(y)-np.median(y), 1e-3), np.median(y), 0.]
    p, _ = curve_fit(mod, x, y, p0=p0, sigma=s, maxfev=40000,
                     bounds=([z0-0.02, 30., 0., -np.inf, -np.inf], [z0+0.02, 600., np.inf, np.inf, np.inf]))
    return p[0], p[1]

def build_grids(w, lam_ha, Rcurve, osamp=10, vmax=9000.):
    """velocity grids: data pixels + 10x oversampled uniform grid (fit window only)."""
    v_all = C_KMS*(w*1e4 - lam_ha)/lam_ha
    keep = np.abs(v_all) < vmax
    v_pix = v_all[keep]
    dv = np.median(np.diff(v_pix))
    vg = np.arange(v_pix.min()-3*dv, v_pix.max()+3*dv, dv/osamp)
    R = np.interp(lam_ha/1e4, Rcurve[0], Rcurve[1])
    sig_lsf = C_KMS/(1.8*R)/2.3548
    return keep, v_pix, vg, sig_lsf, dv

def kernel(step, half_width, prof):
    """short symmetric kernel sampled at the grid step, truncated at half_width."""
    nk = max(int(np.ceil(half_width/step)), 2)
    vc = np.arange(-nk, nk+1)*step
    k = prof(vc); s = k.sum()
    return k/s if s > 0 else k

def tau_of(theta, vg):
    (_,_,_,_,_,_,_,_, tau0, v_a, sig_a, gam_a, g0, _,_) = theta
    V = voigt_profile(vg-v_a, max(sig_a,15.), max(gam_a,0.1))
    V = V/voigt_profile(0., max(sig_a,15.), max(gam_a,0.1))
    return tau0*V*(1 + erf(g0*(vg-v_a)/(np.sqrt(2)*max(sig_a,15.))))

def model_on_pixels(theta, vg, v_pix, sig_lsf, dv, with_nii=True):
    (A_I, v_I, fw_I, fw_E, tau_e, Cc, A_N, A_N2, tau0, v_a, sig_a, gam_a, g0, sigN, vN) = theta
    sig_I = fw_I/2.3548
    step = vg[1]-vg[0]
    I = A_I*np.exp(-0.5*((vg-v_I)/sig_I)**2)
    wE = max(fw_E, 60.)/(2*np.log(2))
    E = kernel(step, min(8.*wE, (len(vg)-3)*step/2.), lambda x: np.exp(-np.abs(x)/wE))
    IE = fftconvolve(I, E, mode='same')
    scat = np.exp(-tau_e)
    cont_plus_broad = scat*I + (1-scat)*IE + Cc
    tau = tau_of(theta, vg)
    absorbed = cont_plus_broad*np.exp(-np.clip(tau, 0, 50))
    # narrow + [N II], tied width/z
    N = A_N*np.exp(-0.5*((vg-vN)/sigN)**2)
    if with_nii:
        for lam, amp in ((L_NII_R, A_N2), (L_NII_B, A_N2/3.049)):
            voff = C_KMS*(lam-L_HA)/L_HA + vN
            N = N + amp*np.exp(-0.5*((vg-voff)/sigN)**2)
    tot = absorbed + N
    L = kernel(step, 5.*sig_lsf, lambda x: np.exp(-0.5*(x/sig_lsf)**2))
    tot = fftconvolve(tot, L, mode='same')
    osamp = int(round(dv/step))
    box = np.ones(osamp)/osamp
    tot = fftconvolve(tot, box, mode='same')
    return np.interp(v_pix, vg, tot), tau

def components_of(theta, vg):
    (A_I, v_I, fw_I, fw_E, tau_e, Cc, A_N, A_N2, tau0, v_a, sig_a, gam_a, g0, sigN, vN) = theta
    sig_I = fw_I/2.3548
    step = vg[1]-vg[0]
    I = A_I*np.exp(-0.5*((vg-v_I)/sig_I)**2)
    wE = max(fw_E, 60.)/(2*np.log(2))
    E = kernel(step, min(8.*wE, (len(vg)-3)*step/2.), lambda x: np.exp(-np.abs(x)/wE))
    IE = fftconvolve(I, E, mode='same')
    scat = np.exp(-tau_e)
    U = scat*I + (1-scat)*IE + Cc
    N = A_N*np.exp(-0.5*((vg-vN)/sigN)**2)
    for lam, amp in ((L_NII_R, A_N2), (L_NII_B, A_N2/3.049)):
        voff = C_KMS*(lam-L_HA)/L_HA + vN
        N = N + amp*np.exp(-0.5*((vg-voff)/sigN)**2)
    return U, N

def v95_of(theta, vg, mode='full'):
    tau = tau_of(theta, vg)
    if mode == 'tau':
        T = np.exp(-tau)
    else:
        U, N = components_of(theta, vg)
        T = (U*np.exp(-np.clip(tau,0,50)) + N)/np.maximum(U + N, 1e-9)
    v_a = theta[9]
    below = T < 0.95
    if not below.any(): return None
    ia = int(np.argmin(np.abs(vg - v_a)))
    if not below[ia]:
        cand = np.where(below)[0]
        ia = cand[np.argmin(np.abs(vg[cand] - v_a))]
    i0 = ia
    while i0 > 0 and below[i0-1]: i0 -= 1
    if i0 == 0: return float(vg[0])
    return float(np.interp(0.95, [T[i0], T[i0-1]], [vg[i0], vg[i0-1]]))

def fit_source(name, z0, fname, plotdir=None, seed=0, tau0_max=8.0, oiii_fname=None, gauss_abs=False, pin_va=None, mask_feii=False):
    w, f, e, Rcurve, g = load(fname)
    oiii_flag = 'fit'
    try:
        if oiii_fname:
            wo, fo, eo, _, _ = load(oiii_fname, clamp=False)
            z_sys, sigN_oiii = fit_oiii(wo, fo, eo, z0)
        else:
            try:
                z_sys, sigN_oiii = fit_oiii(w, f, e, z0)
            except Exception:
                wo, fo, eo, _, _ = load(fname, clamp=False)
                z_sys, sigN_oiii = fit_oiii(wo, fo, eo, z0)
                oiii_flag = 'edge-unclamped'
    except Exception as ex:
        z_sys, sigN_oiii, oiii_flag = z0, 150.0, f'fallback({ex})' 
    lam_ha = L_HA*(1+z_sys)
    keep, v_pix, vg, sig_lsf, dv = build_grids(w, lam_ha, Rcurve)
    # global polynomial continuum over the clamped range, emission lines masked
    LINES = [(6564.632, 6000.), (4862.7, 4000.), (5008.24, 2500.), (4960.295, 2500.),
             (6725.0, 2500.), (5877.25, 2500.), (6302.0, 2000.), (7067.1, 2000.),
             (7325.0, 2000.), (9071.1, 2000.), (9533.2, 2000.), (10833.2, 3000.),
             (10052.1, 2000.), (10941.1, 2000.), (12821.6, 2000.), (18756.1, 2500.)]
    cm = np.ones(len(w), bool)
    for lam0, hw in LINES:
        lc = lam0*(1+z_sys)/1e4
        cm &= np.abs(C_KMS*(w-lc)/lc) > hw
    deg = 2 if cm.sum() > 40 else 1
    cp = np.polyfit(w[cm], f[cm], deg, w=1./np.maximum(e[cm], 1e-9))
    fsub = f - np.polyval(cp, w)
    vv, ff, ee = v_pix, fsub[keep], e[keep]
    if mask_feii:
        # Matthee Fig 5: regions possibly affected by [Fe II] excluded from the fits
        km = np.ones(len(vv), bool)
        for lam_fe in (6456.4, 6516.1):
            vfe = C_KMS*(lam_fe-L_HA)/L_HA
            km &= np.abs(vv - vfe) > 600.
        vv, ff, ee = vv[km], ff[km], ee[km]
    pk = np.nanmax(ff[np.abs(vv) < 1500.]); ff, ee = ff/pk, ee/pk
    sigN = sigN_oiii/1.0  # narrow width tied to [O III]
    # theta layout: A_I, v_I, fw_I, fw_E, tau_e, C, A_N, A_N2, tau0, v_a, sig_a, gam_a, g0  (+ tied sigN, vN=0)
    def full_theta(t): return np.concatenate([t, [sigN, 0.]])
    def res(t):
        mod, _ = model_on_pixels(full_theta(t), vg, vv, sig_lsf, dv)
        return (mod - ff)/ee
    lo = [0.05, -1000., 100.,  400., 0.0, -0.5, 0.0, 0.0, 0.05, -1000.,  40., 0.1, -8.]
    hi = [5.00,  1000., 2500., 3000., 6.0,  0.5, 3.0, 2.0, tau0_max,  1000., 500., 300., 8.]
    if gauss_abs: lo[11], hi[11] = 0.1, 0.3
    if pin_va is not None: lo[9], hi[9] = pin_va-60., pin_va+60.
    rng = np.random.default_rng(seed)
    sols = []
    starts = [(-60.,3.,130.,1.,0.), (-200.,3.,130.,1.,0.), (-400.,3.,130.,1.,0.), (-100.,1.5,200.,1.,2.),
              (-30.,5.,90.,1.,-2.), (-600.,3.,130.,1.,0.), (-150.,6.,60.,30.,-4.), (-300.,2.,250.,1.,1.),
              (-20.,5.,260.,1.,-1.), (-100.,2.,300.,1.,0.),
              (-25.,6.,300.,1.,-4.), (-50.,7.,400.,1.,-6.), (-30.,4.,200.,1.,-3.)]
    if pin_va is not None:
        starts = [(pin_va, t0, sg, 1., sk) for t0 in (2., 5., 7.)
                  for sg in (120., 260., 400.) for sk in (-4., -1., 1.)]
    for va0, t00, s0, g0v, sk0 in starts:
        p0 = np.array([1., 50., 500., 1400., 1.0, 0.0, 0.3, 0.05, t00, va0, s0, g0v, sk0])
        p0 = np.clip(p0*(1+0.05*rng.standard_normal(13)), lo, hi)
        try:
            r = least_squares(res, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        sols.append(r)
    sols.sort(key=lambda r: r.cost)
    best = sols[0]
    top = []
    seenv = []
    for r in sols:
        vv95 = v95_of(full_theta(r.x), vg, mode='full')
        key = None if vv95 is None else round(vv95/40)
        if key in seenv: continue
        seenv.append(key)
        top.append(dict(cost=float(r.cost), v95=vv95, v_abs=float(r.x[9]), tau0=float(r.x[8]),
                        sig=float(r.x[10]), gamL=float(r.x[11]), skew=float(r.x[12])))
        if len(top) >= 3: break
    t = best.x; chi2 = 2*best.cost/ (len(ff)-13)
    v95 = v95_of(full_theta(t), vg, mode='full')
    v95t = v95_of(full_theta(t), vg, mode='tau')
    tauf = tau_of(full_theta(t), vg)
    dlam_rest = (vg[1]-vg[0])/C_KMS*L_HA
    ew_rest = float(np.sum(1. - np.exp(-np.clip(tauf, 0, 50)))*dlam_rest)
    out = dict(name=name, z_sys=float(z_sys), sig_narrow=float(sigN), chi2r=float(chi2),
               v95=v95, v95_tau=v95t, v_abs=float(t[9]), sig_abs=float(t[10]), gam_abs=float(t[11]),
               skew=float(t[12]), tau0=float(t[8]), fw_exp=float(t[3]), fw_int=float(t[2]),
               tau_e=float(t[4]), grating=g, oiii=oiii_flag, ew_rest=ew_rest, top_minima=top)
    if plotdir:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        mod, tau = model_on_pixels(full_theta(t), vg, vv, sig_lsf, dv)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.2, 1]))
        a1.step(vv, ff, where='mid', color='0.3', lw=0.8)
        a1.fill_between(vv, ff-ee, ff+ee, step='mid', color='0.3', alpha=0.15, lw=0)
        a1.plot(vv, mod, color='#CC3311', lw=1.4)
        a1.set_ylabel('normalized flux')
        a1.set_title(f"{name} Halpha  z={z_sys:.4f}  chi2r={chi2:.2f}  v95={v95 and round(v95)} km/s", fontsize=10, loc='left')
        T = np.exp(-tau)
        a2.plot(vg, T, color='#0077BB', lw=1.3); a2.axhline(0.95, color='k', ls=':', lw=0.9)
        if v95 is not None: a2.axvline(v95, color='#CC3311', ls='--', lw=1.1)
        a2.set_xlim(-4000, 4000); a2.set_xlabel('velocity [km/s]'); a2.set_ylabel('e^-tau')
        for a in (a1, a2): a.tick_params(direction='in', top=True, right=True)
        fig.tight_layout(); fig.savefig(os.path.join(plotdir, f'meq1_{name}.png'), dpi=140)
        plt.close('all')
    return out

if __name__ == '__main__':
    name, z0, fname = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    plotdir = sys.argv[sys.argv.index('--plot')+1] if '--plot' in sys.argv else None
    t0m = float(sys.argv[sys.argv.index('--tau0max')+1]) if '--tau0max' in sys.argv else 8.0
    oiii = sys.argv[sys.argv.index('--oiii')+1] if '--oiii' in sys.argv else None
    o = fit_source(name, z0, fname, plotdir=plotdir, tau0_max=t0m, oiii_fname=oiii, gauss_abs=('--gauss' in sys.argv))
    print(json.dumps(o, indent=1))
