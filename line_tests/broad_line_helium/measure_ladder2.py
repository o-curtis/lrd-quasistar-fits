#!/usr/bin/env python3
"""
Rigorous hardness-ladder line measurement (v2).

Fixes over experiments/ladder/measure_ladder.py:
  (i)   ratio lines measured ONLY in medium/high gratings (G140M/G235M/G395M/G395H).
        PRISM (R~30-300) cannot separate [OII]3727 from [NeIII]3869 (dlambda_rest=141A)
        nor isolate HeII4686 from [ArIV]4711/40 -> PRISM used for z-scan only, never ratios.
  (ii)  a ratio is a DETECTION only if BOTH lines exceed 3sigma; otherwise a limit or dropped.
  (iii) contaminant masking in every continuum sideband: [ArIV]4711.26/4740.12 by HeII;
        HeI3889/H8 by [NeIII]3869; [NeIII]3967+Heps3970 kept clear of Hgamma etc.
  (iv)  no crash on missing coverage / empty windows (all guarded); high-z safe.

v_esc is NOT computed here (needs the fit chains; done in the figure step from
Teff/logg/logL posteriors). This module returns z, narrow sigma, per-line fluxes+S/N,
and the two hardness ratios with explicit detection/limit flags.

Usage:  python measure_ladder2.py NAME file1.spec.fits [file2 ...]
        python measure_ladder2.py --batch spectra_root out.json
"""
import numpy as np, sys, os, glob, json
from astropy.io import fits
from scipy.optimize import curve_fit

C = 2.998e5          # km/s
CC = 2.998e10        # cm/s
MEDHIGH = ('G140M', 'G235M', 'G395M', 'G395H')   # never PRISM for ratios

# rest wavelengths (Angstrom, vacuum)
L_OII   = 3727.4     # [OII] 3726/3729 doublet (blended at these R)
L_NEIII = 3868.76
L_HB    = 4862.7
L_OIII  = 5006.8
L_HEII  = 4685.7
# contaminants to mask (rest A)
MASK_HEII   = [4711.26, 4740.12]              # [ArIV] doublet, red of HeII
MASK_NEIII  = [3889.05, 3888.6, 3967.47, 3970.07]  # HeI3889/H8, [NeIII]3967/Heps (blue/red guards)


BANDPASS = {'G140M': (0.70, 1.89), 'G235M': (1.66, 3.07),
            'G395M': (2.87, 5.30), 'G395H': (2.87, 5.30), 'PRISM': (0.55, 5.35)}

def load(files):
    """Return {GRATING: (wave_um, flux_uJy, err_uJy, wmin, wmax)} for finite pixels.
    Pixels outside the grating's PHYSICAL bandpass are dropped — DJA pads arrays
    with the full wavelength grid and out-of-band pixels are junk."""
    sp = {}
    for f in files:
        key = [k for k in ('g140m', 'g235m', 'g395m', 'g395h', 'prism') if k in f.lower()]
        if not key:
            continue
        g = key[0].upper()
        with fits.open(f) as h:
            d = h['SPEC1D'].data
            w = d['wave'].astype(float); fn = d['flux'].astype(float); en = d['err'].astype(float)
        bp = BANDPASS[g]
        ok = np.isfinite(w) & np.isfinite(fn) & np.isfinite(en) & (en > 0) \
             & (w >= bp[0]) & (w <= bp[1])
        if ok.sum() < 8:
            continue
        w, fn, en = w[ok], fn[ok], en[ok]
        srt = np.argsort(w)
        w, fn, en = w[srt], fn[srt], en[srt]
        sp[g] = (w, fn, en, w.min(), w.max())
    return sp


def covers(sp, g, lam_obs_um, margin=0.02):
    """Does grating g cover lam_obs with room for sidebands?"""
    if g not in sp:
        return False
    _, _, _, wmin, wmax = sp[g]
    return (wmin + margin) < lam_obs_um < (wmax - margin)


def snr_quick(w, fn, en, lam_obs_um, half_kms=250.):
    """Rough boxcar S/N for z-scan / grating pick."""
    if not (w.min() < lam_obs_um < w.max()):
        return 0.0
    v = C * (w - lam_obs_um) / lam_obs_um
    core = np.abs(v) < half_kms
    side = (np.abs(v) > 2.5 * half_kms) & (np.abs(v) < 6 * half_kms)
    if core.sum() < 2 or side.sum() < 4:
        return 0.0
    cont = np.median(fn[side])
    return np.sum(fn[core] - cont) / np.sqrt(np.sum(en[core] ** 2) + 1e-30)


def zscan(sp, zrange=(2.8, 9.2), step=0.0003):
    """Multi-line z from strongest available diagnostics (never single-pixel)."""
    lines = [6564.6, L_OIII, 4958.9, L_HB, L_NEIII, L_OII]
    best = (0.0, -1e30)
    for zt in np.arange(zrange[0], zrange[1], step):
        tot = 0.0
        for l0 in lines:
            lam = l0 * (1 + zt) / 1e4
            for g in ('G395M', 'G235M', 'G140M', 'G395H', 'PRISM'):
                if g in sp and sp[g][0].min() < lam < sp[g][0].max():
                    tot += max(0.0, snr_quick(*sp[g][:3], lam)); break
        if tot > best[1]:
            best = (zt, tot)
    return best[0]


def narrow_sigma(sp, z):
    """Narrow sigma (km/s) from a Gaussian fit to [OIII]5007 in its best med/high grating."""
    lam = L_OIII * (1 + z) / 1e4
    cand = [g for g in MEDHIGH if covers(sp, g, lam)]
    if not cand:
        return 150.0, z, None          # PRISM-only: fall back, flag grating None
    g = max(cand, key=lambda gg: snr_quick(*sp[gg][:3], lam))
    w, fn, en = sp[g][:3]
    sel = np.abs(w - lam) < 0.012
    if sel.sum() < 6:
        return 150.0, z, g

    def gauss(x, a, x0, s, c0, c1):
        return c0 + c1 * (x - lam) + a * np.exp(-0.5 * ((x - x0) / s) ** 2)

    try:
        p0 = [np.nanmax(fn[sel]) - np.nanmedian(fn[sel]), lam, 0.0015, np.nanmedian(fn[sel]), 0.0]
        pp, _ = curve_fit(gauss, w[sel], fn[sel], p0=p0, sigma=en[sel],
                          absolute_sigma=True, maxfev=30000)
        sig_um = abs(pp[2]); zc = pp[1] / (L_OIII / 1e4) - 1.0
        sig_kms = C * sig_um / lam
        if not (10 < sig_kms < 1200):     # sanity clamp
            sig_kms, zc = 150.0, z
        return sig_kms, zc, g
    except Exception:
        return 150.0, z, g


def line_flux(sp, l0, z, sig_kms, masks=(), require_medhigh=True, kcore=2.2, kside=(3.0, 7.0)):
    """
    Integrated line flux + 1sigma error in a med/high grating, with masked sidebands.
    Returns dict(F, E, snr, grating) or None if uncoverable.
    """
    lam = l0 * (1 + z) / 1e4
    grset = MEDHIGH if require_medhigh else MEDHIGH + ('PRISM',)
    cand = [g for g in grset if covers(sp, g, lam)]
    if not cand:
        return None
    g = max(cand, key=lambda gg: snr_quick(*sp[gg][:3], lam, half_kms=max(2 * sig_kms, 200)))
    w, fn, en = sp[g][:3]
    half = max(kcore * sig_kms, 160.0)                     # km/s core half-width
    v = C * (w - lam) / lam
    core = np.abs(v) < half
    side = (np.abs(v) > kside[0] * half) & (np.abs(v) < kside[1] * half)
    # mask contaminants (in observed um) from the sideband continuum estimate
    for mrest in masks:
        mlam = mrest * (1 + z) / 1e4
        side &= np.abs(C * (w - mlam) / mlam) > half        # drop pixels within +-half of a contaminant
    if core.sum() < 2 or side.sum() < 4:
        return dict(F=np.nan, E=np.nan, snr=0.0, grating=g)
    # linear continuum from masked sidebands
    p = np.polyfit(w[side], fn[side], 1)
    cont = np.polyval(p, w[core])
    conv = 1e-29 * CC / (w[core] * 1e-4) ** 2               # uJy -> erg/s/cm2/Hz * (c/lam^2)
    dl = np.gradient(w)[core] * 1e-4                        # um -> cm? (consistent units cancel in ratio)
    F = np.sum((fn[core] - cont) * conv * dl)
    E = np.sqrt(np.sum((en[core] * conv * dl) ** 2))
    return dict(F=F, E=E, snr=(F / E if E > 0 else 0.0), grating=g)


def measure(name, files, zrange=(2.8, 9.2), z_prior=None, zwin=0.04):
    """z_prior (Hviding/manifest) anchors the scan; a free scan without an anchor
    can latch onto a G395M noise peak at the wrong z (G2 rule)."""
    sp = load(files)
    if not sp:
        return dict(name=name, error='no_spectra')
    if z_prior is not None:
        z0 = zscan(sp, (z_prior - zwin, z_prior + zwin), step=0.0002)
    else:
        z0 = zscan(sp, zrange)
    sig, zc, g5 = narrow_sigma(sp, z0)
    # if the [OIII] refine ran off (PRISM-only / no line), keep the anchor
    if z_prior is not None and abs(zc - z_prior) > zwin:
        zc = z_prior
    out = dict(name=name, z=round(float(zc), 4), sig_kms=round(float(sig), 1),
               oiii_grating=g5, z_prior=z_prior,
               z_flag=(None if z_prior is None else round(float(abs(zc - z_prior)), 4)))

    F = {}
    F['OII']   = line_flux(sp, L_OII,   zc, sig, masks=[])
    F['NeIII'] = line_flux(sp, L_NEIII, zc, sig, masks=MASK_NEIII)
    F['Hb']    = line_flux(sp, L_HB,    zc, sig, masks=[])
    F['HeII']  = line_flux(sp, L_HEII,  zc, sig, masks=MASK_HEII)

    def snr(k):
        return F[k]['snr'] if F[k] else None
    def cov(k):
        return (F[k] is not None) and np.isfinite(F[k]['F'])

    # ---- [NeIII]/[OII] hardness ----
    ne = dict(value=None, kind='no_coverage')
    if cov('NeIII') and cov('OII'):
        s_ne, s_o2 = F['NeIII']['snr'], F['OII']['snr']
        r = F['NeIII']['F'] / F['OII']['F']
        if s_ne >= 3 and s_o2 >= 3 and r > 0:
            # crude ratio error via fractional quadrature
            fr = np.sqrt((1 / s_ne) ** 2 + (1 / s_o2) ** 2)
            ne = dict(value=float(r), err=float(r * fr), kind='detection',
                      snr_neiii=float(s_ne), snr_oii=float(s_o2))
        elif s_o2 >= 3 and F['NeIII']['E'] > 0:      # NeIII undetected -> upper limit
            ne = dict(value=float(3 * F['NeIII']['E'] / F['OII']['F']), kind='upper_limit',
                      snr_neiii=float(s_ne), snr_oii=float(s_o2))
        elif s_ne >= 3 and F['OII']['E'] > 0:        # OII undetected -> lower limit
            ne = dict(value=float(F['NeIII']['F'] / (3 * F['OII']['E'])), kind='lower_limit',
                      snr_neiii=float(s_ne), snr_oii=float(s_o2))
        else:
            ne = dict(value=None, kind='both_undetected',
                      snr_neiii=float(s_ne), snr_oii=float(s_o2))

    # ---- HeII/Hb ----
    he = dict(value=None, kind='no_coverage')
    if cov('Hb') and (F['HeII'] is not None):
        s_hb = F['Hb']['snr']; s_he = F['HeII']['snr']
        if s_hb >= 3:
            if s_he >= 3 and F['HeII']['F'] > 0:
                he = dict(value=float(F['HeII']['F'] / F['Hb']['F']), kind='detection',
                          snr_heii=float(s_he), snr_hb=float(s_hb))
            elif F['HeII']['E'] > 0:                 # 3sigma UL on HeII/Hb
                he = dict(value=float(3 * F['HeII']['E'] / F['Hb']['F']), kind='upper_limit',
                          snr_heii=float(s_he), snr_hb=float(s_hb))
        else:
            he = dict(value=None, kind='hb_undetected', snr_hb=float(s_hb))

    out['fluxes'] = {k: (None if F[k] is None else
                         {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                          for kk, vv in F[k].items()}) for k in F}
    out['NeIII_OII'] = ne
    out['HeII_Hb'] = he
    return out


def _print(o):
    print(f"\n=== {o['name']}  z={o.get('z')}  sigma={o.get('sig_kms')} km/s  ([OIII] in {o.get('oiii_grating')}) ===")
    if 'error' in o:
        print("  ERROR:", o['error']); return
    for k in ('OII', 'NeIII', 'Hb', 'HeII'):
        fk = o['fluxes'][k]
        if fk is None:
            print(f"  {k:<6} : no med/high coverage")
        else:
            print(f"  {k:<6} : F={fk['F']:.3e}  S/N={fk['snr']:.1f}  [{fk['grating']}]")
    ne, he = o['NeIII_OII'], o['HeII_Hb']
    print(f"  --> [NeIII]/[OII] : {ne['kind']:<14} value={ne['value']}")
    print(f"  --> HeII/Hb       : {he['kind']:<14} value={he['value']}")


if __name__ == '__main__':
    if sys.argv[1] == '--batch':
        root, outp = sys.argv[2], sys.argv[3]
        manifest = sys.argv[4] if len(sys.argv) > 4 else 'lrd36_manifest.json'
        zmap = {}
        try:
            m = json.load(open(manifest))
            rows = m if isinstance(m, list) else m.get('sources', [])
            for r in rows:
                if isinstance(r, dict) and r.get('name'):
                    zmap[r['name']] = r.get('z') or r.get('z_hviding') or r.get('z_dja')
        except Exception as e:
            print("WARN: no manifest z prior:", e)
        res = []
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            if not os.path.isdir(d):
                continue
            files = glob.glob(os.path.join(d, '*.spec.fits'))
            if not files:
                continue
            name = os.path.basename(d)
            try:
                o = measure(name, files, z_prior=zmap.get(name))
            except Exception as e:
                o = dict(name=name, error=repr(e))
            res.append(o); _print(o)
        json.dump(res, open(outp, 'w'), indent=1)
        print(f"\nwrote {outp}  ({len(res)} sources)")
    else:
        name = sys.argv[1]; files = sys.argv[2:]
        o = measure(name, files)
        _print(o)
        json.dump(o, open(f'ladder2_{name}.json', 'w'), indent=1)
