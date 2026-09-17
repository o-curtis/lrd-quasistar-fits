#!/usr/bin/env python3
"""
fit_rs_m3.py — Rosetta Stone GN-28074 (z=2.26), adopted Egg M3 port.

JADES NIRSpec medium gratings G140M/G235M/G395M, native pixels -> maggies (flux already in
erg/s/cm^2/Å). CaT 8400-8750 A (rest) weighted x15, NOT masked. F_SYS=0.10. Model/likelihood
otherwise identical to the Egg M3 (see m3port.py).

LOSVD: per user (2026-06-12) sigma_smooth is FIXED (not free) to an estimate from the 2nd CaT
line (Ca II 8542, the cleanest of the triplet). Fitting a Gaussian to the absorption dip in
G235M (rest) gives sigma_obs ~ 190-212 km/s across stable windows; deconvolving the grating
resolution (R~1000, sigma_inst ~127 km/s) in quadrature -> sigma_LOSVD ~ 155 km/s. (Derivation:
rosetta_stone/cat_losvd_estimate.png.) xi_mtb FIXED=2.

READY TO SUBMIT — do NOT auto-run.
    python fit_rs_m3.py --pool N
    python fit_rs_m3.py --dry-run
"""
import os, argparse
import numpy as np
from astropy.io import fits
import m3port as M

RS    = '/storage/group/jtw13/default/oCurtisWorkDir/lrdmesa/water-dot-tests/stack/rosetta_stone'
OUT   = os.path.dirname(os.path.abspath(__file__))
Z     = 2.26
LABEL = 'Rosetta Stone GN-28074'
STEM  = 'RS_m3b'                # Plan A + [M/H]=-1 fixed + CaT weight restored + Calzetti
DUST_LAW = 'calzetti'           # was SMC in m3a; matches old RS fiducial (Calzetti)

R_GRAT      = 1000.0
SIGMA_INST  = M.F._CKMS / (R_GRAT * 2.355)       # ~127 km/s, all three gratings
WEIGHTS     = [(8400.0, 8750.0, 15.0)]           # CaT x15 (restored; m3a had dropped it)

# model knobs (2026-06-13): sigma FREE (old free fit ~415 km/s), cap A_V<0.5,
# **[M/H] FIXED -1** (closes the cool-T interp artifact that corrupted m3a), T floor 2000.
MODEL_KWARGS = dict(sigma_prior=(50.0, 500.0), teff_lo=2000.0, mh_fix=1.0, av_cap=0.5)

# per-arm rest-frame fit windows (from fit_rs.py)
ARMS = [
    ('f070lp-g140m', 2200.0,  3900.0),
    ('f170lp-g235m', 5092.0,  9000.0),
    ('f290lp-g395m', 9000.0, 15500.0),
]

# per-arm emission-line masks (rest Å) — finalized in fit_rs.py; CaT NOT masked
LINES_140 = [(2798,200),(3426,80),(3727,80),(3808,4),(3869,60),(3934,50),(3968,50)]  # 3808: closes [O II]/[Ne III] gap (−3.5σ noise point)
LINES_235 = [(4861,100),(4959,50),(5007,50),(5876,40),(6300,40),(6548,30),(6563,150),
             (6583,30),(6678,30),(6716,30),(6731,30),(7065,30),(7136,30),(7325,45),
             (8446,25),(9069,40)]
LINES_395 = [(9015,100),(9123,16),(9229,100),(9532,40),(9546,100),(10049,120),(10313,25),
             (10830,250),(10938,200),(11287,150),(12820,250)]
LINES = {'f070lp-g140m': LINES_140, 'f170lp-g235m': LINES_235, 'f290lp-g395m': LINES_395}


def _load(tag):
    fn = os.path.join(RS, f'hlsp_jades_jwst_nirspec_goods-n-mediumhst-00028074_{tag}_v1.0_x1d.fits')
    with fits.open(fn) as h:
        d = h[1].data
        w_obs = d['WAVELENGTH'].astype(float) * 1e4    # um -> Å (observed); flux in erg/s/cm^2/Å
        flam  = d['FLUX'].astype(float)
        elam  = d['FLUX_ERR'].astype(float)
    return w_obs, flam, elam


def build_obs():
    obs_list = []
    for tag, lo, hi in ARMS:
        w_obs, flam, elam = _load(tag)
        w_rest = w_obs / (1.0 + Z)
        elam = np.sqrt(elam**2 + (M.F_SYS * np.abs(flam))**2)
        flux = M.flam_to_maggies(w_obs, flam)
        unc  = M.flam_to_maggies(w_obs, elam)
        good = np.isfinite(flux) & np.isfinite(unc) & (unc > 0) & (flam != 0)
        mask = (M.line_mask(w_rest, LINES[tag])
                & (w_rest >= lo) & (w_rest <= hi) & good)
        print(f"  {tag}: {int(mask.sum())} fit pixels", flush=True)
        obs_list.append(dict(name=tag, wave_obs=w_obs, wave_rest=w_rest,
                             flux=flux, unc=unc, mask=mask, sigma_inst=SIGMA_INST))
    return obs_list


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    M.fit(STEM, Z, MODEL_KWARGS, build_obs, WEIGHTS,
          out_dir=OUT, dust_law=DUST_LAW, pool_n=a.pool, dry_run=a.dry_run, source_label=LABEL)
