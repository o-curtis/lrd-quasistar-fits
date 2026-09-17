#!/usr/bin/env python3
"""
fit_egg_prospector_tlusty.py — Two-component Prospector fit of The Egg.

Source: J1025+1402 (The Egg), z = 0.1007
Data  : MODS Red (3900–7400 Å rest) + FIRE (7400–15500 Å rest)

=============================================================================
MODEL
=============================================================================

Two additive components, each contributing to the same observed spectrum:

  1. HOST GALAXY  — FastStepBasis (FSPS/C3K) with continuity non-parametric SFH.
                    Chabrier IMF.  Noll+09 two-component dust (dust_type=4):
                      dust1      — birth-cloud τ_V  (young stars only)
                      dust2_gal  — diffuse    τ_V  (all stars)
                      dust_index — slope modifier of diffuse attenuation curve
                    No nebular emission (lines are masked; continuum secondary).
                    6 SFH age bins spanning 30 Myr → age of universe at z=0.1007.

  2. LRD PHOTOSPHERE — TLUSTYBasis (Liu+2026 TLUSTY grid, C3K-consistent).
                    SMC extinction (Gordon+2003) applied inside TLUSTYBasis.
                    Stefan-Boltzmann normalization via logL_star.
                    sigma_smooth broadens photosphere features only (not galaxy).

Dust conventions
----------------
  • Galaxy    : Noll+09 / Calzetti (dust_type=4).  dust1 for birth cloud,
                dust2_gal → FSPS dust2 for diffuse, dust_index free.
  • Photosphere: SMC Gordon+2003 (av_smc), applied inside TLUSTYBasis.

Smoothing convention
--------------------
  sigma_smooth (LOSVD of the LRD photosphere) is applied only to the TLUSTY
  spectrum before summing.  The galaxy continuum is broadened by instrumental
  resolution only — applying a km/s LOSVD to old stellar populations is not
  physically meaningful here.

=============================================================================
FREE PARAMETERS (16)
=============================================================================
  Photosphere (6):
    teff         : T_eff [K]              Uniform [3500, 6750]  ← TLUSTY grid extent
    logg         : log g                 Uniform [-3.5, 0.5]   ← TLUSTY grid extent
    mh_idx       : metallicity index     Uniform [0, 2]   (0=[M/H]=-2, 2=solar)
    logL_star    : log10(L*/Lsun)        Uniform [5, 14]
    av_smc       : A_V SMC (photosphere) Uniform [0, 8] mag
    sigma_smooth : LOSVD sigma (LRD only) Uniform [0, 600] km/s

  Host galaxy (10):
    logmass      : log10(M_gal/Msun)     Uniform [7, 11.5]
    logzsol      : log(Z/Zsun)           Uniform [-2.0, 0.19]
    dust2_gal    : diffuse τ_V (Noll+09) Uniform [0, 4]
    dust1        : birth-cloud τ_V       Uniform [0, 4]
    dust_index   : attenuation slope n   Uniform [-2.2, 0.4]
    logsfr_ratio_{0..4} : SFH shape      StudentT(0, 0.3, df=2) each

=============================================================================
OUTPUTS (all written to egg_analysis/)
=============================================================================
  egg_dynesty_prospector_tlusty.pkl
  egg_chain_prospector_tlusty.npy
  egg_corner_prospector_tlusty.png
  egg_spectra_prospector_tlusty.png
  egg_bic_prospector_tlusty.txt
"""

import os
import sys
import importlib.util
import time
import pickle
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo

warnings.filterwarnings('ignore')

# SPS_HOME must be set before importing fsps or prospect
_SPS_HOME = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
os.environ['SPS_HOME'] = _SPS_HOME

import dynesty
from dynesty.utils import resample_equal

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, 'egg_data',
                             'Re_ J1025+1402 LBT _ Magellan spectra')
SPEC_DIR     = ('/home/omc5226/work/lrdmesa/liu2026_library/'
                'LRD_synthetic_spectral_library-main/specs')
TLUSTY_CACHE = os.path.join(SCRIPT_DIR, 'egg_tlusty_r10k.npz')

_TLUSTY_MODULE = (
    '/home/omc5226/prospector_tlusty_dev/prospector/'
    'prospect/sources/tlusty_basis.py'
)
_PROSPECT_DIR = '/home/omc5226/prospector_tlusty_dev/prospector'

def _import_tlusty():
    """Load TLUSTYBasis directly to avoid SPS_HOME check in agnssp_basis."""
    spec = importlib.util.spec_from_file_location('tlusty_basis', _TLUSTY_MODULE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TLUSTYBasis

# ── Source properties ──────────────────────────────────────────────────────────
Z_EGG    = 0.1007
F_SYS    = 0.05             # systematic flux error floor (fraction)
MODSR_LO = 3900.0           # rest-frame AA
MODSR_HI = 7400.0
FIRE_LO  = 7400.0
FIRE_HI  = 15500.0

# Instrument / library resolving powers
R_MODS   = 9100
R_FIRE   = 7000
TARGET_R = 10000
_CKMS    = 2.998e5
SIGMA_LIB  = _CKMS / (TARGET_R * 2.355)   # ≈ 12.7 km/s
SIGMA_MODS = _CKMS / (R_MODS  * 2.355)    # ≈ 14.0 km/s
SIGMA_FIRE = _CKMS / (R_FIRE  * 2.355)    # ≈ 18.2 km/s

# CGS / unit conversion
_lsun       = 3.846e33
_pc_cm      = 3.0857e18
_jansky_cgs = 1e-23
_c_AA       = 2.998e18
_to_cgs     = _lsun / (4.0 * np.pi * (10.0 * _pc_cm)**2)
_PYPEIT_SCALE = 1e-17   # erg/s/cm²/Å per PyPeIt flux unit

# Emission line masks (rest-frame AA, half-width AA)
MODS_LINES = [
    (6563, 110), (4861, 100), (4340, 50), (4102, 30),
    (5007, 30),  (4959, 20),  (6583, 25), (6548, 20),
    (6716, 20),  (6731, 20),  (3727, 30), (3869, 20),
    (5876, 25),  (4686, 25),
]
FIRE_LINES = [
    (10830, 250), (10938, 200), (12820, 250), (9015, 150),
    (9229, 150),  (9546, 150),  (10049, 200), (9069, 40),
    (9532, 40),   (6563, 110),
]

# ── Continuity SFH age bins ────────────────────────────────────────────────────
# 6 bins spanning 10 Myr to the age of the universe at z=0.1007.
# Edges in log10(yr); bin N-1 ends at log10(t_H).
N_BINS_SFH = 6
N_RATIOS   = N_BINS_SFH - 1   # = 5 free logsfr_ratios

def _make_agebins(zred):
    t_H_yr = cosmo.age(zred).to('yr').value
    return np.array([
        [0.0,   7.47],
        [7.47,  8.0 ],
        [8.0,   8.5 ],
        [8.5,   9.0 ],
        [9.0,   9.5 ],
        [9.5,   np.log10(t_H_yr)],
    ])

AGEBINS = _make_agebins(Z_EGG)

# ── Dynesty settings ───────────────────────────────────────────────────────────
# 16 params: 6 photosphere + 5 galaxy scalar + 5 SFH ratios
NDIM        = 6 + 5 + N_RATIOS
NLIVE       = 600
NLIVE_INIT  = 400
NLIVE_BATCH = 600
DLOGZ_INIT  = 0.01
N_EFFECTIVE = 8000

LABELS = (
    ['teff', 'logg', 'mh_idx', 'logL_star', 'av_smc', 'sigma_smooth',
     'logmass', 'logzsol', 'dust2_gal', 'dust1', 'dust_index']
    + [f'logsfr_ratio_{i}' for i in range(N_RATIOS)]
)


# =============================================================================
# TLUSTYPlusGalaxyBasis — composite SPS (photosphere + host galaxy)
# =============================================================================

class TLUSTYPlusGalaxyBasis:
    """
    Composite SPS for the two-component Egg model.

    Internally holds:
      self.tlusty  : TLUSTYBasis  — LRD photosphere
      self.galaxy  : FastStepBasis — host galaxy with continuity SFH

    get_galaxy_spectrum() sums both components on the TLUSTY wavelength grid.

    Dust conventions
    ----------------
    Photosphere : SMC (Gordon+2003) controlled by params['av_smc'].
                  Applied inside TLUSTYBasis.get_galaxy_spectrum() by mapping
                  av_smc → dust2 before calling that method.
    Galaxy      : Charlot & Fall / Calzetti (FSPS dust_type=0) controlled by
                  params['dust2_gal'], passed as dust2 to FSPS via update().

    Normalisation
    -------------
    Photosphere : TLUSTYBasis uses Stefan-Boltzmann to set the absolute
                  luminosity from logL_star.  Output is in Lsun/Hz.
    Galaxy      : FastStepBasis.get_galaxy_spectrum() returns Lsun/Hz per
                  solar mass formed.  We multiply by sum(params['mass']) to
                  get total Lsun/Hz.  params['mass'] is the per-bin mass array
                  derived from logmass + logsfr_ratios (computed externally in
                  _logl before calling here).
    """

    def __init__(self, spec_dir, target_R, cache_file, verbose):
        # Load TLUSTY photosphere library
        TLUSTYBasis = _import_tlusty()
        self.tlusty = TLUSTYBasis(
            spec_dir=spec_dir,
            target_R=target_R,
            wave_lo=2000.0,
            wave_hi=25000.0,
            verbose=verbose,
            cache_file=cache_file,
        )
        self.wave = self.tlusty.wave   # master grid

        # Load FSPS FastStepBasis (galaxy)
        sys.path.insert(0, _PROSPECT_DIR)
        from prospect.sources import FastStepBasis
        self.galaxy = FastStepBasis(
            zcontinuous=1,   # interpolate SSPs in log Z
            compute_vega_mags=False,
        )
        if verbose:
            print('  FastStepBasis (FSPS galaxy) initialised.', flush=True)

        # Spectral resolution attribute used by SpecModel (library sigma, km/s)
        self.spectral_resolution = self.tlusty.spectral_resolution

    @property
    def wavelengths(self):
        return self.wave

    def get_spectra_components(self, **params):
        """
        Return photosphere and galaxy spectra separately on self.wave (Lsun/Hz).

        The caller is responsible for smoothing each component independently
        before summing — sigma_smooth applies to the photosphere only.

        Returns
        -------
        wave     : ndarray (N_pix,) AA — common wavelength grid
        spec_tl  : ndarray (N_pix,) Lsun/Hz — photosphere, or NaN array if OOB
        spec_gal : ndarray (N_pix,) Lsun/Hz — galaxy
        mfrac    : float — surviving/formed mass fraction from FSPS
        """
        # ── photosphere ──────────────────────────────────────────────────────
        tlusty_params = dict(params)
        tlusty_params['dust2'] = float(params.get('av_smc', 0.0))
        _, spec_tl, _ = self.tlusty.get_galaxy_spectrum(**tlusty_params)

        # ── galaxy ───────────────────────────────────────────────────────────
        galaxy_params = dict(params)
        galaxy_params['dust2'] = float(params.get('dust2_gal', 0.0))
        wave_gal, spec_per_mass, mfrac = self.galaxy.get_galaxy_spectrum(
            **galaxy_params)
        mtot = np.atleast_1d(params.get('mass', np.array([1.0]))).sum()
        spec_gal_interp = np.interp(
            self.wave, wave_gal, spec_per_mass * mtot, left=0.0, right=0.0)

        return self.wave, spec_tl, spec_gal_interp, mfrac

    def get_galaxy_spectrum(self, **params):
        """Summed spectrum — kept for API compatibility."""
        wave, spec_tl, spec_gal, mfrac = self.get_spectra_components(**params)
        if np.any(np.isnan(spec_tl)):
            return wave, np.full(len(wave), np.nan), 1.0
        return wave, spec_tl + spec_gal, mfrac

    def get_galaxy_elines(self):
        return np.array([]), np.array([])


# =============================================================================
# build_sps
# =============================================================================

def build_sps(spec_dir=SPEC_DIR, target_R=TARGET_R,
              cache_file=TLUSTY_CACHE, verbose=True, **kwargs):
    """Build TLUSTYPlusGalaxyBasis (photosphere + FSPS galaxy)."""
    return TLUSTYPlusGalaxyBasis(
        spec_dir=spec_dir,
        target_R=target_R,
        cache_file=cache_file,
        verbose=verbose,
    )


# =============================================================================
# build_obs
# =============================================================================

def build_obs(data_dir=DATA_DIR, **kwargs):
    """
    Load and pre-process The Egg spectra (MODS Red + FIRE).

    Returns
    -------
    obs_mods, obs_fire : dict
        Keys: wave_obs, wave_rest, flux [maggies], unc [maggies],
              mask [bool], name [str]
    """
    def load_pypeit(fname):
        with fits.open(os.path.join(data_dir, fname)) as h:
            s    = h['SPECTRUM'].data
            wobs = s['wave'].astype(float)
            flux = s['flux'].astype(float)
            ivar = s['ivar'].astype(float)
            msk  = s['mask'].astype(int)
        good  = (msk == 1) & (ivar > 0) & np.isfinite(flux)
        sigma = np.where(good, 1.0 / np.sqrt(np.where(ivar > 0, ivar, np.inf)),
                         np.nan)
        return wobs, flux, sigma, good

    def flam_to_maggies(w_obs_aa, f_lam_pypeit):
        f_lam = f_lam_pypeit * _PYPEIT_SCALE
        f_nu  = f_lam * w_obs_aa**2 / _c_AA
        return f_nu / (3631.0 * _jansky_cgs)

    def make_line_mask(wave_rest, lines):
        m = np.ones(len(wave_rest), dtype=bool)
        for cen, hw in lines:
            m &= ~((wave_rest >= cen - hw) & (wave_rest <= cen + hw))
        return m

    wm_obs, fm, sm, gm = load_pypeit(
        'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits')
    wf_obs, ff, sf, gf = load_pypeit(
        'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits')

    wm_rest = wm_obs / (1.0 + Z_EGG)
    wf_rest = wf_obs / (1.0 + Z_EGG)

    sm = np.sqrt(sm**2 + (F_SYS * np.abs(fm))**2)
    sf = np.sqrt(sf**2 + (F_SYS * np.abs(ff))**2)

    fm_mag = flam_to_maggies(wm_obs, fm)
    sm_mag = flam_to_maggies(wm_obs, sm)
    ff_mag = flam_to_maggies(wf_obs, ff)
    sf_mag = flam_to_maggies(wf_obs, sf)

    range_m = (wm_rest >= MODSR_LO) & (wm_rest <= MODSR_HI)
    range_f = (wf_rest >= FIRE_LO)  & (wf_rest <= FIRE_HI)
    lines_m = make_line_mask(wm_rest, MODS_LINES)
    lines_f = make_line_mask(wf_rest, FIRE_LINES)
    good_m  = gm & np.isfinite(fm_mag) & np.isfinite(sm_mag) & (sm_mag > 0)
    good_f  = gf & np.isfinite(ff_mag) & np.isfinite(sf_mag) & (sf_mag > 0)

    obs_mods = dict(wave_obs=wm_obs, wave_rest=wm_rest,
                    flux=fm_mag, unc=sm_mag,
                    mask=range_m & lines_m & good_m, name='MODS_R')
    obs_fire = dict(wave_obs=wf_obs, wave_rest=wf_rest,
                    flux=ff_mag, unc=sf_mag,
                    mask=range_f & lines_f & good_f, name='FIRE')

    print(f"  MODS R fit pixels: {obs_mods['mask'].sum()} | "
          f"FIRE fit pixels: {obs_fire['mask'].sum()}", flush=True)
    return obs_mods, obs_fire


# =============================================================================
# build_model
# =============================================================================

def build_model(mesa_teff=5293., mesa_teff_sigma=800.,
                mesa_logg=-2.255, mesa_logg_sigma=0.5,
                logL_init=10.5, **kwargs):
    """
    Prospector-style parameter dict for the two-component model.

    Returns a dict of dicts (N, isfree, init, prior, units) mirroring the
    Prospector TemplateLibrary convention.  This is for documentation and
    initialisation; the actual dynesty prior is in _prior_transform().
    """
    lumdist = cosmo.luminosity_distance(Z_EGG).to('Mpc').value

    model_params = {
        # ── Photosphere (TLUSTYBasis) ─────────────────────────────────────
        "teff": {
            "N": 1, "isfree": True, "init": mesa_teff,
            "prior": "Uniform(3500, 6750)", "units": "K",
        },
        "logg": {
            "N": 1, "isfree": True, "init": mesa_logg,
            "prior": "Uniform(-3.5, 0.5)", "units": "log10(cm/s2)",
        },
        "mh_idx": {
            "N": 1, "isfree": True, "init": 1.5,
            "prior": "Uniform(0, 2)",
            "units": "index: 0=[M/H]=-2, 1=-1, 2=0",
        },
        "logL_star": {
            "N": 1, "isfree": True, "init": logL_init,
            "prior": "Uniform(5, 14)", "units": "log10(L_star/L_sun)",
        },
        "av_smc": {
            "N": 1, "isfree": True, "init": 2.0,
            "prior": "Uniform(0, 8)", "units": "A_V (SMC, Gordon+2003)",
        },
        "sigma_smooth": {
            "N": 1, "isfree": True, "init": 100.0,
            "prior": "Uniform(0, 600)", "units": "km/s (LOSVD sigma)",
        },

        # ── Host galaxy (FastStepBasis + continuity SFH) ──────────────────
        "logmass": {
            "N": 1, "isfree": True, "init": 9.0,
            "prior": "Uniform(7, 11.5)", "units": "log10(M*/Msun)",
        },
        "logzsol": {
            "N": 1, "isfree": True, "init": -0.5,
            "prior": "Uniform(-2.0, 0.19)", "units": "log(Z/Zsun)",
        },
        "dust2_gal": {
            "N": 1, "isfree": True, "init": 0.3,
            "prior": "Uniform(0, 4)", "units": "Noll+09 diffuse tau_V",
        },
        "dust1": {
            "N": 1, "isfree": True, "init": 0.0,
            "prior": "Uniform(0, 4)", "units": "birth-cloud tau_V (young stars)",
        },
        "dust_index": {
            "N": 1, "isfree": True, "init": 0.0,
            "prior": "Uniform(-2.2, 0.4)",
            "units": "Noll+09 slope modifier (0=Calzetti)",
        },
        "logsfr_ratios": {
            "N": N_RATIOS, "isfree": True,
            "init": [0.0] * N_RATIOS,
            "prior": "StudentT(mean=0, scale=0.3, df=2) each",
            "units": "log10(SFR_j / SFR_{j+1}), j=0 most recent",
        },

        # ── Fixed bookkeeping ─────────────────────────────────────────────
        "sfh":              {"N": 1, "isfree": False, "init": 3},
        "imf_type":         {"N": 1, "isfree": False, "init": 2},   # Chabrier
        "dust_type":        {"N": 1, "isfree": False, "init": 4},   # Noll+09
        "dust1_index":      {"N": 1, "isfree": False, "init": -1.0},# birth-cloud slope
        "add_neb_emission": {"N": 1, "isfree": False, "init": False},
        "add_neb_continuum":{"N": 1, "isfree": False, "init": False},
        "add_dust_emission":{"N": 1, "isfree": False, "init": False},
        "agebins":          {"N": N_BINS_SFH, "isfree": False,
                             "init": AGEBINS.tolist()},
        "mass":             {"N": 1, "isfree": False, "init": 1e9,
                             "units": "derived from logmass+logsfr_ratios"},
        "zred":             {"N": 1, "isfree": False, "init": float(Z_EGG)},
        "lumdist":          {"N": 1, "isfree": False, "init": float(lumdist),
                             "units": "Mpc"},
    }
    return model_params


# =============================================================================
# build_all
# =============================================================================

def build_all(**kwargs):
    print("Building SPS …", flush=True)
    sps = build_sps(**kwargs)
    print("Loading observations …", flush=True)
    obs_mods, obs_fire = build_obs(**kwargs)
    print("Building model …", flush=True)
    model_params = build_model(**kwargs)
    return sps, obs_mods, obs_fire, model_params


# =============================================================================
# Forward model utilities
# =============================================================================

def _flux_norm(zred, lumdist):
    """Prospector flux_norm: Lsun/Hz → maggies (includes 1+z factor)."""
    dfactor   = (lumdist * 1e5)**2
    unit_conv = _to_cgs / (3631.0 * _jansky_cgs) * (1.0 + zred)
    return unit_conv / dfactor


def _smooth_and_interp(wave_rest, spec_lsun_hz, obs_wave_obs,
                       sigma_smooth_kms, sigma_inst_kms, sigma_lib_kms,
                       zred):
    """
    Convert Lsun/Hz → maggies, apply LOSVD + instrumental smoothing in
    log-lam space, redshift, and interpolate onto the data wavelength grid.
    """
    sigma_add2 = max(0.0, sigma_inst_kms**2 - sigma_lib_kms**2)
    sigma_eff  = np.sqrt(sigma_smooth_kms**2 + sigma_add2)

    norm = _flux_norm(zred, cosmo.luminosity_distance(zred).to('Mpc').value)
    spec = spec_lsun_hz * norm

    if sigma_eff > 0.5:
        dloglam   = np.log(wave_rest[1] / wave_rest[0])
        sigma_pix = (sigma_eff / _CKMS) / dloglam
        spec = gaussian_filter1d(spec, sigma_pix)

    wave_obs_model = wave_rest * (1.0 + zred)
    spec_obs       = spec / (1.0 + zred)
    return np.interp(obs_wave_obs, wave_obs_model, spec_obs,
                     left=np.nan, right=np.nan)


# =============================================================================
# Likelihood and prior
# =============================================================================

def _logl(theta, sps, obs_mods, obs_fire):
    """
    Log-likelihood for the two-component model (16 parameters).

    theta: [teff, logg, mh_idx, logL_star, av_smc, sigma_smooth,   (0-5)
            logmass, logzsol, dust2_gal, dust1, dust_index,         (6-10)
            logsfr_ratio_0, ..., logsfr_ratio_4]                    (11-15)

    sigma_smooth is applied only to the photosphere.  The galaxy continuum
    uses instrumental smoothing only — a km/s LOSVD is not appropriate for
    old stellar populations.
    """
    from prospect.models.transforms import logsfr_ratios_to_masses

    teff, logg, mh_idx, logL_star, av_smc, sigma_smooth = theta[:6]
    logmass, logzsol, dust2_gal, dust1, dust_index       = theta[6:11]
    logsfr_ratios                                        = theta[11:]

    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=AGEBINS)

    params = dict(
        # photosphere
        teff=teff, logg=logg, mh_idx=mh_idx,
        logL_star=logL_star, av_smc=av_smc,
        # galaxy
        logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1, dust_index=dust_index,
        logsfr_ratios=logsfr_ratios, mass=mass, agebins=AGEBINS,
        # fixed FSPS settings
        sfh=3, imf_type=2, dust_type=4, dust1_index=-1.0,
        add_neb_emission=False, add_neb_continuum=False,
        add_dust_emission=False, zred=Z_EGG,
    )

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**params)

    if np.any(np.isnan(spec_tl)):
        return -1e300

    # ── MODS Red ─────────────────────────────────────────────────────────────
    # Photosphere: sigma_smooth (LRD LOSVD) + MODS instrumental broadening
    # Galaxy: MODS instrumental broadening only
    pred_tl_m = _smooth_and_interp(wave, spec_tl, obs_mods['wave_obs'],
                                    sigma_smooth, SIGMA_MODS, SIGMA_LIB, Z_EGG)
    pred_gal_m = _smooth_and_interp(wave, spec_gal, obs_mods['wave_obs'],
                                     0.0, SIGMA_MODS, SIGMA_LIB, Z_EGG)
    pred_m = pred_tl_m + pred_gal_m
    mask_m = obs_mods['mask'] & np.isfinite(pred_m) & (pred_m > 0)
    if mask_m.sum() < 10:
        return -1e300

    # ── FIRE ──────────────────────────────────────────────────────────────────
    pred_tl_f = _smooth_and_interp(wave, spec_tl, obs_fire['wave_obs'],
                                    sigma_smooth, SIGMA_FIRE, SIGMA_LIB, Z_EGG)
    pred_gal_f = _smooth_and_interp(wave, spec_gal, obs_fire['wave_obs'],
                                     0.0, SIGMA_FIRE, SIGMA_LIB, Z_EGG)
    pred_f = pred_tl_f + pred_gal_f
    mask_f = obs_fire['mask'] & np.isfinite(pred_f) & (pred_f > 0)
    if mask_f.sum() < 10:
        return -1e300

    res_m = obs_mods['flux'][mask_m] - pred_m[mask_m]
    res_f = obs_fire['flux'][mask_f] - pred_f[mask_f]
    lnp   = (-0.5 * np.sum((res_m / obs_mods['unc'][mask_m])**2)
             - 0.5 * np.sum((res_f / obs_fire['unc'][mask_f])**2))
    return float(lnp)


def _prior_transform(u):
    """
    Map 16-d unit hypercube to physical parameters.

    Indices:
      0    teff          Uniform [3500, 6750] K   — matches TLUSTY grid extent
      1    logg          Uniform [-3.5, 0.5]      — matches TLUSTY grid extent
      2    mh_idx        Uniform [0, 2]
      3    logL_star     Uniform [5, 14]
      4    av_smc        Uniform [0, 8] mag
      5    sigma_smooth  Uniform [0, 600] km/s    — LRD photosphere LOSVD only
      6    logmass       Uniform [7, 11.5]
      7    logzsol       Uniform [-2.0, 0.19]
      8    dust2_gal     Uniform [0, 4]            — Noll+09 diffuse τ_V
      9    dust1         Uniform [0, 4]            — birth-cloud τ_V
      10   dust_index    Uniform [-2.2, 0.4]       — attenuation slope modifier
      11–15 logsfr_ratios StudentT(0, 0.3, df=2) each
    """
    theta = np.zeros(NDIM)
    theta[0]  = 3500.0 + 3250.0 * u[0]          # teff  [3500, 6750]
    theta[1]  = -3.5   + 4.0   * u[1]           # logg  [-3.5, 0.5]
    theta[2]  =          2.0   * u[2]           # mh_idx
    theta[3]  =  5.0   + 9.0   * u[3]           # logL_star
    theta[4]  =          8.0   * u[4]           # av_smc
    theta[5]  =        600.0   * u[5]           # sigma_smooth
    theta[6]  =  7.0   + 4.5   * u[6]           # logmass [7, 11.5]
    theta[7]  = -2.0   + 2.19  * u[7]           # logzsol
    theta[8]  =          4.0   * u[8]           # dust2_gal
    theta[9]  =          4.0   * u[9]           # dust1
    theta[10] = -2.2   + 2.6   * u[10]          # dust_index [-2.2, 0.4]
    for i in range(N_RATIOS):
        theta[11 + i] = student_t.ppf(u[11 + i], df=2, loc=0.0, scale=0.3)
    return theta


# =============================================================================
# Plotting
# =============================================================================

def plot_corner(samples, out_path):
    try:
        import corner
        display_labels = [
            r'$T_{\rm eff}$ [K]',
            r'$\log g$',
            r'$[{\rm M/H}]_{\rm idx}$',
            r'$\log_{10}(L_*/L_\odot)$',
            r'$A_V^{\rm SMC}$',
            r'$\sigma_v$ [km/s]',
            r'$\log_{10}(M_{\rm gal}/M_\odot)$',
            r'$\log(Z/Z_\odot)$',
            r'$\tau_V^{\rm diff}$',
            r'$\tau_V^{\rm BC}$',
            r'$n_{\rm dust}$',
        ] + [r'$\log(\dot{M}_{%d}/\dot{M}_{%d})$' % (i, i+1)
             for i in range(N_RATIOS)]
        fig = corner.corner(samples, labels=display_labels,
                            quantiles=[0.16, 0.50, 0.84],
                            show_titles=True, title_kwargs={'fontsize': 9})
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Corner plot → {out_path}", flush=True)
    except ImportError:
        print("  corner not installed; skipping.", flush=True)


def plot_spectra(samples, sps, obs_mods, obs_fire, out_path):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from prospect.models.transforms import logsfr_ratios_to_masses

        med = np.median(samples, axis=0)
        teff, logg, mh_idx, logL_star, av_smc, sigma_smooth = med[:6]
        logmass, logzsol, dust2_gal, dust1, dust_index       = med[6:11]
        logsfr_ratios                                        = med[11:]

        mass = logsfr_ratios_to_masses(
            logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=AGEBINS)
        params = dict(teff=teff, logg=logg, mh_idx=mh_idx,
                      logL_star=logL_star, av_smc=av_smc,
                      logmass=logmass, logzsol=logzsol,
                      dust2_gal=dust2_gal, dust1=dust1, dust_index=dust_index,
                      logsfr_ratios=logsfr_ratios, mass=mass, agebins=AGEBINS,
                      sfh=3, imf_type=2, dust_type=4, dust1_index=-1.0,
                      add_neb_emission=False, add_neb_continuum=False,
                      add_dust_emission=False, zred=Z_EGG)
        wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**params)

        pred_m = (_smooth_and_interp(wave, spec_tl, obs_mods['wave_obs'],
                                     sigma_smooth, SIGMA_MODS, SIGMA_LIB, Z_EGG)
                  + _smooth_and_interp(wave, spec_gal, obs_mods['wave_obs'],
                                       0.0, SIGMA_MODS, SIGMA_LIB, Z_EGG))
        pred_f = (_smooth_and_interp(wave, spec_tl, obs_fire['wave_obs'],
                                     sigma_smooth, SIGMA_FIRE, SIGMA_LIB, Z_EGG)
                  + _smooth_and_interp(wave, spec_gal, obs_fire['wave_obs'],
                                       0.0, SIGMA_FIRE, SIGMA_LIB, Z_EGG))

        def mag2flam(f_mag, w_obs):
            return f_mag * 3631.0 * _jansky_cgs * _c_AA / w_obs**2

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        wm, ok = obs_mods['wave_rest'], obs_mods['mask']
        ax0.plot(wm[ok], mag2flam(obs_mods['flux'][ok], obs_mods['wave_obs'][ok]),
                 'k-', lw=0.5, alpha=0.6, label='MODS R data')
        ax0.plot(wm[ok], mag2flam(pred_m[ok], obs_mods['wave_obs'][ok]),
                 'r-', lw=1.2, label='Model (median)')
        ax0.set_xlabel('Rest wavelength [Å]')
        ax0.set_ylabel(r'$F_\lambda$ [erg/s/cm²/Å]')
        ax0.set_title(f'MODS  Teff={teff:.0f}K  logg={logg:.2f}  '
                      f'logL={logL_star:.2f}  Av={av_smc:.2f}')
        ax0.legend(fontsize=8)

        wf, ok = obs_fire['wave_rest'], obs_fire['mask']
        ax1.plot(wf[ok], mag2flam(obs_fire['flux'][ok], obs_fire['wave_obs'][ok]),
                 'k-', lw=0.5, alpha=0.6, label='FIRE data')
        ax1.plot(wf[ok], mag2flam(pred_f[ok], obs_fire['wave_obs'][ok]),
                 'b-', lw=1.2, label='Model (median)')
        ax1.set_xlabel('Rest wavelength [Å]')
        ax1.set_ylabel(r'$F_\lambda$ [erg/s/cm²/Å]')
        ax1.set_title(f'FIRE  logM={logmass:.2f}  logZ={logzsol:.2f}  '
                      f'σ={sigma_smooth:.0f} km/s')
        ax1.legend(fontsize=8)

        fig.suptitle('The Egg — TLUSTY photosphere + continuity SFH host galaxy',
                     fontsize=11)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Spectra plot → {out_path}", flush=True)
        plt.close(fig)
    except Exception as e:
        print(f"  Spectra plot failed: {e}", flush=True)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import functools

    print(f'\n{"="*60}', flush=True)
    print('The Egg — TLUSTY photosphere + continuity SFH host galaxy', flush=True)
    print(f'MODS Red {MODSR_LO:.0f}–{MODSR_HI:.0f} Å + '
          f'FIRE {FIRE_LO:.0f}–{FIRE_HI:.0f} Å', flush=True)
    print(f'ndim={NDIM}  |  SFH bins={N_BINS_SFH}  |  '
          f'SFH ratios={N_RATIOS}', flush=True)
    print(f'{"="*60}\n', flush=True)

    sps, obs_mods, obs_fire, model_params = build_all()

    logl_fn  = functools.partial(_logl, sps=sps,
                                  obs_mods=obs_mods, obs_fire=obs_fire)
    prior_fn = _prior_transform

    print('\nStarting DynamicNestedSampler …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl_fn, prior_fn, NDIM,
        nlive=NLIVE, bound='multi', sample='rwalk')
    t0 = time.time()
    sampler.run_nested(
        dlogz_init=DLOGZ_INIT,
        nlive_init=NLIVE_INIT,
        nlive_batch=NLIVE_BATCH,
        wt_kwargs={'pfrac': 1.0},
        n_effective=N_EFFECTIVE,
        print_progress=True)
    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f} s', flush=True)

    res  = sampler.results
    wts  = np.exp(res.logwt - res.logz[-1])
    samp = resample_equal(res.samples, wts)

    stem     = 'egg_prospector_tlusty'
    pkl_path = os.path.join(SCRIPT_DIR, f'egg_dynesty_{stem}.pkl')
    npy_path = os.path.join(SCRIPT_DIR, f'egg_chain_{stem}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}')

    n_pix  = obs_mods['mask'].sum() + obs_fire['mask'].sum()
    ll_max = res.logl.max()
    lz, lze = res.logz[-1], res.logzerr[-1]
    bic    = NDIM * np.log(n_pix) - 2.0 * ll_max

    lines = [
        'The Egg — TLUSTY photosphere + continuity SFH host galaxy',
        f'ndim={NDIM}  SFH bins={N_BINS_SFH}',
        f'Data: MODS R ({obs_mods["mask"].sum()} pix) + '
        f'FIRE ({obs_fire["mask"].sum()} pix) = {n_pix} total',
        '',
        f'  log Z = {lz:.2f} ± {lze:.2f}',
        f'  BIC   = {bic:.2f}',
        f'  max ln L = {ll_max:.2f}',
        f'  Elapsed  = {elapsed:.0f} s',
        '',
        'Parameter posteriors (16–50–84 percentiles):',
    ]
    for i, lbl in enumerate(LABELS):
        p16, p50, p84 = np.percentile(samp[:, i], [16, 50, 84])
        lines.append(f'  {lbl:20s}: {p50:.3f}  +{p84-p50:.3f}/-{p50-p16:.3f}')
    report = '\n'.join(lines)
    print('\n' + report)

    bic_path = os.path.join(SCRIPT_DIR, f'egg_bic_{stem}.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'\nBIC report → {bic_path}')

    corner_path  = os.path.join(SCRIPT_DIR, f'egg_corner_{stem}.png')
    spectra_path = os.path.join(SCRIPT_DIR, f'egg_spectra_{stem}.png')
    plot_corner(samp, corner_path)
    plot_spectra(samp, sps, obs_mods, obs_fire, spectra_path)

    print('\nDone.')
