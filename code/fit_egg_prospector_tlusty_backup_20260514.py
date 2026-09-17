#!/usr/bin/env python3
"""
fit_egg_prospector_tlusty.py  —  Two-component spectroscopic fit of The Egg.

SOURCE: J1025+1402 ("The Egg"), z = 0.1007
DATA  : MODS Red (3900–7400 Å rest)  +  FIRE (7400–15500 Å rest)

=============================================================================
PROSPECTOR INTEGRATION
=============================================================================

This script interfaces with Prospector (Leja+2017, Johnson+2021) as directly
as possible while supporting the custom two-component SPS model described
below.

  • ProspectorParams  — parameter state, prior objects, prior_transform.
    build_model() returns a ProspectorParams instance with actual Prior
    objects (Uniform, StudentT from prospect.models.priors).  Dynesty is
    driven by model.prior_transform directly — no bespoke _prior_transform.

  • logsfr_ratios_to_masses, zred_to_agebins  — Prospector transforms used
    as-is.

  • TLUSTYPlusGalaxyBasis  — custom composite SPS that wraps TLUSTYBasis
    (Liu+2026 photosphere grid) and FastStepBasis (FSPS/C3K galaxy).
    It intentionally bypasses SpecModel.predict() because the two components
    require independent velocity smoothing (sigma_smooth applies only to the
    photosphere); SpecModel supports only a single sigma_smooth for the full
    summed spectrum.  _logl constructs the predicted spectrum manually from
    get_spectra_components().

  • Emission-line masking  — neither component produces nebular emission
    (add_neb_emission=False; TLUSTY is a pure stellar atmosphere).  Rest-frame
    line masks are applied in build_obs() before fitting; masked pixels are
    excluded from chi-squared.  Masking is mandatory: unmasked emission pixels
    would bias continuum parameters (logL_star, dust2, logmass) since the
    model cannot reproduce them.

=============================================================================
MODEL  (Prospector-alpha framework, Leja+2017)
=============================================================================

Two additive spectral components:

  1. HOST GALAXY  — FastStepBasis (FSPS/C3K), continuity non-parametric SFH
     (Leja+2019).  N_BINS_SFH=6 age bins; edges set by zred_to_agebins().
     Chabrier IMF (imf_type=2).
     Noll+09 two-component dust (dust_type=4):
       dust1      — birth-cloud optical depth (young stars <30 Myr)
       dust2_gal  → FSPS dust2: diffuse optical depth (all stars)
       dust_index — slope modifier on Calzetti baseline (0 = pure Calzetti)
       dust1_index — fixed at −1.0 (Charlot & Fall 2000 power law)
     No nebular emission.  Masked line pixels contain no information for the
     pure-continuum model, so masking them loses nothing.

  2. LRD PHOTOSPHERE — TLUSTYBasis (Liu+2026 TLUSTY stellar atmosphere grid,
     ~167 models at [M/H] = −2, −1, 0).  SMC Gordon+2003 extinction applied
     inside TLUSTYBasis (av_smc → dust2).  Stefan-Boltzmann normalization via
     logL_star.  LOSVD sigma_smooth broadens photosphere features only.

Dust conventions
----------------
  • Galaxy:      Noll+09 ISM law (dust_type=4).  Constrained by the broad
                 spectral shape of the galaxy continuum.
  • Photosphere: SMC Gordon+2003 (av_smc).  Constrained by the photospheric
                 colour temperature gradient vs. the Planck/Rayleigh-Jeans
                 slope.  These two laws are physically and observationally
                 distinct; separate dust treatments are physically motivated.

Smoothing convention
--------------------
  sigma_smooth (km/s, LOSVD sigma) is applied only to the TLUSTY photosphere
  spectrum before summing with the galaxy.  The galaxy continuum receives only
  instrument PSF convolution (sigma_smooth_kms=0 in _smooth_and_interp).
  Applying a km/s LOSVD to integrated old stellar populations is incorrect:
  the apparent width of galaxy spectral features is set by the instrument, not
  by stellar kinematics of the unresolved host.

Separate metallicities
----------------------
  mh_idx controls the photosphere atmosphere composition via the TLUSTY
  grid (three [M/H] slices).  logzsol controls the galaxy stellar-population
  composition via FSPS/C3K.  These are physically distinct: the LRD
  photosphere metallicity traces the quasi-star surface; the galaxy
  metallicity reflects the integrated stellar population formed over Gyr
  timescales.  Separate parameters are correct.

=============================================================================
FREE PARAMETERS  (16 total)
=============================================================================
  Photosphere (6):
    teff         : T_eff [K]            Uniform [3500, 6750]  TLUSTY grid extent
    logg         : log g                Uniform [-3.5, 0.5]   TLUSTY grid extent
    mh_idx       : metallicity index    Uniform [0, 2]   0→[M/H]=-2, 2→[M/H]=0
    logL_star    : log10(L*/Lsun)       Uniform [5, 14]
    av_smc       : A_V (SMC, photosph.) Uniform [0, 8]
    sigma_smooth : LOSVD sigma (LRD)    Uniform [0, 600] km/s

  Host galaxy (10):
    logmass      : log10(M*/Msun)       Uniform [7, 11.5]
    logzsol      : log(Z/Zsun)          Uniform [-2.0, 0.19]  C3K grid range
    dust2_gal    : diffuse τ_V          Uniform [0, 4]
    dust1        : birth-cloud τ_V      Uniform [0, 4]
    dust_index   : attenuation slope    Uniform [-2.2, 0.4]
    logsfr_ratio_{0..4}: SFH shape      StudentT(mean=0, scale=0.3, df=2) each

=============================================================================
OUTPUTS  (written to egg_analysis/)
=============================================================================
  egg_dynesty_prospector_tlusty.pkl   — full dynesty results object
  egg_chain_prospector_tlusty.npy     — equal-weight posterior samples
  egg_corner_prospector_tlusty.png    — corner plot
  egg_spectra_prospector_tlusty.png   — data/model/components overlay
  egg_bic_prospector_tlusty.txt       — BIC, log Z, and posteriors

=============================================================================
MULTI-SOURCE DESIGN
=============================================================================
  To adapt for another source change only the SOURCE CONFIGURATION block:
    Z_SOURCE, MODSR_LO/HI, FIRE_LO/HI, DATA_DIR, MODS_FNAME, FIRE_FNAME,
    TLUSTY_CACHE, and the initial values in build_model().
  Line masks are defined in rest-frame Å and require no modification.
  ProspectorParams, the SFH, dust, and likelihood logic are source-agnostic.
"""

import os
import sys
import importlib.util
import time
import pickle
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo

warnings.filterwarnings('ignore')

# SPS_HOME must be set before any FSPS or Prospector import
_SPS_HOME = '/home/omc5226/prospector_tlusty_dev/fsps_fresh/fsps-master'
os.environ['SPS_HOME'] = _SPS_HOME

import dynesty
from dynesty.utils import resample_equal

# ── Prospector imports ────────────────────────────────────────────────────────
_PROSPECT_DIR  = '/home/omc5226/prospector_tlusty_dev/prospector'
_TLUSTY_MODULE = (
    '/home/omc5226/prospector_tlusty_dev/prospector/'
    'prospect/sources/tlusty_basis.py'
)
sys.path.insert(0, _PROSPECT_DIR)

from prospect.models.parameters import ProspectorParams
from prospect.models.priors     import Uniform, StudentT
from prospect.models.transforms import logsfr_ratios_to_masses, zred_to_agebins


# =============================================================================
# SOURCE CONFIGURATION  ←  change these when running on a different target
# =============================================================================
SOURCE_NAME  = 'TheEgg'
Z_SOURCE     = 0.1007            # spectroscopic redshift
F_SYS        = 0.05              # systematic flux error floor (fraction of |flux|)

MODSR_LO     = 3900.0            # rest-frame Å — fit window edges
MODSR_HI     = 7400.0
FIRE_LO      = 7400.0
FIRE_HI      = 15500.0

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, 'egg_data',
                             'Re_ J1025+1402 LBT _ Magellan spectra')
MODS_FNAME   = 'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits'
FIRE_FNAME   = 'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits'

SPEC_DIR     = ('/home/omc5226/work/lrdmesa/liu2026_library/'
                'LRD_synthetic_spectral_library-main/specs')
TLUSTY_CACHE = os.path.join(SCRIPT_DIR, 'egg_tlusty_r10k.npz')

# ── Instrument resolving powers ───────────────────────────────────────────────
R_MODS   = 9100
R_FIRE   = 7000
TARGET_R = 10000                          # TLUSTYBasis cache resolution
_CKMS    = 2.998e5                        # km/s
SIGMA_LIB  = _CKMS / (TARGET_R * 2.355)  # library  σ ≈ 12.7 km/s
SIGMA_MODS = _CKMS / (R_MODS   * 2.355)  # MODS-R   σ ≈ 14.0 km/s
SIGMA_FIRE = _CKMS / (R_FIRE   * 2.355)  # FIRE     σ ≈ 18.2 km/s

# ── CGS / unit constants ──────────────────────────────────────────────────────
_lsun         = 3.846e33      # erg/s
_pc_cm        = 3.0857e18     # cm/pc
_jansky_cgs   = 1e-23         # erg/s/Hz/cm²
_c_AA         = 2.998e18      # Å/s
_to_cgs       = _lsun / (4.0 * np.pi * (10.0 * _pc_cm)**2)  # Lsun→erg/s/cm² @10pc
_PYPEIT_SCALE = 1e-17         # erg/s/cm²/Å per PyPeIt unit

# ── Precomputed cosmological quantities at Z_SOURCE ───────────────────────────
# Computed once at module load; _smooth_and_interp receives _LUMDIST as a
# constant to avoid calling cosmo.luminosity_distance at every likelihood eval.
_LUMDIST = cosmo.luminosity_distance(Z_SOURCE).to('Mpc').value

# ── Rest-frame emission-line masks ─────────────────────────────────────────────
# Standard rest-frame mask list for the LRD MESA spectroscopic programme.
# Applied to all sources; no modification needed when changing Z_SOURCE because
# masks are applied after computing wave_rest = wave_obs / (1+z).
#
# Mask widths are consistent with prior single-component TLUSTY fits:
#   MODS-R: fit_egg_modsr.py, fit_egg_mods_joint.py
#   FIRE  : fit_egg_fire.py   (note ±150 Å on [S III] to protect Pa-series)
#
# Broad Balmer wings (Hα ±110 Å ≈ ±5000 km/s, Hβ ±90 Å ≈ ±5500 km/s) cover
# the BLR emission that the photosphere model cannot reproduce.  Narrow
# forbidden lines use tighter windows.  He I 10830 uses ±250 Å because it is
# known to be extremely broad in LRDs (possibly P Cygni wind absorption).
# Hα (6563 Å) is listed in FIRE_LINES for generality at higher z; it falls
# outside FIRE coverage for The Egg (z=0.1007 → λ_obs = 7224 Å < FIRE start).
MODS_LINES = [
    # Broad Balmer (BLR origin — generous masks)
    (6563, 110),   # Hα  ±5000 km/s
    (4861,  90),   # Hβ
    (4340,  50),   # Hγ  (partial BLR contamination)
    (4102,  30),   # Hδ
    (3970,  30),   # Hε / Ca H blend
    # Narrow forbidden
    (5007,  30),   # [O III]
    (4959,  20),   # [O III]
    (6583,  25),   # [N II]
    (6548,  20),   # [N II]
    (6716,  20),   # [S II]
    (6731,  20),   # [S II]
    (3727,  30),   # [O II]
    (3869,  20),   # [Ne III]
    (9069,  40),   # [S III] — outside MODS-R for The Egg; kept for generality
    (9532,  40),   # [S III]
    # He recombination
    (5876,  25),   # He I
    (4686,  25),   # He II
]
FIRE_LINES = [
    (10830, 250),  # He I 1.083 µm — very broad in LRDs; ±250 Å ≈ ±7000 km/s
    (10938, 200),  # Pa δ
    (12820, 250),  # Pa β
    ( 9015, 150),  # Pa η
    ( 9229, 150),  # Pa ζ
    ( 9546, 150),  # Pa ε
    (10049, 200),  # Pa δ (alt.)
    ( 9069, 150),  # [S III] — ±150 Å matches fit_egg_fire.py to protect Pa-series
    ( 9532, 150),  # [S III]
    ( 6563, 110),  # Hα — outside FIRE for The Egg; included for high-z sources
]

# ── SFH configuration ─────────────────────────────────────────────────────────
N_BINS_SFH = 6
N_RATIOS   = N_BINS_SFH - 1    # 5 free log-SFR ratios

# Initial bin-edge skeleton passed to zred_to_agebins().  The function keeps
# the first bin ([0, 7.47] = 0–30 Myr), linearly spaces the intermediate
# edges in log(age), and sets the oldest bin upper edge to log10(t_univ).
_AGEBINS_INIT = np.array([
    [0.0,  7.47],   # youngest: 0–30 Myr (matches dust1 birth-cloud threshold)
    [7.47, 8.0 ],
    [8.0,  8.5 ],
    [8.5,  9.0 ],
    [9.0,  9.5 ],
    [9.5,  10.0],   # upper edge overwritten by zred_to_agebins
])
AGEBINS = zred_to_agebins(zred=Z_SOURCE, agebins=_AGEBINS_INIT)

# ── Dynesty settings ──────────────────────────────────────────────────────────
NLIVE       = 600
NLIVE_INIT  = 400
NLIVE_BATCH = 600
DLOGZ_INIT  = 0.01
N_EFFECTIVE = 8000


# =============================================================================
# Import TLUSTYBasis without triggering FSPS-dependent module-level code
# =============================================================================

def _import_tlusty():
    """Load TLUSTYBasis directly from its source file.

    Direct loading avoids prospect.sources.__init__ importing agnssp_basis,
    which inspects SPS_HOME at import time.  We set SPS_HOME above, but the
    import-order dependency is fragile; loading via importlib.util is robust.
    """
    spec = importlib.util.spec_from_file_location('tlusty_basis', _TLUSTY_MODULE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TLUSTYBasis


# =============================================================================
# TLUSTYPlusGalaxyBasis — composite two-component SPS
# =============================================================================

class TLUSTYPlusGalaxyBasis:
    """
    Composite SPS for fitting a two-component LRD spectrum: stellar-atmosphere
    photosphere plus host-galaxy continuum.

    Why this class exists
    ---------------------
    Prospector's standard pipeline routes SPS output through SpecModel.predict(),
    which applies a single sigma_smooth LOSVD to the entire model spectrum.  For
    the LRD TLUSTY model, sigma_smooth describes the kinematic broadening of
    photospheric absorption features and must NOT be applied to the galaxy
    continuum, whose apparent spectral width is determined by the instrument
    PSF alone.  Applying sigma_smooth to old stellar-population features is
    also physically incorrect for an unresolved host galaxy.

    To enforce component-specific smoothing, this class exposes
    get_spectra_components(), which returns the photosphere and galaxy spectra
    on the same wavelength grid but separately.  The calling code in _logl
    applies sigma_smooth to spec_tl only, then sums before comparing to data.

    Because we bypass SpecModel.predict(), the following Prospector pieces are
    used directly instead:

        ProspectorParams.prior_transform   → drives dynesty prior
        logsfr_ratios_to_masses            → SFH mass decomposition
        zred_to_agebins                    → age-bin edges at z_source
        Uniform, StudentT (Prior objects)  → encoded in ProspectorParams

    Component 1 — LRD photosphere (TLUSTYBasis, Liu+2026)
    -------------------------------------------------------
    ~167 stellar-atmosphere models at three metallicities ([M/H] = −2, −1, 0).
    Each model is a TLUSTY non-LTE calculation stored at native R~430,000 and
    convolved to R=10,000 (TARGET_R) during cache construction.  The cache is a
    log-spaced wavelength grid (wave_lo–wave_hi Å).

    Interpolation: trilinear in (Teff, logg) for each bracketing [M/H] slice,
    then linear across metallicity index (mh_idx).  If EITHER bracketing [M/H]
    slice is outside the convex hull of available (Teff, logg) points,
    get_galaxy_spectrum returns a NaN spectrum — _logl detects NaN and returns
    -1e300, steering dynesty away from out-of-bounds regions without crashing.

    Normalization: TLUSTY .spec files store the Eddington flux H_λ.
        H_ν = H_λ · λ²/c
        4π R² = L_star / (σ_SB T_eff⁴)    ← Stefan-Boltzmann
        L_ν = 4π R² · 4π H_ν
        spec [Lsun/Hz] = L_ν / L_sun

    Dust: SMC Gordon+2003 extinction applied inside TLUSTYBasis when dust2>0.
    Mapped as av_smc → dust2 before calling tlusty.get_galaxy_spectrum().

    Component 2 — Host galaxy (FastStepBasis, FSPS/C3K)
    -----------------------------------------------------
    Continuity non-parametric SFH (Leja+2019).  FastStepBasis converts the
    per-bin mass array (shape N_BINS_SFH) to a tabular SFH for FSPS.
    get_galaxy_spectrum returns specific luminosity [Lsun/Hz per M_sun formed];
    we multiply by the total per-bin mass (derived from logmass + logsfr_ratios
    via logsfr_ratios_to_masses) to get total Lsun/Hz.

    C3K spectral library (Conroy+), native R~3000 at optical, interpolated
    continuously in log(Z) (zcontinuous=1).  Metallicity range: [−2.0, +0.19]
    dex solar — both endpoints are the C3K grid limits.

    Dust: Noll+09 two-component (FSPS dust_type=4):
      dust2_gal → FSPS dust2: diffuse attenuation optical depth, all stars.
      dust1     → FSPS dust1: extra birth-cloud optical depth, stars <30 Myr.
      dust_index → FSPS dust_index: slope modifier on Calzetti baseline.
      dust1_index: fixed at −1.0 (Charlot & Fall 2000 birth-cloud slope).
    dust2_gal is re-keyed to 'dust2' before calling galaxy.get_galaxy_spectrum.

    Wavelength grid
    ---------------
    self.wave: the TLUSTYBasis log-spaced grid (2000–25,000 Å, ~25,000 pix at
    R=10,000).  Galaxy output (native FSPS grid) is interpolated onto self.wave
    via np.interp before returning.

    Spectral resolution attribute
    -----------------------------
    self.spectral_resolution = c / (TARGET_R × 2.355) km/s (σ).  Kept for
    API completeness in case the object is inspected by Prospector utilities,
    but not consumed by our _logl pipeline.

    Parameters
    ----------
    spec_dir : str
        Directory containing Liu+2026 TLUSTY .spec files.
    target_R : int or float
        Target resolving power for the in-memory spectral cache.
    cache_file : str
        Path to .npz cache.  Loaded if it exists; built and saved if not.
    verbose : bool
        Print init progress.
    """

    def __init__(self, spec_dir, target_R, cache_file, verbose=True):
        TLUSTYBasis = _import_tlusty()
        self.tlusty = TLUSTYBasis(
            spec_dir=spec_dir,
            target_R=target_R,
            wave_lo=2000.0,
            wave_hi=25000.0,
            verbose=verbose,
            cache_file=cache_file,
        )
        self.wave = self.tlusty.wave      # master log-spaced grid (Å)

        from prospect.sources import FastStepBasis
        self.galaxy = FastStepBasis(zcontinuous=1, compute_vega_mags=False)
        if verbose:
            print('  FastStepBasis (FSPS/C3K galaxy) initialised.', flush=True)

        # Prospector API attribute: library spectral resolution σ in km/s.
        self.spectral_resolution = self.tlusty.spectral_resolution

    @property
    def wavelengths(self):
        """Alias for Prospector utility compatibility."""
        return self.wave

    def get_spectra_components(self, **params):
        """
        Return photosphere and galaxy spectra separately on self.wave (Lsun/Hz).

        Separating the components before returning is the primary purpose of
        this class.  The caller applies sigma_smooth to spec_tl only, then
        sums the two components for comparison to data.  If the components were
        summed here (as in get_galaxy_spectrum), the caller would have to
        re-separate them or apply a single broadening to the total — neither
        of which is correct.

        Required params keys
        --------------------
        teff, logg, mh_idx, logL_star, av_smc  (photosphere)
        logmass, logzsol, dust2_gal, dust1, dust_index, mass, agebins
        sfh, imf_type, dust_type, dust1_index, add_neb_emission,
        add_neb_continuum, add_dust_emission, zred                (FSPS fixed)

        Returns
        -------
        wave     : ndarray (N_pix,), AA
        spec_tl  : ndarray (N_pix,), Lsun/Hz — photosphere (NaN array if OOB)
        spec_gal : ndarray (N_pix,), Lsun/Hz — galaxy on self.wave
        mfrac    : float — surviving / initial mass fraction from FSPS
        """
        # ── Photosphere ──────────────────────────────────────────────────────
        # av_smc → dust2: TLUSTYBasis applies Gordon+2003 SMC extinction when
        # dust2 > 0.  The mapping is explicit so av_smc never accidentally
        # reaches FSPS as a dust parameter.
        tlusty_params = dict(params)
        tlusty_params['dust2'] = float(params.get('av_smc', 0.0))
        _, spec_tl, _ = self.tlusty.get_galaxy_spectrum(**tlusty_params)

        # ── Galaxy ───────────────────────────────────────────────────────────
        # dust2_gal → dust2: FSPS diffuse optical depth.  params['mass'] must
        # be the per-bin mass array (shape N_BINS_SFH), set by the caller via
        # logsfr_ratios_to_masses before calling this method.
        galaxy_params = dict(params)
        galaxy_params['dust2'] = float(params.get('dust2_gal', 0.0))
        wave_gal, spec_per_mass, mfrac = self.galaxy.get_galaxy_spectrum(
            **galaxy_params)
        mtot = np.atleast_1d(params.get('mass', np.array([1.0]))).sum()
        spec_gal = np.interp(
            self.wave, wave_gal, spec_per_mass * mtot, left=0.0, right=0.0)

        return self.wave, spec_tl, spec_gal, mfrac

    def get_galaxy_spectrum(self, **params):
        """
        Summed spectrum — preserves the Prospector SPS API.

        Returns a NaN spectrum if the photosphere is OOB so that callers
        relying on this API can check for NaN.  The primary inference path
        uses get_spectra_components() to keep the components separate.
        """
        wave, spec_tl, spec_gal, mfrac = self.get_spectra_components(**params)
        if np.any(np.isnan(spec_tl)):
            return wave, np.full(len(wave), np.nan), 1.0
        return wave, spec_tl + spec_gal, mfrac

    def get_galaxy_elines(self):
        """Return empty arrays — neither component produces emission lines."""
        return np.array([]), np.array([])


# =============================================================================
# build_sps / build_obs / build_model / build_all
# =============================================================================

def build_sps(spec_dir=SPEC_DIR, target_R=TARGET_R,
              cache_file=TLUSTY_CACHE, verbose=True, **kwargs):
    """Instantiate the composite two-component SPS."""
    return TLUSTYPlusGalaxyBasis(
        spec_dir=spec_dir, target_R=target_R,
        cache_file=cache_file, verbose=verbose)


def build_obs(data_dir=DATA_DIR, mods_fname=MODS_FNAME,
              fire_fname=FIRE_FNAME, **kwargs):
    """
    Load and pre-process The Egg spectra (MODS-R + FIRE).

    Processing pipeline
    -------------------
    1. Read PyPeIt coadd FITS; honour the pipeline pixel-quality mask.
    2. Add systematic error floor F_SYS in quadrature to pipeline σ.
    3. Convert f_λ [PyPeIt units = 1e-17 erg/s/cm²/Å] to maggies (F_ν / 3631 Jy).
    4. Apply rest-frame emission-line masks (MODS_LINES, FIRE_LINES).
    5. Clip to instrument wavelength range and require finite, positive σ.

    Emission-line masking
    ---------------------
    Neither model component (TLUSTY photosphere, FSPS galaxy) produces
    emission lines.  Unmasked line pixels would force the model continuum to
    over-predict the emission, biasing logL_star and dust parameters upward.
    Masking the lines loses no continuum information because the windows are
    narrow relative to the total spectral coverage (3518 + 10346 pixels).

    Returns
    -------
    obs_mods, obs_fire : dict
        wave_obs  : observed-frame wavelengths (Å)
        wave_rest : rest-frame wavelengths (Å)
        flux      : flux in maggies
        unc       : uncertainty in maggies
        mask      : True = pixel included in the fit
        name      : label string ('MODS_R' or 'FIRE')
    """
    def load_pypeit(fname):
        with fits.open(os.path.join(data_dir, fname)) as h:
            s    = h['SPECTRUM'].data
            wobs = s['wave'].astype(float)
            flux = s['flux'].astype(float)
            ivar = s['ivar'].astype(float)
            msk  = s['mask'].astype(int)
        good  = (msk == 1) & (ivar > 0) & np.isfinite(flux)
        sigma = np.where(good,
                         1.0 / np.sqrt(np.where(ivar > 0, ivar, np.inf)),
                         np.nan)
        return wobs, flux, sigma, good

    def flam_to_maggies(w_obs_aa, f_lam_pypeit):
        f_lam = f_lam_pypeit * _PYPEIT_SCALE   # erg/s/cm²/Å
        f_nu  = f_lam * w_obs_aa**2 / _c_AA    # erg/s/cm²/Hz
        return f_nu / (3631.0 * _jansky_cgs)   # maggies

    def make_line_mask(wave_rest, lines):
        m = np.ones(len(wave_rest), dtype=bool)
        for cen, hw in lines:
            m &= ~((wave_rest >= cen - hw) & (wave_rest <= cen + hw))
        return m

    wm_obs, fm, sm, gm = load_pypeit(mods_fname)
    wf_obs, ff, sf, gf = load_pypeit(fire_fname)

    wm_rest = wm_obs / (1.0 + Z_SOURCE)
    wf_rest = wf_obs / (1.0 + Z_SOURCE)

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


def build_model(zred=Z_SOURCE,
                teff_init=5293., logg_init=-2.255, logL_init=10.5,
                **kwargs):
    """
    Return a ProspectorParams instance with genuine Prior objects.

    model.prior_transform is passed directly to dynesty — no bespoke
    _prior_transform function is needed.  model.theta_index (a dict of
    name → slice) is used in _logl for read-only, thread-safe parameter
    extraction from the theta vector.

    Age bins are computed via zred_to_agebins() from Prospector's transforms
    module: intermediate bin edges are evenly spaced in log(age) between 30 Myr
    and 0.85 × t_univ(zred), and the oldest bin runs to t_univ(zred).

    Note on StudentT prior length warning
    --------------------------------------
    ProspectorParams.map_theta() checks len(prior) == N for each free param.
    StudentT with scalar params has len() = 1; for logsfr_ratios (N=5) this
    triggers a RuntimeWarning that the prior length does not match N.  The
    warning is suppressed by warnings.filterwarnings('ignore') at module level
    and has no functional effect: StudentT.unit_transform broadcasts correctly
    over the 5-element unit-cube slice it receives.

    Parameters
    ----------
    zred : float
        Source redshift.  Sets agebins and lumdist; stored as fixed param.
    teff_init, logg_init, logL_init : float
        Initial guesses for the photosphere parameters (e.g. from MESA).

    Returns
    -------
    model : ProspectorParams
    """
    lumdist = cosmo.luminosity_distance(zred).to('Mpc').value
    agebins = zred_to_agebins(zred=zred, agebins=_AGEBINS_INIT.copy())

    model_params = {
        # ── Photosphere (TLUSTYBasis) ─────────────────────────────────────
        'teff': {
            'N': 1, 'isfree': True, 'init': float(teff_init),
            'prior': Uniform(mini=3500., maxi=6750.),
            'units': 'K — TLUSTY grid extent [3500, 6750]',
        },
        'logg': {
            'N': 1, 'isfree': True, 'init': float(logg_init),
            'prior': Uniform(mini=-3.5, maxi=0.5),
            'units': 'log10(g / cm s^-2) — TLUSTY grid extent [-3.5, 0.5]',
        },
        'mh_idx': {
            'N': 1, 'isfree': True, 'init': 1.5,
            'prior': Uniform(mini=0., maxi=2.),
            'units': 'metallicity index: 0=[M/H]=-2, 1=-1, 2=0',
        },
        'logL_star': {
            'N': 1, 'isfree': True, 'init': float(logL_init),
            'prior': Uniform(mini=5., maxi=14.),
            'units': 'log10(L_star / L_sun)',
        },
        'av_smc': {
            'N': 1, 'isfree': True, 'init': 2.0,
            'prior': Uniform(mini=0., maxi=8.),
            'units': 'V-band extinction, SMC Gordon+2003, photosphere only',
        },
        'sigma_smooth': {
            'N': 1, 'isfree': True, 'init': 100.,
            'prior': Uniform(mini=0., maxi=600.),
            'units': 'km/s — LOSVD sigma, photosphere absorption only',
        },

        # ── Host galaxy (FastStepBasis + continuity SFH) ──────────────────
        'logmass': {
            'N': 1, 'isfree': True, 'init': 9.0,
            'prior': Uniform(mini=7., maxi=11.5),
            'units': 'log10(M_* / M_sun)',
        },
        'logzsol': {
            'N': 1, 'isfree': True, 'init': -0.5,
            'prior': Uniform(mini=-2.0, maxi=0.19),
            'units': 'log(Z / Z_sun) — C3K grid range [-2.0, +0.19]',
        },
        'dust2_gal': {
            'N': 1, 'isfree': True, 'init': 0.3,
            'prior': Uniform(mini=0., maxi=4.),
            'units': 'Noll+09 diffuse optical depth tau_V (all stars)',
        },
        'dust1': {
            'N': 1, 'isfree': True, 'init': 0.0,
            'prior': Uniform(mini=0., maxi=4.),
            'units': 'birth-cloud optical depth tau_V (stars < 30 Myr)',
        },
        'dust_index': {
            'N': 1, 'isfree': True, 'init': 0.0,
            'prior': Uniform(mini=-2.2, maxi=0.4),
            'units': 'Noll+09 attenuation slope modifier (0 = Calzetti)',
        },
        'logsfr_ratios': {
            'N': N_RATIOS, 'isfree': True, 'init': np.zeros(N_RATIOS),
            'prior': StudentT(mean=0., scale=0.3, df=2),
            'units': 'log10(SFR_j / SFR_{j+1}), j=0 youngest (continuity SFH)',
        },

        # ── Fixed parameters ──────────────────────────────────────────────
        'sfh':               {'N': 1, 'isfree': False, 'init': 3},
        'imf_type':          {'N': 1, 'isfree': False, 'init': 2},    # Chabrier
        'dust_type':         {'N': 1, 'isfree': False, 'init': 4},    # Noll+09
        'dust1_index':       {'N': 1, 'isfree': False, 'init': -1.0}, # C&F 2000
        'add_neb_emission':  {'N': 1, 'isfree': False, 'init': False},
        'add_neb_continuum': {'N': 1, 'isfree': False, 'init': False},
        'add_dust_emission': {'N': 1, 'isfree': False, 'init': False},
        'agebins':           {'N': N_BINS_SFH, 'isfree': False, 'init': agebins},
        'mass':              {'N': 1,  'isfree': False, 'init': 1e9,
                              'units': 'per-bin array; set by logsfr_ratios_to_masses in _logl'},
        'zred':              {'N': 1,  'isfree': False, 'init': float(zred)},
        'lumdist':           {'N': 1,  'isfree': False, 'init': float(lumdist),
                              'units': 'Mpc'},
    }
    return ProspectorParams(model_params)


def build_all(**kwargs):
    print("Building SPS …", flush=True)
    sps = build_sps(**kwargs)
    print("Loading observations …", flush=True)
    obs_mods, obs_fire = build_obs(**kwargs)
    print("Building model …", flush=True)
    model = build_model(**kwargs)
    return sps, obs_mods, obs_fire, model


# =============================================================================
# Forward-model utilities
# =============================================================================

def _flux_norm(zred, lumdist):
    """
    Lsun/Hz → maggies conversion factor.

    Equivalent to Prospector's SpecModel.flux_norm() for mass=1:
        norm = (L_sun / 4π(10 pc)²) × (1+z) / (d_L/10 pc)² / (3631 Jy)

    The (1+z) numerator accounts for the redshifting of photon energies
    (F_ν = L_ν (1+z) / 4π d_L²).  The 1/(d_L/10pc)² term is the
    inverse-square law factor.  lumdist is precomputed at module load to
    avoid repeated cosmology calls during the fit.
    """
    dfactor   = (lumdist * 1e5)**2              # (d_L / 10 pc)²
    unit_conv = _to_cgs / (3631.0 * _jansky_cgs) * (1.0 + zred)
    return unit_conv / dfactor


def _smooth_and_interp(wave_rest, spec_lsun_hz, obs_wave_obs,
                       sigma_smooth_kms, sigma_inst_kms, sigma_lib_kms,
                       zred, lumdist):
    """
    Convert Lsun/Hz → maggies, apply combined LOSVD + instrument smoothing in
    constant-velocity (log-λ) space, redshift the grid, and interpolate onto
    the observed-frame data wavelength array.

    Smoothing is performed in log-λ space (one pixel = one dloglam step),
    which is equivalent to sedpy.smoothspec with smoothtype='vel' on a
    log-spaced wavelength grid.  The effective sigma is:
        σ_add  = sqrt(max(0, σ_inst² − σ_lib²))
        σ_eff  = sqrt(σ_smooth² + σ_add²)
    so the library resolution and instrument PSF are added in quadrature with
    the requested LOSVD broadening.  Pass sigma_smooth_kms=0 for the galaxy
    component to apply instrument convolution only.

    Parameters
    ----------
    wave_rest          : rest-frame wavelength grid, log-spaced (Å)
    spec_lsun_hz       : spectrum in Lsun/Hz
    obs_wave_obs       : observed-frame wavelength grid of the spectrograph (Å)
    sigma_smooth_kms   : LOSVD sigma to add (0 for galaxy component)
    sigma_inst_kms     : instrument σ (SIGMA_MODS or SIGMA_FIRE)
    sigma_lib_kms      : library σ (SIGMA_LIB = c / (TARGET_R × 2.355))
    zred               : source redshift
    lumdist            : luminosity distance in Mpc (pass _LUMDIST)

    Returns
    -------
    spec_obs : ndarray (len(obs_wave_obs),) in maggies; NaN outside model range
    """
    sigma_add2 = max(0.0, sigma_inst_kms**2 - sigma_lib_kms**2)
    sigma_eff  = np.sqrt(sigma_smooth_kms**2 + sigma_add2)

    norm = _flux_norm(zred, lumdist)
    spec = spec_lsun_hz * norm           # maggies, rest-frame grid

    if sigma_eff > 0.5:
        dloglam   = np.log(wave_rest[1] / wave_rest[0])
        sigma_pix = (sigma_eff / _CKMS) / dloglam
        spec = gaussian_filter1d(spec, sigma_pix)

    # λ_obs = λ_rest (1+z).  The additional 1/(1+z) on spec converts the
    # observed bandwidth from dλ_rest to dλ_obs (standard Prospector convention
    # matching SpecModel.predict_spec).
    wave_obs_model = wave_rest * (1.0 + zred)
    spec_obs       = spec / (1.0 + zred)
    return np.interp(obs_wave_obs, wave_obs_model, spec_obs,
                     left=np.nan, right=np.nan)


# =============================================================================
# Log-likelihood
# =============================================================================

def _logl(theta, sps, obs_mods, obs_fire, model):
    """
    Log-likelihood for the 16-parameter two-component model.

    Parameter extraction uses model.theta_index — a read-only dict set at
    init that maps each free parameter name to its slice in the theta vector.
    Fixed parameters (agebins, zred, lumdist) are read from model.params,
    which holds the init values and is never mutated inside this function.
    This design is safe for dynesty's multiprocessing pool (each worker
    inherits a copy of model at fork time; no shared mutable state).

    sigma_smooth is applied only to spec_tl (photosphere).
    The galaxy receives instrument PSF convolution only (sigma_smooth_kms=0).
    """
    idx = model.theta_index       # read-only after ProspectorParams.__init__

    # ── Photosphere ───────────────────────────────────────────────────────────
    teff         = float(theta[idx['teff']])
    logg         = float(theta[idx['logg']])
    mh_idx       = float(theta[idx['mh_idx']])
    logL_star    = float(theta[idx['logL_star']])
    av_smc       = float(theta[idx['av_smc']])
    sigma_smooth = float(theta[idx['sigma_smooth']])

    # ── Galaxy ────────────────────────────────────────────────────────────────
    logmass       = float(theta[idx['logmass']])
    logzsol       = float(theta[idx['logzsol']])
    dust2_gal     = float(theta[idx['dust2_gal']])
    dust1         = float(theta[idx['dust1']])
    dust_index    = float(theta[idx['dust_index']])
    logsfr_ratios = theta[idx['logsfr_ratios']]    # ndarray, length N_RATIOS

    # ── Fixed (read-only from model.params) ───────────────────────────────────
    agebins = np.array(model.params['agebins'])    # (N_BINS_SFH, 2)
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])

    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)

    sps_params = dict(
        teff=teff, logg=logg, mh_idx=mh_idx,
        logL_star=logL_star, av_smc=av_smc,
        logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1, dust_index=dust_index,
        logsfr_ratios=logsfr_ratios, mass=mass, agebins=agebins,
        sfh=3, imf_type=2, dust_type=4, dust1_index=-1.0,
        add_neb_emission=False, add_neb_continuum=False,
        add_dust_emission=False, zred=zred,
    )

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)

    # NaN from TLUSTYBasis signals (teff,logg) outside the TLUSTY convex hull.
    if np.any(np.isnan(spec_tl)):
        return -1e300

    # ── MODS Red ──────────────────────────────────────────────────────────────
    pred_tl_m  = _smooth_and_interp(wave, spec_tl,  obs_mods['wave_obs'],
                                     sigma_smooth, SIGMA_MODS, SIGMA_LIB, zred, lumdist)
    pred_gal_m = _smooth_and_interp(wave, spec_gal, obs_mods['wave_obs'],
                                     0.0, SIGMA_MODS, SIGMA_LIB, zred, lumdist)
    pred_m = pred_tl_m + pred_gal_m
    mask_m = obs_mods['mask'] & np.isfinite(pred_m) & (pred_m > 0)
    if mask_m.sum() < 10:
        return -1e300

    # ── FIRE ──────────────────────────────────────────────────────────────────
    pred_tl_f  = _smooth_and_interp(wave, spec_tl,  obs_fire['wave_obs'],
                                     sigma_smooth, SIGMA_FIRE, SIGMA_LIB, zred, lumdist)
    pred_gal_f = _smooth_and_interp(wave, spec_gal, obs_fire['wave_obs'],
                                     0.0, SIGMA_FIRE, SIGMA_LIB, zred, lumdist)
    pred_f = pred_tl_f + pred_gal_f
    mask_f = obs_fire['mask'] & np.isfinite(pred_f) & (pred_f > 0)
    if mask_f.sum() < 10:
        return -1e300

    res_m = obs_mods['flux'][mask_m] - pred_m[mask_m]
    res_f = obs_fire['flux'][mask_f] - pred_f[mask_f]
    lnp   = (-0.5 * np.sum((res_m / obs_mods['unc'][mask_m])**2)
             - 0.5 * np.sum((res_f / obs_fire['unc'][mask_f])**2))
    return float(lnp)


# =============================================================================
# Plotting
# =============================================================================

def plot_corner(samples, model, out_path):
    try:
        import corner
        latex_labels = [
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
        ] + [r'$\log(\dot{M}_{%d}/\dot{M}_{%d})$' % (i, i + 1)
             for i in range(N_RATIOS)]
        fig = corner.corner(samples, labels=latex_labels,
                            quantiles=[0.16, 0.50, 0.84],
                            show_titles=True, title_kwargs={'fontsize': 9})
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Corner → {out_path}", flush=True)
    except ImportError:
        print("  corner not installed; skipping.", flush=True)


def plot_spectra(samples, sps, obs_mods, obs_fire, model, out_path):
    """Plot data vs. model (total + individual components) at the posterior median."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        idx = model.theta_index
        med = np.median(samples, axis=0)

        teff          = float(med[idx['teff']])
        logg          = float(med[idx['logg']])
        mh_idx        = float(med[idx['mh_idx']])
        logL_star     = float(med[idx['logL_star']])
        av_smc        = float(med[idx['av_smc']])
        sigma_smooth  = float(med[idx['sigma_smooth']])
        logmass       = float(med[idx['logmass']])
        logzsol       = float(med[idx['logzsol']])
        dust2_gal     = float(med[idx['dust2_gal']])
        dust1         = float(med[idx['dust1']])
        dust_index    = float(med[idx['dust_index']])
        logsfr_ratios = med[idx['logsfr_ratios']]

        agebins = np.array(model.params['agebins'])
        zred    = float(model.params['zred'])
        lumdist = float(model.params['lumdist'])

        mass = logsfr_ratios_to_masses(
            logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)
        sps_params = dict(
            teff=teff, logg=logg, mh_idx=mh_idx, logL_star=logL_star,
            av_smc=av_smc, logmass=logmass, logzsol=logzsol,
            dust2_gal=dust2_gal, dust1=dust1, dust_index=dust_index,
            logsfr_ratios=logsfr_ratios, mass=mass, agebins=agebins,
            sfh=3, imf_type=2, dust_type=4, dust1_index=-1.0,
            add_neb_emission=False, add_neb_continuum=False,
            add_dust_emission=False, zred=zred,
        )
        wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)

        def _pred(obs, sigma_inst):
            tl  = _smooth_and_interp(wave, spec_tl,  obs['wave_obs'],
                                      sigma_smooth, sigma_inst, SIGMA_LIB, zred, lumdist)
            gal = _smooth_and_interp(wave, spec_gal, obs['wave_obs'],
                                      0.0, sigma_inst, SIGMA_LIB, zred, lumdist)
            return tl, gal, tl + gal

        pred_tl_m, pred_gal_m, pred_m = _pred(obs_mods, SIGMA_MODS)
        pred_tl_f, pred_gal_f, pred_f = _pred(obs_fire, SIGMA_FIRE)

        def mag2flam(f_mag, w_obs):
            return f_mag * 3631.0 * _jansky_cgs * _c_AA / w_obs**2

        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
        for ax, obs, ptot, ptl, pgal, sinst, label in [
            (axes[0], obs_mods, pred_m, pred_tl_m, pred_gal_m, SIGMA_MODS, 'MODS R'),
            (axes[1], obs_fire, pred_f, pred_tl_f, pred_gal_f, SIGMA_FIRE, 'FIRE'),
        ]:
            ok = obs['mask']
            wr = obs['wave_rest'][ok]
            wo = obs['wave_obs'][ok]
            ax.plot(wr, mag2flam(obs['flux'][ok], wo),
                    'k-', lw=0.5, alpha=0.5, label='Data')
            ax.plot(wr, mag2flam(ptot[ok], wo),
                    'r-', lw=1.2, label='Total model')
            ax.plot(wr, mag2flam(ptl[ok],  wo),
                    'b--', lw=0.8, alpha=0.7, label='Photosphere')
            ax.plot(wr, mag2flam(pgal[ok], wo),
                    'g--', lw=0.8, alpha=0.7, label='Galaxy')
            ax.set_xlabel('Rest wavelength [Å]')
            ax.set_ylabel(r'$F_\lambda$ [erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
            ax.set_title(label)
            ax.legend(fontsize=8)

        fig.suptitle(
            f'{SOURCE_NAME}  —  TLUSTY photosphere + continuity SFH host galaxy\n'
            f'Teff={teff:.0f} K  logg={logg:.2f}  logL={logL_star:.2f}  '
            f'Av_SMC={av_smc:.2f}  logM={logmass:.2f}  σ={sigma_smooth:.0f} km/s',
            fontsize=10)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Spectra → {out_path}", flush=True)
        plt.close(fig)
    except Exception as e:
        print(f"  Spectra plot failed: {e}", flush=True)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import functools

    print(f'\n{"="*60}', flush=True)
    print(f'{SOURCE_NAME}  —  TLUSTY photosphere + continuity SFH host galaxy',
          flush=True)
    print(f'MODS R {MODSR_LO:.0f}–{MODSR_HI:.0f} Å  +  '
          f'FIRE {FIRE_LO:.0f}–{FIRE_HI:.0f} Å  |  z = {Z_SOURCE}', flush=True)
    print(f'{"="*60}\n', flush=True)

    sps, obs_mods, obs_fire, model = build_all()

    # Derive ndim and labels from the model — no hardcoded constants needed.
    NDIM   = model.ndim
    LABELS = model.theta_labels()

    print(f'\nndim={NDIM}  |  SFH bins={N_BINS_SFH}  |  ratios={N_RATIOS}',
          flush=True)
    print(f'AGEBINS log10(yr):\n{AGEBINS}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = _logl(th_mid, sps, obs_mods, obs_fire, model)
    print(f'\nMid-prior theta sample:', flush=True)
    for lbl, val in zip(LABELS, th_mid):
        print(f'  {lbl:25s}: {val:.3f}', flush=True)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    u_oob     = u_mid.copy()
    u_oob[0]  = 0.9999            # teff → near 6750 K (likely OOB for some logg)
    th_oob    = model.prior_transform(u_oob)
    ll_oob    = _logl(th_oob, sps, obs_mods, obs_fire, model)
    print(f'OOB check (teff={th_oob[0]:.0f} K) ln L = {ll_oob:.1f}', flush=True)

    # ── DynamicNestedSampler ──────────────────────────────────────────────────
    # model.prior_transform replaces the bespoke _prior_transform function.
    logl_fn = functools.partial(_logl, sps=sps,
                                 obs_mods=obs_mods, obs_fire=obs_fire, model=model)

    print('\nStarting DynamicNestedSampler …', flush=True)
    sampler = dynesty.DynamicNestedSampler(
        logl_fn, model.prior_transform, NDIM,
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

    report_lines = [
        f'{SOURCE_NAME}  —  TLUSTY photosphere + continuity SFH host galaxy',
        f'ndim={NDIM}  SFH bins={N_BINS_SFH}  z={Z_SOURCE}',
        f'Data: MODS R ({obs_mods["mask"].sum()} pix) + '
        f'FIRE ({obs_fire["mask"].sum()} pix) = {n_pix} total',
        '',
        f'  log Z    = {lz:.2f} ± {lze:.2f}',
        f'  BIC      = {bic:.2f}',
        f'  max ln L = {ll_max:.2f}',
        f'  Elapsed  = {elapsed:.0f} s',
        '',
        'Parameter posteriors (16–50–84 percentiles):',
    ]
    for i, lbl in enumerate(LABELS):
        p16, p50, p84 = np.percentile(samp[:, i], [16, 50, 84])
        report_lines.append(
            f'  {lbl:25s}: {p50:.4f}  +{p84-p50:.4f}/-{p50-p16:.4f}')
    report = '\n'.join(report_lines)
    print('\n' + report)

    bic_path = os.path.join(SCRIPT_DIR, f'egg_bic_{stem}.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'\nBIC report → {bic_path}')

    corner_path  = os.path.join(SCRIPT_DIR, f'egg_corner_{stem}.png')
    spectra_path = os.path.join(SCRIPT_DIR, f'egg_spectra_{stem}.png')
    plot_corner(samp, model, corner_path)
    plot_spectra(samp, sps, obs_mods, obs_fire, model, spectra_path)

    print('\nDone.')
