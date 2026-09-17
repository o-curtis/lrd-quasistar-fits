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
     No nebular emission.

  2. LRD PHOTOSPHERE — TLUSTYBasis (Liu+2026 TLUSTY stellar atmosphere grid,
     ~167 models at [M/H] = −2, −1, 0).  Stefan-Boltzmann normalization via
     logL_star.  LOSVD sigma_smooth broadens photosphere features only.

Dust conventions
----------------
  Both components use the same foreground ISM dust screen: dust2_gal (A_V)
  and dust_index (Noll+09 slope modifier) are applied to the photosphere via
  _calzetti_transmission, matching the law FSPS applies to the galaxy
  internally (dust_type=4).  Birth-cloud dust (dust1) is NOT applied to the
  photosphere; it affects only young galaxy stars.  No extra free parameter
  is required.  To use an independent dust law for the photosphere, set
  phot_dust_mode='independent' in build_sps() and add av_phot to build_model().

Smoothing convention (scalable across instruments)
--------------------------------------------------
  _smooth_and_interp applies a Gaussian in log-λ (constant-velocity) space
  with effective sigma:
      σ_add  = sqrt(max(0, σ_inst² − σ_lib²))
      σ_eff  = sqrt(σ_smooth² + σ_add²)

  For the PHOTOSPHERE: σ_lib = SIGMA_LIB (TLUSTY cache at R=10000, ~12.7 km/s).
    σ_smooth (free) dominates — this is the physical LOSVD from electron
    scattering in the quasi-star atmosphere.

  For the GALAXY: σ_lib = SIGMA_C3K (C3K native, ~42 km/s optical; σ_smooth=0).
    At MODS/FIRE resolution (σ_inst~14–18 km/s): σ_inst < σ_lib, so σ_add=0.
    No additional convolution is applied — the C3K library is already coarser
    than both instruments.
    At PRISM resolution (R~100, σ_inst~1270 km/s): σ_add≈1269 km/s.
    The galaxy is correctly convolved down to PRISM resolution. Scalable.

Separate metallicities
----------------------
  mh_idx controls the photosphere atmosphere composition via the TLUSTY
  grid (three [M/H] slices).  logzsol controls the galaxy stellar-population
  composition via FSPS/C3K.  These are physically distinct.

=============================================================================
FREE PARAMETERS  (14 total)
=============================================================================
  Photosphere (5):
    teff         : T_eff [K]            Uniform [3500, 6750]  TLUSTY grid extent
    logg         : log g                Uniform [-3.5, 0.5]   TLUSTY grid extent
    logL_star    : log10(L*/Lsun)       Uniform [5, 14]
    sigma_smooth : LOSVD sigma (LRD)    Uniform [115, 135] km/s
    mh_idx       : [M/H] index          Uniform [0, 2]  (0=−2, 1=−1, 2=0)

  Host galaxy (9):
    logmass      : log10(M*/Msun)       Uniform [7, 11.5]
    logzsol      : log(Z/Zsun)          Uniform [-2.0, 0.19]  C3K grid range
    dust2_gal    : shared A_V [mag]     Uniform [0, 4]  SMC applied to BOTH components
    dust1        : birth-cloud τ_V      Uniform [0, 4]  galaxy only (inside FSPS)
    logsfr_ratio_{0..4}: SFH shape      StudentT(mean=0, scale=0.3, df=2) each

=============================================================================
MULTI-INSTRUMENT DESIGN
=============================================================================
  build_obs() returns a list of obs dicts, one per spectrograph.  Each dict
  carries sigma_inst (instrument resolution in km/s), so _logl and plot_spectra
  iterate over the list without hardcoded instrument assumptions.

  To add MODS Blue or a JWST grism: append an obs dict with the correct
  sigma_inst, wave range, filename, and line mask to the list in build_obs().

  Flux calibration is assumed correct as delivered by PyPeIt — no inter-
  instrument renormalization is applied.  If MODS-R and FIRE have a flux
  offset at 7400 Å, it will appear as a visible jump in the data but will
  not prevent convergence (the model predicts both ranges simultaneously).
  For grism data with uncertain absolute calibration, add a multiplicative
  scale factor per spectrograph as a free parameter (isfree=True in
  model_params).

=============================================================================
OUTPUTS  (written to egg_analysis/)
=============================================================================
  dynesty_{SOURCE_TAG}_prospector_tlusty.pkl   — full dynesty results object
  chain_{SOURCE_TAG}_prospector_tlusty.npy     — equal-weight posterior samples
  corner_{SOURCE_TAG}_prospector_tlusty.png    — corner plot
  spectra_{SOURCE_TAG}_prospector_tlusty.png   — data/model/components overlay
  bic_{SOURCE_TAG}_prospector_tlusty.txt       — BIC, log Z, and posteriors
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

warnings.filterwarnings('ignore', category=RuntimeWarning)   # StudentT len mismatch + FSPS numerics

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
from prospect.models.priors     import Uniform, StudentT, ClippedNormal
from prospect.models.transforms import logsfr_ratios_to_masses, zred_to_agebins


# =============================================================================
# SOURCE CONFIGURATION  ←  change these when running on a different target
# =============================================================================
SOURCE_NAME  = 'TheEgg'          # used in plot titles and BIC reports
SOURCE_TAG   = 'egg'             # short token for filenames; change per target
Z_SOURCE     = 0.1007            # spectroscopic redshift
F_SYS        = 0.05              # systematic flux error floor (fraction of |flux|)

# Photosphere metallicity is now a FREE parameter (mh_idx ∈ [0, 2]).
# mh_idx mapping: 0=[M/H]=−2, 1=[M/H]=−1, 2=[M/H]=0 (linear interpolation between slices).
# Separable bilinear interpolation in tlusty_basis.py handles arbitrary mh_idx correctly.
MH_TAG = 'mhfree'

MODSB_LO     = 2715.0            # rest-frame Å — MODS-B fit window; full blue extent (noise at <3000 Å won't drive fit)
MODSB_HI     = 5500.0            # handoff from MODS-B to MODS-R
MODSR_LO     = 5500.0            # rest-frame Å — MODS-R fit window (G670L); 5500 avoids overlap with MODS-B
MODSR_HI     = 7400.0
FIRE_LO      = 7400.0
FIRE_HI      = 22886.0   # full FIRE array; Brγ (21660) + K-band tail; CO bandheads start ~22935
# Flux re-scaling: FIRE is 9–10% fainter than MODS-R in the 7550–8450 Å overlap.
# Measured from median flux ratios in three emission-line-free windows:
#   7550–7900 Å: FIRE/MODS-R = 0.9043  (-9.6%)
#   7900–8200 Å: FIRE/MODS-R = 0.8396  (-16.0%, possible telluric contamination)
#   8200–8450 Å: FIRE/MODS-R = 0.9064  (-9.4%)
# Robust median: 0.9043; correction = 1/0.9043 = 1.1057.
# MODS-B vs MODS-R in 5100–5900 Å overlap: +0.97% — no correction applied.
SCALE_FIRE   = 1.0 / 0.9043   # multiply FIRE flux and unc by this factor

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, f'{SOURCE_TAG}_data',
                             'Re_ J1025+1402 LBT _ Magellan spectra')
MODSB_FNAME  = 'J1025+1402_MODSB_coadd1d_tellcorr_slitcorr_dered.fits'
MODS_FNAME   = 'J1025+1402_MODSR_coadd1d_tellcorr_slitcorr_dered.fits'
FIRE_FNAME   = 'J1025+1402_FIRE_coadd1d_tellcorr_calib_dered.fits'

SPEC_DIR     = ('/home/omc5226/work/lrdmesa/liu2026_library/'
                'LRD_synthetic_spectral_library-main/specs')

# ── Instrument resolving powers ───────────────────────────────────────────────
R_MODSB  = 1850     # MODS-B G400L, 1.0" slit
R_MODSR  = 2250     # MODS-R G670L, 1.0" slit
R_MODS   = R_MODSR  # backward-compat alias used in SIGMA_MODS below
R_FIRE   = 6000     # CANONICAL: Magellan/FIRE echelle delivered R (0.6" slit; verified vs
                    # MIT FIRE instrument specs). All future Egg runs MUST use 6000 — do not
                    # revert to the old placeholder 7000 (see X. Lin, priv. comm.).
TARGET_R = 10000                          # TLUSTYBasis cache resolution

# TLUSTY cache is source-independent (same library for all targets).
# Named by library resolution only so it is shared across all fitting runs.
TLUSTY_CACHE  = os.path.join(SCRIPT_DIR, f'tlusty_cache_r{TARGET_R // 1000}k_40k.npz')
XI10_CACHE    = os.path.join(SCRIPT_DIR, f'tlusty_cache_r{TARGET_R // 1000}k_40k_v10.npz')
_CKMS    = 2.998e5                        # km/s

# Spectral resolution σ = c / (R × 2.355) in km/s
SIGMA_LIB  = _CKMS / (TARGET_R * 2.355)  # TLUSTY library  σ ≈ 12.7 km/s
SIGMA_MODSB = _CKMS / (R_MODSB  * 2.355) # MODS-B G400L    σ ≈ 68.8 km/s
SIGMA_MODSR = _CKMS / (R_MODSR  * 2.355) # MODS-R G670L    σ ≈ 56.5 km/s
SIGMA_MODS  = SIGMA_MODSR                # backward-compat alias
SIGMA_FIRE  = _CKMS / (R_FIRE   * 2.355) # FIRE            σ ≈ 21.2 km/s (R=6000)

# C3K native spectral resolution used for galaxy component smoothing.
# C3K R~3000 in the optical → σ~42 km/s.  This is the correct library sigma
# to pass to _smooth_and_interp for the galaxy so that:
#   • At MODS/FIRE (σ_inst~14–18 km/s < σ_C3K): σ_add=0, no convolution.
#   • At PRISM (σ_inst~1270 km/s >> σ_C3K): σ_add≈1269 km/s, correctly
#     convolves the galaxy spectrum down to prism resolution.
SIGMA_C3K  = _CKMS / (3000.0  * 2.355)  # C3K optical     σ ≈ 42.4 km/s

# ── CGS / unit constants ──────────────────────────────────────────────────────
_lsun         = 3.846e33      # erg/s
_pc_cm        = 3.0857e18     # cm/pc
_jansky_cgs   = 1e-23         # erg/s/Hz/cm²
_c_AA         = 2.998e18      # Å/s
_to_cgs       = _lsun / (4.0 * np.pi * (10.0 * _pc_cm)**2)  # Lsun→erg/s/cm² @10pc
_PYPEIT_SCALE = 1e-17         # erg/s/cm²/Å per PyPeIt unit

# ── Rest-frame emission-line masks ─────────────────────────────────────────────
# Standard rest-frame mask list for the LRD MESA spectroscopic programme.
# Applied to all sources; no modification needed when changing Z_SOURCE because
# masks are applied after computing wave_rest = wave_obs / (1+z).
#
# Mask widths are consistent with prior single-component TLUSTY fits:
#   MODS-R: fit_egg_modsr.py, fit_egg_mods_joint.py
#   FIRE  : fit_egg_fire.py   (note ±150 Å on [S III] to protect Pa-series)
#
# Broad Balmer wings (Hα ±150 Å ≈ ±6800 km/s, Hβ ±90 Å ≈ ±5500 km/s) cover
# the BLR emission that the photosphere model cannot reproduce.  Narrow
# forbidden lines use tighter windows.  He I 10830 uses ±250 Å because it is
# known to be extremely broad in LRDs (possibly P Cygni wind absorption).
# Hα (6563 Å) is listed in FIRE_LINES for generality at higher z; it falls
# outside FIRE coverage for The Egg (z=0.1007 → λ_obs = 7224 Å < FIRE start).
# H-K telluric gap at rest-frame 16500–17500 Å (SNR≈1.4) is masked via a
# wide ±500 Å window centred at 17000 Å in FIRE_LINES.
MODS_LINES = [
    # ── Broad Balmer (BLR origin — generous masks) ───────────────────────────
    (6563, 150),   # Hα  ±6800 km/s (widened: broad wings visible ±150 Å)
    (4861,  90),   # Hβ  (also covers He I 4922 at ±90)
    (4340,  50),   # Hγ  (also covers [O III] 4363 at ±50)
    (4102,  30),   # Hδ
    (3970,  30),   # Hε / Ca H blend; also covers [Ne III] 3967
    # ── Higher Balmer (near Balmer break) — CRITICAL ─────────────────────────
    # Unmasked Balmer emission pushes sigma_smooth to unphysical values and
    # creates a pseudo-continuum that depresses the apparent Balmer break.
    (3889,  30),   # H8  = n=8→2; overlaps edge of [Ne III] 3869 mask
    (3835,  40),   # H9  = n=9→2
    (3798,  40),   # H10 = n=10→2
    (3771,  40),   # H11 = n=11→2
    # H12 (3750) and H13 (3734) are inside the H11 ±40 mask above.
    # H14–H17 (3722–3697 Å) fall just below the H11 mask and must be masked
    # separately; the dominant BLR emission here creates −6σ spikes in residuals.
    (3711,  30),   # H14–H17 = n=14→2 through n=17→2 (3697–3722 Å)
    # ── He I optical forest ───────────────────────────────────────────────────
    (3820,  20),   # He I 3819.6
    (4026,  20),   # He I 4026.2
    (4471,  30),   # He I 4471.5  (widened ±20→±30: −17σ spike at 4491 = old mask edge)
    # ── High-ionization (AGN/LRD) ────────────────────────────────────────────
    (3095,  20),   # [Fe IV] 3094.96
    (3346,  20),   # [Ne V] 3346
    (3426,  20),   # [Ne V] 3426
    # ── [S II] doublet (pre-Hδ region) ───────────────────────────────────────
    (4068,  15),   # [S II] 4068.6
    (4076,  15),   # [S II] 4076.4
    # ── [Fe V] coronal ────────────────────────────────────────────────────────
    (4227,  20),   # [Fe V] 4227.2
    # ── [Fe III] + Bowen N III fluorescence ──────────────────────────────────
    (4288,  15),   # [Fe III] 4287.4 forbidden emission (+8.5σ unmasked spike)
    (4416,  12),   # N III Bowen fluorescence λ4413–4417 (+7.9σ unmasked spike)
    # ── [Ar IV]+[Ne IV] complex — CRITICAL ───────────────────────────────────
    # Between He II 4686 and Hβ; unmasked these drive sigma_smooth unphysical.
    (4711,  20),   # [Ar IV] 4711.3
    (4724,  20),   # [Ne IV] 4724.2
    (4740,  20),   # [Ar IV] 4740.1
    # ── Narrow forbidden ─────────────────────────────────────────────────────
    (5007,  30),   # [O III] 5007; also covers He I 5015
    (4959,  20),   # [O III] 4959
    (6583,  25),   # [N II] 6583
    (6548,  20),   # [N II] 6548
    (6716,  20),   # [S II] 6716
    (6731,  20),   # [S II] 6731
    (6300,  20),   # [O I] 6300
    (6364,  15),   # [O I] 6364; also covers [Fe X] 6375
    (7065,  20),   # He I 7065 — +9σ spike between [S II] and [Ar III]
    (7136,  25),   # [Ar III] 7136; widened to ±25 to cover 7157 Å red wing
    (7325,  35),   # [O II] 7320/7330 doublet; also covers [Ar IV] 7332
    (3727,  30),   # [O II] 3727/3729 doublet
    (3869,  20),   # [Ne III] 3869; also covers He I 3867
    (9069,  40),   # [S III] 9069 — outside MODS-R for The Egg; kept for generality
    (9532,  40),   # [S III] 9532
    # ── Additional forbidden/recombination ───────────────────────────────────
    (5199,  15),   # [N I] 5197.9/5200.3 doublet (combined)
    (5528,  25),   # [Cl III] 5517.7+5537.9 doublet (combined)
    (5577,  15),   # O I 5577.3 (telluric sky residual)
    (5755,  20),   # [N II] 5754.6 auroral
    (6312,  15),   # [S III] 6312.1
    (6678,  20),   # He I 6678.2
    (7006,  20),   # [Ar V] 7005.8
    (7281,  20),   # He I 7281.4
    # ── He recombination ─────────────────────────────────────────────────────
    (5876,  25),   # He I 5876
    (4686,  25),   # He II 4686
]
FIRE_LINES = [
    (10830, 130),  # He I 1.083 µm — blue edge at 10700 Å
    (10938, 170, 55),  # Pa γ = n=6→3
    (12820, 720, 535),  # Pa β + J-H gap: 12100–13355 Å (full J-H telluric band)
    ( 9015,  18),  # Pa η = n=10→3; narrowed from ±75 (thin line)
    ( 9180,  15),  # unidentified NIR feature at 9180 Å (not in NJC list; 4σ spike blueward of Pa ζ)
    ( 9229,  26),  # Pa ζ = n=9→3; red edge at 9255 Å
    ( 9546,  20),  # Pa ε = n=8→3; thin line
    (10105, 130, 65),  # Pa δ = n=7→3; 9975–10234 Å (blue edge at 9975, red narrowed ~30 Å total)
    (10400,  10,  20),  # unidentified NIR feature; 10390–10420 Å
    # ── [S III] ───────────────────────────────────────────────────────────────
    ( 9069,  18),  # [S III] 9069; narrowed from ±100 (thin line)
    ( 9532,  20),  # [S III] 9532; thin line
    # ── Other NIR lines ───────────────────────────────────────────────────────
    ( 7751,  25),  # [Ar III] 7751.1
    ( 8446,  15),  # O I 8446 triplet (emission confined to ±8 Å; ±15 conservative)
    (11287,  60),  # O I 1.1287 µm fluorescence (just red of He I+Paγ mask)
    # ── Extended NIR (H-band, K-band) ─────────────────────────────────────────
    (17000, 675, 500),  # H-K telluric gap: 16325–17500 Å
    (18751, 250),  # Pa α = n=4→3 — strong in LRDs; ±250 Å ≈ ±4000 km/s
    (18403,  60),  # OH sky residual at 1.840 µm (between H-K telluric gap and Pa α)
    (19446, 200),  # Br δ = n=8→4
    (21660, 200),  # Br γ = n=7→4 (in K-band; only present when FIRE_HI ≥ 21660)
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
# Note: pass _AGEBINS_INIT.copy() to zred_to_agebins to avoid in-place mutation.
AGEBINS = zred_to_agebins(zred=Z_SOURCE, agebins=_AGEBINS_INIT.copy())

# ── Dynesty settings ──────────────────────────────────────────────────────────
NLIVE       = 600
NLIVE_INIT  = 400
NLIVE_BATCH = 600
DLOGZ_INIT  = 0.01
N_EFFECTIVE = 8000

# ── Plot colours (matching fits_broadband/final/ convention) ──────────────────
_C_MOD = '#AA3377'   # total model
_C_PHO = '#CC3311'   # LRD photosphere component
_C_GAL = '#CCAA00'   # host galaxy component


# =============================================================================
# Import tlusty_basis without triggering FSPS-dependent module-level code
# =============================================================================

def _import_tlusty_module():
    """Load tlusty_basis.py directly and return the module.

    Direct loading avoids prospect.sources.__init__ importing agnssp_basis,
    which inspects SPS_HOME at import time.  We set SPS_HOME above, but the
    import-order dependency is fragile; loading via importlib.util is robust.
    Both TLUSTYBasis and TLUSTYPlusGalaxyBasis are accessed from the returned
    module object.
    """
    spec = importlib.util.spec_from_file_location('tlusty_basis', _TLUSTY_MODULE)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# build_sps / build_obs / build_model / build_all
# =============================================================================

def build_sps(spec_dir=SPEC_DIR, target_R=TARGET_R,
              cache_file=TLUSTY_CACHE, include_xi10=False, verbose=True, **kwargs):
    """Instantiate TLUSTYPlusGalaxyBasis (defined in tlusty_basis.py).

    include_xi10=True loads the v10 (ξ=10 km/s) supplemental cache so that
    xi_mtb becomes a meaningful free parameter (M2/M3 model variants).
    """
    mod = _import_tlusty_module()
    xi10 = XI10_CACHE if include_xi10 else None
    return mod.TLUSTYPlusGalaxyBasis(
        spec_dir=spec_dir, target_R=target_R,
        cache_file=cache_file, xi10_cache_file=xi10, verbose=verbose,
        phot_dust_mode='shared', phot_dust_law='smc')


def build_obs(data_dir=DATA_DIR, modsb_fname=MODSB_FNAME,
              mods_fname=MODS_FNAME, fire_fname=FIRE_FNAME, **kwargs):
    """
    Load and pre-process The Egg spectra (MODS-B + MODS-R + FIRE).

    Returns a list of obs dicts — one per spectrograph — so that _logl and
    plot_spectra can iterate over an arbitrary number of instruments without
    hardcoded assumptions.

    Each obs dict contains:
        wave_obs  : observed-frame wavelengths (Å)
        wave_rest : rest-frame wavelengths (Å)
        flux      : flux in maggies (F_ν / 3631 Jy)
        unc       : uncertainty in maggies
        mask      : True = pixel included in the fit
        sigma_inst: instrument spectral resolution σ in km/s
        name      : label string

    Processing pipeline
    -------------------
    1. Read PyPeIt coadd FITS; honour the pipeline pixel-quality mask.
    2. Add systematic error floor F_SYS in quadrature to pipeline σ.
    3. Convert f_λ [PyPeIt units = 1e-17 erg/s/cm²/Å] to maggies (F_ν / 3631 Jy).
    4. Apply rest-frame emission-line masks (MODS_LINES, FIRE_LINES).
    5. Clip to instrument wavelength range and require finite, positive σ.

    Flux calibration assumption
    ---------------------------
    FIRE fluxes are multiplied by SCALE_FIRE (= 1/0.9043 ≈ 1.106) to correct
    a ~10% calibration offset relative to MODS-R measured in the 7550–8450 Å
    overlap region.  MODS-B and MODS-R agree to <1% in their 5100–5900 Å
    overlap and need no correction.

    Emission-line masking
    ---------------------
    Neither model component (TLUSTY photosphere, FSPS galaxy) produces
    emission lines.  Unmasked line pixels would force the model continuum to
    over-predict the emission, biasing logL_star and dust parameters upward.
    Masking the lines loses no continuum information because the windows are
    narrow relative to the total spectral coverage.
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
        """Convert f_λ [PyPeIt 1e-17 erg/s/cm²/Å] to F_ν maggies."""
        f_lam = f_lam_pypeit * _PYPEIT_SCALE   # erg/s/cm²/Å
        f_nu  = f_lam * w_obs_aa**2 / _c_AA    # erg/s/cm²/Hz
        return f_nu / (3631.0 * _jansky_cgs)   # maggies

    def make_line_mask(wave_rest, lines):
        m = np.ones(len(wave_rest), dtype=bool)
        for entry in lines:
            if len(entry) == 3:
                cen, hw_lo, hw_hi = entry
            else:
                cen, hw_lo = entry
                hw_hi = hw_lo
            m &= ~((wave_rest >= cen - hw_lo) & (wave_rest <= cen + hw_hi))
        return m

    wb_obs, fb, sb, gb = load_pypeit(modsb_fname)
    wm_obs, fm, sm, gm = load_pypeit(mods_fname)
    wf_obs, ff, sf, gf = load_pypeit(fire_fname)

    wb_rest = wb_obs / (1.0 + Z_SOURCE)
    wm_rest = wm_obs / (1.0 + Z_SOURCE)
    wf_rest = wf_obs / (1.0 + Z_SOURCE)

    # Systematic error floor in quadrature (in f_λ units before conversion)
    sb = np.sqrt(sb**2 + (F_SYS * np.abs(fb))**2)
    sm = np.sqrt(sm**2 + (F_SYS * np.abs(fm))**2)
    sf = np.sqrt(sf**2 + (F_SYS * np.abs(ff))**2)

    fb_mag = flam_to_maggies(wb_obs, fb)
    sb_mag = flam_to_maggies(wb_obs, sb)
    fm_mag = flam_to_maggies(wm_obs, fm)
    sm_mag = flam_to_maggies(wm_obs, sm)
    ff_mag = flam_to_maggies(wf_obs, ff) * SCALE_FIRE
    sf_mag = flam_to_maggies(wf_obs, sf) * SCALE_FIRE

    range_b = (wb_rest >= MODSB_LO) & (wb_rest <= MODSB_HI)
    range_m = (wm_rest >= MODSR_LO) & (wm_rest <= MODSR_HI)
    range_f = (wf_rest >= FIRE_LO)  & (wf_rest <= FIRE_HI)
    lines_b = make_line_mask(wb_rest, MODS_LINES)
    lines_m = make_line_mask(wm_rest, MODS_LINES)
    lines_f = make_line_mask(wf_rest, FIRE_LINES)
    good_b  = gb & np.isfinite(fb_mag) & np.isfinite(sb_mag) & (sb_mag > 0)
    good_m  = gm & np.isfinite(fm_mag) & np.isfinite(sm_mag) & (sm_mag > 0)
    good_f  = gf & np.isfinite(ff_mag) & np.isfinite(sf_mag) & (sf_mag > 0)

    obs_modsb = dict(
        wave_obs=wb_obs, wave_rest=wb_rest,
        flux=fb_mag, unc=sb_mag,
        mask=range_b & lines_b & good_b,
        sigma_inst=SIGMA_MODSB,
        name='MODS_B',
    )
    obs_mods = dict(
        wave_obs=wm_obs, wave_rest=wm_rest,
        flux=fm_mag, unc=sm_mag,
        mask=range_m & lines_m & good_m,
        sigma_inst=SIGMA_MODSR,
        name='MODS_R',
    )
    obs_fire = dict(
        wave_obs=wf_obs, wave_rest=wf_rest,
        flux=ff_mag, unc=sf_mag,
        mask=range_f & lines_f & good_f,
        sigma_inst=SIGMA_FIRE,
        name='FIRE',
    )

    obs_list = [obs_modsb, obs_mods, obs_fire]
    n_pix_total = sum(o['mask'].sum() for o in obs_list)
    for obs in obs_list:
        print(f"  {obs['name']} fit pixels: {obs['mask'].sum()}", flush=True)
    print(f"  Total fit pixels: {n_pix_total}", flush=True)
    return obs_list


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
    warning is suppressed by warnings.filterwarnings('ignore', RuntimeWarning) at module level
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
        'logL_star': {
            'N': 1, 'isfree': True, 'init': float(logL_init),
            'prior': Uniform(mini=5., maxi=14.),
            'units': 'log10(L_star / L_sun)',
        },
        'sigma_smooth': {
            'N': 1, 'isfree': True, 'init': 100.,
            'prior': Uniform(mini=115., maxi=135.),
            'units': 'km/s — LOSVD sigma, photosphere absorption only',
        },
        'mh_idx': {
            'N': 1, 'isfree': True, 'init': 1.0,
            'prior': Uniform(mini=0., maxi=2.),
            'units': '[M/H] index: 0=[M/H]=-2, 1=[M/H]=-1, 2=[M/H]=0',
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
            'units': 'shared A_V [mag] — SMC applied to both photosphere and galaxy',
        },
        'dust1': {
            'N': 1, 'isfree': True, 'init': 0.0,
            'prior': Uniform(mini=0., maxi=4.),
            'units': 'birth-cloud optical depth tau_V (galaxy stars < 30 Myr)',
        },
        'logsfr_ratios': {
            'N': N_RATIOS, 'isfree': True, 'init': np.zeros(N_RATIOS),
            'prior': StudentT(mean=0., scale=0.3, df=2),
            'units': 'log10(SFR_j / SFR_{j+1}), j=0 youngest (continuity SFH)',
        },

        # ── Fixed parameters ──────────────────────────────────────────────
        'sfh':               {'N': 1, 'isfree': False, 'init': 3},
        'imf_type':          {'N': 1, 'isfree': False, 'init': 2},    # Chabrier
        'dust_type':         {'N': 1, 'isfree': False, 'init': 4},    # Noll+09 for dust1
        'dust1_index':       {'N': 1, 'isfree': False, 'init': -1.0}, # C&F 2000
        'add_neb_emission':  {'N': 1, 'isfree': False, 'init': False},
        'add_neb_continuum': {'N': 1, 'isfree': False, 'init': False},
        'add_dust_emission': {'N': 1, 'isfree': False, 'init': False},
        'agebins':           {'N': N_BINS_SFH, 'isfree': False, 'init': agebins},
        'mass':              {'N': 1,  'isfree': False, 'init': 1e9,
                              'units': 'per-bin array; set by logsfr_ratios_to_masses in _logl'},
        'zred':              {'N': 1,  'isfree': False, 'init': float(zred)},
        'lumdist':           {'N': 1,  'isfree': False, 'init': float(lumdist),
                              'units': 'Mpc — luminosity distance at zred'},
    }
    return ProspectorParams(model_params)


def build_model_variant(model_type='M1', sigma_mode='free', zsol_mode='free',
                        mh_mode='free', neb_mode='off', **kwargs):
    """
    Build a ProspectorParams for one of the model-finalization variants.

    model_type : 'M1' (base), 'M2' (+xi_mtb), 'M3' (+xi_mtb+duste)
    sigma_mode : 'free' → Uniform(115,135) km/s | 'fixed' → 123.0 km/s (not in theta)
    zsol_mode  : 'free' → Uniform(-2, 0.19)      | 'fixed' → derived = mh_idx−2 (not in theta)
    mh_mode    : 'free' → Uniform(0,2)            | 'fixed' → 1.0 ([M/H]=−1, Liu+2026)
    neb_mode   : 'off'  → add_neb_emission=False  | 'on'   → add_neb_emission=True,
                          gas_logu free Uniform(-4,-1), gas_logz fixed=-1.0
    """
    if model_type not in ('M1', 'M2', 'M3'):
        raise ValueError(f"model_type must be M1/M2/M3, got '{model_type}'")
    if sigma_mode not in ('free', 'fixed'):
        raise ValueError(f"sigma_mode must be free/fixed, got '{sigma_mode}'")
    if zsol_mode not in ('free', 'fixed'):
        raise ValueError(f"zsol_mode must be free/fixed, got '{zsol_mode}'")
    if mh_mode not in ('free', 'fixed'):
        raise ValueError(f"mh_mode must be free/fixed, got '{mh_mode}'")
    if neb_mode not in ('off', 'on'):
        raise ValueError(f"neb_mode must be off/on, got '{neb_mode}'")

    base = build_model(**kwargs)
    mp   = base.config_dict

    if sigma_mode == 'fixed':
        mp['sigma_smooth']['isfree'] = False
        mp['sigma_smooth']['init']   = 123.0

    if zsol_mode == 'fixed':
        mp['logzsol']['isfree'] = False
        mp['logzsol']['init']   = -1.0     # placeholder; _logl computes mh_idx−2

    if mh_mode == 'fixed':
        mp['mh_idx']['isfree'] = False
        mp['mh_idx']['init']   = 1.0       # [M/H]=−1 (Liu+2026 photosphere metallicity)

    if model_type in ('M2', 'M3'):
        mp['xi_mtb'] = {
            'N': 1, 'isfree': True, 'init': 2.0,
            'prior': Uniform(mini=2.0, maxi=10.0),
            'units': 'km/s — microturbulence; v10 grid active at T=4000-5000K, [M/H]=-1 only',
        }

    if model_type == 'M3':
        mp['add_dust_emission'] = {'N': 1, 'isfree': False, 'init': True}
        mp['duste_umin'] = {
            'N': 1, 'isfree': True, 'init': 1.0,
            'prior': Uniform(mini=0.1, maxi=25.),
            'units': 'Draine & Li U_min',
        }
        mp['duste_qpah'] = {
            'N': 1, 'isfree': True, 'init': 2.0,
            'prior': ClippedNormal(mean=2.0, sigma=2.0, mini=0.0, maxi=7.0),
            'units': 'PAH mass fraction (%)',
        }
        mp['duste_gamma'] = {
            'N': 1, 'isfree': True, 'init': 0.01,
            'prior': Uniform(mini=0.0, maxi=0.15),
            'units': 'Fraction of dust in high-U PDRs',
        }

    if neb_mode == 'on':
        mp['add_neb_emission']  = {'N': 1, 'isfree': False, 'init': True}
        mp['add_neb_continuum'] = {'N': 1, 'isfree': False, 'init': True}
        # gas_logu: ionization parameter free (Liu+2026 prior [-4,-1], best-fit -2)
        mp['gas_logu'] = {
            'N': 1, 'isfree': True, 'init': -2.0,
            'prior': Uniform(mini=-4.0, maxi=-1.0),
            'units': 'log ionization parameter (gas_logu)',
        }
        # gas_logz: nebular metallicity fixed at -1 (matching [M/H]=-1 atmosphere/galaxy)
        # Emission lines are masked, so gas_logz only affects nebular continuum shape.
        mp['gas_logz'] = {'N': 1, 'isfree': False, 'init': -1.0}

    m = ProspectorParams(mp)
    m.configure(reset=True)
    return m


def build_all(**kwargs):
    print("Building SPS …", flush=True)
    sps = build_sps(**kwargs)
    print("Loading observations …", flush=True)
    obs_list = build_obs(**kwargs)
    print("Building model …", flush=True)
    model = build_model(**kwargs)
    return sps, obs_list, model


# =============================================================================
# Forward-model utilities
# =============================================================================

def _flux_norm(zred, lumdist):
    """
    Lsun/Hz → maggies conversion factor.

    Matches Prospector's SpecModel.flux_norm() exactly (mass=1 assumed; caller
    applies mass separately):
        norm = (1+z) × to_cgs / (3631 Jy) / (d_L / 10 pc)²

    Derivation: F_ν_obs(λ_obs) = (1+z) × L_ν_em(λ_rest) / (4π d_L²)
    The (1+z) factor accounts for photon energy redshift.  The spectrum values
    are F_ν_obs on the rest-frame wavelength grid; relabelling the x-axis to
    λ_obs = λ_rest × (1+z) does NOT change the y-values (see _smooth_and_interp).

    Verified against Prospector sedmodel.py lines 418–438:
        unit_conversion = to_cgs / (3631 * jansky_cgs) * (1 + self._zred)
        return mass * unit_conversion / dfactor
    Our _flux_norm returns the same factor without the mass term.
    """
    dfactor   = (lumdist * 1e5)**2              # (d_L / 10 pc)²
    unit_conv = _to_cgs / (3631.0 * _jansky_cgs) * (1.0 + zred)
    return unit_conv / dfactor


def _smooth_and_interp(wave_rest, spec_lsun_hz, obs_wave_obs,
                       sigma_smooth_kms, sigma_inst_kms, sigma_lib_kms,
                       zred, lumdist):
    """
    Convert Lsun/Hz → maggies, apply Gaussian smoothing in constant-velocity
    (log-λ) space, redshift the wavelength grid, and interpolate onto the
    observed-frame data wavelength array.

    Flux normalization convention (Prospector-consistent)
    ------------------------------------------------------
    _flux_norm() returns F_ν_obs = (1+z) × L_ν × to_cgs / (3631 Jy) / (d_L/10pc)².
    After multiplying spec_lsun_hz × norm, the result is F_ν_obs [maggies] on
    the REST-FRAME wavelength grid.  Relabelling the x-axis to the observed
    frame (wave_obs = wave_rest × (1+z)) does NOT change the y-values — they
    remain F_ν_obs.  There is no additional (1+z) divisor.

    This matches exactly what Prospector's SpecModel.predict_spec does:
        _norm_spec  = spec × flux_norm()          # F_ν_obs on rest-frame grid
        obs_wave    = wave_rest × (1+z)           # x-axis relabelled
        inst_spec   = instrumental_smoothing(obs_wave, _norm_spec)  # no /(1+z)

    Smoothing
    ---------
    The effective sigma applied is:
        σ_add  = sqrt(max(0, σ_inst² − σ_lib²))
        σ_eff  = sqrt(σ_smooth² + σ_add²)
    Smoothing is skipped when σ_eff < 0.5 km/s.

    For the photosphere: pass sigma_lib=SIGMA_LIB, sigma_smooth=sigma_smooth.
    For the galaxy:      pass sigma_lib=SIGMA_C3K, sigma_smooth=0.0.
      At MODS/FIRE: σ_C3K (42 km/s) > σ_inst (14-18 km/s) → σ_add=0, no conv.
      At PRISM R~100: σ_PRISM (~1270 km/s) >> σ_C3K → correct large conv.

    Parameters
    ----------
    wave_rest        : rest-frame wavelength grid, log-spaced (Å)
    spec_lsun_hz     : spectrum in Lsun/Hz
    obs_wave_obs     : observed-frame wavelength array of the spectrograph (Å)
    sigma_smooth_kms : LOSVD sigma to add in quadrature (km/s)
    sigma_inst_kms   : instrument resolution σ (km/s)
    sigma_lib_kms    : library resolution σ (km/s) — SIGMA_LIB or SIGMA_C3K
    zred             : source redshift
    lumdist          : luminosity distance in Mpc (from model.params['lumdist'])

    Returns
    -------
    spec_obs : ndarray, same length as obs_wave_obs, in maggies; NaN outside range
    """
    sigma_add2 = max(0.0, sigma_inst_kms**2 - sigma_lib_kms**2)
    sigma_eff  = np.sqrt(sigma_smooth_kms**2 + sigma_add2)

    norm = _flux_norm(zred, lumdist)
    spec = spec_lsun_hz * norm           # F_ν_obs [maggies] on rest-frame grid

    if sigma_eff > 0.5:
        dloglam   = np.log(wave_rest[1] / wave_rest[0])
        sigma_pix = (sigma_eff / _CKMS) / dloglam
        spec = gaussian_filter1d(spec, sigma_pix)

    # Relabel x-axis: rest-frame → observed-frame.  y-values unchanged.
    wave_obs_model = wave_rest * (1.0 + zred)
    return np.interp(obs_wave_obs, wave_obs_model, spec,
                     left=np.nan, right=np.nan)


# =============================================================================
# Log-likelihood
# =============================================================================

def _logl(theta, sps, obs_list, model):
    """
    Log-likelihood for the 14-parameter two-component model.

    Parameter extraction uses model.theta_index — a read-only dict set at
    init that maps each free parameter name to its slice in the theta vector.
    Fixed parameters (agebins, zred, lumdist) are read from
    model.params, which holds the init values and is never mutated inside
    this function.  This design is safe for dynesty's multiprocessing pool
    (each worker inherits a copy of model at fork time; no shared mutable state).

    sigma_smooth is applied only to spec_tl (photosphere).
    The galaxy uses SIGMA_C3K as sigma_lib, giving zero additional smoothing
    at MODS/FIRE resolution and correct convolution at PRISM resolution.

    obs_list is iterated so the function scales to any number of spectrographs.
    Each obs dict must contain: wave_obs, flux, unc, mask, sigma_inst.
    """
    idx = model.theta_index       # read-only after ProspectorParams.__init__

    # ── Photosphere ───────────────────────────────────────────────────────────
    teff         = float(theta[idx['teff']])
    logg         = float(theta[idx['logg']])
    logL_star    = float(theta[idx['logL_star']])
    sigma_smooth = (float(theta[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                    else float(model.params['sigma_smooth']))
    mh_idx       = (float(theta[idx['mh_idx']]) if 'mh_idx' in idx
                    else float(model.params['mh_idx']))
    xi_mtb       = (float(theta[idx['xi_mtb']]) if 'xi_mtb' in idx
                    else float(model.params.get('xi_mtb', 2.0)))

    # ── Galaxy ────────────────────────────────────────────────────────────────
    logmass       = float(theta[idx['logmass']])
    logzsol       = (float(theta[idx['logzsol']]) if 'logzsol' in idx
                     else mh_idx - 2.0)
    dust2_gal     = float(theta[idx['dust2_gal']])
    dust1         = float(theta[idx['dust1']])
    logsfr_ratios = theta[idx['logsfr_ratios']]    # ndarray, length N_RATIOS

    # ── Fixed (read-only from model.params) ───────────────────────────────────
    agebins = np.array(model.params['agebins'])    # (N_BINS_SFH, 2)
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])

    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)

    add_dust_emission = 'duste_umin' in idx
    add_neb_emission  = 'gas_logu' in idx
    if add_neb_emission:
        gas_logu = float(theta[idx['gas_logu']])
        gas_logz = float(model.params.get('gas_logz', -1.0))
    sps_params = dict(
        teff=teff, logg=logg, mh_idx=mh_idx, xi_mtb=xi_mtb,
        logL_star=logL_star,
        logmass=logmass, logzsol=logzsol,
        dust2_gal=dust2_gal, dust1=dust1,
        logsfr_ratios=logsfr_ratios, mass=mass, agebins=agebins,
        sfh=int(model.params['sfh']),
        imf_type=int(model.params['imf_type']),
        dust_type=int(model.params['dust_type']),
        dust1_index=float(model.params['dust1_index']),
        add_neb_emission=add_neb_emission,
        add_neb_continuum=add_neb_emission,
        add_dust_emission=add_dust_emission, zred=zred,
    )
    if add_neb_emission:
        sps_params['gas_logu'] = gas_logu
        sps_params['gas_logz'] = gas_logz
    if add_dust_emission:
        sps_params['duste_umin']  = float(theta[idx['duste_umin']])
        sps_params['duste_qpah']  = float(theta[idx['duste_qpah']])
        sps_params['duste_gamma'] = float(theta[idx['duste_gamma']])

    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_params)

    # NaN from TLUSTYBasis signals teff outside the clipped T range (should not occur).
    if np.any(np.isnan(spec_tl)):
        return -1e300

    lnp = 0.0
    for obs in obs_list:
        sigma_inst = obs['sigma_inst']

        pred_tl  = _smooth_and_interp(wave, spec_tl,  obs['wave_obs'],
                                       sigma_smooth, sigma_inst, SIGMA_LIB,
                                       zred, lumdist)
        pred_gal = _smooth_and_interp(wave, spec_gal, obs['wave_obs'],
                                       0.0, sigma_inst, SIGMA_C3K,
                                       zred, lumdist)
        pred = pred_tl + pred_gal
        mask = obs['mask'] & np.isfinite(pred) & (pred > 0)
        if mask.sum() < 10:
            return -1e300

        res     = obs['flux'][mask] - pred[mask]
        cat_sel = ((obs['wave_rest'][mask] >= 8400.) &
                   (obs['wave_rest'][mask] <= 8750.))
        weights = np.where(cat_sel, 15.0, 1.0)
        lnp += -0.5 * np.sum(weights * (res / obs['unc'][mask])**2)

    return float(lnp)


# =============================================================================
# Plotting helpers
# =============================================================================

def _make_sps_params(theta_or_med, model):
    """
    Extract all sps_params from a theta vector and the fixed model params.
    Used by plot_spectra to avoid duplicating parameter extraction logic.
    """
    idx = model.theta_index
    th  = theta_or_med
    agebins = np.array(model.params['agebins'])
    logmass       = float(th[idx['logmass']])
    mh_idx        = (float(th[idx['mh_idx']]) if 'mh_idx' in idx
                     else float(model.params['mh_idx']))
    logsfr_ratios = th[idx['logsfr_ratios']]
    mass = logsfr_ratios_to_masses(
        logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=agebins)
    add_dust_emission = 'duste_umin' in idx
    add_neb_emission  = 'gas_logu' in idx
    p = dict(
        teff         = float(th[idx['teff']]),
        logg         = float(th[idx['logg']]),
        mh_idx       = mh_idx,
        xi_mtb       = (float(th[idx['xi_mtb']]) if 'xi_mtb' in idx
                        else float(model.params.get('xi_mtb', 2.0))),
        logL_star    = float(th[idx['logL_star']]),
        sigma_smooth = (float(th[idx['sigma_smooth']]) if 'sigma_smooth' in idx
                        else float(model.params['sigma_smooth'])),
        logmass      = logmass,
        logzsol      = (float(th[idx['logzsol']]) if 'logzsol' in idx
                        else mh_idx - 2.0),
        dust2_gal    = float(th[idx['dust2_gal']]),
        dust1        = float(th[idx['dust1']]),
        logsfr_ratios= logsfr_ratios,
        mass         = mass,
        agebins      = agebins,
        sfh          = int(model.params['sfh']),
        imf_type     = int(model.params['imf_type']),
        dust_type    = int(model.params['dust_type']),
        dust1_index  = float(model.params['dust1_index']),
        add_neb_emission  = add_neb_emission,
        add_neb_continuum = add_neb_emission,
        add_dust_emission = add_dust_emission,
        zred         = float(model.params['zred']),
    )
    if add_neb_emission:
        p['gas_logu'] = float(th[idx['gas_logu']])
        p['gas_logz'] = float(model.params.get('gas_logz', -1.0))
    if add_dust_emission:
        p['duste_umin']  = float(th[idx['duste_umin']])
        p['duste_qpah']  = float(th[idx['duste_qpah']])
        p['duste_gamma'] = float(th[idx['duste_gamma']])
    return p


def _eval_model(sps_p, sps, obs_list, model):
    """
    Evaluate the model for a single theta and return predictions per obs.

    Returns
    -------
    wave        : rest-frame grid (Å)
    spec_tl     : photosphere Lsun/Hz
    spec_gal    : galaxy Lsun/Hz
    preds       : list of (pred_tl, pred_gal) arrays in maggies, one per obs
    """
    zred    = float(model.params['zred'])
    lumdist = float(model.params['lumdist'])
    wave, spec_tl, spec_gal, _ = sps.get_spectra_components(**sps_p)
    if np.any(np.isnan(spec_tl)):
        return None
    sigma_smooth = sps_p['sigma_smooth']
    preds = []
    for obs in obs_list:
        si = obs['sigma_inst']
        ptl  = _smooth_and_interp(wave, spec_tl,  obs['wave_obs'],
                                   sigma_smooth, si, SIGMA_LIB,  zred, lumdist)
        pgal = _smooth_and_interp(wave, spec_gal, obs['wave_obs'],
                                   0.0,          si, SIGMA_C3K,  zred, lumdist)
        preds.append((ptl, pgal))
    return wave, spec_tl, spec_gal, preds


# =============================================================================
# Plotting
# =============================================================================

def plot_corner(samples, labels, out_path):
    """Corner plot of posterior samples."""
    try:
        import corner
        # Build latex labels matching the free-parameter list
        latex_labels = [
            r'$T_{\rm eff}$ [K]',
            r'$\log g$',
            r'$\log_{10}(L_*/L_\odot)$',
            r'$\sigma_v$ [km/s]',
            r'$[{\rm M/H}]_{\rm idx}$',
            r'$\log_{10}(M_{\rm gal}/M_\odot)$',
            r'$\log(Z/Z_\odot)$',
            r'$A_V$ [mag] (SMC, shared)',
            r'$\tau_V^{\rm BC}$',
        ] + [r'$\log(\dot{M}_{%d}/\dot{M}_{%d})$' % (i, i + 1)
             for i in range(N_RATIOS)]
        if len(latex_labels) != samples.shape[1]:
            latex_labels = labels   # fallback to plain names
        fig = corner.corner(samples, labels=latex_labels,
                            quantiles=[0.16, 0.50, 0.84],
                            show_titles=True, title_kwargs={'fontsize': 9})
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Corner → {out_path}", flush=True)
    except ImportError:
        print("  corner not installed; skipping.", flush=True)


def plot_spectra(samples, sps, obs_list, model, out_path, n_post=50):
    """
    Spectrum plot matching the style of fits_broadband/final/*.

    Layout
    ------
    GridSpec(2,1, height_ratios=[3.5, 1.0], hspace=0.05):
      ax     — main panel: data + model + posterior envelope
      ax_res — residual panel: (data−model)/σ in σ units

    Style (matching reference plots)
    ---------------------------------
    Data  : fill_between (±1σ, alpha=0.18) + step (alpha=0.85)
    Model : continuous lines — total (#AA3377, solid), photosphere (#CC3311, --)
            galaxy (#CCAA00, -.)  No gaps from masking.
    Posterior envelope : n_post random draws → [16,84] fill_between (alpha=0.15)
    Residuals : scatter, coloured for in-fit pixels, gray (0.70) for masked.

    Normalization
    -------------
    All fluxes are normalized by the median maggies in 9000–11000 Å rest-frame
    (falls in FIRE for The Egg), matching the reference plot convention.

    The model is evaluated at ALL obs pixels (not just masked), so the model
    line is continuous with no gaps.  Masked pixels appear only in the residual
    panel as gray points.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        zred    = float(model.params['zred'])
        lumdist = float(model.params['lumdist'])

        # ── Normalization scale ───────────────────────────────────────────────
        ns = 1.0
        for obs in obs_list:
            sel = ((obs['wave_rest'] >= 9000.) & (obs['wave_rest'] <= 11000.)
                   & np.isfinite(obs['flux']) & (obs['flux'] > 0))
            if sel.sum() > 3:
                ns = float(np.nanmedian(obs['flux'][sel]))
                break

        # ── Median model ─────────────────────────────────────────────────────
        med     = np.median(samples, axis=0)
        sps_p   = _make_sps_params(med, model)
        result  = _eval_model(sps_p, sps, obs_list, model)
        if result is None:
            print("  Spectra plot: median model OOB — skipping.", flush=True)
            return
        wave, spec_tl, spec_gal, preds_med = result

        # ── Posterior envelope (n_post draws) ────────────────────────────────
        # Each draw calls FSPS once (galaxy) and TLUSTYBasis once (photosphere).
        # n_post=50 takes ~20-60s depending on machine; increase for publication.
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(samples), size=min(n_post, len(samples)),
                             replace=False)
        post_tot = [[] for _ in obs_list]  # list-of-lists: [obs_idx][draw_idx]
        for ii in indices:
            sps_pi  = _make_sps_params(samples[ii], model)
            res_i   = _eval_model(sps_pi, sps, obs_list, model)
            if res_i is None:
                continue
            _, _, _, preds_i = res_i
            for oi, (ptl_i, pgal_i) in enumerate(preds_i):
                tot_i = ptl_i + pgal_i
                if np.any(np.isfinite(tot_i)):
                    post_tot[oi].append(tot_i)

        def _envelope(arr_list):
            """[16, 84] percentile envelope from a list of arrays."""
            if len(arr_list) < 3:
                return None, None
            stack = np.vstack(arr_list)
            return (np.nanpercentile(stack, 16, axis=0),
                    np.nanpercentile(stack, 84, axis=0))

        # ── Assemble combined wavelength arrays for residuals ─────────────────
        all_wr   = np.concatenate([o['wave_rest'] for o in obs_list])
        all_mask = np.concatenate([o['mask']      for o in obs_list])
        all_flux = np.concatenate([o['flux']       for o in obs_list])
        all_unc  = np.concatenate([o['unc']        for o in obs_list])
        all_pred = np.concatenate([pt + pg for pt, pg in preds_med])

        resid    = (all_flux - all_pred) / all_unc
        in_fit   = all_mask & np.isfinite(resid)
        masked   = ~all_mask & np.isfinite(resid) & np.isfinite(all_flux)
        rms_fit  = np.nanstd(resid[in_fit]) if in_fit.sum() > 0 else np.nan

        # ── Figure layout ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 8))
        gs  = gridspec.GridSpec(2, 1, figure=fig,
                                height_ratios=[3.5, 1.0], hspace=0.05)
        ax     = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax)

        # ── Main panel: data ──────────────────────────────────────────────────
        # Colors and ARM_DISPLAY match replot_egg_tlusty.py exactly.
        _arm_colors = {'MODS_B': '#5599CC', 'MODS_R': '#33AA55', 'FIRE': '#EE7722'}
        _arm_display = {'MODS_B': (2700., 5500.), 'MODS_R': (5500., 7400.),
                        'FIRE':   (7400., 22900.)}
        _smooth_pix  = {'MODS_B': 14, 'MODS_R': 14, 'FIRE': 5}
        for obs in obs_list:
            arm_name = obs.get('name', '')
            ic = _arm_colors.get(arm_name, '0.50')
            d_lo, d_hi = _arm_display.get(arm_name, (2700., 22900.))
            w, f, e = obs['wave_rest'], obs['flux'], obs['unc']
            ok = (np.isfinite(f) & np.isfinite(e) & (e > 0)
                  & (w >= d_lo) & (w <= d_hi))
            if ok.sum() < 3:
                continue
            wr = w[ok] * 1e-4
            fr = f[ok] / ns;  er = e[ok] / ns
            ax.fill_between(wr, fr - er, fr + er, color=ic, alpha=0.12, zorder=2)
            ax.plot(wr, fr, color=ic, lw=0.7, alpha=0.30, zorder=3)
            sp = _smooth_pix[arm_name]
            fr_sm = gaussian_filter1d(fr, sigma=sp)
            ax.plot(wr, fr_sm, color=ic, lw=2.2, alpha=0.90, zorder=4,
                    label=arm_name)

        # ── Main panel: posterior envelope ────────────────────────────────────
        for oi, obs in enumerate(obs_list):
            lo, hi = _envelope(post_tot[oi])
            if lo is not None:
                wr = obs['wave_rest'] * 1e-4
                ax.fill_between(wr, lo / ns, hi / ns,
                                color=_C_MOD, alpha=0.15, zorder=4)

        # ── Main panel: model components (continuous — no masking gaps) ───────
        for oi, (obs, (ptl, pgal)) in enumerate(zip(obs_list, preds_med)):
            wr = obs['wave_rest'] * 1e-4
            ok = np.isfinite(ptl) & np.isfinite(pgal)
            if oi == 0:
                lbl_tl  = f'Photosphere  T={sps_p["teff"]:.0f} K  logg={sps_p["logg"]:.2f}'
                lbl_gal = f'Galaxy  logM={sps_p["logmass"]:.2f}'
                lbl_tot = 'Total model'
            else:
                lbl_tl = lbl_gal = lbl_tot = '_nolegend_'
            ax.plot(wr[ok], ptl[ok]  / ns, color=_C_PHO, lw=1.2, ls='--',
                    alpha=0.70, zorder=5, label=lbl_tl)
            ax.plot(wr[ok], pgal[ok] / ns, color=_C_GAL, lw=1.2, ls='-.',
                    alpha=0.70, zorder=5, label=lbl_gal)
            ax.plot(wr[ok], (ptl[ok] + pgal[ok]) / ns,
                    color=_C_MOD, lw=2.2, ls='-', alpha=0.90, zorder=6,
                    label=lbl_tot)

        ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.2)
        ax.set_xlim(0.1, 2.5)
        xlo, xhi = 0.1, 2.5
        yvals = all_flux[all_mask & np.isfinite(all_flux)] / ns
        if len(yvals) > 0:
            ax.set_ylim(-0.3, np.nanpercentile(yvals, 98) * 1.35)
        ax.set_ylabel('Relative flux  [norm. 9000–11000 Å]', fontsize=11)
        ax.tick_params(labelsize=10, labelbottom=False, direction='in',
                       top=True, right=True)
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', top=True, right=True, length=3)
        ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92, ncol=2)

        # Annotate key absorption/emission features
        _ABS = {'FeH 9900': 9900, r'H$_2$O 13500': 13500}
        _EM  = {r'H$\alpha$': 6563, r'H$\beta$': 4861,
                r'He I 10830': 10830, r'Pa$\beta$': 12820}
        for name, wc in _ABS.items():
            if xlo < wc * 1e-4 < xhi:
                ax.axvline(wc * 1e-4, color='#9467bd', lw=1.0, ls='--', alpha=0.40)
                ax.text(wc * 1e-4, 0.72, name, rotation=90, ha='center',
                        va='bottom', fontsize=7, color='#9467bd',
                        transform=ax.get_xaxis_transform())
        for name, wc in _EM.items():
            if xlo < wc * 1e-4 < xhi:
                ax.axvline(wc * 1e-4, color='#d62728', lw=1.0, ls='-', alpha=0.30)
                ax.text(wc * 1e-4, 0.87, name, rotation=90, ha='center',
                        va='bottom', fontsize=7, color='#d62728',
                        transform=ax.get_xaxis_transform())

        # Title
        p50_teff  = np.median(samples[:, model.theta_index['teff']])
        p50_logg  = np.median(samples[:, model.theta_index['logg']])
        p50_logL  = np.median(samples[:, model.theta_index['logL_star']])
        p50_sig   = np.median(samples[:, model.theta_index['sigma_smooth']])
        p50_logM  = np.median(samples[:, model.theta_index['logmass']])
        ax.set_title(
            f'{SOURCE_NAME}  (z={zred})  —  TLUSTY photosphere + continuity SFH host\n'
            f'$T_{{\\rm eff}}={p50_teff:.0f}$ K  '
            f'log $g={p50_logg:.2f}$  '
            f'log $L={p50_logL:.2f}$  '
            f'$\\sigma={p50_sig:.0f}$ km/s  '
            f'log $M={p50_logM:.2f}$  '
            f'[$n_{{\\rm post}}={min(n_post, len(samples))}$ draws]',
            fontsize=10)

        # ── Residual panel ────────────────────────────────────────────────────
        # In-fit pixels: coloured; masked pixels: gray (same style as reference)
        for obs, (ptl, pgal) in zip(obs_list, preds_med):
            arm_name = obs.get('name', '')
            ic = _arm_colors.get(arm_name, '0.50')
            d_lo, d_hi = _arm_display.get(arm_name, (2700., 22900.))
            w = obs['wave_rest']
            sel_win = ((w >= d_lo) & (w <= d_hi)
                       & np.isfinite(obs['flux']) & np.isfinite(obs['unc'])
                       & (obs['unc'] > 0) & np.isfinite(ptl + pgal))
            res_o  = (obs['flux'] - (ptl + pgal)) / obs['unc']
            msk_o  = sel_win &  obs['mask'] & np.isfinite(res_o)
            nmsk_o = sel_win & ~obs['mask'] & np.isfinite(res_o)
            ax_res.scatter(w[msk_o]  * 1e-4, np.clip(res_o[msk_o],  -7, 7),
                           s=6, color=ic, alpha=0.75, zorder=3)
            ax_res.scatter(w[nmsk_o] * 1e-4, np.clip(res_o[nmsk_o], -7, 7),
                           s=4, color='0.65', alpha=0.45, zorder=2,
                           label='masked (lines)' if oi == 0 else '_nolegend_')

        ax_res.axhline(0,  color='k',    lw=1.0, ls='--', alpha=0.60)
        ax_res.axhline( 3, color='0.55', lw=0.8, ls=':')
        ax_res.axhline(-3, color='0.55', lw=0.8, ls=':')
        ax_res.set_ylim(-7, 7)
        ax_res.set_xlim(0.1, 2.5)
        ax_res.set_xlabel(r'Rest-frame wavelength ($\mu$m)', fontsize=11)
        ax_res.set_ylabel(r'$(d-m)/\sigma$', fontsize=10)
        ax_res.tick_params(labelsize=9, direction='in', top=True, right=True)
        ax_res.minorticks_on()
        ax_res.tick_params(which='minor', direction='in', top=True, right=True, length=3)
        ax_res.text(0.01, 0.93, f'RMS (fitted) = {rms_fit:.2f}σ',
                    transform=ax_res.transAxes, fontsize=8,
                    color='#333333', va='top')
        if masked.sum() > 0:
            ax_res.legend(fontsize=7, loc='upper right', framealpha=0.85)

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Spectra → {out_path}", flush=True)
        plt.close(fig)
    except Exception as e:
        import traceback
        print(f"  Spectra plot failed: {e}", flush=True)
        traceback.print_exc()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import functools, argparse

    _parser = argparse.ArgumentParser()
    _args = _parser.parse_args()

    print(f'\n{"="*60}', flush=True)
    print(f'{SOURCE_NAME}  —  TLUSTY photosphere + continuity SFH host galaxy',
          flush=True)
    print(f'MODS-B {MODSB_LO:.0f}–{MODSB_HI:.0f} Å  +  '
          f'MODS-R {MODSR_LO:.0f}–{MODSR_HI:.0f} Å  +  '
          f'FIRE {FIRE_LO:.0f}–{FIRE_HI:.0f} Å  |  z = {Z_SOURCE}', flush=True)
    print(f'{"="*60}\n', flush=True)

    sps, obs_list, model = build_all()

    # Derive ndim and labels from the model — no hardcoded constants needed.
    NDIM   = model.ndim
    LABELS = model.theta_labels()
    n_pix  = sum(o['mask'].sum() for o in obs_list)

    print(f'\nndim={NDIM}  |  SFH bins={N_BINS_SFH}  |  ratios={N_RATIOS}',
          flush=True)
    print(f'AGEBINS log10(yr):\n{AGEBINS}', flush=True)
    print(f'Free parameters: {LABELS}', flush=True)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    u_mid  = 0.5 * np.ones(NDIM)
    th_mid = model.prior_transform(u_mid)
    ll_mid = _logl(th_mid, sps, obs_list, model)
    print(f'\nMid-prior theta sample:', flush=True)
    for lbl, val in zip(LABELS, th_mid):
        print(f'  {lbl:25s}: {val:.3f}', flush=True)
    print(f'Mid-prior ln L = {ll_mid:.1f}', flush=True)

    u_oob    = u_mid.copy()
    u_oob[0] = 0.9999            # teff → near 6750 K (likely OOB for some logg)
    th_oob   = model.prior_transform(u_oob)
    ll_oob   = _logl(th_oob, sps, obs_list, model)
    print(f'OOB check (teff={th_oob[0]:.0f} K) ln L = {ll_oob:.1f}', flush=True)

    # ── DynamicNestedSampler ──────────────────────────────────────────────────
    logl_fn = functools.partial(_logl, sps=sps, obs_list=obs_list, model=model)

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

    stem     = f'{SOURCE_TAG}_{MH_TAG}_prospector_tlusty'
    pkl_path = os.path.join(SCRIPT_DIR, f'dynesty_{stem}.pkl')
    npy_path = os.path.join(SCRIPT_DIR, f'chain_{stem}.npy')
    with open(pkl_path, 'wb') as fh:
        pickle.dump(res, fh)
    np.save(npy_path, samp)
    print(f'\nSaved:\n  {pkl_path}\n  {npy_path}')

    ll_max   = res.logl.max()
    lz, lze  = res.logz[-1], res.logzerr[-1]
    bic      = NDIM * np.log(n_pix) - 2.0 * ll_max

    data_str = ' + '.join(f"{o['name']} ({o['mask'].sum()} pix)" for o in obs_list)
    p50_mhidx = np.median(samp[:, model.theta_index['mh_idx']])
    p50_mh    = p50_mhidx - 2.0   # convert index to [M/H] dex: idx=0→-2, 1→-1, 2→0
    report_lines = [
        f'{SOURCE_NAME}  —  TLUSTY photosphere ([M/H] free) + continuity SFH host galaxy',
        f'ndim={NDIM}  SFH bins={N_BINS_SFH}  z={Z_SOURCE}  [M/H] free (mh_idx∈[0,2]; median {p50_mh:+.2f})',
        f'Dust: shared SMC(dust2_gal) on both components + dust1 FSPS birth-cloud',
        f'Data: {data_str} = {n_pix} total',
        '',
        f'  log Z    = {lz:.2f} ± {lze:.2f}',
        f'  BIC      = {bic:.2f}',
        f'  max ln L = {ll_max:.2f}',
        f'  Elapsed  = {elapsed:.0f} s',
        '',
        'Parameter posteriors (16–50–84 percentiles):',
        '  Note: T_eff uncertainty is formal posterior only;',
        '        true uncertainty ≥ grid spacing (~250–500 K)',
    ]
    for i, lbl in enumerate(LABELS):
        p16, p50, p84 = np.percentile(samp[:, i], [16, 50, 84])
        report_lines.append(
            f'  {lbl:25s}: {p50:.4f}  +{p84-p50:.4f}/-{p50-p16:.4f}')
    report = '\n'.join(report_lines)
    print('\n' + report)

    bic_path = os.path.join(SCRIPT_DIR, f'bic_{stem}.txt')
    with open(bic_path, 'w') as fh:
        fh.write(report)
    print(f'\nBIC report → {bic_path}')

    corner_path  = os.path.join(SCRIPT_DIR, f'corner_{stem}.png')
    spectra_path = os.path.join(SCRIPT_DIR, f'spectra_{stem}.png')
    plot_corner(samp, LABELS, corner_path)
    plot_spectra(samp, sps, obs_list, model, spectra_path)

    print('\nDone.')
