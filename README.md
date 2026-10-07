# Posterior chains and line tests for *A Spectral Framework for Testing the Quasi-Star Hypothesis in Little Red Dots I: Weighing LRDs by Their Super-Eddington Luminosity Ratios—No Signs of Overmassive Black Holes*

> **If you use these data, or the [`prospector-tlusty`](https://github.com/o-curtis/prospector-tlusty) code that produced them, please cite the paper:**
>
> Curtis, O., Cleri, N. J., Helton, J. M., Leja, J., & Wright, J. T. 2026, arXiv e-prints, arXiv:2609.09265, doi:10.48550/arXiv.2609.09265
>
> [NASA ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260909265C/abstract) · [arXiv](https://arxiv.org/abs/2609.09265) · BibTeX in the [Citation](#citation) section below

Every spectral fit and every emission-line and absorption-feature test reported in
Curtis et al. (2026), *A Spectral Framework for Testing the Quasi-Star Hypothesis in Little Red Dots I: Weighing LRDs by Their Super-Eddington Luminosity Ratios—No Signs of Overmassive Black Holes*
([arXiv:2609.09265](https://arxiv.org/abs/2609.09265)), released as posterior samples with the driver
script that produced each one and, where one was rendered, the fitted spectrum.

The fits are run with [`prospector-tlusty`](https://github.com/o-curtis/prospector-tlusty),
a fork of [Prospector](https://github.com/bd-j/prospector) that adds a TLUSTY photosphere
basis alongside the code's stock stellar population sources. That repository holds the code;
this one holds the results.

Contact: Olivia Curtis, ocurtis@psu.edu

---

## Quick start

```python
import numpy as np, json

run   = 'chains/primary/J1025+1402'
chain = np.load(f'{run}/chain.npy')          # (n_samples, n_parameters), equal weight
meta  = json.load(open(f'{run}/run.json'))

print(meta['parameters'])                     # column names, in column order
i = meta['parameters'].index('teff')
print(np.percentile(chain[:, i], [16, 50, 84]))
```

Every chain is a set of **equal-weight posterior draws**, so a plain percentile over rows is
the posterior quantile. No importance weights are needed and none are supplied.

The Eddington ratio quoted throughout the paper follows from two columns,

```python
kappa_es, sigma_sb, c = 0.34, 5.670374419e-5, 2.99792458e10
T   = chain[:, meta['parameters'].index('teff')]
g   = chain[:, meta['parameters'].index('logg')]
phi = kappa_es * sigma_sb * T**4 / (10**g * c)
```

---

## Layout

```
MANIFEST.csv                     one row per run: id, family, target, paper section, driver, figure, path
summary/
  photospheric_parameters.csv    16th, 50th and 84th percentiles of T_eff, log g, log L, phi
  evidence.csv                   ln Z and its uncertainty for every run
code/                            every driver script and the shared modules they import
chains/
  primary/<target>/
  supplementary/<mode>/<source>/
  experiments/<experiment>/<run>/
line_tests/<test>/               emission-line and absorption-feature measurements
```

Each run directory holds its chain, its metadata, and its figures.

| file | contents |
|---|---|
| `chain.npy` | equal-weight posterior draws, `float32`, shape `(n_samples, n_parameters)` |
| `run.json` | parameter names in column order, percentile summary, driver filename, paper section, sample counts |
| `spectrum.pdf` or `spectrum.png` | the fitted spectrum drawn over the data, where one was rendered |
| `corner.png` | posterior corner plot, Egg runs only |
| `spectrum_model_finalization.png` | model-by-model spectral comparison, Egg runs only |

---

## What is here, and where it appears in the paper

Every posterior in this release was computed with rest-frame 3400-4100 A excluded from the
likelihood, the Balmer-break mask that Section 2.3 of the paper describes. The Egg fit also
excludes the Ca II H and K cores. No fit that kept the break region is released; the fits of
that kind in earlier versions of this repository were superseded when the paper adopted the
mask and have been removed.

| directory | runs | spectra | paper |
|---|---|---|---|
| `chains/primary/` | 5 | 5 | Section 4.1, Table 1 |
| `chains/supplementary/prism/` | 53 | 53 | Sections 2.3 and 4.4, Figures 5-7 |
| `chains/supplementary/desi/` | 28 | 28 | Sections 2.3 and 4.4, Figures 5-7 |
| `chains/supplementary/grating/` | 1 | 1 | Sections 2.3 and 4.4, Figures 5-7 |
| `chains/experiments/host_prior/` | 89 | 2 | Section 4.4, Figures 5 and 6 |
| `chains/experiments/photometric_anchoring/` | 7 | 7 | Section 2.1 |
| `chains/experiments/dust_law_and_configuration_sweep/` | 4 | 1 | Section 2.3 |
| `chains/experiments/egg_model_finalization/` | 6 | 0 | Appendix C |
| `chains/experiments/two_component_test/` | 46 | 0 | not quoted in v2 (v1 Section 4.4) |
| `chains/experiments/injection_recovery/prism/` | 5 | 0 | not quoted in v2 (v1 Section 4.4) |
| `chains/experiments/injection_recovery/desi/` | 4 | 0 | not quoted in v2 (v1 Section 4.4) |
| `chains/experiments/injection_recovery/egg/` | 2 | 0 | not quoted in v2 (v1 Appendix C) |
| `chains/experiments/duplicate_reduction/` | 1 | 1 | not quoted in v2 (v1 Section 4.4) |
| **total** | **251** | **98** | |

Section and figure numbers refer to arXiv:2609.09265v2. Families marked "not quoted in v2"
were discussed in the first arXiv version and are kept so that those statements remain
reproducible under the adopted mask.

Spectrum figures exist for the five primary sources, every supplementary source, the
photometric-anchoring variants, the UNCOVER-A2744-20698 `_xmask` variant on the spectrum as
observed, the duplicate reduction of A2744-QSO1, and the two water-dot young-host fits that were
rendered during the analysis. The
supplementary figures draw the model at the posterior median over the data, with masked pixels
in light grey. The remaining experiment refits reuse the same data as their parent fit and were
compared through parameters and evidence rather than by eye, so no overlay was drawn for them.

### The primary fits

| target | directory | data | free parameters |
|---|---|---|---|
| J1025+1402 (The Egg) | `chains/primary/J1025+1402` | LBT/MODS-B, MODS-R and Magellan/FIRE, rest-frame 3400-4100 A and the Ca II H and K cores excluded | 18 |
| GN-28074 (the Rosetta Stone) | `chains/primary/GN-28074` | JWST/NIRSpec G140M, G235M and G395M, rest-frame 3400-4100 A excluded | 17 |
| WIDE-EGS-2974 | `chains/primary/WIDE-EGS-2974` | JWST/NIRSpec PRISM as observed, two photospheres, rest-frame 3400-4100 A excluded | 17 |
| UNCOVER-A2744-20698 | `chains/primary/UNCOVER-A2744-20698` | JWST/NIRSpec PRISM at R = 300, rescaled to its NIRCam photometry, two photospheres, rest-frame 3400-4100 A excluded | 17 |
| CAPERS-UDS-23216 | `chains/primary/CAPERS-UDS-23216` | JWST/NIRSpec PRISM, rescaled to its NIRCam photometry, two photospheres, rest-frame 3400-4100 A excluded | 17 |

The two-photosphere fits carry the cold component in columns 14, 15 and 16 (`teff_cold`,
`logg_cold`, `logL_cold`). `run.json` gives the names for every column, so read them from
there rather than assuming a layout.

### The supplementary sample

The 82 supplementary sources of Figures 5-7 are 53 archival PRISM spectra
(11 from Naidu et al. 2026, 30 RUBIES from Hviding et al. 2025, 8 from Sok et al. 2026, and
A2744-QSO1, MoM-BH\*-1, CEERS-6126 and UDS-31092), 28 DESI spectra from Lin et al. 2026, and
the lensed GLIMPSE-17775 fitted on its coadded G395M continuum. The run identifiers carry the
mask in their prefix: `BMKA_` for the archival PRISM fits, `lrds2M_` for the DESI fits, and
`BMK_45924` for A2744-45924, which was fitted alongside the strongest-break sources. The
GLIMPSE-17775 fit window (rest-frame 6400-11300 A) lies entirely redward of the mask, so its
`BMKA_` run repeats the configuration unchanged.

Each run directory keeps the name of its source, so `chains/supplementary/prism/MoMBH1` holds
the fit of MoM-BH\*-1 and `chains/supplementary/desi/J000927` the fit of J000927+081109.

Three of these carry red flags in Figure 6 and enter no statistic in the paper. They are kept
here so that the flag can be checked: `GN9771` presses against the gravity floor of the prior,
`GN68797` truncates at its ceiling, and `RUBIES-EGS966323` is host dominated.

### The experiments

**Host star-formation history (Section 4.4, Figures 5 and 6).** Every fit in the paper
repeated with the host's star formation forced into its youngest bins, which sets the
0.16-0.32 dex spread in the Eddington ratio that the population figures add in quadrature and
the host-mass systematic that the mass-scale figure draws. The archival PRISM refits are
`BMKY_<source>` (shared driver `fit_breakmask_arch_young.py`) or `BMKA_<source>_youngal`
(built by `_variant.py` from the source's own driver), the DESI refits are `lrds2MY_<source>`,
and the primaries are the `BMKP_*_youngal` runs. Runs without a driver of their own name the
driver they are built from in `variant_of`.

**Photometric anchoring (Section 2.1).** The PRISM spectra of UNCOVER-A2744-20698 and
CAPERS-UDS-23216 depart from their NIRCam photometry in shape, so each is multiplied by a
low-order curve fitted to the catalog-to-spectrum flux ratios of its long-wavelength bands
before fitting. The primary fits use a straight line through the UNCOVER DR2 photometry and a
smooth quadratic through the DJA PRIMER-UDS photometry. This directory holds the same
break-masked fit on each spectrum as observed (`BMKP_UNCOVER_2c_smc_r300_ujy`,
`BMKP_CAPERS_2c_smc_ujy`) and under the alternatives: for UNCOVER-A2744-20698 a piecewise-linear
(`_photcal3`) and a smooth (`_photcal4`) curve through the DJA v7 photometry, and for
CAPERS-UDS-23216 a straight line (`_photcal`), a band-interpolated curve (`_photcal2`) and a
piecewise-linear curve (`_photcal3`). Each reader module writes out the curve it applies.

**Dust law and configuration sweep (Section 2.3).** The Egg configuration of the primary fit
under Calzetti (`DLAW_egg_duste_sfree_zfree_calz_bmkp`) and Milky Way
(`DLAW_egg_duste_sfree_zfree_mw_bmkp`) attenuation in place of the Small Magellanic Cloud law,
and the UNCOVER-A2744-20698 primary configuration with two isolated residual regions excluded
in addition to the break (`_xmask`, on the spectrum as observed and after photometric
rescaling). No individual run is cited in the text.

**Egg model finalization (Appendix C).** The adopted Egg configuration repeated with one
prior changed at a time: the birth-cloud dust `dust1` capped at 1 (`d1cap`), fixed at zero
(`d1fix0`) or clamped at the adopted 0.047 (`d1pin`), every host parameter pinned at its
adopted value (`hostpin`), the host metallicity pinned (`zsolpin`), and the star-formation-history
ratios clipped to [-3, 3] (`sfhclamp`). The adopted fit itself, with the Ca II H and K cores
also excluded, is `chains/primary/J1025+1402`. Each driver prints the change it applies.

**Two-component test (not quoted in v2).** Every archival PRISM source refit with the
hot-plus-cold variant under the break mask (`BMKC_<source>_2c`). Pair each with the
single-photosphere fit of the same source under `supplementary/prism/` and take the evidence
difference from `summary/evidence.csv`.

**Injection and recovery (not quoted in v2).** Photospheres of known temperature and gravity
injected into real instrument noise and refit blind under the break mask: four PRISM mocks at
the bright and faint tiers (`BMKA_MOCK_R5_*`), a mock at the Cliff's noise (`BMKA_MOCK_cliff_m3`),
four DESI mocks (`lrds2M_MOCK_R5D_*`), and the Egg injection, which ships its injected truth
vector as a separate run directory containing a one-dimensional array in the same column order
as its chain.

**Duplicate reduction (not quoted in v2).** A2744-QSO1 refit on its second available reduction
(`BMKA_A2744QSO1B`).

## Line and absorption feature tests

`line_tests/` holds the measurements made directly on the spectra rather than through the
photospheric fits. Each directory carries the script that made the measurement, its numerical
result, and every figure drawn from it. `test.json` names the paper section. Entries marked v1 refer to the line analysis of the first arXiv version, which the second version no longer carries.

| directory | paper | contents |
|---|---|---|
| `hbeta_component_ladder` | v1 Section 4.6 | the model-selection suite behind the four-component decomposition |
| `egg_hbeta_absorber` | v1 Section 4.1 | the Hβ absorber investigation, including the joint and two-absorber refits |
| `egg_balmer_and_helium_absorption` | v1 Section 4.1, Figure 3 | Hα, Hβ and He I λ10830 troughs in The Egg |
| `egg_paschen_absorption` | v1 Section 4.1 | Paγ and Paδ absorption at −428 and −355 km/s |
| `balmer_progression` | v1 Section 4.1 | absorption minima up the Balmer series and the two scenarios they distinguish |
| `broad_line_helium` | v1 Section 4.6 | broad He I λ5876, the He II λ4686 limits and stack, the λ10830/λ20581 singlet bound |
| `rosetta_stone_lines` | v1 Section 4.6 | the GN-28074 joint two-grating line fit and its chromosphere test |
| `narrow_line_diagnostics` | v1 Section 4.6 | [O II] densities, Balmer decrements, the narrow-line budget, the Shirazi and OHNO diagnostics |
| `uds40579_helium` | v1 Section 4.6 | the UDS-40579 He I trough |
| `vblue95_forecast` | Section 5.2, Figure 7 | the uniform v_blue,95 wind-launch measurements and P Cygni profile fits for every point in the wind-launch panel |

### The component-selection suite

Section 4.6 of the first arXiv version states that the four-component decomposition is selected by model comparison
rather than imposed. The suite behind that sentence is in `line_tests/hbeta_component_ladder`.
It fits The Egg's Hβ complex over ±4500 km/s against a quadratic continuum, with the absorber
present in every model, and compares every combination of components on the same pixels:

| model | BIC | against the adopted model |
|---|---|---|
| narrow + absorber | 4921 | +2682 |
| narrow + broad Gaussian + absorber | 2311 | +72 |
| narrow + broad exponential + absorber | 2548 | +309 |
| **narrow + outflow + broad Gaussian + absorber** | **2239** | adopted |
| narrow + outflow + two broad Gaussians + absorber | 2259 | +20 |

The same criterion applied per source returns a two-component model for three of the seven
broad-line sources, so the choice is made source by source rather than imposed as a template.

The broad-profile family is settled separately, in the same directory. With the absorber
fixed to the independently measured Hα notch kinematics, a Gaussian broad component beats a
single-scattering exponential by Δχ² = 102 at equal complexity
(`definitive_egg_ladder_fixedabs.json`). Letting the absorber kinematics float instead opens a
degenerate emission and absorption pair that can make either family appear to win, which is
why the component identities are anchored externally: the narrow width comes from [O III], the
outflow from the shared-kinematics [O III]/Hβ ratio, and the absorber from the Hα notch.

---

## Reproducing a fit

`code/` holds every driver plus the shared modules they import. A driver names its own run:

```bash
python code/fit_rs_m3c_smc.py --pool 8
```

Two things are not redistributed here. The observed spectra belong to their original archives
(MAST for the JWST programmes, the DESI data releases, and the LBT and Magellan programmes for
The Egg), and the TLUSTY model library is the one published by Liu et al. 2026. A driver
expects both to be present at the paths set at the top of the shared modules.

Most archival PRISM fits share one driver, `code/fit_breakmask_arch.py`, which takes the source
stem and reads the redshift and spectrum from that source's own configuration module
(`code/fit_<stem>.py`, kept for that purpose):

```bash
python code/fit_breakmask_arch.py RUBIES_EGS28812 --pool 3
```

The sources whose spectra come from a different archive have self-contained drivers
(`code/bmka_fit_<stem>.py`), as do the mocks, the Cliff and GLIMPSE-17775. The 28 DESI fits
share `code/fit_desi_lrds2_breakmask.py`, which selects a source by name, and A2744-45924 was
fitted with `code/fit_breakmask.py`. The young-host refits use `fit_breakmask_arch_young.py`,
`fit_desi_lrds2_youngal_bmk.py`, or `code/_variant.py` applied to the driver named in their
`variant_of` field,

```bash
python code/_variant.py bmkp_fit_wide_2c_smc_ujy youngal --pool 8
```

Every other run has its own driver, named in its `run.json`. The unmasked configuration
modules that several masked drivers import (`fit_<stem>.py`, `fit_desi_lrds2_roar.py`,
`fit_glimpse17775.py`, `fit_MOCK_*.py`) are kept as code only; no run made with them is
released.

---

## Conventions and caveats

**Thinning.** Chains are released at up to 50,000 draws in `float32`. Nested sampling produced
these as equal-weight resamples in which rows repeat, and no original chain contains more than
46,973 *unique* draws, so the released set carries the full unique content of every posterior.
Where an original exceeded 50,000 rows the release takes a systematic subsample of the ordered
sequence, which preserves the distribution exactly. `run.json` records both counts.

**Precision.** Samples are stored at `float32`, roughly seven significant digits, which is far
finer than any posterior width in this work but is not bit-identical to the `float64` originals.

**Parameter names.** Taken from the run log of each fit where one survives, which records
the model's own parameter list at build time. Where it does not, the masked run uses the
parameter list of the configuration it clones, which builds an identical model and differs only
in the pixels entering the likelihood; the chain dimension was checked against that list in
every case. `run.json` records the provenance in `parameter_source`.

**Evidence.** `summary/evidence.csv` gives ln Z and its sampling uncertainty for all 250
posterior runs. The dynesty result objects themselves are not redistributed. They run to
several gigabytes and require a matching dynesty version to unpickle, and every evidence
comparison in the paper is reproducible from the table.

**Units.** `teff` is in kelvin, `logg` is base-ten log of surface gravity in cgs, `logL_star`
and `logL_cold` are base-ten log of luminosity in solar units, `sigma_smooth` is in km/s,
`mh_idx` indexes the library metallicity grid, and `dust2_gal` is the V-band optical depth of
the shared screen.

**Evidence comparisons across the break mask.** Evidence values are comparable only between
runs fitted to the same pixels. The Egg primary fit excludes the Ca II H and K cores as well as
the break, so its ln Z is not on the same footing as the Egg finalization variants.

**Not included.** Fits that kept rest-frame 3400-4100 A in the likelihood, the geometry-tie
refits of the water dots, and the superseded development ladders that predate the adopted
configuration are not released. The paper adopts the break mask throughout, and no number in it
depends on those runs.

---

## Citation

If you use these fits, or the [`prospector-tlusty`](https://github.com/o-curtis/prospector-tlusty)
code that produced them, please cite

Curtis, O., Cleri, N. J., Helton, J. M., Leja, J., & Wright, J. T. 2026, arXiv e-prints, arXiv:2609.09265, doi:10.48550/arXiv.2609.09265 ([NASA ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260909265C/abstract))

```bibtex
@ARTICLE{2026arXiv260909265C,
       author = {{Curtis}, Olivia and {Cleri}, Nikko J. and {Helton}, Jakob M. and {Leja}, Joel and {Wright}, Jason T.},
        title = "{A Spectral Framework for Testing the Quasi-Star Hypothesis in Little Red Dots I: Weighing LRDs by Their Super-Eddington Luminosity Ratios---No Signs of Overmassive Black Holes}",
      journal = {arXiv e-prints},
         year = 2026,
        month = sep,
          eid = {arXiv:2609.09265},
        pages = {arXiv:2609.09265},
          doi = {10.48550/arXiv.2609.09265},
archivePrefix = {arXiv},
       eprint = {2609.09265},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260909265C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

The specific run you used is identified by its `run_id` in `MANIFEST.csv`.

## License

Posterior samples, summary tables and figures are released under CC BY 4.0. The scripts are
released under the MIT license, matching Prospector.
