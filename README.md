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

| directory | runs | spectra | paper |
|---|---|---|---|
| `chains/primary/` | 5 | 5 | Section 4.1, Table 1 |
| `chains/supplementary/prism/` | 53 | 50 | Sections 2.3 and 4.4, Figure 8 |
| `chains/supplementary/desi/` | 28 | 28 | Sections 2.3 and 4.4, Figure 8 |
| `chains/supplementary/grating/` | 1 | 1 | Sections 2.3 and 4.4, Figure 8 |
| `chains/experiments/geometry_ties/` | 6 | 4 | Section 4.2 |
| `chains/experiments/geometry_ties/alternative_configurations/` | 18 | 14 | Section 4.2, alternative configurations |
| `chains/experiments/two_component_test/` | 49 | 0 | Section 4.4 |
| `chains/experiments/injection_recovery/prism/` | 7 | 0 | Section 4.4 |
| `chains/experiments/injection_recovery/desi/` | 4 | 4 | Section 4.4 |
| `chains/experiments/injection_recovery/egg/` | 2 | 1 | Appendix C |
| `chains/experiments/duplicate_reduction/` | 1 | 1 | Section 4.4 |
| `chains/experiments/dust_law_and_configuration_sweep/` | 170 | 145 | Section 2.3 |
| `chains/experiments/photometric_anchoring/` | 7 | 7 | Section 2.1 |
| `chains/experiments/host_prior/` | 8 | 4 | Sections 4.3 and 4.4 |
| `chains/experiments/egg_model_finalization/` | 20 | 17 | Appendix C |
| **total** | **379** | **281** | |

Spectrum figures exist for every fit whose overlay was rendered during the analysis, which
covers the five primary sources, the full supplementary sample except CEERS-6126, MoM-BH\*-1
and UDS-31092, the DESI injections, the Egg ladder, and most of the water-dot experiments. The
archival experiment refits reuse the same data as their parent fit and were compared through
parameters and evidence rather than by eye, so no overlay was drawn for them.

### The primary fits

| target | directory | data | free parameters |
|---|---|---|---|
| J1025+1402 (The Egg) | `chains/primary/J1025+1402` | LBT/MODS-B, MODS-R and Magellan/FIRE, Ca II H and K cores (rest-frame 3900-3980 A) excluded | 18 |
| GN-28074 (the Rosetta Stone) | `chains/primary/GN-28074` | JWST/NIRSpec G140M, G235M and G395M, rest-frame 3400-4100 A excluded | 17 |
| WIDE-EGS-2974 | `chains/primary/WIDE-EGS-2974` | JWST/NIRSpec PRISM as observed, two photospheres, rest-frame 3400-4100 A excluded | 17 |
| UNCOVER-A2744-20698 | `chains/primary/UNCOVER-A2744-20698` | JWST/NIRSpec PRISM at R = 300, rescaled to its NIRCam photometry, two photospheres, rest-frame 3400-4100 A excluded | 17 |
| CAPERS-UDS-23216 | `chains/primary/CAPERS-UDS-23216` | JWST/NIRSpec PRISM, rescaled to its NIRCam photometry, two photospheres, rest-frame 3400-4100 A excluded | 17 |

The two-photosphere fits carry the cold component in columns 14, 15 and 16 (`teff_cold`,
`logg_cold`, `logL_cold`). `run.json` gives the names for every column, so read them from
there rather than assuming a layout.

### The supplementary sample

The 82 supplementary sources of Figure 8 are 53 archival PRISM spectra
(11 from Naidu et al. 2026, 30 RUBIES from Hviding et al. 2025, 8 from Sok et al. 2026, and
A2744-QSO1, MoM-BH\*-1, CEERS-6126 and UDS-31092), 28 DESI spectra from Lin et al. 2026, and
the lensed GLIMPSE-17775 fitted on its coadded G395M continuum.

Three of these carry red flags in Figure 8 and enter no statistic in the paper. They are kept
here so that the flag can be checked: `GN9771` presses against the gravity floor of the prior,
`GN68797` truncates at its ceiling, and `RUBIES-EGS966323` is host dominated with its
photosphere contributing 37 percent of the fitted flux.

### The experiments

**Geometry ties (Section 4.2).** Each water dot is refit with the cold gravity tied two ways,
to the gravity that conserves the Eddington ratio along a radial outflow (`_isophi`) and to the
hot gravity that a starspot requires (`_spot`). Each tie is read against the free two-photosphere
fit of the same configuration, which sits in the configuration sweep, so a tie comparison uses
three runs:

| source | free | wind tie | starspot tie |
|---|---|---|---|
| WIDE-EGS-2974 | `dust_law_and_configuration_sweep/WIDE_2c_smc_ujy` | `geometry_ties/WIDE_2c_smc_isophi_ujy` | `geometry_ties/WIDE_2c_smc_spot_ujy` |
| UNCOVER-A2744-20698 | `dust_law_and_configuration_sweep/UNCOVER_2c_smc_r300_ujy_photcal` | `geometry_ties/UNCOVER_2c_smc_r300_isophi_ujy_photcal` | `geometry_ties/UNCOVER_2c_smc_r300_spot_ujy_photcal` |
| CAPERS-UDS-23216 | `dust_law_and_configuration_sweep/CAPERS_2c_smc_ujy` | `geometry_ties/CAPERS_2c_smc_isophi_ujy` | `geometry_ties/CAPERS_2c_smc_spot_ujy` |

`alternative_configurations/` holds the same two ties under Calzetti attenuation, at
nlive = 2000 (`_n2k`), and, for UNCOVER-A2744-20698, on the spectrum before photometric
rescaling. Those are not the runs the paper reads and should not be used to reproduce a
published number.

**Two-component test (Section 4.4).** Every archival source refit with the hot-plus-cold
variant. Pair each with its single-photosphere counterpart of the same name under
`supplementary/prism/` and take the evidence difference from `summary/evidence.csv`. The paper
quotes 48 constrained sources rather than 49 because `GN9771` is one of the three red-flagged
fits above and enters no statistic. Its formal difference is the largest in the set at
Δln Z = +11.1, which is a property of a prior-railed fit rather than of a real cold component.

**Injection and recovery (Section 4.4, Appendix C).** Photospheres of known temperature and
gravity injected into real instrument noise and refit blind. The Egg injection ships its
injected truth vector as a separate run directory containing a one-dimensional array in the
same column order as its chain.

**Dust law and configuration sweep (Section 2.3).** Fits of each source under alternative
attenuation laws and model configurations, comparing the evidence for each. These span Small
Magellanic Cloud against Calzetti attenuation (`_smc` against `_cz`), cold-component
temperature floors (`_cool2600`,
`_cool2800`, `_cool3000`, `_floor2k`), the PRISM resolution check at R = 300 (`_r300`),
nebular emission on (`_neb`), the extended attenuation ceiling (`_av3`), and single against
two photospheres (`_1c` against `_2c`). The water-dot sweep adds the true PRISM resolution
curve in place of a fixed R = 100 (`_prismR`), the water band left unweighted (`_noupw`) or
masked entirely (`_h2omask`), the emission-line masks turned off (`_nomask`), a free
hot-photosphere metallicity (`_mhfree`), a second photosphere allowed up to 4500 K without the
water-band weight (`_warm`), nlive = 2000 (`_n2k`), the fits on the photometry-rescaled
UNCOVER-A2744-20698 spectrum (`_photcal`), the same configurations with rest-frame 3400-4100 A
excluded from the likelihood (`BMKP_`), and that mask plus two isolated residual regions
(`_xmask`). A name ending in `_long` repeats an identical configuration under a longer
wall-clock limit. `RS_m3c_smc` is the GN-28074 configuration of the primary fit without the
break mask. Each driver's docstring states what its run changes. No individual run is cited in
the text.

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

**Host star-formation history (Sections 4.3 and 4.4).** Fits repeated with the host's star
formation forced into its youngest bins (`_youngal`), which sets the host-mass systematic that
the mass-scale figure draws. These runs have no driver of their own; `run.json` names the
driver they are built from in `variant_of`.

**Egg model finalization (Appendix C).** The ladder that selected the adopted Egg
configuration, spanning the three model families (`base`, `xitb`, `duste`), broadening fixed
or free (`s123`, `sfree`), stellar metallicity fixed or free (`zfix`, `zfree`), and the
uncertainty-treatment variants (`globalfloor`, `jitter_floor`, `speccal1`, `dustsep`).
`egg_duste_sfree_zfree` selected the adopted configuration; the fit in
`chains/primary/J1025+1402` repeats it with the Ca II H and K cores masked, and the run here is
the same configuration with those pixels kept.

---

## Line and absorption feature tests

`line_tests/` holds the measurements made directly on the spectra rather than through the
photospheric fits. Each directory carries the script that made the measurement, its numerical
result, and every figure drawn from it. `test.json` names the paper section.

| directory | paper | contents |
|---|---|---|
| `hbeta_component_ladder` | Section 4.6 | the model-selection suite behind the four-component decomposition |
| `egg_hbeta_absorber` | Section 4.1 | the Hβ absorber investigation, including the joint and two-absorber refits |
| `egg_balmer_and_helium_absorption` | Section 4.1, Figure 3 | Hα, Hβ and He I λ10830 troughs in The Egg |
| `egg_paschen_absorption` | Section 4.1 | Paγ and Paδ absorption at −428 and −355 km/s |
| `balmer_progression` | Section 4.1 | absorption minima up the Balmer series and the two scenarios they distinguish |
| `broad_line_helium` | Section 4.6 | broad He I λ5876, the He II λ4686 limits and stack, the λ10830/λ20581 singlet bound |
| `rosetta_stone_lines` | Section 4.6 | the GN-28074 joint two-grating line fit and its chromosphere test |
| `narrow_line_diagnostics` | Section 4.6 | [O II] densities, Balmer decrements, the narrow-line budget, the Shirazi and OHNO diagnostics |
| `uds40579_helium` | Section 4.6 | the UDS-40579 He I trough |
| `vblue95_forecast` | Section 5.2, Figure 7 | the uniform v_blue,95 wind-launch measurements and P Cygni profile fits for every point in the wind-launch panel |

### The component-selection suite

Section 4.6 states that the four-component decomposition is selected by model comparison
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

The 28 DESI fits share one parameterized driver, `code/fit_desi_lrds2_roar.py`, which selects
a source by name. The host star-formation-history runs are built by `code/_variant.py` from the
driver named in their `variant_of` field,

```bash
python code/_variant.py bmkp_fit_wide_2c_smc_ujy youngal --pool 8
```

Every other run has its own driver, named in its `run.json`.

---

## Conventions and caveats

**Thinning.** Chains are released at up to 50,000 draws in `float32`. Nested sampling produced
these as equal-weight resamples in which rows repeat, and no original chain contains more than
46,973 *unique* draws, so the released set carries the full unique content of every posterior.
Where an original exceeded 50,000 rows the release takes a systematic subsample of the ordered
sequence, which preserves the distribution exactly. `run.json` records both counts.

**Precision.** Samples are stored at `float32`, roughly seven significant digits, which is far
finer than any posterior width in this work but is not bit-identical to the `float64` originals.

**Parameter names.** Taken from the run log of each fit, which records the model's own
parameter list at build time. Five runs had no surviving log. Four take their names from a run
of identical configuration and identical dimension, and GLIMPSE-17775 takes its names from the
model rebuilt directly from its driver. `run.json` records the provenance in `parameter_source`.

**Evidence.** `summary/evidence.csv` gives ln Z and its sampling uncertainty for all 378
posterior runs. The dynesty result objects themselves are not redistributed. They run to
several gigabytes and require a matching dynesty version to unpickle, and every evidence
comparison in the paper is reproducible from the table.

**Units.** `teff` is in kelvin, `logg` is base-ten log of surface gravity in cgs, `logL_star`
and `logL_cold` are base-ten log of luminosity in solar units, `sigma_smooth` is in km/s,
`mh_idx` indexes the library metallicity grid, and `dust2_gal` is the V-band optical depth of
the shared screen.

**Known issue.** `egg_model_finalization/egg_liu2026_exact.BADPRIOR` was run with a prior
specification error found afterwards and is superseded by `egg_liu2026_corrected`. It is kept
so the correction can be seen, and `run.json` records the issue. Do not use it for science.

**Not included.** Superseded development ladders that predate the adopted configuration
(the `m3a` through `m3d` model-development series and the earlier Egg `*_prospector_tlusty`
fits) are not released. No number in the paper depends on them.

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
