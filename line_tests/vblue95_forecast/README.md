# v_blue,95 wind-launch measurements (Section 5.2, Figure 7)

This directory holds every absorption-line velocity that enters the right panel of
Figure 7, the wind-launch test. Each velocity is a `v_blue,95`: the blueshifted
velocity at which the transmission through the absorber recovers to 95%, following the
definition of Naidu et al. 2026. It traces the fastest outflowing material along the
line of sight, and we take it as a proxy for the wind terminal velocity.

## What is here

- `profiles/` — one panel per line fit, named `v95_<source>_<line>.png`. Each panel
  shows the continuum-normalized profile (gray), the best-fit model (red), and, below,
  the model transmission with the `v_blue,95` crossing marked. These are the P Cygni
  profiles behind every point in the figure.
- `vblue95_measurements.csv` — the adopted velocity for each source, with its
  provenance (measured here, or, for two sources with no public spectrum, the published
  value of Naidu et al. 2026).
- `fit_matthee_eq1.py`, `fit_v95_all.py` — the fitting code. `fit_matthee_eq1.py` builds
  the model; `fit_v95_all.py` runs it on the DESI, JWST-grating, and ground-based
  echelle spectra.
- `egs42046_pcygni.{pdf,png}` — the paper's worked example (Figure of Section 5.2),
  RUBIES-EGS-42046 Hα. `gn28074_pcygni.*` and `egg_pcygni.*` are the same figure for two
  other sources.
- `balmer_forecast_1x2.pdf` — the assembled Figure 7.

## The model

We fit each continuum-subtracted line profile with Equation 1 of Matthee et al. 2026: a
Gaussian emission line and its electron-scattering-broadened counterpart, attenuated by a
skewed-Gaussian absorber, plus an unabsorbed narrow component (with [N II] for Hα) whose
width and redshift are tied to the [O III] doublet where it is covered. The full model is
convolved with each instrument's line-spread function. `v_blue,95` is read from the
composite transmission, the model divided by the same model with the absorber removed.

Two deliberate departures from the letter of Matthee et al. 2026 are documented in the
paper: the absorber is fit with a Gaussian core (the damping wings are pinned) to avoid a
known degeneracy, and the optimizer is least-squares rather than MCMC. The pipeline
reproduces four of the seven `v_blue,95` that Naidu et al. 2026 report for sources with
public spectra to within a few percent; the remaining three profiles admit more than one
decomposition.

## Provenance notes

- Every velocity in the figure is measured here except A2744-45924 and GS-13971, whose
  custom reductions are not public; for those two we adopt the published value.
- A2744-QSO1 has no public grating spectrum and does not appear in the right panel.
- For five sources (Cliff, GLIMPSE-17775, MoM-BH*-1, RUBIES-EGS-29489, and the DESI
  source J094411) the absorber improves the fit without being decisively required
  (ΔBIC > −10); we plot our measured value for uniformity.
- The two He I λ10830 velocities (The Egg, UDS-40579) are metastable-helium wind tracers,
  discussed in Paper II, not Balmer `v_blue,95`. The Egg's He I profile is a deep P Cygni
  whose absorption drops below the continuum, outside the emission-times-absorber model;
  its trough measurement lives in `../egg_balmer_and_helium_absorption`.
