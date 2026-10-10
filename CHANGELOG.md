# Changelog

All notable changes to this repository are documented in this file.

## v1.2.0-revision (2026-10-10)

R1 revision archive for Chaos, Solitons & Fractals manuscript CHAOS-D-26-09420.

### Added

- Nine revision scripts under `experiments/`:
  - `R_defensible.py` 鈥?defensibility labels over all 98 specifications (Table 11, Fig. 9).
  - `R_t4.py` 鈥?window admissibility and grid comparison (local intervals in Table 11).
  - `R_indirect.py`, `R_indirect_summary.py`, `R_indirect_matched.py` 鈥?calibrated exogenous log-OU benchmark (Table 9).
  - `R_moderes.py` 鈥?mode-resolved bounds and Appendix D simulations (Table 10, Table D.15, Fig. 8, Section 5.11).
  - `R_tick_np.py` 鈥?nonparametric Wiener-Hopf kernel via `tick==0.8.0.2` (Table 3).
  - `R_bivar_resid.py` 鈥?bivariate residual diagnostics (bivariate rows in Table 6).
  - `R_pooled_vr.py` 鈥?pooled signed variance ratio, with and without within-sample centring (Section 5.11 pooled VR-, Table 10 pooled row).
- `tables_rev/R_defensible_labels.csv` and all frozen outputs of the nine new scripts.
- Three revision figures: `figures_rev/R_fig_modes.pdf` (Fig. 6), `R_fig_moderes.pdf` (Fig. 8), `R_fig_scatter.pdf` (Fig. 9).
- `requirements-revision.txt` now pins `tick==0.8.0.2`.

### Changed

- Existing scripts (`R_revision.py`, `R_summaries*.py`, `R_make_figures.py`, `A_csf_additions.py`, `E1_bund_hawkes.py`, `make_figures.py`, `src/hawkes.py`) and frozen CSVs updated to the author's final revision outputs; no path or entry-point changes were required beyond those already present.
- `README.md`, `REPRODUCIBILITY.md`, `CITATION.cff` updated to the revised manuscript title, contact email (xby040909@outlook.com), concept DOI 10.5281/zenodo.22774223, and v1.2.0-revision metadata.
- `SHA256SUMS.txt` regenerated over the full file set.

### Added

- `.gitattributes` with `* -text` to disable all line-ending conversion; SHA256SUMS.txt regenerated over LF-normalized files.

### Not changed

- `data/bund.npz` is byte-identical to the upstream tick-datasets snapshot (SHA-256 fa1d4679鈥d8c), BSD-3-Clause.
- Random seed remains `20260914`; `LICENSE` (MIT) and `DATA_REDISTRIBUTION_NOTE.md` unchanged.
- No force-push; `v1.0.0-submission` still points at `09ec2c8` and `v1.1.0-revision` is untouched.

## v1.1.0-revision (2026-10-08)

First revision archive; missing several scripts and figures, superseded by v1.2.0-revision.

## v1.0.0-submission (2026-09)

Original submission archive, commit `09ec2c8`, DOI 10.5281/zenodo.22774224.
