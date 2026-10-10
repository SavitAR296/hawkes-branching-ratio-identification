# Changelog

All notable changes to this repository are documented in this file.

## v1.2.2-revision

Adds one script and its output for the revised Section 5.11 and fixes two documentation errors. No existing script, CSV, JSON or PDF changes, and no number reported in v1.2.1 moves.

### Added

- `experiments/R_pooled_vr_long.py` and `tables_rev/R_pooled_vr_long.json`: the pooled signed variance ratio of `R_pooled_vr.py` at bin widths of 1 h, 2 h, 3.5 h, 7 h and one full session, with the finite-bin-width bound of Corollary 2 and the share of full-session signed variance carried by the three most trending sessions (Section 5.11). Same estimator, bootstrap and seed (20260914) as `R_pooled_vr.py`; the 1-h row reproduces the value already reported.

### Fixed

- README: the Table 9 headline read "Slow cross-excitation slows self-excitation"; it now states that slow cross-excitation exceeds slow self-excitation (slow cross-minus-self mass).
- CHANGELOG: em dashes and an ellipsis had been saved in the wrong encoding (Windows code page); restored to UTF-8.

## v1.2.1-revision (2026-10-10)

Metadata and reproducibility-only patch over v1.2.0-revision. No script, CSV, JSON, or PDF changes; no numbers, figures or conclusions move.

### Fixed

- `SHA256SUMS.txt` previously listed its own hash, causing one `FAILED` line under `sha256sum -c`; regenerated to exclude itself (and `.gitattributes`/`.gitignore`, which git manages directly).
- `requirements-revision.txt` listed `python==3.12.3`, which is not a PyPI package and made `pip install -r` / `uv pip install -r` fail; replaced with a comment `# Python 3.12.3 (interpreter version; not installable by pip)`. Added `.python-version` pinning 3.12.3.

### Changed

- `README.md`: added `python -m pip install -r requirements-revision.txt` and `sha256sum -c SHA256SUMS.txt` as the first two lines of the revision-analyses code block; listed the nine revision helper scripts explicitly in "What this repository contains"; expanded the `R_revision.py` task list to the full 16 tasks actually run.
- `CITATION.cff`: version bumped to `1.2.1-revision`; title uses the en-dash form "Euro–Bund" to match the manuscript; `date-released` set to 2026-10-10.
- `REPRODUCIBILITY.md`: Archive identifiers now record v1.2.0-revision DOI 10.5281/zenodo.23273209 and explain why it is superseded by this patch.

### Removed

- `figs.sha256`, a temporary verification file that was committed by mistake.

## v1.2.0-revision (2026-10-10)

R1 revision archive for Chaos, Solitons & Fractals manuscript CHAOS-D-26-09420.

### Added

- Nine revision scripts under `experiments/`:
  - `R_defensible.py` — defensibility labels over all 98 specifications (Table 11, Fig. 9).
  - `R_t4.py` — window admissibility and grid comparison (local intervals in Table 11).
  - `R_indirect.py`, `R_indirect_summary.py`, `R_indirect_matched.py` — calibrated exogenous log-OU benchmark (Table 9).
  - `R_moderes.py` — mode-resolved bounds and Appendix D simulations (Table 10, Table D.15, Fig. 8, Section 5.11).
  - `R_tick_np.py` — nonparametric Wiener-Hopf kernel via `tick==0.8.0.2` (Table 3).
  - `R_bivar_resid.py` — bivariate residual diagnostics (bivariate rows in Table 6).
  - `R_pooled_vr.py` — pooled signed variance ratio, with and without within-sample centring (Section 5.11 pooled VR-, Table 10 pooled row).
- `tables_rev/R_defensible_labels.csv` and all frozen outputs of the nine new scripts.
- Three revision figures: `figures_rev/R_fig_modes.pdf` (Fig. 6), `R_fig_moderes.pdf` (Fig. 8), `R_fig_scatter.pdf` (Fig. 9).
- `requirements-revision.txt` now pins `tick==0.8.0.2`.

### Changed

- Existing scripts (`R_revision.py`, `R_summaries*.py`, `R_make_figures.py`, `A_csf_additions.py`, `E1_bund_hawkes.py`, `make_figures.py`, `src/hawkes.py`) and frozen CSVs updated to the author's final revision outputs; no path or entry-point changes were required beyond those already present.
- `README.md`, `REPRODUCIBILITY.md`, `CITATION.cff` updated to the revised manuscript title, contact email (xby040909@outlook.com), concept DOI 10.5281/zenodo.22774223, and v1.2.0-revision metadata.
- `SHA256SUMS.txt` regenerated over the full file set.
- Line-ending normalization: `.gitattributes` set to `* -text`; all existing files that appeared changed versus `v1.1.0-revision` (`src/hawkes.py`, `R_summaries*.py`, `tables/*.csv`, `tables_csf/*.csv`) are byte-identical after LF normalization — the apparent diff was purely CRLF/LF, not content.
- `R_revision.py` adds six tasks for the operational defensibility rule (Section 3.1): `heldout_grids`, `cv_const`, `heldout_windows`, `cv_tight`, `k11_variants`, `k11_tight` (forward held-out log-likelihood, constant-baseline cross-validation, K=11 grid variants).
- `R_make_figures.py` reads the new `R_cv_tight.csv` and `R_cv_constant.csv` for the penalty/λ scan; minor legend position/fontsize/ylim adjustments for Fig. 4.

### Added

- `.gitattributes` with `* -text` to disable all line-ending conversion; SHA256SUMS.txt regenerated over LF-normalized files.

### Not changed

- `data/bund.npz` is byte-identical to the upstream tick-datasets snapshot (SHA-256 fa1d4679…bd8c), BSD-3-Clause.
- Random seed remains `20260914`; `LICENSE` (MIT) and `DATA_REDISTRIBUTION_NOTE.md` unchanged.
- No force-push; `v1.0.0-submission` still points at `09ec2c8` and `v1.1.0-revision` is untouched.

## v1.1.0-revision (2026-10-08)

First revision archive; missing several scripts and figures, superseded by v1.2.0-revision.

## v1.0.0-submission (2026-09)

Original submission archive, commit `09ec2c8`, DOI 10.5281/zenodo.22774224.
