# Hawkes Branching-Ratio Identification

Reproducibility repository for the manuscript:

**What Hawkes branching ratios measure in high-frequency markets: specification sensitivity, mode-resolved bounds and critical slowing down in a Euro–Bund case study** (revised version, CSF manuscript CHAOS-D-26-09420)

Boyan Xing — Department of Mathematical Sciences, Faculty of Science and Technology, Beijing Normal-Hong Kong Baptist University, Zhuhai, Guangdong, China.

Submitted to *Chaos, Solitons & Fractals*; this version accompanies the revision (R1).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22774223.svg)](https://doi.org/10.5281/zenodo.22774223) (concept DOI; the submission version is 10.5281/zenodo.22774224)

## What this repository contains

- `src/` — Hawkes likelihoods, sum-of-exponentials (SoE) model and supporting estimators.
- `experiments/` — the core Bund experiment (`E1_bund_hawkes.py`), all CSF robustness additions (`A_csf_additions.py`) and figure generation (`make_figures.py`).
- `data/` — the public Euro-Bund futures event snapshot (`bund.npz`) distributed with the research workspace (X-DataInitiative/tick-datasets; Bacry–Jaisson–Muzy data lineage).
- `tables/` and `tables_csf/` — frozen CSV outputs used by every headline table.
- `figures_csf/` — figures of the original submission.
- `experiments/R_revision.py` — all analyses added in revision (tasks: `timestamp`, `history`, `nested`, `bivar`, `sim`, `adeq`, `boot`, `bootfix`, `soecheck`, `penalty`), resumable per session.
- `experiments/R_summaries.py`, `R_summaries2.py`, `R_summaries3.py`, `R_summaries4.py` — deterministic summaries of the revision tables; `experiments/R_make_figures.py` — revision figures.
- `tables_rev/` and `figures_rev/` — frozen revision outputs and vector figures.

## Reproduce the headline results

Tested with Python 3.13.

```bash
python -m pip install -r requirements.txt
python experiments/E1_bund_hawkes.py
python experiments/A_csf_additions.py all --reps 30
python experiments/make_figures.py
```

Revision analyses (about three hours of single-core computation, measured; every task resumes where it stopped):

```bash
for t in timestamp history nested bivar sim adeq boot bootfix soecheck penalty heldout_grids cv_const heldout_windows k11_variants cv_tight k11_tight; do python experiments/R_revision.py $t; done
python experiments/R_indirect.py calibrate && python experiments/R_indirect.py simulate && python experiments/R_indirect_summary.py && python experiments/R_indirect_matched.py
python experiments/R_moderes.py && python experiments/R_bivar_resid.py && python experiments/R_pooled_vr.py && python experiments/R_tick_np.py main 100 && python experiments/R_t4.py && python experiments/R_defensible.py && CADM=10 python experiments/R_defensible.py
python experiments/R_summaries.py && python experiments/R_summaries2.py && python experiments/R_summaries3.py && python experiments/R_summaries4.py
python experiments/R_make_figures.py csd bivariate stress penalty robustness moderes scatter modes
```

The revision analyses were run with the versions in `requirements-revision.txt` (Python 3.12.3); the headline full-session medians are identical under both environments.

The robustness script is computationally heavier than figure generation; the frozen CSVs used by the manuscript are included so tables and figures can be rebuilt without rerunning every fit.

## Locked headline results

- 859,467 Euro-Bund events (20 April-2014 sessions).
- Full-session median branching ratios: Exp 0.577 / SoE 0.936 (mid-price); Exp 0.478 / SoE 0.950 (trades).
- Flexible positive B-spline background: Exp 0.577 / SoE 0.870 (mid-price); Exp 0.412 / SoE 0.859 (trades).
- SoE basis K = 5–11 leaves the high branching-ratio conclusion stable over the principal support.
- Known-n multiscale simulations show severe downward pseudo-true bias under a one-scale exponential kernel.
- Symmetric bivariate Hawkes fit: median activity mode n+ = 0.578; signed mode n- = 0.040.

## Revision headline results

- Kernel dimension dominates a session-paired decomposition: Exp→SoE +0.363 [0.340, 0.374] (mid-price), +0.477 [0.308, 0.612] (trades).
- Nested free-rate kernels: n rises 0.577 → 0.779 → 0.863 → 0.936 (K = 1, 2, 3, grid; mid-price); every multiscale kernel wins out of sample in 20/20 sessions.
- Non-self-exciting Cox / regime controls give spurious n ≈ 0.91–0.94 under both kernels (0.82–0.89 with a 30-min spline).
- Bivariate SoE: n+ = 0.935, n− = −0.259 (negative in 20/20 sessions); cross-excitation carries all slow mass.
- Refitted parametric bootstrap: every candidate lies outside its null band in every session; within-model SE(n) 0.004–0.010.
- Nonparametric pooled kernel norm (mid-price, Table 3): 0.726 / 0.874 / 0.934 at 1 / 10 / 100 s support; up/down cross-kernel norm 0.59 [0.57, 0.62] at 1 s and 0.86 [0.82, 0.90] at 10 s.
- Slow cross-excitation slows self-excitation (Table 9): 0.228 in the data; +0.235 [0.211, 0.259] after subtracting the log-OU control.
- Mode-resolved lower bounds on n+ (Table 10): 0.62 [0.59, 0.67] at W* = 60 s, 0.75 [0.62, 0.81] at 300 s, pooled 0.82 [0.71, 0.92].

## Randomness

The controlled stress map, timestamp randomisation, all revision simulations and the bootstrap use seed `20260914` (one derived seed per replication in the bootstrap). Optimisation is deterministic conditional on the data and the initialisation in the scripts.

## AI-assisted code development

OpenAI ChatGPT (GPT-5.6 Sol; OpenAI, accessed September 2026) assisted with drafting/refactoring robustness-analysis and plotting code for the original submission; Anthropic Claude assisted with drafting the revision analysis code (`experiments/R_*.py`). All reported results were generated by executing Python code against the supplied data, frozen to CSV, and checked against the manuscript tables. The author retains full responsibility.

## License

Code is released under the MIT License (see `LICENSE`). See `DATA_REDISTRIBUTION_NOTE.md` for the provenance and redistribution terms of the bundled data snapshot.

## Citation

See `CITATION.cff`. Once the manuscript is accepted, add the journal DOI.
