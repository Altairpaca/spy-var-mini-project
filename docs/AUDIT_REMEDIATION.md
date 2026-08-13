# Audit Remediation Log

Audit session: 2026-08-13. Branch: `audit/correctness-rerun` (based on `research/full-experiment@d10c762`).

This document records every audit finding, declares which prior results are invalid, and specifies the remediation. It is part of the permanent research record; old experiments remain recoverable from Git history.

## 1. INVALIDATION STATEMENT

The following prior final-test conclusions are declared **INVALIDATED BY AUDIT — DO NOT USE FOR MODEL SELECTION OR FINAL CLAIMS**:

- GARCH(1,1)-Student-t final VaR series (M1)
- GJR-GARCH(1,1,1)-Student-t final VaR series (M4)
- All failure rates, Kupiec / Christoffersen / DQ statistics, pinball losses, DM and bootstrap comparisons, and regime stratifications involving M1 or M4
- Any headline such as "best model at tail X" derived from the old frozen run
- Any statement about neural-vs-classical ranking that relied on GARCH-family baselines

Reason: the Student-t VaR construction used the **unstandardized** `scipy.stats.t.ppf(alpha, nu)`, while `arch`'s Student-t innovation is a **unit-variance standardized** t distribution. The correct quantile is `StudentsT().ppf(alpha, nu)`, mathematically equal to `stats.t.ppf(alpha, nu) * sqrt((nu - 2) / nu)`. At the fitted degrees of freedom observed in the data (nu roughly 5-12) this overstates |VaR| by approximately sqrt(nu/(nu-2))-1, i.e. about 10-25% at the 1% tail. The direction and size of the error are data-dependent (they vary with the fitted nu per rolling window), so no old GARCH-family number can be salvaged by a constant correction factor.

Explicitly NOT invalidated (unaffected):
- Historical Simulation (M0), Linear/HAR Quantile Regression (M2), Multi-Quantile MLP (M3), GRU (M5) prediction panels
- Data audit, feature engineering, rolling-engine semantics, leakage tests
- Development-period window comparison and neural search evidence (these never read the old final test)
- Gaussian-GARCH diagnostic (uses norm.ppf, scale-correct)

## 2. AUDIT FINDINGS AND REMEDIATION PLAN

### 2.1 P0 — Student-t VaR scale error (correctness)
- Finding: `stats.t.ppf(alpha, nu)` used without the standardization factor `sqrt((nu-2)/nu)`.
- Fix: use `arch.univariate.StudentsT().ppf` for the innovation quantile in M1 and M4.
- Tests: `tests/test_garch_t_scale.py` (arch-vs-scaled-scipy equality at nu in {4.5, 6, 10, 30}, unit-variance property, regression guard against the naive form, end-to-end consistency of all three tails with the back-solved mean, Gaussian unaffected).
- Status: FIXED.

### 2.2 P0 — Target-date partition boundary (correctness)
- Finding: origin-based date filtering put the 2007-12-31 origin (target 2008-01-02) into development; the partition must be defined by the **forecast target date**.
- Fix: `scripts/common.py::forecast_origins` now filters on `target = index[pos + 1]`.
- Tests: `tests/test_target_dates.py` (no dev target >= split, no final target < split, disjoint and gapless coverage, cross-boundary origin assigned to final, first final target = 2008-01-02).
- Status: FIXED.

### 2.3 P1 — MLP non-crossing head initialization (numerical quality)
- Finding: the final-layer bias init yields initial softplus gaps of log(1+exp(0)) ≈ 0.693, far above the scale of daily return quantile gaps (roughly 0.5-1.5 percentage points).
- Fix (chosen): Scheme B — train on strictly train-only standardized targets (subtract train mean, divide by train std), predict standardized quantiles, invert the transform for the reported VaR. Rationale: scale-invariant by construction, no dependence on a development quantile-gap estimate, and it also stabilizes the pinball loss gradient scale. Recorded in RESEARCH_LOG.
- Status: PENDING (next).

### 2.4 P1 — Freeze enforcement (provenance)
- Finding: the freeze gate recorded data/config SHA256 and HEAD, but did not verify (a) working tree cleanliness, (b) code/evaluator signature, (c) the *effective* data path when `--data` override is used.
- Fix: gate now verifies working-tree clean (`git status --porcelain` empty), adds an evaluator/code signature (hash of src/ + scripts/ + tests/), and freezes the actual data file hash. Final run forbids `--data` override unless `--force` with explicit hash re-check.
- Status: PENDING (next).

### 2.5 P1 — Stale artifact silent reuse (provenance)
- Finding: "output exists -> skip" could silently reuse artifacts from a different config/data/commit.
- Fix: every artifact carries an experiment signature (data hash, config hash, git HEAD, model, feature set, window, seed, target-date range, env signature); reuse only on exact signature match; formal audited runs use `--clean-run` writing under `outputs/runs/<freeze_id>/`; evaluation/report read only the canonical freeze run.
- Status: PENDING (next).

### 2.6 P1 — Neural hyperparameter search provenance (provenance)
- Finding: the development search grid was predeclared, but the pipeline did not force the search -> decision artifact -> final.yaml chain to actually run in the rerun; `update_final_config.py` was not a formal pipeline stage.
- Fix: rerun development pipeline with explicit stages; require non-empty `outputs/development/neural_search.csv` and `neural_search_decision.json`; `update_final_config` writes final.yaml only from these artifacts. Grid unchanged (no expansion).
- Status: PENDING (rerun).

### 2.7 P1 — Window-selection aggregation (statistical)
- Finding: summing raw pinball losses across tails biases toward tails with larger loss scale and gives models with more feature configurations higher weight.
- Fix: per (model, feature-set, tail) candidate-relative normalized loss, equal-weight average; report per-configuration winners, pairwise win counts, and sensitivity. Candidates stay {1000, 1500}.
- Status: PENDING (next).

### 2.8 P1 — Statistical reporting upgrades (statistical)
- DQ: keep as auxiliary diagnostic; add unit tests (power against clustered violations, size under i.i.d. violations, edge cases); explicitly labeled auxiliary in the report.
- DM: predeclare headline comparison families (GARCH-t vs MLP, LinQR vs MLP, MLP vs GRU, GARCH-t vs GJR-t) with Holm family-wise correction; full pairwise matrix stays in the appendix; block bootstrap reported alongside, discrepancies stated plainly.
- Calibration language: "closest empirical failure rate" (unconditional frequency) is distinguished from conditional adequacy (Kupiec UC + Christoffersen IND/CC); a model with correct frequency but clustered violations is never called conditionally well calibrated.
- Seed robustness: seeds {7, 42, 2026} all reported (n_seeds=3), mean/std per neural model x feature set x tail; any GRU seed sensitivity stated plainly.
- Quantile crossing of Linear QR reported per feature set; MLP/GRU structural non-crossing presented as a genuine method advantage; post-hoc monotone rearrangement allowed only as an auxiliary diagnostic, not as a new primary model.
- Status: PENDING (next).

### 2.9 P1 — Report generator (reporting)
- Remove all hard-coded result sentences; every number read from frozen output tables.
- Forecast count from the canonical panel, not from a unique-count of a metric column.
- Escape all literal `%` as `\%`; add `\hypersetup{hidelinks}`; use the repository author metadata (no placeholder author).
- Add explicit math for GARCH(1,1), GJR-GARCH, linear quantile regression, pinball loss, MLP non-crossing head, and the GRU role.
- Status: PENDING (later stage).

### 2.10 P1 — Regime language (reporting)
- Report two-sided regime adaptation for HS when the data supports it (over-violation at crisis onset, under-violation after calm resumes) rather than only "underestimation during crises".
- Status: PENDING (after rerun).

## 3. NON-NEGOTIABLE PROTOCOL FOR THIS RERUN

1. No post-hoc model shopping: model set, feature sets F0-F3, window candidates {1000, 1500}, seeds {7, 42, 2026} (primary 42), loss functions, and the research questions are unchanged from the original predeclared protocol.
2. Allowed changes come only from (a) the correctness fixes above, (b) development-only evidence, (c) reproducibility/reporting fixes.
3. Sequence: data audit/tests -> development rerun -> window decision -> neural search -> decision artifacts -> update final.yaml -> freeze commit -> clean final rerun -> statistics -> figures -> report -> QA -> merge.
4. No old final-test number is used at any decision point in this rerun.
