# Correction Pass Log — final-pass

Session: 2026-08-13 (post-merge). Branch: `correction/final-pass` (based on `main@c85a8dd`).

This log tracks the independent (ChatGPT) audit findings and their disposition. Every finding is fixed in code/report/docs, then the protocol order `dev → select → search@chosen W → update → freeze → clean final rerun` is re-executed. No models, features, search grids, seeds, or losses are changed.

## Findings and fixes

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| 1 | Neural hyperparameters searched at W=1000 (`min(candidates)`) while final window is 1500 — protocol violation (search must run at the chosen window) | P1-high | `neural_search.py` reads the chosen window from `outputs/development/window_decision.json`; pipeline order fixed to `dev → select → search → update`; search re-run at W=1500 with the identical predeclared grid (M3 16 configs, M5 8 configs) |
| 2 | README "one-command full reproduction" does not actually work (final.yaml has no `neural_search` section; run_all would overwrite development artifacts with empty search) | P0 reproducibility doc | README rewritten as an explicit three-phase protocol: development pipeline → Git freeze boundary → frozen OOS run; report-only regeneration kept separate |
| 3 | PDF: conclusion reads 5% tail rows but claims "strong 1% tail calibration" | P0 report fact | `_conclusion_latex` reads the 1% rows for the 1% sentence; wording replaced by "closest 1% failure rate among primary models, although unconditional coverage still rejects the nominal 1% rate at the 5% level" |
| 4 | "over-conservative" direction reversed (1% rates 1.5% > 1% are under-conservative; GRU 3.2% strongly under-conservative) | P0 concept | PDF, FINAL_SUMMARY_ZH and conclusion text corrected to under-conservative / underestimates left-tail risk |
| 5 | FINAL_SUMMARY_ZH: "GARCH 族 1%/5% Ind/CC 未被拒绝" is false (GARCH-t CC 1% p=0.017 rejected, 5% p=0.007 rejected; GJR-t 1% p=0.089 not rejected, 5% p=0.008 rejected) | P0 fact | Rewritten per-model: independence not clearly rejected at 1%/5%, but conditional coverage passes only for GJR-t at 1% |
| 6 | PDF: "10% CC rejected for the GARCH family" — GJR-t CC 10% p=0.074 is NOT rejected | P0 fact | Split per model: GARCH-t fails 10% CC at 5% level, GJR-t does not |
| 7 | "GJR proves leverage effect" over-claimed (1% Holm p=0.071 not significant; loss comparison, not parameter evidence) | P1 claim | "consistent with an economically relevant leverage/asymmetry effect"; significance claimed only at 5%/10% |
| 8 | Neural negative result wording too strong while search window mismatched; 5% tail not significant | P1 claim | Softened to "no consistent absolute OOS improvements under the pre-specified family and development-only tuning"; re-verified after W=1500 search |
| 9 | NN early stopping drops the most recent 10% of each window from gradient training (effective ~1350 obs at W=1500) — hidden disadvantage vs GARCH/QR | P2 limitation | NOT changed this pass (final already seen; changing training now would be test-driven); written as an explicit limitation in report and FINAL_SUMMARY_ZH |
| 10 | Neural search power limited: 101 origins, unnormalized mean pinball across tails | P2 note | Acknowledged in report (selection based on sparse grid and unnormalized tail-average loss; grid predeclared, unchanged) |
| 11 | Ablation conclusion should state RV is the only clearly useful realized-measure block; BV|RV ≈ 0; F3 block slightly negative | P1 wording | Report and summary rewritten with marginal deltas (ΔRV, ΔBV|RV, Δstructured) |
| 12 | Table 4 renders `textbfPair` (double backslash) and `->` renders wrongly | P0 visual | `_dm_latex` header rebuilt with single-backslash commands; arrows replaced by `$\rightarrow$`; page-by-page render QA |
| 13 | Table 4 packs all 45 pairwise tests in the body | P1 readability | Body keeps only the 12 headline tests (4 pairs × 3 tails); full 45-row matrix moved to the appendix |
| 14 | Report artifact paths stale (`outputs/predictions/*.parquet`, `outputs/tables/seed_robustness.csv`) | P1 provenance | Paths generated from `outputs/manifests/current_run.json` (canonical run dir); seed summary filename corrected |
| 15 | Freeze naming: `git_commit` in manifest is the freeze-generation HEAD, its child is the manifest commit — both called "freeze commit" | P2 doc | Renamed in docs as `freeze_base_commit` / `freeze_manifest_commit`; manifest field kept as `git_commit` (already documented semantics) |
| 16 | H1–H7 ratings updated per audit (H1 strongly supported with two-sided lag; H2 supported but calibration mixed; H3a supported at 5/10%; H3b unsupported; H4/H5 unsupported/inconclusive pending W=1500 search; H6 supported; H7 not identified) | P1 | Ratings rewritten in report hypothesis section and FINAL_SUMMARY_ZH after the rerun |

## Protocol for this pass (unchanged research decisions)

- Models M0–M5, features F0–F3, window candidates {1000, 1500}, seeds {7, 42, 2026} (primary 42), pinball loss, early-stopping algorithm: **unchanged**.
- Order: data audit/tests → development rerun → window selection → **neural search at chosen window** → decision artifacts → update final.yaml → full tests → freeze commit → clean final rerun → statistics/figures/report/summary → visual QA → clean reproduction → merge.
- No old final-test number is used at any decision point.
