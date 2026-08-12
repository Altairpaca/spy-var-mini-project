# AGENTS.md — Research and Coding Guardrails

This repository is a research submission. Correct temporal alignment and reproducibility take priority over speed or model complexity.

## Non-negotiable rules
1. **No future information.** A forecast for day `t+1` may use only information observable by the end of day `t`.
2. **No global preprocessing.** Scalers, transforms, feature selectors, PCA, clipping thresholds, and similar learned preprocessing must be fitted only on the allowed training window.
3. **No test-set tuning.** Do not alter architecture, hyperparameters, window length, feature set, or evaluation choices because of final test results.
4. **Common forecast panel.** All compared models must be evaluated on identical forecast dates unless a model failure is explicitly documented.
5. **Single evaluation implementation.** Coverage, exceedance indicators, quantile loss, and VaR tests must use shared code.
6. **Preserve VaR sign convention.** VaR is the return quantile itself: lower-tail thresholds are typically negative. A violation occurs when realized return <= forecast VaR.
7. **Quantile ordering.** Neural or direct multi-quantile models should satisfy q_1% <= q_5% <= q_10%, or report and explicitly handle crossing.
8. **Reproducibility.** Every reported number must be regenerable from scripts and a committed config.
9. **Do not cherry-pick seeds.** Predeclare the primary seed. Robustness may report multiple seeds as mean/std.
10. **Do not silently drop failures.** Optimization failures, non-convergence, NaNs, or missing forecasts must be logged.

## Required tests before trusting results
- Target/feature date alignment.
- Rolling-window boundaries.
- No-leakage preprocessing test.
- Forecast-date equality across models.
- VaR exceedance sign convention.
- Quantile ordering/crossing rate.
- Deterministic rerun for deterministic components.

## Engineering rules
- Notebooks are for exploration only. Final results must come from scripts.
- Prefer config-driven experiments.
- Keep model implementation separate from rolling orchestration and evaluation.
- Save raw per-date forecasts before aggregating metrics.
- Save experiment metadata: git commit, config, seed, package versions, runtime, and hardware.
- Do not overwrite frozen final-test outputs without an explicit new experiment ID.

## Suggested resource policy on altair-server
For CPU-parallel rolling classical models, benchmark 16/32/64 workers before increasing concurrency. Set BLAS/OpenMP threads per worker to 1 to avoid oversubscription. Use the RTX 3060 for neural experiments; avoid launching many GPU workers simultaneously.
