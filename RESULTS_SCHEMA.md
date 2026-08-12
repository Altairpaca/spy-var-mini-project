# Results and Output Contract

Final numeric claims in the report must be traceable to machine-generated outputs following this contract.

## 1. Forecast panel
One row per forecast date and model, or a wide equivalent with lossless conversion.

Required fields:
- `forecast_date`
- `target_date`
- `model_id`
- `experiment_id`
- `q_001`
- `q_005`
- `q_010`
- `realized_log_ret`
- `violation_001`
- `violation_005`
- `violation_010`
- `fit_status`

Recommended metadata:
- rolling window start/end dates;
- seed;
- fit/runtime information.

## 2. Metric table
For each `(experiment_id, model_id, alpha)`:
- `n_forecasts`
- `n_violations`
- `expected_violations`
- `failure_rate`
- `target_alpha`
- `mean_pinball_loss`
- `kupiec_lr`
- `kupiec_pvalue`
- `christoffersen_ind_lr`
- `christoffersen_ind_pvalue`
- `conditional_coverage_lr`
- `conditional_coverage_pvalue`
- `crossing_rate` where applicable

Optional enhanced fields:
- DQ statistic/p-value;
- loss-comparison statistic/p-value;
- regime label.

## 3. Required figures
At minimum, generate reproducibly:
1. realized returns with dynamic VaR overlays for each alpha/model or selected comparison panels;
2. violation markers through time;
3. model comparison of failure rates vs target alpha;
4. quantile-loss comparison;
5. data exploration showing return tails and volatility clustering.

## 4. Integrity checks
Before report generation:
- identical target dates across primary models;
- no duplicate forecast rows;
- violation columns recomputed from realized return and q columns;
- q_001 <= q_005 <= q_010 check;
- no NaN/inf in primary result rows unless explicitly documented;
- metric table regenerated from forecast panel, never hand-edited.
