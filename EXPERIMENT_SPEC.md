# Experiment Specification

Status: **DRAFT — must be frozen before final test evaluation**

## 1. Research question
Can conditional volatility information and nonlinear sequence models improve one-day-ahead SPY VaR forecasts relative to simple and classical statistical baselines at alpha = 1%, 5%, and 10%?

Secondary questions:
- Does realized volatility (`rv5`) add value beyond return history?
- Does bipower variation (`bv`) and a jump proxy add incremental information beyond realized volatility?
- Are gains, if any, stable across tail levels and market regimes?
- Does a neural model improve calibrated quantile forecasts, rather than merely changing failure rates?

## 2. Target
For each forecast origin t, predict the conditional return quantile for t+1:

`q_alpha,t+1` such that `P(r_t+1 <= q_alpha,t+1 | F_t) = alpha`, alpha in {0.01, 0.05, 0.10}.

Violation indicator:
`I_alpha,t+1 = 1[r_t+1 <= q_alpha,t+1]`.

## 3. Dataset
Daily SPY observations with:
- `log_ret`: logarithmic daily return.
- `rv5`: realized volatility/variance measure constructed from 5-minute data.
- `bv`: bipower variation.

Dataset currently contains 4,640 daily observations from 2000-01-04 through 2018-06-27.

Before modeling, verify date uniqueness, monotonicity, missingness, finite values, and the empirical scale/definition of `rv5` and `bv`.

## 4. Information set
Candidate causal inputs available at t:
- current and lagged `log_ret`;
- absolute/squared/negative-return transforms;
- log-transformed `rv5` and `bv`;
- jump proxy such as `max(rv5 - bv, 0)` where justified;
- rolling/HAR-style daily-weekly-monthly volatility summaries using only data through t.

No feature may use t+1 information.

## 5. Data split
**To be frozen after validation design discussion and before test evaluation.**

Candidate protocol A:
- initial model-development/training era: 2000–2005;
- validation/model-selection era: 2006–2007;
- frozen final test: 2008–2018.

Candidate protocol B:
- training: 2000–2006;
- validation: 2007–2010;
- frozen final test: 2011–2018.

Final choice must state the reason and acknowledge statistical-power tradeoffs for the 1% tail.

## 6. Rolling protocol
All models must use a rolling-window framework.

Candidate window lengths for validation only: 1000, 1500, 2000 trading days. Select/freeze one primary window before final test evaluation. Robustness may report alternatives if predeclared.

For forecast t+1:
1. construct the allowed rolling training sample ending at t;
2. fit all learned preprocessing on that sample only;
3. fit/update the model according to the predeclared rule;
4. generate q_1%, q_5%, q_10%;
5. save the prediction before reading t+1 outcome in the experiment logic.

Neural retraining policy (full retrain, periodic retrain, or warm start) must be explicitly specified. If warm starts retain information from observations that have rolled out of the window, this must be analyzed and not silently treated as a pure fixed-window estimator.

## 7. Model ladder
### M0 — Historical Simulation
Rolling empirical quantiles. Purpose: transparent non-parametric baseline.

### M1 — GARCH-t
GARCH(1,1) with Student-t innovations as the primary volatility/conditional-distribution baseline. Gaussian GARCH may be a robustness comparator.

### M2 — Direct conditional quantile model
Choose one primary interpretable direct-quantile model after validation: linear quantile regression, CAViaR, or HAR-style quantile regression using realized measures.

### M3 — Neural quantile model
Small MLP-QR or GRU-QR model with joint 1/5/10% outputs and non-crossing design/penalty. Architecture must be modest relative to the ~4.6k-day sample size.

## 8. Neural ablations
Prefer information ablations over architecture proliferation:
- returns only;
- returns + RV;
- returns + RV + BV;
- returns + RV + BV + jump/HAR features.

Primary seed is predeclared. Robustness: 3–5 seeds, reporting mean and standard deviation. Never select the best seed as the headline result.

## 9. Evaluation
For every model and alpha:
- number of forecasts;
- number of violations;
- empirical failure rate;
- expected violations;
- Kupiec unconditional coverage test;
- Christoffersen independence test;
- conditional coverage test;
- mean pinball/quantile loss;
- quantile crossing rate where relevant.

Candidate enhanced evaluation:
- Dynamic Quantile test;
- Diebold–Mariano or block-bootstrap comparison of quantile losses;
- crisis/non-crisis or regime-stratified analysis.

Failure rate alone is never sufficient to rank models.

## 10. Primary comparison rules
- Compare all models on the same frozen test dates.
- Rank/calibrate separately at 1%, 5%, and 10%; do not assume one universal winner.
- Distinguish calibration from sharpness/loss.
- Negative results are valid: neural models are not required to outperform classical methods.

## 11. Final-test freeze checklist
Before opening final test results, commit:
- split dates;
- rolling window length;
- feature definitions;
- model families and primary hyperparameters;
- retraining schedule;
- seeds;
- evaluation metrics/tests;
- primary plots/tables.

Tag or otherwise identify this commit as the frozen specification.
