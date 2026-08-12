# Assignment Summary

Source: provided PhD pre-screening mini-project instructions.

## Task
Forecast one-day-ahead Value-at-Risk for SPY log returns using daily data and a strict rolling-window framework.

## Data
Daily SPY observations with:
- `log_ret`: logarithmic returns
- `rv5`: realized volatility
- `bv`: bipower variation

The candidate may choose the training/validation/test split, but the rolling-window design must be strictly followed.

## Forecast targets
Alpha in `{1%, 5%, 10%}`, with VaR defined by:
`P(r_{t+1} <= VaR_alpha) = alpha`.

## Methodological minimum
Implement:
- at least one non-neural method (examples given: historical simulation, GARCH-type models, quantile regression);
- at least one neural-network method (examples given: MLP, LSTM/GRU, QRNN, Transformer).

## Submission
Submit:
- one PDF report;
- corresponding Python scripts;
- CV.

The report should include:
- data exploration;
- detailed model descriptions and motivation;
- rolling-window settings and computational details;
- tables/figures of VaR forecasts, failure rates, and statistical hypothesis tests;
- comparison of model performance at each tail level and neural vs classical methods;
- conclusion and potential improvements;
- evidence of individual strengths in mathematics, programming, and/or domain knowledge.

Suggested stack in the assignment: Python 3.8+, pandas, numpy, matplotlib, statsmodels, `arch`, scikit-learn, PyTorch.

## Internal interpretation
The assignment does not require a particular architecture, GPU, window length, or split. Research quality will therefore depend heavily on causal experiment design, fair model comparison, correct VaR backtesting, reproducibility, and clear interpretation.
