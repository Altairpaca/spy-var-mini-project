# SPY VaR Mini-Project

PhD pre-screening mini-project on one-day-ahead Value-at-Risk forecasting for SPY log returns.

## Objective
Forecast conditional VaR for next-day SPY log returns at alpha = 1%, 5%, and 10%, using a strict rolling-window framework and comparing at least one classical method with at least one neural-network method.

## Research principles
- Strictly causal feature construction and preprocessing.
- Frozen validation/test protocol before final evaluation.
- Identical forecast dates and evaluation code across models.
- Reproducible experiment configs, seeds, tables, and figures.
- Statistical calibration and forecast-loss evaluation; failure rate alone is not sufficient.
- Prefer interpretable, well-controlled experiments over unnecessary model complexity.

## Initial model ladder
1. Rolling Historical Simulation.
2. GARCH(1,1) with Student-t innovations.
3. Direct conditional quantile baseline (Quantile Regression / CAViaR / HAR-Quantile candidate).
4. Small neural quantile model (MLP-QR or GRU-QR candidate) with non-crossing outputs.

The exact final specification must be frozen in `EXPERIMENT_SPEC.md` before the final test set is evaluated.

## Repository layout
```text
.
├── README.md
├── AGENTS.md
├── EXPERIMENT_SPEC.md
├── ENVIRONMENT.md
├── RESEARCH_LOG.md
├── RESULTS_SCHEMA.md
├── .gitignore
├── docs/
│   └── ASSIGNMENT_SUMMARY.md
├── configs/          # experiment configs
├── data/             # raw data kept local unless explicitly approved
├── notebooks/        # EDA only; no authoritative final results
├── src/              # reusable implementation
├── scripts/          # reproducible experiment entrypoints
├── tests/            # leakage/alignment/rolling/evaluation tests
├── outputs/          # generated predictions/tables/figures
└── report/           # final report source
```

## Intended compute environment
Primary development and experiment environment: `altair-server` (Linux, 256 CPU cores, 128 GB RAM, RTX 3060), because Codex is directly usable there. The project is small enough that the RTX 3060 is more than sufficient; most rolling classical fits are CPU-bound.

See `ENVIRONMENT.md` for execution rules.
