# Research Log

Use this file for decision history, not raw console output. Every material methodological change should state why it was made and whether final-test results had already been observed.

## 2026-08-11 — Project initialization
### Assignment interpretation
The task is one-day-ahead SPY VaR forecasting at 1%, 5%, and 10% tails with a strict rolling-window evaluation. At least one non-neural and one neural method are required.

### Initial methodological stance
- Treat leakage prevention and temporal alignment as first-class research requirements.
- Use a model ladder: Historical Simulation -> GARCH-t -> direct conditional quantile model -> small neural quantile model.
- Evaluate calibration and quantile loss; do not rank models by failure rate alone.
- Prefer feature/information ablations over a large architecture zoo.
- Do not require neural methods to win.

### Compute decision
Primary environment: altair-server (256 CPU cores, 128 GB RAM, RTX 3060, Codex available). SUPER-26 is optional batch compute only.

### Open decisions before coding the final experiment
- final train/validation/test split;
- primary rolling-window length;
- direct-quantile baseline choice;
- neural architecture (MLP-QR vs GRU-QR);
- neural retraining schedule;
- exact statistical test implementation and significance reporting.
