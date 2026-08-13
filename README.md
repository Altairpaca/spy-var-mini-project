# SPY VaR Mini-Project

PhD pre-screening mini-project: one-day-ahead conditional VaR forecasts (1% / 5% / 10%)
for SPY log returns under a strictly causal rolling-window framework, with a fair
comparison of classical and small neural models.

## Research protocol summary

- **Zero future information leakage**: the forecast for `t+1` uses only information
  observable at `t`; scalers, early stopping and hyperparameter selection are confined
  to the training window.
- **development** (< 2008-01-01): rolling-origin validation, validation years
  2005/2006/2007 (1500-day window valid for 2006-2007; see RESEARCH_LOG.md).
- **final test** (>= 2008-01-01): must be frozen before first run
  (`configs/final.yaml` + `docs/FREEZE_MANIFEST.md`; the gate is enforced by tests
  and scripts).
- **Models**: M0 Historical Simulation, M1 GARCH(1,1)-t, M2 Linear/HAR Quantile,
  M3 Multi-Quantile MLP (structurally non-crossing), M4 GJR-GARCH-t, M5 GRU.
- **Feature ablation**: F0 (returns) -> F1 (+RV) -> F2 (+BV) -> F3 (+jump + downside
  asymmetry); M2/M3 share the information sets.
- **Seeds**: primary 42 (predeclared), robustness {7, 2026}; main results use the
  primary seed, not an ensemble.
- **Evaluation**: failure rate, Kupiec, Christoffersen ind/cc, pinball loss, crossing
  rate, violation clustering, regime breakdown, DM + block bootstrap; DQ as an
  additional diagnostic.

## Environment

```bash
uv sync --frozen --python 3.11   # exact environment used for the clean-clone verification
```

Requirements: Python 3.11, about 4 GB RAM, CPU only (the NN models are tiny); GPU
optional. BLAS thread limits are set automatically inside the scripts
(OMP/MKL/OPENBLAS=1).

## Reproduction (three-phase research protocol)

The research lifecycle is split into three phases separated by Git boundaries; the
freeze is a mandatory boundary between development and frozen OOS. It is deliberately
not compressed into a single command.

### Phase 1 - Development (repeatable)

Run each development stage with the **development** config:

```bash
python scripts/audit_data.py --config configs/development.yaml
python scripts/run_development.py --config configs/development.yaml
python scripts/select_window.py --config configs/development.yaml
python scripts/neural_search.py --config configs/development.yaml
```

Artifacts (`outputs/development/`): data audit, development panels, window selection
(`window_decision.json`), neural search (`neural_search.csv` +
`neural_search_decision.json`, executed on the *selected* window - the script fails
closed if `window_decision.json` is missing). `neural_search.py` refuses to fall back
to a candidate window.

Then write the development-only decisions into the **final** config (never into the
development config):

```bash
python scripts/update_final_config.py --config configs/final.yaml
```

**Commit these development decisions** (`window_decision.json`,
`neural_search_decision.json`, updated `configs/final.yaml`) before proceeding.

`scripts/run_all.py` also supports running the development stages together; the
`update` stage always targets `configs/final.yaml` regardless of the `--config`
argument:

```bash
python scripts/run_all.py --config configs/development.yaml --stages audit,dev,select,search,update
```

### Phase 2 - Freeze (one-time Git boundary)

```bash
python scripts/freeze_final.py --config configs/final.yaml   # validates data/config/code signatures + clean tree
git add configs/final.yaml docs/FREEZE_MANIFEST.md docs/freeze.json outputs/manifests/freeze.json
git commit   # freeze commit (SHA is preserved in history)
```

Any code/config change after freezing makes the gate reject further runs until a new
freeze.

### Phase 3 - Frozen OOS (run once, after freeze)

```bash
python scripts/run_all.py --config configs/final.yaml --stages final,robust,eval,figures,report
```

(or stage by stage: `run_final.py --clean-run` -> `seed_robustness.py` ->
`evaluate.py` -> `make_figures.py` -> `make_report.py`.)

### Report-only regeneration (no retraining)

```bash
python scripts/make_report.py --config configs/final.yaml
python scripts/generate_final_summary.py
```

Tests:

```bash
python -m pytest tests/
```

## Artifacts

```text
outputs/runs/<freeze_id>/  canonical frozen run: predictions (parquet) + manifests
outputs/development/       development decisions (window selection, neural search)
outputs/manifests/         freeze manifests and data hashes
docs/DATA_AUDIT.md         data audit (auto-generated)
docs/FREEZE_MANIFEST.md    final-test freeze manifest (auto-generated)
docs/FINAL_SUMMARY_ZH.md   Chinese audit summary (auto-generated)
report/final_report.pdf    formal English report (auto-generated)
```

All report numbers are recomputed by scripts from the artifacts; none are
hand-entered.

## Repository layout

```text
src/spyvar/       data/features/rolling engine/models/evaluation/freeze machinery
scripts/          reproducible experiment entry points (run_all.py orchestrates)
tests/            leakage/alignment/rolling/sign-convention/statistical-test tests
configs/          experiment protocol (development.yaml / final.yaml)
docs/             audits, freeze, Chinese summary, assignment summary
report/           English PDF report
```

## Data

`data/raw/spy_data.csv` (columns: date, log_ret, rv5, bv) is a read-only input; the
loader validates integrity and records the SHA256 at freeze time. The data file is
provided by the assignment.
