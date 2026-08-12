# Experimental Environment

## Primary environment
Use `altair-server` as the canonical development and execution environment.

Known resources:
- Linux server
- 256 CPU cores
- 128 GB RAM
- NVIDIA RTX 3060
- Codex available

This is sufficient by a wide margin for a 4,640-observation daily VaR project. Most classical rolling fits are CPU-bound; the RTX 3060 is adequate for the proposed small neural models.

## Recommended software baseline
- Python 3.11
- numpy
- pandas
- scipy
- statsmodels
- arch
- scikit-learn
- PyTorch
- matplotlib
- pytest
- PyYAML
- optional: Jupyter for EDA only

Prefer a lockable environment (`uv` + `pyproject.toml`/`uv.lock`, or an equivalent reproducible conda/mamba setup).

## CPU parallelism
Do not default to all 256 cores. Benchmark worker counts such as 16, 32, and 64.

For process-parallel rolling fits, start with:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```
This avoids nested BLAS/OpenMP oversubscription.

## GPU policy
Use the RTX 3060 for neural-network training. The dataset is small; model size should remain modest. Prefer one GPU experiment process at a time unless profiling demonstrates a reason otherwise.

## SUPER-26 policy
SUPER-26 is not part of the primary development loop because Codex cannot be used there. Use it only as an optional, frozen-code batch-compute node if later robustness work becomes genuinely CPU-intensive (e.g., very large bootstrap grids). Do not introduce cross-server complexity without evidence that altair-server is a bottleneck.

## Reproducibility metadata
Each experiment should record:
- experiment ID;
- timestamp;
- git commit SHA;
- config path/content hash;
- Python/package versions;
- seed(s);
- device;
- worker count;
- runtime;
- forecast date range;
- model convergence/failure counts.
