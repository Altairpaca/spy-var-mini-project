# Final Test Freeze Manifest

本文件由 `scripts/freeze_final.py` 生成；任何数值禁止手改。

```json
{
  "freeze_created_at": "2026-08-12T22:07:15+0800",
  "git_commit": "a818290413a819ff88ee9c315503e248f1598ea9",
  "config_path": "configs/final.yaml",
  "config_sha256": "ed67a63430953f2cace0a251945f179ed26992ec71d98224916bca5764406bb3",
  "data_path": "data/raw/spy_data.csv",
  "data_sha256": "277406a832c1418de30221396bd8dbf12a444c3f5430ad88c7cc75d7a7f573a3",
  "models": [
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5"
  ],
  "feature_sets": [
    "F0",
    "F1",
    "F2",
    "F3"
  ],
  "primary_window": 1500,
  "seeds": [
    42,
    7,
    2026
  ],
  "evaluation_metrics": [
    "failure_rate",
    "kupiec_lr",
    "christoffersen_ind",
    "christoffersen_cc",
    "dq_test",
    "mean_pinball",
    "crossing_rate",
    "violation_runs",
    "dm_test",
    "block_bootstrap"
  ],
  "final_test_start": "2008-01-01",
  "development_end": "2007-12-31"
}
```
