# Final Test Freeze Manifest

本文件由 `scripts/freeze_final.py` 生成；任何数值禁止手改。

```json
{
  "freeze_created_at": "2026-08-13T15:38:33+0800",
  "git_commit": "0b76a5a7e8d804b290c4bdcbe1180dbbb5de1125",
  "config_path": "configs/final.yaml",
  "config_sha256": "09a350fea09580175262ea8dc06e1c4416eaabae4dd311184a34a838fd3027b9",
  "data_path": "data/raw/spy_data.csv",
  "data_sha256": "277406a832c1418de30221396bd8dbf12a444c3f5430ad88c7cc75d7a7f573a3",
  "code_signature": "875ef27a64cebc10aac6d108500a66be5db289c6074096a861e3d7417635ef44",
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
