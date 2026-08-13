"""实验产物 IO：预测面板、指标表、清单、元数据。

所有实验输出带 sidecar manifest（config 哈希、git commit、
包版本、种子、窗口、运行时），保证任意数字可追溯。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .config import Config

PANEL_COLUMNS = [
    "forecast_date",
    "target_date",
    "model_id",
    "feature_set",
    "window",
    "seed",
    "q_001",
    "q_005",
    "q_010",
    "realized_log_ret",
    "violation_001",
    "violation_005",
    "violation_010",
    "fit_status",
]


def git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 —— 非 git 环境返回 unknown
        return "unknown"


def package_versions() -> dict[str, str]:
    import arch
    import matplotlib
    import numpy
    import pandas
    import scipy
    import sklearn
    import statsmodels
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "arch": arch.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
    }


def make_manifest(
    config: Config,
    *,
    experiment_id: str,
    model_id: str,
    feature_set: str,
    window: int,
    seed: int,
    device: str,
    workers: int,
    runtime_seconds: float,
    n_forecasts: int,
    date_range: tuple[str, str] | None,
    extra: dict | None = None,
) -> dict:
    """构造实验 manifest（JSON 可序列化）。"""
    m = {
        "experiment_id": experiment_id,
        "model_id": model_id,
        "feature_set": feature_set,
        "window": window,
        "seed": seed,
        "config_sha256": config.sha256,
        "config_path": config.config_path,
        "git_commit": git_commit_sha(),
        "device": device,
        "workers": workers,
        "runtime_seconds": runtime_seconds,
        "n_forecasts": n_forecasts,
        "date_range": list(date_range) if date_range else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "packages": package_versions(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if config.data_sha256:
        m["data_sha256"] = config.data_sha256
    if extra:
        m.update(extra)
    return m


def save_panel(panel: pd.DataFrame, path: str | Path, manifest: dict) -> Path:
    """保存预测面板（parquet）+ manifest（json）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(p, index=False)
    mp = p.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_panel(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """保存指标/对比表（csv，UTF-8）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def save_json(obj: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return p


def ensure_panel_schema(panel: pd.DataFrame) -> pd.DataFrame:
    """校验面板列完整并按 schema 排序（RESULTS_SCHEMA 一致性）。"""
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"面板缺少 schema 列: {missing}")
    return panel[PANEL_COLUMNS]
