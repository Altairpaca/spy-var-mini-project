"""脚本共享工具：环境线程限制、参数解析、数据加载、模型工厂。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须在任何数值库 import 之前设置：限制嵌套 BLAS/OpenMP 线程
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd

from spyvar.config import Config, load_config
from spyvar.data.loader import load_spy_data
from spyvar.models import make_model
from spyvar.rolling import run_rolling

DEFAULT_CONFIG = "configs/final.yaml"


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="实验配置文件路径")
    p.add_argument("--data", default=None, help="覆盖数据文件路径（测试/合成数据用）")
    p.add_argument("--workers", type=int, default=None, help="覆盖并行 worker 数")
    p.add_argument("--out-root", default="outputs", help="输出根目录")
    p.add_argument("--docs-dir", default=None, help="冻结清单文档目录（默认仓库 docs/）")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.workers is not None:
        raw = dict(cfg.raw)
        raw["parallel"] = dict(raw.get("parallel", {}))
        raw["parallel"]["workers"] = args.workers
        cfg = Config(raw=raw, config_path=cfg.config_path, sha256=cfg.sha256)
    return cfg


def load_data(cfg: Config, data_override: str | None = None) -> pd.DataFrame:
    return load_spy_data(data_override or cfg.data_path)


def model_factory_for(cfg: Config, model_id: str) -> object:
    """返回可调用工厂：每次调用生成新模型实例。"""
    return lambda: make_model(model_id, cfg.models)


def feature_names_for(cfg: Config, feature_set: str) -> list[str] | None:
    if feature_set == "none":
        return None
    if feature_set not in cfg.feature_sets:
        raise KeyError(f"未知特征集 {feature_set}")
    return cfg.feature_sets[feature_set]


def forecast_origins(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    window: int,
    min_history: int = 0,
) -> np.ndarray:
    """按目标日期返回预测原点位置（整数位置）。

    研究分区按被预测的 target date 定义：origin 位置 pos 的
    target 是 df.index[pos + 1]。返回所有 target 落在
    [start_date, end_date] 内且历史充足的 origin。
    审计修复（2026-08-13）：旧实现按 origin date 过滤，
    导致 2007-12-31 的 origin（target 2008-01-02）被错误归入 development。
    """
    dates = df.index
    first = window + min_history
    positions = np.arange(first, len(df) - 1)
    targets = dates[positions + 1]
    mask = (targets >= pd.Timestamp(start_date)) & (targets <= pd.Timestamp(end_date))
    return positions[mask]


def run_experiment(
    cfg: Config,
    df: pd.DataFrame,
    *,
    model_id: str,
    feature_set: str,
    window: int,
    seed: int,
    origins: np.ndarray,
    out_path: str,
    experiment_id: str,
    device: str = "cpu",
) -> pd.DataFrame:
    """统一实验运行入口：滚动预测 + 面板保存 + manifest。"""
    from spyvar.io import make_manifest, save_panel

    factory = model_factory_for(cfg, model_id)
    names = feature_names_for(cfg, feature_set)
    panel = run_rolling(
        df, factory, origins, window, cfg.tails, seed,
        cfg.models.get(model_id, {}), feature_names=names,
        feature_set_label=feature_set, workers=cfg.workers,
    )
    panel.insert(0, "experiment_id", experiment_id)
    manifest = make_manifest(
        cfg,
        experiment_id=experiment_id,
        model_id=model_id,
        feature_set=feature_set,
        window=window,
        seed=seed,
        device=device,
        workers=cfg.workers,
        runtime_seconds=panel.attrs.get("runtime_seconds", float("nan")),
        n_forecasts=len(panel),
        date_range=(str(panel["target_date"].min()), str(panel["target_date"].max())),
        extra={
            "fit_failures": int((panel["fit_status"] != "ok").sum()),
            "feature_set_names": names,
        },
    )
    save_panel(panel, out_path, manifest)
    return panel


def q_col(alpha: float) -> str:
    return f"q_{int(alpha * 100):03d}"


def violation_col(alpha: float) -> str:
    return f"violation_{int(alpha * 100):03d}"


def require_real_data(cfg: Config) -> None:
    """真实数据存在性检查（防止把合成数据当真实结果）。"""
    if not Path(cfg.data_path).exists():
        sys.exit(f"数据文件不存在: {cfg.data_path} —— 请将 spy_data.csv 放入 data/raw/")


def save_artifact(df: pd.DataFrame, path: str) -> None:
    from spyvar.io import save_table

    save_table(df, path)
