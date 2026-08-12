"""统一滚动预测引擎。

对每个预测原点 t（交易日期）：
1. 取允许的训练窗口 [t-window+1, t]（严格 window 行）；
2. 在窗口切片上计算特征（因果，见 features.py）；
3. 调用模型的 fit(window_data)，模型内部完成全部学习
   （包括 scaler / early-stopping 划分），全程不接触窗口外数据；
4. 记录 t+1 的分位数预测与实现收益（实现收益在预测记录时写入
   t+1 行对应观测，供评估复用；不参与任何拟合）。

所有模型共享本引擎，输出统一宽表 schema（RESULTS_SCHEMA.md）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import build_features


@dataclass
class WindowData:
    """传给模型 fit 的窗口数据；模型只允许读取这些字段。"""

    dates: pd.DatetimeIndex
    returns: np.ndarray
    features: pd.DataFrame | None
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    origin_date: pd.Timestamp
    target_date: pd.Timestamp
    seed: int
    model_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantileForecast:
    """单个预测原点输出的 1/5/10% 分位数预测。"""

    q_001: float
    q_005: float
    q_010: float
    fit_status: str = "ok"
    meta: dict[str, Any] = field(default_factory=dict)


class Model:
    """模型接口：fit(window) -> QuantileForecast。"""

    model_id: str = "base"

    def fit(self, window: WindowData) -> QuantileForecast:
        raise NotImplementedError


def make_window(
    df: pd.DataFrame,
    origin_pos: int,
    window: int,
    feature_names: list[str] | None,
    seed: int,
    model_config: dict[str, Any],
) -> WindowData:
    """构造原点 origin_pos 的训练窗口（不向前看任何数据）。"""
    start = origin_pos - window + 1
    if start < 0:
        raise ValueError(f"历史不足: 原点 {df.index[origin_pos].date()} 需要 {window} 行")
    dates = df.index[start : origin_pos + 1]
    returns = df["log_ret"].to_numpy()[start : origin_pos + 1]
    features = None
    if feature_names is not None:
        features = build_features(df.iloc[start : origin_pos + 1], feature_names)
    return WindowData(
        dates=dates,
        returns=returns,
        features=features,
        window_start=dates[0],
        window_end=dates[-1],
        origin_date=dates[-1],
        target_date=df.index[origin_pos + 1],
        seed=seed,
        model_config=model_config,
    )


def _fit_one(
    model_factory: Any,
    df: pd.DataFrame,
    origin_pos: int,
    window: int,
    feature_names: list[str] | None,
    seed: int,
    model_config: dict[str, Any],
    tails: list[float],
) -> dict[str, Any]:
    """单个原点的完整流程：窗口构造 -> 拟合 -> 记录（含 fit_status）。"""
    w = make_window(df, origin_pos, window, feature_names, seed, model_config)
    realized = float(df["log_ret"].iloc[origin_pos + 1])
    row: dict[str, Any] = {
        "forecast_date": w.origin_date,
        "target_date": w.target_date,
        "realized_log_ret": realized,
    }
    try:
        model = model_factory()
        fc = model.fit(w)
        row["q_001"] = float(fc.q_001)
        row["q_005"] = float(fc.q_005)
        row["q_010"] = float(fc.q_010)
        row["fit_status"] = fc.fit_status
        for k, v in fc.meta.items():
            row[f"meta_{k}"] = v
        qs = np.array([row["q_001"], row["q_005"], row["q_010"]])
        if not np.isfinite(qs).all():
            row["fit_status"] = "nan_output"
    except Exception as e:  # noqa: BLE001 —— 失败必须记录，绝不静默
        row["q_001"] = row["q_005"] = row["q_010"] = np.nan
        row["fit_status"] = f"failed:{type(e).__name__}:{e}"
    for tail in tails:
        key = f"q_{int(tail * 100):03d}"
        q = row[key]
        row[f"violation_{int(tail * 100):03d}"] = (
            int(row["realized_log_ret"] <= q) if pd.notna(q) else np.nan
        )
    return row


def run_rolling(
    df: pd.DataFrame,
    model_factory: Any,
    origin_positions: np.ndarray,
    window: int,
    tails: list[float],
    seed: int,
    model_config: dict[str, Any],
    feature_names: list[str] | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """对给定原点列表运行统一滚动预测，返回宽表 DataFrame。

    origin_positions 为 df 中的整数位置；模型结果与实现收益
    均在此对齐。workers=1 时串行（保证确定性顺序）。
    """
    t0 = time.perf_counter()
    fn = delayed(_fit_one)
    args = [
        (model_factory, df, pos, window, feature_names, seed, model_config, tails)
        for pos in origin_positions
    ]
    rows = Parallel(n_jobs=workers, prefer="processes")(fn(*a) for a in args)
    panel = pd.DataFrame(rows)
    panel.insert(0, "model_id", model_factory().model_id)
    panel.insert(1, "feature_set", "none" if feature_names is None else "|".join(feature_names))
    panel.insert(2, "window", window)
    panel.insert(3, "seed", seed)
    panel.attrs["runtime_seconds"] = time.perf_counter() - t0
    return panel
