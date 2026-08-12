"""因果特征工程：F0–F3 信息集。

所有特征对日期 s 只使用 <= s 的信息（滞后、rolling 聚合均不向前看）。
特征在窗口切片上计算，配合 min_periods=pad 保证训练行的每个特征
完全落在窗口内部 —— 不存在"窗口外历史"参与拟合的模糊地带。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.DataFrame], pd.Series]


def _pos(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0)


def _neg(x: pd.Series) -> pd.Series:
    return (-x).clip(lower=0.0)


def _log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=1e-12))


FEATURE_FUNCTIONS: dict[str, FeatureFn] = {
    "lag_ret_1": lambda d: d["log_ret"].shift(1),
    "lag_ret_2": lambda d: d["log_ret"].shift(2),
    "lag_ret_5": lambda d: d["log_ret"].shift(5),
    "lag_ret_22": lambda d: d["log_ret"].shift(22),
    "abs_ret_1": lambda d: d["log_ret"].abs(),
    "sq_ret_1": lambda d: d["log_ret"] ** 2,
    "abs_ret_5d": lambda d: d["log_ret"].abs().rolling(5, min_periods=5).mean(),
    "down_mean_5d": lambda d: _neg(d["log_ret"]).rolling(5, min_periods=5).mean(),
    "log_rv5": lambda d: _log(d["rv5"]),
    "rv5_scale": lambda d: np.sqrt(d["rv5"].clip(lower=1e-12)),
    "log_rv5_5d": lambda d: _log(d["rv5"].rolling(5, min_periods=5).mean()),
    "log_rv5_22d": lambda d: _log(d["rv5"].rolling(22, min_periods=22).mean()),
    "log_bv": lambda d: _log(d["bv"]),
    "log_bv_5d": lambda d: _log(d["bv"].rolling(5, min_periods=5).mean()),
    "log_bv_22d": lambda d: _log(d["bv"].rolling(22, min_periods=22).mean()),
    "jump": lambda d: _pos(d["rv5"] - d["bv"]),
    "jump_rel": lambda d: _pos(d["rv5"] - d["bv"]) / d["rv5"].clip(lower=1e-12),
}


def required_pad(feature_names: list[str]) -> int:
    """该特征集所需的最小历史天数（训练行安全裁剪边界）。"""
    lags = [int(n.split("_")[-1]) for n in feature_names if n.startswith("lag_ret_")]
    windows = [22 if "22d" in n else 5 if "5d" in n else 1 for n in feature_names]
    return max([max(lags, default=0), max(windows, default=1)])


def build_features(
    window: pd.DataFrame,
    feature_names: list[str],
    pad: int | None = None,
) -> pd.DataFrame:
    """在窗口切片上计算特征矩阵（行 = 窗口内日期，列 = 特征）。

    window 必须含列 log_ret / rv5 / bv 与 DatetimeIndex。
    返回仅含非 NaN 行的 DataFrame（前 pad 行因历史不足被丢弃）。
    """
    missing = [n for n in feature_names if n not in FEATURE_FUNCTIONS]
    if missing:
        raise ValueError(f"未知特征: {missing}")
    out = pd.DataFrame(index=window.index)
    for n in feature_names:
        out[n] = FEATURE_FUNCTIONS[n](window)
    pad = required_pad(feature_names) if pad is None else pad
    out = out.iloc[pad:]
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"特征存在 NaN（实现缺陷，不应发生）: {bad}")
    return out


FEATURE_SETS: dict[str, list[str]] = {
    "F0": [
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_5",
        "lag_ret_22",
        "abs_ret_1",
        "sq_ret_1",
        "abs_ret_5d",
    ],
    "F1": [
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_5",
        "lag_ret_22",
        "abs_ret_1",
        "sq_ret_1",
        "abs_ret_5d",
        "log_rv5",
        "rv5_scale",
        "log_rv5_5d",
        "log_rv5_22d",
    ],
    "F2": [
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_5",
        "lag_ret_22",
        "abs_ret_1",
        "sq_ret_1",
        "abs_ret_5d",
        "log_rv5",
        "rv5_scale",
        "log_rv5_5d",
        "log_rv5_22d",
        "log_bv",
        "log_bv_5d",
        "log_bv_22d",
    ],
    "F3": [
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_5",
        "lag_ret_22",
        "abs_ret_1",
        "sq_ret_1",
        "abs_ret_5d",
        "log_rv5",
        "rv5_scale",
        "log_rv5_5d",
        "log_rv5_22d",
        "log_bv",
        "log_bv_5d",
        "log_bv_22d",
        "jump",
        "jump_rel",
        "down_mean_5d",
    ],
}
