"""基础评价指标：覆盖率、分位数损失、交叉率、违例聚集。

violation 约定（AGENTS.md 第 6 条）：VaR 是收益分位数本身
（左尾通常为负），violation = 实现收益 <= VaR。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def violation_stats(violations: np.ndarray, alpha: float) -> dict[str, float]:
    """覆盖率统计：n / 违例数 / 期望违例数 / 经验失败率。"""
    v = np.asarray(violations, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    x = int(np.sum(v))
    return {
        "n_forecasts": n,
        "n_violations": x,
        "expected_violations": n * alpha,
        "failure_rate": x / n if n else np.nan,
    }


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """平均 pinball 损失：E[(y-q)(alpha - 1[y<q])]。"""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if not mask.any():
        return float("nan")
    diff = y[mask] - q[mask]
    return float(np.mean(diff * (alpha - (diff < 0))))


def crossing_rate(q_001: np.ndarray, q_005: np.ndarray, q_010: np.ndarray) -> float:
    """分位数交叉率：违反 q_01<=q_05<=q_10 的日期占比。"""
    a = np.asarray(q_001, dtype=float)
    b = np.asarray(q_005, dtype=float)
    c = np.asarray(q_010, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if not valid.any():
        return float("nan")
    crossed = ((a[valid] > b[valid]) | (b[valid] > c[valid])).mean()
    return float(crossed)


def violation_runs(violations: np.ndarray) -> dict[str, float]:
    """违例聚集诊断：违例游程数与最长游程。"""
    v = np.asarray(violations, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n_runs": np.nan, "max_run": np.nan, "mean_run": np.nan}
    runs = np.diff(np.concatenate([[0.0], v, [0.0]]))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    lengths = ends - starts if len(starts) else np.array([], dtype=int)
    return {
        "n_runs": len(lengths),
        "max_run": int(lengths.max()) if len(lengths) else 0,
        "mean_run": float(lengths.mean()) if len(lengths) else 0.0,
    }


def regime_labels(dates, regimes: dict[str, tuple[str, str]]) -> pd.Series:
    """按预定义日期区间标记 regime（区间互斥，按配置顺序首个命中）。

    dates 可为 DatetimeIndex 或含日期列的 Series；返回 Series 的
    索引与输入对齐。
    """
    dts = dates if isinstance(dates, pd.DatetimeIndex) else pd.DatetimeIndex(dates)
    out = pd.Series("outside", index=dts)
    for name, (start, end) in regimes.items():
        mask = (dts >= pd.Timestamp(start)) & (dts <= pd.Timestamp(end))
        out[mask] = name
    return out
