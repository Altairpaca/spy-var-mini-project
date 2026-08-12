"""原始数据加载与校验。

数据文件为只读输入：加载过程不修改文件；校验失败抛出 DataValidationError。
冻结协议要求记录数据文件 SHA256，本模块提供 sha256_file 供 freeze manifest 使用。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "log_ret", "rv5", "bv"]


class DataValidationError(ValueError):
    """数据缺失、格式错误或语义违反（重复日期、非单调、NaN 等）。"""


def sha256_file(path: str | Path) -> str:
    """计算数据文件 SHA256（分块读取，不载入内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spy_data(path: str | Path) -> pd.DataFrame:
    """加载并校验 SPY 数据。

    返回 DataFrame（索引为升序 DatetimeIndex），列为
    log_ret / rv5 / bv。任何缺失列、重复日期、乱序日期、
    NaN/inf 都会抛出 DataValidationError，绝不静默修补。
    """
    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"数据文件不存在: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise DataValidationError(f"数据文件无法解析: {p}: {e}") from e

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if "date" in missing:
        # 兼容日期列无表头的格式（第一列为日期但列名为空/Unnamed: 0）
        candidates = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
        for cand in candidates:
            try:
                pd.to_datetime(df[cand], errors="raise")
            except Exception:  # noqa: BLE001
                continue
            df = df.rename(columns={cand: "date"})
            missing.remove("date")
            break
    if missing:
        raise DataValidationError(f"数据缺少必需列 {missing}; 实际列: {list(df.columns)}")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise DataValidationError("date 列存在无法解析的日期")
    df = df.copy()
    df["date"] = dates
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        dup = df.loc[df["date"].duplicated(), "date"].dt.date.unique()[:5]
        raise DataValidationError(f"存在重复日期（前几个）: {list(dup)}")

    for col in ("log_ret", "rv5", "bv"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"列 {col} 不是数值类型")
        if df[col].isna().any() or not np.isfinite(df[col]).all():
            raise DataValidationError(f"列 {col} 存在 NaN/inf")

    if (df["rv5"] < 0).any():
        raise DataValidationError("rv5 存在负值（方差度量应为非负）")
    if (df["bv"] < 0).any():
        raise DataValidationError("bv 存在负值（bipower variation 应为非负）")

    df = df.set_index("date", drop=False)
    return df[REQUIRED_COLUMNS]
