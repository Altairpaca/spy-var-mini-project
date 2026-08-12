"""数据审计：对真实数据执行 §14 全部检查并生成 docs/DATA_AUDIT.md。

所有数字由代码计算并落盘（outputs/tables/data_audit.json），
禁止手工录入；文档由本脚本生成，不是手写。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts.common import load_data, parse_common_args, resolve_config
from spyvar.data.loader import sha256_file


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= lag:
        return float("nan")
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return float("nan")
    return float(np.dot(xc[lag:], xc[:-lag]) / denom)


def run_audit(df: pd.DataFrame, cfg) -> dict:
    """计算全部审计统计量。"""
    out: dict = {}
    out["row_count"] = len(df)
    out["date_range"] = [str(df.index.min().date()), str(df.index.max().date())]
    out["duplicate_dates"] = int(df.index.duplicated().sum())
    out["monotonic"] = bool(df.index.is_monotonic_increasing)
    out["missing_or_inf"] = {
        col: int((~np.isfinite(df[col])).sum()) for col in ("log_ret", "rv5", "bv")
    }
    r = df["log_ret"].to_numpy(dtype=float)
    out["log_ret"] = {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "min": float(r.min()),
        "max": float(r.max()),
        "skewness": float(pd.Series(r).skew()),
        "excess_kurtosis": float(pd.Series(r).kurtosis()),
        "q001": float(np.quantile(r, 0.01)),
        "q005": float(np.quantile(r, 0.05)),
        "q010": float(np.quantile(r, 0.10)),
        "n_below_q001": int((r <= np.quantile(r, 0.01)).sum()),
        "n_below_q005": int((r <= np.quantile(r, 0.05)).sum()),
        "n_below_q010": int((r <= np.quantile(r, 0.10)).sum()),
    }
    rv, bv = df["rv5"].to_numpy(dtype=float), df["bv"].to_numpy(dtype=float)
    out["rv5"] = {
        "mean": float(rv.mean()),
        "median": float(np.median(rv)),
        "std": float(rv.std(ddof=1)),
        "min": float(rv.min()),
        "max": float(rv.max()),
        "pct_zeros": float((rv == 0).mean()),
    }
    out["bv"] = {
        "mean": float(bv.mean()),
        "median": float(np.median(bv)),
        "std": float(bv.std(ddof=1)),
        "min": float(bv.min()),
        "max": float(bv.max()),
        "pct_zeros": float((bv == 0).mean()),
    }
    out["sqrt_rv5"] = {
        "mean": float(np.sqrt(rv).mean()),
        "corr_with_abs_ret": float(np.corrcoef(np.sqrt(rv), np.abs(r))[0, 1]),
        "corr_with_log_ret2": float(np.corrcoef(np.sqrt(rv), r**2)[0, 1]),
    }
    out["rv_bv"] = {
        "corr": float(np.corrcoef(rv, bv)[0, 1]),
        "corr_log": float(np.corrcoef(np.log(rv), np.log(bv))[0, 1]),
    }
    out["autocorr"] = {
        "log_ret_lag1": _autocorr(r, 1),
        "abs_ret_lag1": _autocorr(np.abs(r), 1),
        "abs_ret_lag5": _autocorr(np.abs(r), 5),
        "sq_ret_lag1": _autocorr(r**2, 1),
        "log_rv5_lag1": _autocorr(np.log(rv), 1),
        "log_rv5_lag5": _autocorr(np.log(rv), 5),
        "log_bv_lag1": _autocorr(np.log(bv), 1),
    }
    jump = np.maximum(rv - bv, 0.0)
    out["jump_proxy"] = {
        "mean": float(jump.mean()),
        "median": float(np.median(jump)),
        "pct_positive": float((jump > 0).mean()),
        "corr_with_future_rv5": float(
            np.corrcoef(jump[:-1], rv[1:])[0, 1]
        ),
    }
    neg = (r < 0).astype(float)
    out["negative_return_vol_link"] = {
        "corr_negind_with_next_abs_ret": float(np.corrcoef(neg[:-1], np.abs(r[1:]))[0, 1]),
        "corr_negind_with_next_rv5": float(np.corrcoef(neg[:-1], rv[1:])[0, 1]),
        "corr_negret_with_next_rv5": float(
            np.corrcoef(np.minimum(r[:-1], 0.0), rv[1:])[0, 1]
        ),
    }
    return out


def render_markdown(audit: dict) -> str:
    """渲染 DATA_AUDIT.md（数字全部来自 audit dict）。"""
    L = []
    L.append("# 数据审计报告")
    L.append("")
    L.append("由 `scripts/audit_data.py` 自动生成；所有数字来自 `outputs/tables/data_audit.json`。")
    L.append("")
    L.append("## 基本结构")
    L.append("")
    L.append(f"- 行数: {audit['row_count']}")
    L.append(f"- 日期范围: {audit['date_range'][0]} ~ {audit['date_range'][1]}")
    L.append(f"- 重复日期: {audit['duplicate_dates']}")
    L.append(f"- 日期单调递增: {audit['monotonic']}")
    L.append(f"- 缺失/inf: {audit['missing_or_inf']}")
    L.append("")
    L.append("## log_ret 描述统计")
    L.append("")
    lr = audit["log_ret"]
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    for k in ("mean", "std", "min", "max", "skewness", "excess_kurtosis", "q001", "q005", "q010"):
        L.append(f"| {k} | {lr[k]:.6g} |")
    L.append("")
    L.append("经验左尾样本数（`realized <= quantile`）:")
    L.append("")
    for k in ("n_below_q001", "n_below_q005", "n_below_q010"):
        L.append(f"- {k}: {lr[k]}")
    L.append("")
    L.append("## rv5 / bv 尺度")
    L.append("")
    L.append("| 统计量 | rv5 | bv |")
    L.append("|---|---|---|")
    for k in ("mean", "median", "std", "min", "max", "pct_zeros"):
        L.append(f"| {k} | {audit['rv5'][k]:.6g} | {audit['bv'][k]:.6g} |")
    L.append("")
    L.append("## sqrt(rv5) 合理性（应接近日波动率量级并与 |r| 相关）")
    L.append("")
    for k, v in audit["sqrt_rv5"].items():
        L.append(f"- {k}: {v:.6g}")
    L.append("")
    L.append("## RV/BV 相关")
    L.append("")
    for k, v in audit["rv_bv"].items():
        L.append(f"- {k}: {v:.6g}")
    L.append("")
    L.append("## 自相关")
    L.append("")
    L.append("| 序列 | lag1 | lag5 |")
    L.append("|---|---|---|")
    L.append(f"| log_ret | {audit['autocorr']['log_ret_lag1']:.4g} | - |")
    L.append(f"| abs_ret | {audit['autocorr']['abs_ret_lag1']:.4g} | {audit['autocorr']['abs_ret_lag5']:.4g} |")
    L.append(f"| sq_ret | {audit['autocorr']['sq_ret_lag1']:.4g} | - |")
    L.append(f"| log_rv5 | {audit['autocorr']['log_rv5_lag1']:.4g} | {audit['autocorr']['log_rv5_lag5']:.4g} |")
    L.append(f"| log_bv | {audit['autocorr']['log_bv_lag1']:.4g} | - |")
    L.append("")
    L.append("## jump proxy `max(rv5-bv, 0)` 分布")
    L.append("")
    for k, v in audit["jump_proxy"].items():
        L.append(f"- {k}: {v:.6g}")
    L.append("")
    L.append("## 负收益与未来波动的关系")
    L.append("")
    for k, v in audit["negative_return_vol_link"].items():
        L.append(f"- {k}: {v:.6g}")
    L.append("")
    return "\n".join(L)


def main() -> None:
    args = parse_common_args("SPY 数据审计")
    cfg = resolve_config(args)
    if args.data is None and not Path(cfg.data_path).exists():
        sys.exit(f"数据文件不存在: {cfg.data_path}")
    df = load_data(cfg, args.data)
    audit = run_audit(df, cfg)
    out_root = Path(args.out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    (out_root / "manifests").mkdir(parents=True, exist_ok=True)
    (out_root / "tables" / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if args.data is None:
        (out_root / "manifests" / "data_sha256.txt").write_text(
            sha256_file(cfg.data_path) + "\n", encoding="utf-8"
        )
    doc = render_markdown(audit)
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "DATA_AUDIT.md").write_text(doc, encoding="utf-8")
    print(f"审计完成: {audit['row_count']} 行, {audit['date_range'][0]} ~ {audit['date_range'][1]}")
    print(f"DATA_AUDIT.md -> {docs / 'DATA_AUDIT.md'}")


if __name__ == "__main__":
    main()
