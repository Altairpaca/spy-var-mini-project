"""development 窗口选择：从 dev 面板计算 1000 vs 1500 对比表。

决策规则（预先声明，development 证据驱动）：
- 主指标：三个 tail 的 mean pinball loss 之和（窗口内取较小者）；
- 并列（差异 < 1% 相对差）时选 1500 —— 长窗口对 1% tail
  有效尾部样本更多、经验分位数更稳定（报告中论证）；
- 覆盖率（|failure_rate - alpha|）作为诊断展示，不作主规则。
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

from scripts.common import parse_common_args, q_col, resolve_config, violation_col
from spyvar.evaluation.metrics import pinball_loss


def _model_label(model_id: str, fset: str) -> str:
    return f"{model_id}-{fset}"


def collect_dev_panels(out_root: Path) -> pd.DataFrame:
    pred_dir = out_root / "predictions"
    frames = []
    for p in sorted(pred_dir.glob("dev-*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit(f"没有 dev 预测面板: {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_common_args("development 窗口选择")
    cfg = resolve_config(args)
    panels = collect_dev_panels(Path(args.out_root))
    rows = []
    for (window, model, fset), g in panels.groupby(["window", "model_id", "feature_set"]):
        for alpha in cfg.tails:
            v = g[violation_col(alpha)].to_numpy(dtype=float)
            y = g["realized_log_ret"].to_numpy(dtype=float)
            q = g[q_col(alpha)].to_numpy(dtype=float)
            rate = float(np.nanmean(v)) if len(v) else np.nan
            rows.append({
                "window": int(window),
                "model": str(model),
                "feature_set": str(fset),
                "tail": alpha,
                "n": len(g),
                "failure_rate": rate,
                "abs_coverage_gap": abs(rate - alpha) if np.isfinite(rate) else np.nan,
                "mean_pinball": pinball_loss(y, q, alpha),
            })
    tbl = pd.DataFrame(rows)
    out_root = Path(args.out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_root / "tables" / "window_comparison.csv", index=False)

    # 主规则：公共模型集（M0/M1/M2 各特征集）的 pinball 之和
    common_models = tbl.groupby(["window", "model", "feature_set"]).size().reset_index()
    n_models_per_window = common_models.groupby("window").size()
    usable = n_models_per_window[n_models_per_window == n_models_per_window.max()]
    usable_windows = set(usable.index)
    sub = tbl[tbl["window"].isin(usable_windows)]
    score = sub.groupby("window")["mean_pinball"].sum()
    w1, w2 = sorted(score.index)
    win_small = score.idxmin()
    rel_diff = abs(score[w1] - score[w2]) / min(score[w1], score[w2])
    chosen = win_small if rel_diff >= 0.01 else max(w1, w2)
    decision = {
        "window_pinball_sum": {int(k): float(v) for k, v in score.items()},
        "relative_diff": float(rel_diff),
        "rule": "min_sum_pinball; tie-break (rel diff < 1%) -> 1500",
        "chosen_window": int(chosen),
        "note": "基于 development 证据；覆盖率为诊断列。",
    }
    (out_root / "tables" / "window_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"对比表 -> {out_root / 'tables' / 'window_comparison.csv'}")


if __name__ == "__main__":
    main()
