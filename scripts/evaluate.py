"""统一评估：从 final 预测面板再生全部指标表。

生成（全部由面板重算，禁止手录数字）：
- metrics.csv            每 (experiment, tail)：覆盖/Kupiec/Christoffersen/
                          pinball/交叉率/DQ/游程
- regime_metrics.csv     按预定义 regime 分段的失败率与 pinball
- ablation.csv           M2/M3 按特征集 F0-F3 的 pinball 与覆盖率
- seed_robustness.csv    M3/M5 不同 seed 的指标（mean/std 汇总）
- dm_comparison.csv      每 tail 的 DM + block bootstrap 显著性
- common_panel_check.json 所有主模型目标日期一致性检查结果
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

from scripts.common import (
    parse_common_args,
    q_col,
    resolve_config,
    violation_col,
)
from spyvar.evaluation.backtests import (
    block_bootstrap_pvalue,
    christoffersen_cc,
    christoffersen_ind,
    dm_test,
    dq_test,
    kupiec_lr,
)
from spyvar.evaluation.metrics import (
    crossing_rate,
    pinball_loss,
    regime_labels,
    violation_runs,
    violation_stats,
)

FINAL_MODELS = [
    ("M0", "none"), ("M1", "none"), ("M4", "none"),
    *[("M2", f) for f in ("F0", "F1", "F2", "F3")],
    *[("M3", f) for f in ("F0", "F1", "F2", "F3")],
    ("M5", "F3"),
]


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm family-wise correction (step-down); returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def load_final_panels(out_root: Path) -> pd.DataFrame:
    pred_dir = out_root / "predictions"
    frames = []
    for p in sorted(pred_dir.glob("final-*.parquet")):
        if "rob" in p.name:
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit(f"没有 final 预测面板: {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def load_robustness_panels(out_root: Path) -> pd.DataFrame:
    pred_dir = out_root / "predictions"
    frames = [pd.read_parquet(p) for p in sorted(pred_dir.glob("rob-*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def common_panel_check(panels: pd.DataFrame) -> dict:
    """验证所有主模型目标日期集一致（AGENTS.md 第 4 条）。"""
    per_model = {k: set(g["target_date"].astype(str)) for k, g in panels.groupby(["model_id", "feature_set"])}
    keys = list(per_model)
    base = per_model[keys[0]]
    ok = all(per_model[k] == base for k in keys[1:])
    return {
        "models": sorted(per_model),
        "identical_target_dates": bool(ok),
        "n_dates": len(base),
    }


def _metric_row(g: pd.DataFrame, alpha: float, model: str, fset: str, exp_id: str) -> dict:
    v = g[violation_col(alpha)].to_numpy(dtype=float)
    y = g["realized_log_ret"].to_numpy(dtype=float)
    q = g[q_col(alpha)].to_numpy(dtype=float)
    stats = violation_stats(v, alpha)
    x, n = int(stats["n_violations"]), int(stats["n_forecasts"])
    lr_uc, p_uc = kupiec_lr(x, n, alpha)
    lr_ind, p_ind = christoffersen_ind(v)
    lr_cc, p_cc = christoffersen_cc(v, x, n, alpha)
    dq, p_dq = dq_test(v, alpha, q, lags=4)
    runs = violation_runs(v)
    return {
        "experiment_id": exp_id,
        "model": model,
        "feature_set": fset,
        "tail": alpha,
        "n_forecasts": n,
        "n_violations": x,
        "expected_violations": stats["expected_violations"],
        "failure_rate": stats["failure_rate"],
        "kupiec_lr": lr_uc,
        "kupiec_pvalue": p_uc,
        "christoffersen_ind_lr": lr_ind,
        "christoffersen_ind_pvalue": p_ind,
        "conditional_coverage_lr": lr_cc,
        "conditional_coverage_pvalue": p_cc,
        "dq_stat": dq,
        "dq_pvalue": p_dq,
        "mean_pinball": pinball_loss(y, q, alpha),
        "crossing_rate": crossing_rate(
            g[q_col(0.01)].to_numpy(dtype=float),
            g[q_col(0.05)].to_numpy(dtype=float),
            g[q_col(0.10)].to_numpy(dtype=float),
        ),
        "n_runs": runs["n_runs"],
        "max_run": runs["max_run"],
    }


def main() -> None:
    args = parse_common_args("统一评估")
    cfg = resolve_config(args)
    out_root = Path(args.out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    panels = load_final_panels(out_root)
    check = common_panel_check(panels)
    (out_root / "tables" / "common_panel_check.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"公共日期集一致性: {check['identical_target_dates']} ({check['n_dates']} 个日期)")

    rows = []
    for (model, fset), g in panels.groupby(["model_id", "feature_set"]):
        exp_id = g["experiment_id"].iloc[0]
        for alpha in cfg.tails:
            rows.append(_metric_row(g, alpha, model, fset, exp_id))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_root / "tables" / "metrics.csv", index=False)

    # regime 分段
    regimes = cfg.regimes
    reg_rows = []
    for (model, fset), g in panels.groupby(["model_id", "feature_set"]):
        g_labels = regime_labels(g["target_date"], regimes).to_numpy()
        for alpha in cfg.tails:
            for reg in regimes:
                sel = g[g_labels == reg]
                if len(sel) < 20:
                    continue
                v = sel[violation_col(alpha)].to_numpy(dtype=float)
                y = sel["realized_log_ret"].to_numpy(dtype=float)
                q = sel[q_col(alpha)].to_numpy(dtype=float)
                reg_rows.append({
                    "model": model, "feature_set": fset, "tail": alpha,
                    "regime": reg, "n": len(sel),
                    "failure_rate": float(np.nanmean(v)) if len(v) else np.nan,
                    "mean_pinball": pinball_loss(y, q, alpha),
                })
    pd.DataFrame(reg_rows).to_csv(out_root / "tables" / "regime_metrics.csv", index=False)

    # 消融：M2/M3 按特征集
    abl = metrics[metrics["model"].isin(["M2", "M3"])][
        ["model", "feature_set", "tail", "failure_rate", "mean_pinball", "kupiec_pvalue"]
    ]
    abl.to_csv(out_root / "tables" / "ablation.csv", index=False)

    # DM + block bootstrap：主模型两两对比（F3 信息集 / none）
    cmp_rows = []
    primary = {
        "M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3",
    }
    keep = panels[
        panels.apply(lambda r: primary.get(r["model_id"]) == r["feature_set"], axis=1)
    ]
    models = sorted(keep["model_id"].unique())
    boot_cfg = cfg.section("evaluation").get("bootstrap", {"B": 999, "block": 60})
    # 预声明 headline 对比族（审计修复）：这些是正式假设检验，
    # 其余两两对比仅作 exploratory appendix。
    headline_pairs = {("M1", "M3"), ("M2", "M3"), ("M3", "M5"), ("M1", "M4")}
    for alpha in cfg.tails:
        for a in models:
            for b in models:
                if a >= b:
                    continue
                ga = keep[keep["model_id"] == a].set_index("target_date")
                gb = keep[keep["model_id"] == b].set_index("target_date")
                idx = ga.index.intersection(gb.index)
                if len(idx) < 50:
                    continue
                ya = ga.loc[idx, "realized_log_ret"].to_numpy(dtype=float)
                qa = ga.loc[idx, q_col(alpha)].to_numpy(dtype=float)
                yb = gb.loc[idx, "realized_log_ret"].to_numpy(dtype=float)
                qb = gb.loc[idx, q_col(alpha)].to_numpy(dtype=float)
                loss_a = ya - qa
                loss_b = yb - qb
                la = loss_a * (alpha - (loss_a < 0))
                lb = loss_b * (alpha - (loss_b < 0))
                valid = np.isfinite(la) & np.isfinite(lb)
                dm = dm_test(la, lb)
                bb = block_bootstrap_pvalue(la, lb, B=boot_cfg["B"], block=boot_cfg["block"])
                cmp_rows.append({
                    "tail": alpha, "model_a": a, "model_b": b,
                    "n": dm["n"],
                    "dm_stat": dm["dm_stat"], "dm_pvalue": dm["pvalue"],
                    "bootstrap_pvalue": bb["pvalue"],
                    "pinball_a": float(la[valid].mean()), "pinball_b": float(lb[valid].mean()),
                    "favors": "a" if float(la[valid].mean()) < float(lb[valid].mean()) else "b",
                    "headline": int((a, b) in headline_pairs),
                    "holm_dm_pvalue": np.nan,
                })
    dm_df = pd.DataFrame(cmp_rows)
    # Holm family-wise correction within each tail for headline pairs (audit fix)
    if len(dm_df):
        for alpha in cfg.tails:
            sel = dm_df["tail"] == alpha
            head = dm_df[sel & (dm_df["headline"] == 1)]
            if len(head):
                adjusted = holm_adjust(head["dm_pvalue"].to_numpy(dtype=float))
                dm_df.loc[head.index, "holm_dm_pvalue"] = adjusted
    dm_df.to_csv(out_root / "tables" / "dm_comparison.csv", index=False)

    # seed 稳健性：robustness runs {7, 2026} + primary seed 42 面板 => n_seeds = 3
    rob = load_robustness_panels(out_root)
    rob_rows = []
    if len(rob):
        for (model, fset, seed), g in rob.groupby(["model_id", "feature_set", "seed"]):
            for alpha in cfg.tails:
                rob_rows.append(_metric_row(g, alpha, model, fset, f"rob-{model}-{fset}-s{seed}"))
    for model in ("M3", "M5"):
        for alpha in cfg.tails:
            key = (panels["model_id"] == model) & (panels["feature_set"] == "F3")
            g = panels[key]
            if not len(g):
                continue
            rob_rows.append(_metric_row(g, alpha, model, "F3", f"primary-{model}-F3-s{cfg.primary_seed}"))
    if rob_rows:
        pd.DataFrame(rob_rows).to_csv(out_root / "tables" / "seed_robustness.csv", index=False)
        summ = pd.DataFrame(rob_rows).groupby(["model", "feature_set", "tail"]).agg(
            n_seeds=("seed", "count"),
            failure_rate_mean=("failure_rate", "mean"),
            failure_rate_std=("failure_rate", "std"),
            pinball_mean=("mean_pinball", "mean"),
            pinball_std=("mean_pinball", "std"),
        ).reset_index()
        summ.to_csv(out_root / "tables" / "seed_robustness_summary.csv", index=False)

    print(f"metrics.csv            -> {out_root / 'tables' / 'metrics.csv'}")
    print(f"regime_metrics.csv     -> {out_root / 'tables' / 'regime_metrics.csv'}")
    print(f"ablation.csv           -> {out_root / 'tables' / 'ablation.csv'}")
    print(f"dm_comparison.csv      -> {out_root / 'tables' / 'dm_comparison.csv'}")


if __name__ == "__main__":
    main()
