"""论文级图表生成（10 张，全部来自 outputs 产物，信息密度优先）。

fig01: 全样本 SPY 收益与波动率概览
fig02: rv5 / bv 时间序列（对数尺度）
fig03: final-test 实现收益 + 动态 VaR 曲线（多模型叠加）
fig04: 违例点时间分布（1% tail，主模型）
fig05: failure rate 对比（model x tail）
fig06: pinball loss 对比（model x tail）
fig07: tail x model 热力图（pinball 相对最优）
fig08: 特征消融（M2/M3 按 F0-F3）
fig09: crisis/regime 对比
fig10: NN seed 稳健性
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.common import parse_common_args, q_col, resolve_config, violation_col
from spyvar.evaluation.metrics import pinball_loss

PRIMARY = {
    "M0": "none", "M1": "none", "M2": "F3", "M3": "F3", "M4": "none", "M5": "F3",
}
MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5"]
MODEL_LABELS = {
    "M0": "HS", "M1": "GARCH-t", "M2": "LinQR-F3", "M3": "MLP-F3",
    "M4": "GJR-t", "M5": "GRU-F3",
}


def _load_panels(out_root: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted((out_root / "predictions").glob("final-*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def _primary_panels(panels: pd.DataFrame) -> pd.DataFrame:
    return panels[
        panels.apply(lambda r: PRIMARY.get(r["model_id"]) == r["feature_set"], axis=1)
    ]


def _style(ax):
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_facecolor("white")


def fig01_overview(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(df.index, df["log_ret"], linewidth=0.4, color="#1f77b4")
    axes[0].set_ylabel("log return")
    axes[0].set_title("SPY daily log returns and realized volatility (full sample)")
    axes[1].plot(df.index, np.sqrt(df["rv5"]), linewidth=0.6, color="#d62728", label=r"$\sqrt{rv5}$")
    axes[1].plot(df.index, np.sqrt(df["bv"]), linewidth=0.6, color="#2ca02c", label=r"$\sqrt{bv}$")
    axes[1].set_ylabel("volatility scale")
    axes[1].legend(frameon=False)
    _style(axes[0]); _style(axes[1])
    fig.tight_layout()
    fig.savefig(out / "fig01_overview.png", dpi=150)
    plt.close(fig)


def fig02_rv_bv(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(df.index, np.log(df["rv5"]), linewidth=0.5, color="#d62728", label="log rv5")
    ax.plot(df.index, np.log(df["bv"]), linewidth=0.5, color="#2ca02c", label="log bv")
    ax.set_ylabel("log realized measure")
    ax.set_title("Realized variance (rv5) and bipower variation (bv), log scale")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig02_rv_bv.png", dpi=150)
    plt.close(fig)


def fig03_var_curves(panels: pd.DataFrame, out: Path, cfg) -> None:
    p = _primary_panels(panels)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, alpha in zip(axes, cfg.tails):
        y = p[p["model_id"] == "M0"].set_index("target_date")["realized_log_ret"]
        ax.plot(y.index, y, linewidth=0.35, color="gray", label="realized return")
        for model_id in MODEL_ORDER:
            g = p[p["model_id"] == model_id].set_index("target_date")
            ax.plot(g.index, g[q_col(alpha)], linewidth=0.7, label=MODEL_LABELS[model_id])
        ax.set_ylabel(f"{int(alpha*100)}% VaR")
        ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper right")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        _style(ax)
    axes[0].set_title("Frozen out-of-sample: realized returns and dynamic VaR forecasts")
    fig.tight_layout()
    fig.savefig(out / "fig03_var_curves.png", dpi=150)
    plt.close(fig)


def fig04_violations(panels: pd.DataFrame, out: Path, cfg) -> None:
    p = _primary_panels(panels)
    fig, ax = plt.subplots(figsize=(11, 4))
    for model_id in MODEL_ORDER:
        g = p[p["model_id"] == model_id].set_index("target_date")
        v = g[violation_col(0.01)].to_numpy(dtype=float)
        dates = g.index[np.isfinite(v) & (v == 1)]
        ax.scatter(dates, np.full(len(dates), MODEL_ORDER.index(model_id)),
                   s=4, marker="|", label=MODEL_LABELS[model_id])
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("1% VaR violation points by model (frozen final test)")
    ax.set_ylim(-0.5, len(MODEL_ORDER) - 0.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig04_violations.png", dpi=150)
    plt.close(fig)


def fig05_failure_rates(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics.copy()
    m["label"] = m["model"] + "-" + m["feature_set"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, alpha in zip(axes, [0.01, 0.05, 0.10]):
        sub = m[m["tail"] == alpha]
        keep = sub["label"].isin([f"{k}-{v}" for k, v in PRIMARY.items()])
        sub = sub[keep]
        labels = [MODEL_LABELS[r["model"]] for _, r in sub.iterrows()]
        ax.bar(range(len(sub)), sub["failure_rate"], color="#1f77b4")
        ax.axhline(alpha, color="red", linewidth=1, linestyle="--", label=f"target {alpha}")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel("failure rate")
        ax.set_title(f"{int(alpha*100)}% tail")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle("Empirical failure rates vs target coverage")
    fig.tight_layout()
    fig.savefig(out / "fig05_failure_rates.png", dpi=150)
    plt.close(fig)


def fig06_pinball(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics.copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, alpha in zip(axes, [0.01, 0.05, 0.10]):
        sub = m[(m["tail"] == alpha) & m["model"].isin(MODEL_ORDER)]
        sub = sub[sub.apply(lambda r: PRIMARY.get(r["model"]) == r["feature_set"], axis=1)]
        labels = [MODEL_LABELS[r["model"]] for _, r in sub.iterrows()]
        ax.bar(range(len(sub)), sub["mean_pinball"], color="#ff7f0e")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_ylabel("mean pinball loss")
        ax.set_title(f"{int(alpha*100)}% tail")
        _style(ax)
    fig.suptitle("Mean pinball loss by model (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "fig06_pinball.png", dpi=150)
    plt.close(fig)


def fig07_tail_model_heatmap(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics.copy()
    m = m[m.apply(lambda r: PRIMARY.get(r["model"]) == r["feature_set"], axis=1)]
    pivot = m.pivot_table(index="model", columns="tail", values="mean_pinball")
    pivot = pivot.loc[[x for x in MODEL_ORDER if x in pivot.index]]
    rel = pivot.div(pivot.min(axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(rel.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["1%", "5%", "10%"])
    ax.set_yticks(range(len(rel))); ax.set_yticklabels([MODEL_LABELS[m] for m in rel.index])
    for i in range(len(rel)):
        for j in range(3):
            ax.text(j, i, f"{rel.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pinball loss relative to best model per tail (1.00 = best)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig07_tail_model.png", dpi=150)
    plt.close(fig)


def fig08_ablation(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics[metrics["model"].isin(["M2", "M3"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ["M2", "M3"]):
        sub = m[m["model"] == model]
        for alpha in [0.01, 0.05, 0.10]:
            s = sub[sub["tail"] == alpha]
            ax.plot(s["feature_set"], s["mean_pinball"], marker="o", label=f"{int(alpha*100)}%")
        ax.set_xlabel("feature set")
        ax.set_ylabel("mean pinball")
        ax.set_title(f"{MODEL_LABELS[model]} by information set")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle("Feature ablation F0 (returns) -> F3 (+RV+BV+jump+downside)")
    fig.tight_layout()
    fig.savefig(out / "fig08_ablation.png", dpi=150)
    plt.close(fig)


def fig09_regime(regime_metrics: pd.DataFrame, out: Path) -> None:
    m = regime_metrics.copy()
    m["label"] = m["model"] + "-" + m["feature_set"]
    keep = m["label"].isin([f"{k}-{v}" for k, v in PRIMARY.items()])
    m = m[keep]
    m["model_label"] = m["model"].map(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(11, 4))
    for alpha, style in [(0.01, "-"), (0.05, "--"), (0.10, ":")]:
        sub = m[m["tail"] == alpha].pivot_table(index="regime", columns="model_label", values="failure_rate")
        for col in sub.columns:
            ax.plot(sub.index, sub[col], style, marker=".", linewidth=1.2, label=f"{col} {int(alpha*100)}%")
    ax.axhline(0.05, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("failure rate")
    ax.set_title("Failure rate by regime (dashed gray = 5% reference)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig09_regime.png", dpi=150)
    plt.close(fig)


def fig10_seed_robustness(rob_panels: pd.DataFrame, out: Path, cfg) -> None:
    if not len(rob_panels):
        print("  (无 robustness 面板，跳过 fig10)")
        return
    p = rob_panels.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, model in zip(axes, ["M3", "M5"]):
        sub = p[p["model_id"] == model]
        for alpha in cfg.tails:
            vals = []
            for seed, g in sub.groupby("seed"):
                y = g["realized_log_ret"].to_numpy(dtype=float)
                q = g[q_col(alpha)].to_numpy(dtype=float)
                vals.append((seed, pinball_loss(y, q, alpha)))
            seeds, losses = zip(*vals)
            ax.plot(seeds, losses, marker="o", label=f"{int(alpha*100)}%")
        ax.set_xlabel("seed")
        ax.set_ylabel("mean pinball")
        ax.set_title(f"{MODEL_LABELS[model]} seed robustness")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle("NN seed robustness (frozen final test, F3)")
    fig.tight_layout()
    fig.savefig(out / "fig10_seed_robustness.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_common_args("论文级图表")
    cfg = resolve_config(args)
    out_root = Path(args.out_root)
    figs = out_root / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    if args.data is not None:
        from spyvar.data.loader import load_spy_data
        df = load_spy_data(args.data)
        fig01_overview(df, figs)
        fig02_rv_bv(df, figs)
    elif Path(cfg.data_path).exists():
        from spyvar.data.loader import load_spy_data
        df = load_spy_data(cfg.data_path)
        fig01_overview(df, figs)
        fig02_rv_bv(df, figs)
    else:
        print("  (无数据文件，跳过 fig01/fig02)")
    panels = _load_panels(out_root)
    metrics = pd.read_csv(out_root / "tables" / "metrics.csv")
    regime = pd.read_csv(out_root / "tables" / "regime_metrics.csv")
    rob = pd.concat(
        [pd.read_parquet(p) for p in sorted((out_root / "predictions").glob("rob-*.parquet"))],
        ignore_index=True,
    ) if list((out_root / "predictions").glob("rob-*.parquet")) else pd.DataFrame()

    fig03_var_curves(panels, figs, cfg)
    fig04_violations(panels, figs, cfg)
    fig05_failure_rates(metrics, figs)
    fig06_pinball(metrics, figs)
    fig07_tail_model_heatmap(metrics, figs)
    fig08_ablation(metrics, figs)
    fig09_regime(regime, figs)
    fig10_seed_robustness(rob, figs, cfg)
    print(f"图表 -> {figs}")


if __name__ == "__main__":
    main()
