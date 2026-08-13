"""Development window selection with equal-weight normalized aggregation.

Audit fix (2026-08-13): summing raw pinball losses biased toward tails with
larger loss scale and toward models with more feature configurations. New
aggregation, computed per (model, feature-set, tail):

  norm_loss(w) = loss(w) / mean(loss over candidates)   (candidate-relative)

then the equal-weight average across all (model, feature-set, tail) cells.
Reported alongside: per-cell winner, pairwise win count, and sensitivity
(drop-one-cell). Candidates stay {1000, 1500} (no third window). If the two
summaries disagree, the uncertainty is recorded and the conservative choice
(longer window at the 1% tail) is taken.

Development-only evidence; the final test is never consulted.
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


def collect_dev_panels(out_root: Path) -> pd.DataFrame:
    pred_dir = out_root / "predictions"
    frames = []
    for p in sorted(pred_dir.glob("dev-*.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        sys.exit(f"no dev prediction panels under {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_common_args("development window selection")
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
    dev_dir = out_root / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(dev_dir / "window_selection.csv", index=False)

    # equal-weight normalized aggregation per (model, feature-set, tail)
    cells = []
    for (model, fset, tail), g in tbl.groupby(["model", "feature_set", "tail"]):
        if len(g) < 2:
            continue
        losses = {int(r["window"]): r["mean_pinball"] for _, r in g.iterrows()}
        mean_l = float(np.mean(list(losses.values())))
        cells.append({
            "model": model, "feature_set": fset, "tail": tail,
            **{f"loss_w{w}": losses[w] for w in sorted(losses)},
            **{f"norm_w{w}": losses[w] / mean_l for w in sorted(losses)},
            "winner": min(losses, key=losses.get),
        })
    cell_df = pd.DataFrame(cells)
    cell_df.to_csv(dev_dir / "window_selection_cells.csv", index=False)

    windows = sorted(tbl["window"].unique())
    if len(windows) != 2:
        sys.exit(f"expected exactly 2 window candidates, got {windows}")
    w1, w2 = windows
    mean_norm = cell_df[[f"norm_w{w1}", f"norm_w{w2}"]].mean()
    win_counts = cell_df["winner"].value_counts().to_dict()
    chosen_raw = mean_norm.idxmin().replace("norm_w", "")
    # sensitivity: drop-one-cell winner distribution
    sens = {}
    for i in range(len(cell_df)):
        sub = cell_df.drop(index=i)
        m = sub[[f"norm_w{w1}", f"norm_w{w2}"]].mean()
        sens[i] = int(m.idxmin().replace("norm_w", ""))
    sens_counts = pd.Series(sens).value_counts().to_dict()

    # conservative choice on disagreement: longer window (more 1% tail samples)
    raw_sum = tbl.groupby("window")["mean_pinball"].sum()
    raw_winner = int(raw_sum.idxmin())
    disagreement = raw_winner != int(chosen_raw)
    chosen = int(chosen_raw)
    if disagreement:
        chosen = max(w1, w2)
        note = ("aggregations disagree; conservative choice (longer window) taken "
                "for 1% tail stability")
    else:
        note = "aggregations agree"
    decision = {
        "candidates": [w1, w2],
        "mean_normalized_loss": {int(k): float(v) for k, v in mean_norm.items()},
        "cell_win_counts": {int(k): int(v) for k, v in win_counts.items()},
        "sensitivity_win_counts": {int(k): int(v) for k, v in sens_counts.items()},
        "raw_pinball_sum_winner": int(raw_winner),
        "disagreement": bool(disagreement),
        "rule": ("min equal-weight mean of candidate-relative normalized loss; "
                 "disagreement -> conservative longer window"),
        "chosen_window": chosen,
        "note": note,
    }
    (dev_dir / "window_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f"tables -> {dev_dir}/window_selection.csv, window_decision.json")


if __name__ == "__main__":
    main()
