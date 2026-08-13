"""Neural hyperparameter search (development only, small space).

- Window: read from outputs/development/window_decision.json (the SELECTED
  window; the script fails closed when the file is missing - protocol is
  select window -> search neural, never a fallback to a candidate window);
  validation origins = sparse subset of 2006-2007 (every 5th trading day),
  all < 2008-01-01;
- Grid: the neural_search section of configs/development.yaml;
- Selection criterion: validation mean pinball (mean over the three tails);
- Output: outputs/development/neural_search.csv +
  neural_search_decision.json; final hyperparameters are written into
  configs/final.yaml by update_final_config.py (scripted, no hand edits).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from scripts.common import (
    forecast_origins,
    load_data,
    parse_common_args,
    q_col,
    resolve_config,
    violation_col,
)
from spyvar.evaluation.metrics import pinball_loss
from spyvar.models import make_model
from spyvar.rolling import run_rolling

# 搜索在较小候选窗口上执行（development-only）
ORIGIN_STEP = 5
SEARCH_START = "2006-01-01"
SEARCH_END = "2007-12-31"


def _grid(search_spec: dict) -> list[dict]:
    keys = sorted(search_spec)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[search_spec[k] for k in keys])]


def main() -> None:
    args = parse_common_args("NN 超参搜索（development only）")
    cfg = resolve_config(args)
    df = load_data(cfg, args.data)
    search = cfg.section("neural_search")
    out_root = Path(args.out_root)
    dev_dir = out_root / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    # 协议修复（correction/final-pass）：搜索必须在窗口选择之后、
    # 在 development 选定的窗口上执行（不再用候选中的较小者）。
    decision_path = dev_dir / "window_decision.json"
    if not decision_path.exists():
        raise SystemExit(
            "ERROR: outputs/development/window_decision.json not found. "
            "Research protocol requires 'select window -> search neural'; "
            "run scripts/select_window.py first. Refusing to fall back to a "
            "candidate window."
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    search_window = int(decision["chosen_window"])
    print(f"neural search window: {search_window} (from window_decision.json)")

    all_rows = []
    best_by_model = {}
    for model_id in ("M3", "M5"):
        fset = "F3"
        if model_id not in search:
            continue
        origins = forecast_origins(df, SEARCH_START, SEARCH_END, search_window)
        origins = origins[::ORIGIN_STEP]
        print(f"{model_id}: {len(origins)} 搜索原点")
        grid = _grid(search[model_id])
        for combo in grid:
            model_cfg = {**cfg.models.get(model_id, {}), **combo}
            panel = run_rolling(
                df,
                lambda mid=model_id, mcfg=model_cfg: make_model(mid, {mid: mcfg}),
                origins, search_window, cfg.tails, cfg.primary_seed,
                model_cfg, feature_names=cfg.feature_sets[fset],
                workers=cfg.workers,
            )
            row = {"model_id": model_id, "feature_set": fset, "window": search_window,
                   "seed": cfg.primary_seed, "validation_fold": f"{SEARCH_START}_{SEARCH_END}", **combo}
            losses = []
            for alpha in cfg.tails:
                y = panel["realized_log_ret"].to_numpy(dtype=float)
                q = panel[q_col(alpha)].to_numpy(dtype=float)
                loss = pinball_loss(y, q, alpha)
                row[f"pinball_{int(alpha * 100):03d}"] = loss
                row[f"failrate_{int(alpha * 100):03d}"] = float(
                    np.nanmean(panel[violation_col(alpha)].to_numpy(dtype=float))
                )
                losses.append(loss)
            row["pinball_mean"] = float(np.mean(losses))
            row["selection_criterion"] = "mean pinball over 1/5/10% tails"
            row["fit_failures"] = int((panel["fit_status"] != "ok").sum())
            all_rows.append(row)
            print(f"  {combo}: pinball_mean={row['pinball_mean']:.6g}")
        tbl = pd.DataFrame([r for r in all_rows if r["model_id"] == model_id])
        best_row = tbl.loc[tbl["pinball_mean"].idxmin()]
        best_by_model[model_id] = {
            "best": {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in best_row.items() if k in search[model_id]
            },
            "pinball_mean": float(best_row["pinball_mean"]),
        }
        print(f"  BEST {model_id}: {best_by_model[model_id]}")

    pd.DataFrame(all_rows).to_csv(dev_dir / "neural_search.csv", index=False)
    (dev_dir / "neural_search_decision.json").write_text(
        json.dumps(best_by_model, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"搜索结果 -> {dev_dir / 'neural_search.csv'}")


if __name__ == "__main__":
    main()
