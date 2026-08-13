"""NN seed 稳健性：M3-F3 与 M5-F3 在 robustness seeds 上重跑 final test。

seeds 预先声明（configs: 7, 2026）；主 seed 42 的面板来自 run_final.py。
只报告 mean/std 汇总，不按 seed 挑选主结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.common import (
    forecast_origins,
    load_data,
    parse_common_args,
    resolve_config,
    run_experiment,
)
from spyvar.freeze import check_freeze_ready

ROBUSTNESS_GRID = [
    ("M3", "F3"),
    ("M5", "F3"),
]


def main() -> None:
    args = parse_common_args("NN seed 稳健性")
    cfg = resolve_config(args)
    ok, reason = check_freeze_ready(cfg, Path(args.out_root) / "manifests")
    if not ok:
        sys.exit(f"FINAL TEST GATE 拒绝: {reason}")
    df = load_data(cfg, args.data)
    origins = forecast_origins(df, cfg.final_test_start, "2099-12-31", cfg.primary_window)
    out_root = Path(args.out_root)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    for model_id, fset in ROBUSTNESS_GRID:
        for seed in cfg.robustness_seeds:
            if seed == cfg.primary_seed:
                continue
            exp_id = f"rob-{model_id}-{fset}-w{cfg.primary_window}-s{seed}"
            out_path = out_root / "predictions" / f"{exp_id}.parquet"
            panel = run_experiment(
                cfg, df, model_id=model_id, feature_set=fset,
                window=cfg.primary_window, seed=seed,
                origins=origins, out_path=str(out_path), experiment_id=exp_id,
            )
            print(f"  {exp_id}: {len(panel)} 预测")


if __name__ == "__main__":
    main()
