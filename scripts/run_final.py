"""冻结 final test 运行。

gate 检查（freeze.py::check_freeze_ready）不通过则拒绝运行：
- freeze.json 存在且 config 哈希一致（配置在冻结后未被修改）；
- primary window 已选择；
- 数据文件 SHA256 与冻结值一致。

模型 × 特征矩阵（预定义，冻结后不可增删）：
- M0  HS:           特征 none
- M1  GARCH-t:      none
- M2  Linear QR:    F0, F1, F2, F3
- M3  MLP seed=42:  F0, F1, F2, F3
- M4  GJR-GARCH-t:  none
- M5  GRU seed=42:  F3
种子稳健性（7/2026）由 seed_robustness.py 处理。
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

FINAL_MODELS = [
    ("M0", "none"),
    ("M1", "none"),
    ("M4", "none"),
    *[("M2", f) for f in ("F0", "F1", "F2", "F3")],
    *[("M3", f) for f in ("F0", "F1", "F2", "F3")],
    ("M5", "F3"),
]


def main() -> None:
    args = parse_common_args("冻结 final test")
    cfg = resolve_config(args)
    ok, reason = check_freeze_ready(cfg, Path(args.out_root) / "manifests")
    if not ok:
        sys.exit(f"FINAL TEST GATE 拒绝: {reason}")
    df = load_data(cfg, args.data)
    origins = forecast_origins(df, cfg.final_test_start, "2099-12-31", cfg.primary_window)
    if len(origins) == 0:
        sys.exit("final test 区间内没有可预测日期（数据过短或窗口过大）")
    print(f"final test: {len(origins)} 个预测原点 "
          f"({df.index[origins[0]].date()} -> {df.index[origins[-1] + 1].date()})")
    out_root = Path(args.out_root)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    for model_id, fset in FINAL_MODELS:
        exp_id = f"final-{model_id}-{fset}-w{cfg.primary_window}-s{cfg.primary_seed}"
        out_path = out_root / "predictions" / f"{exp_id}.parquet"
        if out_path.exists():
            print(f"  跳过（已存在）: {exp_id}")
            continue
        panel = run_experiment(
            cfg, df, model_id=model_id, feature_set=fset,
            window=cfg.primary_window, seed=cfg.primary_seed,
            origins=origins, out_path=str(out_path), experiment_id=exp_id,
        )
        fails = int((panel["fit_status"] != "ok").sum())
        print(f"  {exp_id}: {len(panel)} 预测, fit 失败 {fails}")


if __name__ == "__main__":
    main()
