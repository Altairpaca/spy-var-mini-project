"""development 期滚动验证实验。

协议（recorded in RESEARCH_LOG）：
- development 期 = 2000-01-04 ~ 2007-12-31（< 2008-01-01）；
- 主验证年份 2005/2006/2007；窗口 1500 的最早可行原点约在
  2005-12，因此 1500 窗口的验证折叠实际为 2006-01 ~ 2007-12
  （文档化的轻微调整），两窗口在公共原点集上比较；
- 窗口候选 {1000, 1500}（用户协议固定，不扩展网格）；
- 模型：M0 HS、M1 GARCH-t、M1_gauss（诊断）、M2 × F0..F3、M4 GJR；
  NN 由 neural_search.py 另行处理（稀疏原点集）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from scripts.common import (
    load_data,
    parse_common_args,
    resolve_config,
    run_experiment,
)

DEV_MODELS = [
    ("M0", "none"),
    ("M1", "none"),
    ("M1_gauss", "none"),
    ("M4", "none"),
    *[("M2", f) for f in ("F0", "F1", "F2", "F3")],
]


def main() -> None:
    args = parse_common_args("development 期 rolling-origin 验证")
    cfg = resolve_config(args)
    if args.data is None and not Path(cfg.data_path).exists():
        sys.exit(f"数据文件不存在: {cfg.data_path}")
    df = load_data(cfg, args.data)
    dev_end = cfg.development_end
    out_root = Path(args.out_root)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)

    for window in cfg.window_candidates:
        # 验证原点：公共区间（两窗口均可预测的最早日期起）
        max_window = max(cfg.window_candidates)
        origins = forecast_origins_common(df, dev_end, max_window)
        print(f"窗口 {window}: {len(origins)} 个验证原点 "
              f"({df.index[origins[0]].date()} ~ {df.index[origins[-1]].date()})")
        for model_id, fset in DEV_MODELS:
            exp_id = f"dev-{model_id}-{fset}-w{window}"
            out_path = out_root / "predictions" / f"{exp_id}.parquet"
            panel = run_experiment(
                cfg, df, model_id=model_id, feature_set=fset, window=window,
                seed=cfg.primary_seed, origins=origins, out_path=str(out_path),
                experiment_id=exp_id,
            )
            fails = int((panel["fit_status"] != "ok").sum())
            print(f"  {exp_id}: {len(panel)} 预测, fit 失败 {fails}")


def forecast_origins_common(df: pd.DataFrame, dev_end: str, max_window: int):
    """公共原点集：历史足够 max_window 且目标日期 <= dev_end。"""
    from scripts.common import forecast_origins

    return forecast_origins(df, "1999-01-01", dev_end, max_window, min_history=0)


if __name__ == "__main__":
    main()
