"""并行 worker 数 benchmark（用户协议 §16）。

在 development 期子集上对比 16/32/64 workers 的吞吐，
输出 outputs/tables/worker_benchmark.csv；BLAS 线程已限制为 1。
选择吞吐量最好的合理设置写入最终配置。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.common import (
    forecast_origins,
    load_data,
    parse_common_args,
    resolve_config,
)
from spyvar.models.historical import HistoricalSimulation
from spyvar.models.linear_qr import LinearQuantile
from spyvar.rolling import run_rolling

BENCH_WORKERS = [16, 32, 64]
BENCH_ORIGINS = 120  # 每个 worker 配置的原点数（相同任务集）


def bench_once(cfg, df, origins, workers, model_factory, feature_names):
    t0 = time.perf_counter()
    run_rolling(
        df, model_factory, origins, cfg.window_candidates[0], cfg.tails,
        cfg.primary_seed, {}, feature_names=feature_names, workers=workers,
    )
    return time.perf_counter() - t0


def main() -> None:
    args = parse_common_args("worker 吞吐 benchmark")
    cfg = resolve_config(args)
    df = load_data(cfg, args.data)
    origins = forecast_origins(df, "2005-01-01", "2007-12-31", max(cfg.window_candidates))[:BENCH_ORIGINS]
    rows = []
    for model, factory, fnames in [
        ("M0-HS", HistoricalSimulation, None),
        ("M2-LinQR-F3", LinearQuantile, cfg.feature_sets["F3"]),
    ]:
        for workers in BENCH_WORKERS:
            if workers > 64:
                continue
            t = bench_once(cfg, df, origins, workers, factory, fnames)
            rows.append({"model": model, "workers": workers, "origins": len(origins),
                         "runtime_s": round(t, 2), "origins_per_s": round(len(origins) / t, 2)})
            print(f"{model} workers={workers}: {t:.2f}s ({len(origins)/t:.1f} origins/s)")
    tbl = pd.DataFrame(rows)
    out_root = Path(args.out_root)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_root / "tables" / "worker_benchmark.csv", index=False)
    print(f"benchmark -> {out_root / 'tables' / 'worker_benchmark.csv'}")


if __name__ == "__main__":
    main()
