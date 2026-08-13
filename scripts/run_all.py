"""统一可复现入口：python scripts/run_all.py --config configs/final.yaml

阶段（--skip 可跳过任一阶段，用逗号分隔）：
  audit     数据审计 -> docs/DATA_AUDIT.md
  dev       development rolling-origin 验证（窗口 1000/1500）
  search    NN 超参搜索（development only，可跳过以复用已有决策）
  select    窗口选择 -> outputs/tables/window_decision.json
  freeze    冻结 final test（生成 FREEZE_MANIFEST.md）
  final     冻结 final test 全量运行
  robust    NN seed 稳健性
  eval      统一评估（全部指标表）
  figures   论文级图表
  report    PDF 报告

报告再生（不重训）：python scripts/make_report.py --config configs/final.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

ALL_STAGES = ["audit", "dev", "select", "search", "update", "freeze", "final", "robust", "eval", "figures", "report"]

STAGE_SCRIPT = {
    "audit": "scripts/audit_data.py",
    "dev": "scripts/run_development.py",
    "search": "scripts/neural_search.py",
    "select": "scripts/select_window.py",
    "update": "scripts/update_final_config.py",
    "freeze": "scripts/freeze_final.py",
    "final": "scripts/run_final.py",
    "robust": "scripts/seed_robustness.py",
    "eval": "scripts/evaluate.py",
    "figures": "scripts/make_figures.py",
    "report": "scripts/make_report.py",
}


def main() -> None:
    p = argparse.ArgumentParser(description="SPY VaR 项目统一入口")
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--data", default=None, help="数据覆盖（合成测试用）")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--skip", default="", help="逗号分隔的跳过阶段")
    p.add_argument("--stages", default="", help="逗号分隔的仅运行阶段（覆盖默认全量）")
    args = p.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.stages.split(",") if s.strip()}
    stages = [s for s in ALL_STAGES if s not in skip and (not only or s in only)]

    cmd_base = [sys.executable]
    for stage in stages:
        cmd = cmd_base + [STAGE_SCRIPT[stage], "--config", args.config]
        if args.data:
            cmd += ["--data", args.data]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        print(f"\n===== [{stage}] {' '.join(cmd)} =====")
        r = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=False)
        if r.returncode != 0:
            sys.exit(f"阶段 {stage} 失败 (exit {r.returncode})")
    print("\n全部阶段完成。")


if __name__ == "__main__":
    main()
