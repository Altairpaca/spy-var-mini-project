"""final-test 冻结：生成 docs/FREEZE_MANIFEST.md + outputs/manifests/freeze.json。

必须在首次运行 final test 之前执行；此后任何配置/数据改动都会
被 run_final.py 的 gate 拒绝。冻结信息：git commit SHA、config
SHA256、data SHA256、模型列表、特征集、窗口、seeds、指标、日期边界。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.common import parse_common_args, resolve_config
from spyvar.data.loader import sha256_file
from spyvar.freeze import write_freeze_manifest


def main() -> None:
    args = parse_common_args("final-test 冻结")
    cfg = resolve_config(args)
    if not Path(cfg.data_path).exists():
        sys.exit(f"数据文件不存在: {cfg.data_path}")
    if cfg.primary_window is None:
        sys.exit("configs/final.yaml 尚未选择 primary window —— 先运行 select_window 并更新配置")
    data_sha = sha256_file(cfg.data_path)
    model_list = ["M0", "M1", "M2", "M3", "M4", "M5"]
    feature_sets = list(cfg.feature_sets.keys())
    seeds = [cfg.primary_seed] + cfg.robustness_seeds
    metrics = [
        "failure_rate", "kupiec_lr", "christoffersen_ind", "christoffersen_cc",
        "dq_test", "mean_pinball", "crossing_rate", "violation_runs",
        "dm_test", "block_bootstrap",
    ]
    manifest = write_freeze_manifest(
        cfg,
        data_sha256=data_sha,
        model_list=model_list,
        feature_sets=feature_sets,
        primary_window=cfg.primary_window,
        seeds=seeds,
        evaluation_metrics=metrics,
        final_test_start=cfg.final_test_start,
        output_path=Path(args.out_root) / "manifests",
    )
    print("冻结完成:")
    for k in ("git_commit", "config_sha256", "data_sha256", "primary_window"):
        print(f"  {k}: {manifest[k]}")
    print(f"  manifest -> {Path(args.out_root) / 'manifests' / 'FREEZE_MANIFEST.md'}")


if __name__ == "__main__":
    main()
