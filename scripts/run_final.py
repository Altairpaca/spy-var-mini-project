"""Frozen final-test runner (audit-hardened, 2026-08-13).

Gate (freeze.py::check_freeze_ready) must pass before anything runs:
- freeze.json exists, config hash matches, primary window selected;
- *effective* data file SHA256 matches the frozen value (no data-path bypass:
  --data override is rejected unless --force re-validates the actual file);
- code signature matches and the git working tree is clean.

Canonical run isolation: outputs are written under
outputs/runs/<freeze_id>/ where freeze_id is derived from the freeze commit
and config hash; stale artifacts from other runs never mix in. --clean-run
wipes the canonical run directory before running. Artifact reuse inside a run
is allowed only on exact experiment-signature match (common.artifact_current).

Model x feature matrix (predeclared, unchanged by audit):
- M0 HS / M1 GARCH-t / M4 GJR-t: feature none
- M2 Linear QR: F0, F1, F2, F3
- M3 MLP seed=42: F0, F1, F2, F3
- M5 GRU seed=42: F3
Robustness seeds {7, 2026} are handled by seed_robustness.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.common import (
    forecast_origins,
    load_data,
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


def _enabled_final_models(cfg) -> list[tuple[str, str]]:
    return [
        (m, f) for m, f in FINAL_MODELS
        if cfg.models.get(m, {}).get("enabled", False)
    ]


def freeze_id(manifest: dict) -> str:
    """Canonical run id: <freeze-commit-short>-<config-sha-short>."""
    return f"{manifest.get('git_commit', 'unknown')[:12]}-{manifest.get('config_sha256', '')[:8]}"


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen final test (audit-hardened)")
    p.add_argument("--config", default="configs/final.yaml")
    p.add_argument("--data", default=None, help="REJECTED unless --force re-validates the hash")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--docs-dir", default=None, help="ignored (freeze docs live in docs/)")
    p.add_argument("--force", action="store_true", help="allow --data override with hash re-check")
    p.add_argument("--clean-run", action="store_true", help="wipe canonical run dir before running")
    p.add_argument("--allow-dirty", action="store_true", help="test-only: skip clean-tree gate check")
    args = p.parse_args()

    cfg = resolve_config(args)
    freeze_dir = Path(args.out_root) / "manifests"
    freeze_json = freeze_dir / "freeze.json"
    if not freeze_json.exists():
        sys.exit("FINAL TEST GATE 拒绝: freeze.json 不存在（未冻结）")
    manifest = json.loads(freeze_json.read_text(encoding="utf-8"))

    # 数据路径审计：默认只允许冻结路径；--data 需要 --force 且实际哈希匹配冻结值
    effective_data = cfg.data_path
    if args.data is not None:
        if not args.force:
            sys.exit("FINAL TEST GATE 拒绝: --data override 被禁止（审计修复）；请使用冻结路径或 --force")
        from spyvar.data.loader import sha256_file

        actual = sha256_file(args.data)
        if actual != manifest.get("data_sha256"):
            sys.exit(f"FINAL TEST GATE 拒绝: --data 文件哈希 {actual[:16]} 与冻结值不匹配")
        effective_data = args.data

    ok, reason = check_freeze_ready(
        cfg, freeze_dir, effective_data_path=effective_data,
        require_clean_tree=not args.allow_dirty,
    )
    if not ok:
        sys.exit(f"FINAL TEST GATE 拒绝: {reason}")

    df = load_data(cfg, effective_data)
    origins = forecast_origins(df, cfg.final_test_start, "2099-12-31", cfg.primary_window)
    if len(origins) == 0:
        sys.exit("final test 区间内没有可预测日期（数据过短或窗口过大）")

    run_id = freeze_id(manifest)
    run_dir = Path(args.out_root) / "runs" / run_id
    pred_dir = run_dir / "predictions"
    if args.clean_run and pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 记录 canonical run（evaluate/figures/report 读取）
    current = Path(args.out_root) / "manifests" / "current_run.json"
    current.write_text(json.dumps({
        "freeze_id": run_id,
        "run_dir": str(run_dir),
        "git_commit": manifest.get("git_commit"),
        "config_sha256": manifest.get("config_sha256"),
        "data_sha256": manifest.get("data_sha256"),
        "n_origins": len(origins),
        "first_target": str(df.index[origins[0] + 1].date()),
        "last_target": str(df.index[origins[-1] + 1].date()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"final test (canonical run {run_id}): {len(origins)} origins "
          f"({df.index[origins[0] + 1].date()} -> {df.index[origins[-1] + 1].date()})")
    for model_id, fset in _enabled_final_models(cfg):
        exp_id = f"final-{model_id}-{fset}-w{cfg.primary_window}-s{cfg.primary_seed}"
        out_path = pred_dir / f"{exp_id}.parquet"
        panel = run_experiment(
            cfg, df, model_id=model_id, feature_set=fset,
            window=cfg.primary_window, seed=cfg.primary_seed,
            origins=origins, out_path=str(out_path), experiment_id=exp_id,
            force=args.clean_run,
        )
        fails = int((panel["fit_status"] != "ok").sum())
        print(f"  {exp_id}: {len(panel)} forecasts, fit failures {fails}")


if __name__ == "__main__":
    main()
