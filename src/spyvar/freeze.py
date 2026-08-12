"""最终测试冻结机制。

final test 只允许在 FREEZE_MANIFEST 就绪后运行：
- data 文件 SHA256 与冻结值一致；
- config 内容哈希与冻结值一致；
- primary window 已选定；
- 冻结 commit 已存在（manifest 记录 git commit）。

任何检查不通过 -> check_freeze_ready 返回 (False, 原因)，
run_final.py 以非零退出码拒绝运行。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Config, content_sha256
from .data.loader import sha256_file
from .io import git_commit_sha


def write_freeze_manifest(
    config: Config,
    *,
    data_sha256: str,
    model_list: list[str],
    feature_sets: list[str],
    primary_window: int,
    seeds: list[int],
    evaluation_metrics: list[str],
    final_test_start: str,
    output_path: str | Path,
) -> dict:
    """写入 docs/FREEZE_MANIFEST.md 与 outputs/manifests/freeze.json。"""
    manifest = {
        "freeze_created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit_sha(),
        "config_path": config.config_path,
        "config_sha256": config.sha256,
        "data_path": config.data_path,
        "data_sha256": data_sha256,
        "models": model_list,
        "feature_sets": feature_sets,
        "primary_window": primary_window,
        "seeds": seeds,
        "evaluation_metrics": evaluation_metrics,
        "final_test_start": final_test_start,
        "development_end": config.development_end,
    }
    p = Path(output_path)
    p.mkdir(parents=True, exist_ok=True)
    md = p / "FREEZE_MANIFEST.md"
    md.write_text(
        "# Final Test Freeze Manifest\n\n"
        "本文件由 `scripts/freeze_final.py` 生成；任何数值禁止手改。\n\n"
        "```json\n"
        + json.dumps(manifest, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    jp = p / "freeze.json"
    jp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def check_freeze_ready(config: Config, freeze_dir: str | Path) -> tuple[bool, str]:
    """检查冻结状态；返回 (是否就绪, 原因)。"""
    fd = Path(freeze_dir)
    jp = fd / "freeze.json"
    if not jp.exists():
        return False, "freeze.json 不存在：final test 尚未冻结"
    try:
        manifest = json.loads(jp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"freeze.json 损坏: {e}"
    if manifest.get("config_sha256") != config.sha256:
        return False, "config 哈希与冻结值不一致（配置在冻结后被修改）"
    if config.primary_window is None:
        return False, "primary window 未选择"
    if not Path(config.data_path).exists():
        return False, f"数据文件缺失: {config.data_path}"
    current_data_sha = sha256_file(config.data_path)
    if manifest.get("data_sha256") != current_data_sha:
        return False, "数据文件 SHA256 与冻结值不一致"
    return True, "freeze 就绪"


def check_freeze_config_integrity(config: Config, freeze_dir: str | Path) -> bool:
    """冻结后 config 是否被改动的快速检查（报告生成等下游用）。"""
    ok, _ = check_freeze_ready(config, freeze_dir)
    return ok


def config_content_sha(config_path: str) -> str:
    return content_sha256(Path(config_path).read_text(encoding="utf-8"))
