"""Final-test freeze mechanism (audit-hardened, 2026-08-13).

final test may run only after the freeze manifest is ready and valid:
- data file SHA256 matches the frozen value (checked on the *effective* data path);
- config content hash matches the frozen value;
- primary window selected;
- code/evaluator signature (hash of src/ + scripts/ + tests/) matches;
- git working tree is clean at freeze time.

Any failed check -> check_freeze_ready returns (False, reason) and
run_final.py refuses to run (non-zero exit).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .config import Config, content_sha256
from .data.loader import sha256_file
from .io import git_commit_sha

CODE_ROOTS = ("src", "scripts", "tests")


def code_signature() -> str:
    """SHA256 over all python sources (evaluator/code signature)."""
    h = hashlib.sha256()
    for root in CODE_ROOTS:
        base = Path(root)
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            h.update(str(p).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def working_tree_clean() -> tuple[bool, str]:
    """True iff `git status --porcelain` is empty (ignoring .omo state)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=30
        )
    except Exception as e:  # noqa: BLE001
        return False, f"cannot run git status: {e}"
    lines = [l for l in out.stdout.splitlines() if not l.startswith("?? .omo/")]
    if lines:
        return False, f"working tree not clean: {lines[:3]}"
    return True, "clean"


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
    allow_dirty: bool = False,
) -> dict:
    """Write docs/FREEZE_MANIFEST.md and freeze.json (one canonical source)."""
    clean, clean_reason = working_tree_clean()
    if not clean and not allow_dirty:
        raise RuntimeError(f"freeze refused: {clean_reason}")
    manifest = {
        "freeze_created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit_sha(),
        "config_path": config.config_path,
        "config_sha256": config.sha256,
        "data_path": config.data_path,
        "data_sha256": data_sha256,
        "code_signature": code_signature(),
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


def check_freeze_ready(
    config: Config,
    freeze_dir: str | Path,
    effective_data_path: str | None = None,
    require_clean_tree: bool = True,
) -> tuple[bool, str]:
    """Check freeze validity; return (ready, reason).

    effective_data_path: the data file the runner will actually use
    (audit-hardened: prevents freezing data A but running data B).
    """
    fd = Path(freeze_dir)
    jp = fd / "freeze.json"
    if not jp.exists():
        return False, "freeze.json missing: final test not frozen"
    try:
        manifest = json.loads(jp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"freeze.json corrupted: {e}"
    if manifest.get("config_sha256") != config.sha256:
        return False, "config hash mismatch (config changed after freeze)"
    if config.primary_window is None:
        return False, "primary window not selected"
    data_path = Path(effective_data_path or config.data_path)
    if not data_path.exists():
        return False, f"data file missing: {data_path}"
    if manifest.get("data_sha256") != sha256_file(data_path):
        return False, "data file SHA256 mismatch with frozen value"
    if manifest.get("code_signature") != code_signature():
        return False, "code signature mismatch (src/scripts/tests changed after freeze)"
    # git_commit in the manifest is the freeze-generation HEAD; the freeze
    # commit that carries the manifest is its child. The frozen SHA must
    # therefore be the current HEAD or an ancestor of it.
    frozen_sha = manifest.get("git_commit")
    head_sha = git_commit_sha()
    if not frozen_sha:
        return False, "freeze manifest missing git_commit"
    if frozen_sha != head_sha:
        try:
            r = subprocess.run(
                ["git", "merge-base", "--is-ancestor", frozen_sha, head_sha],
                capture_output=True, timeout=30, check=False,
            )
            ancestor_ok = r.returncode == 0
        except Exception:  # noqa: BLE001
            ancestor_ok = False
        if not ancestor_ok:
            return False, "frozen git_commit is neither HEAD nor an ancestor (history moved)"
    if require_clean_tree:
        ok, reason = working_tree_clean()
        if not ok:
            return False, reason
    return True, "freeze ready"


def check_freeze_config_integrity(config: Config, freeze_dir: str | Path) -> bool:
    """冻结后 config 是否被改动的快速检查（报告生成等下游用）。"""
    ok, _ = check_freeze_ready(config, freeze_dir)
    return ok


def config_content_sha(config_path: str) -> str:
    return content_sha256(Path(config_path).read_text(encoding="utf-8"))
