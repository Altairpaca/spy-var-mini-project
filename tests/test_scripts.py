"""脚本级集成测试：final-test gate 在进程级拒绝未冻结运行。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from spyvar.data.synthetic import make_synthetic_spy

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def smoke_env(tmp_path):
    """临时配置 + 合成数据 + 最小模型集。"""
    df = make_synthetic_spy(n=2600, seed=5)
    data = tmp_path / "spy.csv"
    df.to_csv(data, index=False)
    cfg = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
    cfg["data"]["path"] = str(data)
    cfg["window"]["candidates"] = [200, 300]
    cfg["window"]["primary"] = 200
    cfg["models"] = {"M0": {"enabled": True}, "M1": {"enabled": True}}
    cfg["seeds"] = {"primary": 42, "robustness": [7]}
    cfg["parallel"]["workers"] = 2
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    out_root = tmp_path / "out"
    return str(cfg_path), str(data), str(out_root)


def _run(script: str, cfg: str, data: str, out: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script),
         "--config", cfg, "--out-root", out,
         "--docs-dir", str(Path(out).parent / "docs"),
         "--allow-dirty"],
        capture_output=True, text=True, cwd=ROOT, timeout=600, check=False,
    )


def test_canonical_run_dir_rejects_stale_marker(tmp_path):
    """A current_run.json that disagrees with freeze.json must fail closed."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from scripts.common import canonical_run_dir

    out = tmp_path / "out"
    (out / "manifests").mkdir(parents=True)
    run_dir = out / "runs" / "abc123456789-deadbeef"
    run_dir.mkdir(parents=True)
    freeze = {"git_commit": "a" * 40, "config_sha256": "b" * 64, "data_sha256": "c" * 64}
    (out / "manifests" / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    current = {
        "run_dir": str(run_dir),
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "data_sha256": "c" * 64,
    }
    (out / "manifests" / "current_run.json").write_text(json.dumps(current), encoding="utf-8")
    assert canonical_run_dir(out) == run_dir
    current["config_sha256"] = "d" * 64
    (out / "manifests" / "current_run.json").write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(SystemExit):
        canonical_run_dir(out)


def test_run_final_refused_before_freeze(smoke_env):
    cfg, data, out = smoke_env
    r = _run("run_final.py", cfg, data, out)
    assert r.returncode != 0
    assert "GATE" in r.stderr


def test_run_final_allowed_after_freeze(smoke_env):
    cfg, data, out = smoke_env
    r = _run("freeze_final.py", cfg, data, out)
    assert r.returncode == 0, r.stderr
    manifest = Path(out) / "manifests" / "freeze.json"
    assert manifest.exists()
    m = json.loads(manifest.read_text())
    assert m["primary_window"] == 200
    assert len(m["data_sha256"]) == 64
    r = _run("run_final.py", cfg, data, out)
    assert r.returncode == 0, r.stderr
    current = json.loads((Path(out) / "manifests" / "current_run.json").read_text())
    run_dir = Path(current["run_dir"])
    panels = list((run_dir / "predictions").glob("final-*.parquet"))
    assert len(panels) == 2  # M0, M1


def test_run_final_refused_when_data_changed_after_freeze(smoke_env):
    cfg, data, out = smoke_env
    assert _run("freeze_final.py", cfg, data, out).returncode == 0
    df = make_synthetic_spy(n=1200, seed=6)
    df.to_csv(data, index=False)
    r = _run("run_final.py", cfg, data, out)
    assert r.returncode != 0
    assert "SHA256" in r.stderr
