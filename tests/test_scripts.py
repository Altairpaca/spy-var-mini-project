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
         "--config", cfg, "--data", data, "--out-root", out],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )


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
    panels = list((Path(out) / "predictions").glob("final-*.parquet"))
    assert len(panels) == 2  # M0, M1


def test_run_final_refused_when_data_changed_after_freeze(smoke_env):
    cfg, data, out = smoke_env
    assert _run("freeze_final.py", cfg, data, out).returncode == 0
    df = make_synthetic_spy(n=1200, seed=6)
    df.to_csv(data, index=False)
    r = _run("run_final.py", cfg, data, out)
    assert r.returncode != 0
    assert "SHA256" in r.stderr
