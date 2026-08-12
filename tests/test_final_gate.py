"""final-test 冻结门禁测试：未冻结或冻结失效时拒绝运行。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from spyvar.config import load_config
from spyvar.data.synthetic import make_synthetic_spy
from spyvar.freeze import check_freeze_ready, write_freeze_manifest


@pytest.fixture()
def cfg(tmp_path):
    """指向合成数据的测试配置（无冻结状态）。"""
    df = make_synthetic_spy(n=1200)
    df.to_csv(tmp_path / "spy.csv", index=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"path": str(tmp_path / "spy.csv"), "sha256": None},
                "dates": {"development_end": "2007-12-31", "final_test_start": "2008-01-01"},
                "tails": [0.01, 0.05, 0.10],
                "window": {"candidates": [1000, 1500], "primary": 1000},
                "features": {
                    "max_lag": 22,
                    "sets": {"F0": ["lag_ret_1", "abs_ret_1"], "F1": ["lag_ret_1", "log_rv5"]},
                },
                "seeds": {"primary": 42, "robustness": [7]},
                "models": {"M0": {"enabled": True}},
                "parallel": {"workers": 2},
                "evaluation": {"regimes": {"a": ["2000-01-01", "2001-01-01"]}},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return load_config(str(cfg_path))


def test_gate_blocks_without_freeze(cfg, tmp_path):
    ok, reason = check_freeze_ready(cfg, tmp_path / "manifests")
    assert not ok
    assert "尚未冻结" in reason


def test_gate_blocks_on_config_change(cfg, tmp_path):
    from spyvar.data.loader import sha256_file

    md = tmp_path / "manifests"
    md.mkdir()
    write_freeze_manifest(
        cfg,
        data_sha256=sha256_file(cfg.data_path),
        model_list=["M0"],
        feature_sets=["F0"],
        primary_window=1000,
        seeds=[42],
        evaluation_metrics=["kupiec"],
        final_test_start="2008-01-01",
        output_path=md,
    )
    ok, reason = check_freeze_ready(cfg, md)
    assert ok, reason
    # 修改配置内容（模拟冻结后改配置）=> 拒绝
    p = Path(cfg.config_path)
    p.write_text(p.read_text().replace("primary: 1000", "primary: 1500"), encoding="utf-8")
    cfg2 = load_config(str(p))
    ok2, reason2 = check_freeze_ready(cfg2, md)
    assert not ok2
    assert "config 哈希" in reason2


def test_gate_blocks_on_data_change(cfg, tmp_path):
    from spyvar.data.loader import sha256_file

    md = tmp_path / "manifests"
    md.mkdir()
    write_freeze_manifest(
        cfg,
        data_sha256=sha256_file(cfg.data_path),
        model_list=["M0"],
        feature_sets=["F0"],
        primary_window=1000,
        seeds=[42],
        evaluation_metrics=["kupiec"],
        final_test_start="2008-01-01",
        output_path=md,
    )
    df = pd.read_csv(cfg.data_path)
    df.loc[0, "log_ret"] += 1e-6
    df.to_csv(cfg.data_path, index=False)
    ok, reason = check_freeze_ready(cfg, md)
    assert not ok
    assert "数据文件 SHA256" in reason


def test_gate_blocks_without_primary_window(cfg, tmp_path):
    from spyvar.data.loader import sha256_file

    md = tmp_path / "manifests"
    md.mkdir()
    write_freeze_manifest(
        cfg,
        data_sha256=sha256_file(cfg.data_path),
        model_list=["M0"],
        feature_sets=["F0"],
        primary_window=1000,
        seeds=[42],
        evaluation_metrics=["kupiec"],
        final_test_start="2008-01-01",
        output_path=md,
    )
    p = Path(cfg.config_path)
    p.write_text(p.read_text().replace("primary: 1000", "primary: null"), encoding="utf-8")
    cfg2 = load_config(str(p))
    # 同步更新 manifest 中的 config 哈希，使检查到达 primary window 分支
    manifest = md / "freeze.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["config_sha256"] = cfg2.sha256
    manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    ok, reason = check_freeze_ready(cfg2, md)
    assert not ok
    assert "primary window" in reason


def test_gate_passes_when_frozen(cfg, tmp_path):
    from spyvar.data.loader import sha256_file

    md = tmp_path / "manifests"
    md.mkdir()
    write_freeze_manifest(
        cfg,
        data_sha256=sha256_file(cfg.data_path),
        model_list=["M0"],
        feature_sets=["F0"],
        primary_window=1000,
        seeds=[42],
        evaluation_metrics=["kupiec"],
        final_test_start="2008-01-01",
        output_path=md,
    )
    ok, reason = check_freeze_ready(cfg, md)
    assert ok
    assert reason == "freeze 就绪"
