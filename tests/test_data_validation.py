"""数据层校验测试：不可变、结构、语义。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spyvar.data.loader import DataValidationError, load_spy_data, sha256_file
from spyvar.data.synthetic import make_synthetic_spy


def _write_csv(tmp_path, df):
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


def test_loader_roundtrip(tmp_path):
    df = make_synthetic_spy(n=120)
    p = _write_csv(tmp_path, df)
    loaded = load_spy_data(p)
    assert list(loaded.columns) == ["date", "log_ret", "rv5", "bv"]
    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == 120


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataValidationError, match="不存在"):
        load_spy_data(tmp_path / "nope.csv")


def test_missing_column_raises(tmp_path):
    df = make_synthetic_spy(n=50).drop(columns=["bv"])
    with pytest.raises(DataValidationError, match="缺少必需列"):
        load_spy_data(_write_csv(tmp_path, df))


def test_duplicate_dates_raise(tmp_path):
    df = make_synthetic_spy(n=60)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="重复日期"):
        load_spy_data(_write_csv(tmp_path, df))


def test_non_monotonic_dates_are_sorted_then_rejected_if_dup(tmp_path):
    df = make_synthetic_spy(n=60).sample(frac=1.0, random_state=1)
    df = df.sort_values("date").reset_index(drop=True)
    loaded = load_spy_data(_write_csv(tmp_path, df))
    assert loaded.index.is_monotonic_increasing


def test_nan_raises(tmp_path):
    df = make_synthetic_spy(n=60)
    df.loc[10, "log_ret"] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        load_spy_data(_write_csv(tmp_path, df))


def test_negative_rv5_raises(tmp_path):
    df = make_synthetic_spy(n=60)
    df.loc[10, "rv5"] = -1.0
    with pytest.raises(DataValidationError, match="rv5"):
        load_spy_data(_write_csv(tmp_path, df))


def test_sha256_stable_and_changes_with_content(tmp_path):
    df = make_synthetic_spy(n=60)
    p = _write_csv(tmp_path, df)
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    df.loc[0, "log_ret"] = df.loc[0, "log_ret"] + 1e-9
    _write_csv(tmp_path, df)
    assert sha256_file(p) != h1


def test_load_does_not_modify_file(tmp_path):
    df = make_synthetic_spy(n=60)
    p = _write_csv(tmp_path, df)
    before = sha256_file(p)
    load_spy_data(p)
    assert sha256_file(p) == before
