"""特征因果性测试：截断不变性 + 窗口内安全裁剪。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spyvar.features import FEATURE_SETS, build_features, required_pad


def _window(df, origin_pos, window):
    return df.iloc[origin_pos - window + 1 : origin_pos + 1]


def test_truncation_invariance(synth_df):
    """特征在日期 s 的值只依赖 <= s 的数据：截断窗口计算必须一致。"""
    names = FEATURE_SETS["F3"]
    origin = 600
    full = _window(synth_df, origin, 400)
    feats_full = build_features(full, names)
    for s in [origin - 100, origin - 1]:
        row_full = feats_full.loc[synth_df.index[s]]
        trunc = synth_df.iloc[s - 350 : s + 1]
        feats_trunc = build_features(trunc, names)
        row_trunc = feats_trunc.loc[synth_df.index[s]]
        pd.testing.assert_series_equal(row_full, row_trunc, check_names=False)


def test_feature_rows_are_within_window(synth_df):
    names = FEATURE_SETS["F3"]
    origin = 600
    full = _window(synth_df, origin, 400)
    feats = build_features(full, names)
    assert feats.index[0] == full.index[required_pad(names)]
    assert feats.index[-1] == full.index[-1]


def test_no_nan_after_pad(synth_df):
    for names in FEATURE_SETS.values():
        feats = build_features(_window(synth_df, 700, 500), names)
        assert not feats.isna().any().any()


def test_pad_insufficient_raises(synth_df):
    """窗口行数不足以支撑特征滞后时，特征矩阵应为空（后续由模型报错）。"""
    feats = build_features(_window(synth_df, 100, 10), FEATURE_SETS["F3"])
    assert len(feats) == 0


def test_unknown_feature_raises(synth_df):
    with pytest.raises(ValueError, match="未知特征"):
        build_features(_window(synth_df, 300, 200), ["not_a_feature"])


def test_har_aggregates_use_current_day_only(synth_df):
    """log_rv5_5d 在 s 处 = ln(mean(rv5[s-4..s]))，不包含 s+1。"""
    names = ["log_rv5_5d"]
    origin = 500
    full = _window(synth_df, origin, 300)
    feats = build_features(full, names)
    s_pos = origin - 50
    s = synth_df.index[s_pos]
    expected = np.log(synth_df["rv5"].iloc[s_pos - 4 : s_pos + 1].mean())
    assert feats.loc[s, "log_rv5_5d"] == pytest.approx(expected, rel=1e-12)
