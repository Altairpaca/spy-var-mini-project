"""滚动引擎语义测试：窗口边界、目标对齐、因果不变量。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spyvar.features import FEATURE_SETS
from spyvar.models.historical import HistoricalSimulation
from spyvar.rolling import make_window, run_rolling


def test_window_boundaries(synth_df):
    """窗口恰好 window 行，终点是原点日期。"""
    origin = 700
    window = 300
    w = make_window(synth_df, origin, window, None, seed=42, model_config={})
    assert len(w.dates) == window
    assert w.window_start == synth_df.index[origin - window + 1]
    assert w.window_end == synth_df.index[origin]
    assert w.origin_date == synth_df.index[origin]
    assert w.target_date == synth_df.index[origin + 1]
    assert len(w.returns) == window


def test_insufficient_history_raises(synth_df):
    import pytest

    with pytest.raises(ValueError, match="历史不足"):
        make_window(synth_df, 50, 300, None, seed=42, model_config={})


def test_target_alignment(synth_df):
    """实现收益 = 目标日期当日的真实收益；violation 由 q 重算。"""
    origins = np.array([700, 701, 702])
    panel = run_rolling(
        synth_df,
        HistoricalSimulation,
        origins,
        window=300,
        tails=[0.01, 0.05, 0.10],
        seed=42,
        model_config={},
        workers=1,
    )
    assert len(panel) == 3
    for _, row in panel.iterrows():
        pos = synth_df.index.get_loc(row["target_date"])
        assert row["realized_log_ret"] == pytest.approx(synth_df["log_ret"].iloc[pos])
        q = row["q_001"]
        assert row["violation_001"] == int(row["realized_log_ret"] <= q)


def test_predictions_do_not_depend_on_future(synth_df):
    """端到端因果性：截断在 t+1 的数据集与完整数据集在 t 的预测一致。"""
    origin = 899
    window = 400
    names = FEATURE_SETS["F2"]
    panel_full = run_rolling(
        synth_df,
        HistoricalSimulation,
        np.array([origin]),
        window=window,
        tails=[0.05],
        seed=42,
        model_config={},
        feature_names=names,
        workers=1,
    )
    truncated = synth_df.iloc[: origin + 2].copy()
    panel_trunc = run_rolling(
        truncated,
        HistoricalSimulation,
        np.array([origin]),
        window=window,
        tails=[0.05],
        seed=42,
        model_config={},
        feature_names=names,
        workers=1,
    )
    assert panel_full["q_005"].iloc[0] == pytest.approx(panel_trunc["q_005"].iloc[0])
    assert panel_full["fit_status"].iloc[0] == panel_trunc["fit_status"].iloc[0]


def test_parallel_matches_serial(synth_df):
    origins = np.arange(700, 706)
    ser = run_rolling(
        synth_df,
        HistoricalSimulation,
        origins,
        window=300,
        tails=[0.01, 0.05, 0.10],
        seed=42,
        model_config={},
        workers=1,
    )
    par = run_rolling(
        synth_df,
        HistoricalSimulation,
        origins,
        window=300,
        tails=[0.01, 0.05, 0.10],
        seed=42,
        model_config={},
        workers=2,
    )
    pd.testing.assert_frame_equal(
        ser.reset_index(drop=True), par.reset_index(drop=True), check_exact=True
    )


def test_failure_recorded_not_silent(synth_df):
    """模型抛异常时必须记录 fit_status 并保留实现收益。"""

    class Broken:
        model_id = "broken"

        def fit(self, window):
            raise RuntimeError("boom")

    panel = run_rolling(
        synth_df,
        Broken,
        np.array([700]),
        window=300,
        tails=[0.05],
        seed=42,
        model_config={},
        workers=1,
    )
    assert panel["fit_status"].iloc[0].startswith("failed:RuntimeError")
    assert np.isnan(panel["q_005"].iloc[0])
    assert not np.isnan(panel["realized_log_ret"].iloc[0])
