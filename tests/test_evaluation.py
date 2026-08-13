"""评价公式正确性测试：pinball / 覆盖率 / 交叉率 / 游程（手算对照）。"""

from __future__ import annotations

import numpy as np
import pytest

from spyvar.evaluation.metrics import (
    crossing_rate,
    pinball_loss,
    violation_runs,
    violation_stats,
)


def test_pinball_hand_computed():
    y = np.array([-2.0, 1.0])
    q = np.array([-1.0, -1.0])
    alpha = 0.05
    # (y-q)(alpha - 1[y<q]): (-1)(0.05-1) + (2)(0.05) = 0.95 + 0.10 = 1.05
    assert pinball_loss(y, q, alpha) == pytest.approx(1.05 / 2.0)


def test_pinball_quantile_optimality():
    """中位数损失在样本中位数处最小（随机检查）。"""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(400)
    cands = np.linspace(-0.5, 0.5, 21)
    losses = [pinball_loss(y, np.full(400, c), 0.5) for c in cands]
    best = cands[int(np.argmin(losses))]
    assert abs(best - np.median(y)) < 0.1


def test_violation_stats_hand_computed():
    v = np.array([1, 0, 1, 1, 0, 1])
    s = violation_stats(v, alpha=0.5)
    assert s["n_forecasts"] == 6
    assert s["n_violations"] == 4
    assert s["expected_violations"] == 3.0
    assert s["failure_rate"] == pytest.approx(4 / 6)


def test_crossing_rate_hand_computed():
    q1 = np.array([-3.0, -2.0, -4.0])
    q5 = np.array([-2.0, -2.5, -3.0])
    q10 = np.array([-1.0, -1.0, -1.0])
    # 日期2: q1(-2.0) > q5(-2.5) 交叉 => 1/3
    assert crossing_rate(q1, q5, q10) == pytest.approx(1 / 3)


def test_violation_runs_hand_computed():
    v = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
    r = violation_runs(v)
    assert r["n_runs"] == 3
    assert r["max_run"] == 3
    assert r["mean_run"] == pytest.approx(2.0)


def test_metrics_recomputed_from_panel(synth_df):
    """指标必须能由预测面板重算（RESULTS_SCHEMA 完整性）。"""
    from spyvar.models.historical import HistoricalSimulation
    from spyvar.rolling import run_rolling

    panel = run_rolling(
        synth_df,
        HistoricalSimulation,
        np.arange(700, 760),
        window=300,
        tails=[0.01, 0.05, 0.10],
        seed=42,
        model_config={},
        workers=1,
    )
    recomputed = (panel["realized_log_ret"] <= panel["q_005"]).astype(int)
    assert (recomputed == panel["violation_005"]).all()
    s = violation_stats(panel["violation_005"].to_numpy(), 0.05)
    assert s["n_violations"] == int(recomputed.sum())
