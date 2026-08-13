"""VaR 符号约定与违例定义测试（AGENTS.md 第 6 条）。"""

from __future__ import annotations

import numpy as np
import pytest

from spyvar.models.garch import GARCHFamily
from spyvar.models.historical import HistoricalSimulation
from spyvar.rolling import make_window, run_rolling


def test_hs_quantile_on_normal_data():
    """标准正态样本上，HS 经验分位数应接近理论值 -1.28/-1.65/-2.33。"""
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(5000)
    q = np.quantile(returns, [0.01, 0.05, 0.10])
    assert q[0] == pytest.approx(-2.326, abs=0.15)
    assert q[1] == pytest.approx(-1.645, abs=0.1)
    assert q[2] == pytest.approx(-1.282, abs=0.08)


def test_hs_vaR_negative_on_typical_data(synth_df):
    """典型波动数据下 1% VaR 必须为负（左尾约定）。"""
    panel = run_rolling(
        synth_df,
        HistoricalSimulation,
        np.array([700, 800, 900]),
        window=400,
        tails=[0.01, 0.05, 0.10],
        seed=42,
        model_config={},
        workers=1,
    )
    assert (panel["q_001"] < 0).all()
    assert (panel["q_010"] < 0).all()


def test_garch_vaR_sign_and_ordering(synth_df):
    w = make_window(synth_df, 900, 500, None, seed=42, model_config={})
    fc = GARCHFamily(model_id="M1").fit(w)
    assert fc.q_001 < fc.q_005 < fc.q_010
    assert fc.q_010 < 0
    assert fc.fit_status == "ok"


def test_violation_is_realized_leq_var(synth_df):
    """violation == (realized <= VaR)，非 < 比较。"""
    panel = run_rolling(
        synth_df,
        HistoricalSimulation,
        np.array([700]),
        window=400,
        tails=[0.05],
        seed=42,
        model_config={},
        workers=1,
    )
    row = panel.iloc[0]
    expected = int(row["realized_log_ret"] <= row["q_005"])
    assert row["violation_005"] == expected
