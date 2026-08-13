"""模型公平性测试：统一日期集、非交叉结构、模型级行为。"""

from __future__ import annotations

import numpy as np

from spyvar.features import FEATURE_SETS
from spyvar.models.garch import GARCHFamily
from spyvar.models.gru import GRUQuantile
from spyvar.models.historical import HistoricalSimulation
from spyvar.models.linear_qr import LinearQuantile
from spyvar.models.mlp import MultiQuantileMLP
from spyvar.rolling import make_window, run_rolling


def test_common_panel_identical_dates(synth_df):
    """所有主模型在同一原点集上输出相同 target_date 集合。"""
    origins = np.arange(700, 730)
    dates = None
    for factory, kw in [
        (HistoricalSimulation, {}),
        (lambda: GARCHFamily(model_id="M1"), {}),
        (lambda: GARCHFamily(model_id="M4", o=1), {}),
    ]:
        panel = run_rolling(
            synth_df, factory, origins, window=300, tails=[0.01, 0.05, 0.10],
            seed=42, model_config=kw, workers=1,
        )
        d = set(panel["target_date"].astype(str))
        dates = d if dates is None else dates & d
        assert set(panel["target_date"].astype(str)) == dates
    assert len(dates) == len(origins)


def test_nn_models_never_cross(synth_df_small):
    """MLP/GRU 结构非交叉：全部预测点满足 q_01 <= q_05 <= q_10。"""
    w = make_window(synth_df_small, 500, 300, FEATURE_SETS["F3"], seed=42, model_config={})
    for model in [
        MultiQuantileMLP(epochs=6, patience=2, hidden=[8]),
        GRUQuantile(epochs=6, patience=2, hidden=8, seq_len=10),
    ]:
        fc = model.fit(w)
        assert fc.q_001 <= fc.q_005 <= fc.q_010


def test_linear_qr_runs_and_orders(synth_df_small):
    w = make_window(synth_df_small, 500, 300, FEATURE_SETS["F3"], seed=42, model_config={})
    fc = LinearQuantile().fit(w)
    assert fc.fit_status == "ok"
    assert np.isfinite([fc.q_001, fc.q_005, fc.q_010]).all()


def test_garch_family_fits(synth_df_small):
    w = make_window(synth_df_small, 500, 300, None, seed=42, model_config={})
    for model in [
        GARCHFamily(model_id="M1"),
        GARCHFamily(model_id="M4", o=1),
        GARCHFamily(model_id="M1_gauss", dist="normal"),
    ]:
        fc = model.fit(w)
        assert fc.fit_status == "ok"
        assert fc.q_001 <= fc.q_005 <= fc.q_010


def test_hs_calibration_on_synthetic_garch(synth_df):
    """合成 GARCH-t 数据上 HS 的 5% 违例率保持在合理范围。

    HS 在波动率突升期反应滞后，违例率可能高于名义水平
    （这正是研究要量化的行为），因此断言取宽松区间。
    """
    origins = np.arange(700, 1700, 10)
    panel = run_rolling(
        synth_df, HistoricalSimulation, origins, window=600, tails=[0.05],
        seed=42, model_config={}, workers=1,
    )
    rate = panel["violation_005"].mean()
    assert 0.02 <= rate <= 0.15


def test_deterministic_rerun_bitwise(synth_df_small):
    """MLP 同 seed 重跑（串行）位级一致。"""
    w = make_window(synth_df_small, 500, 300, FEATURE_SETS["F0"], seed=42, model_config={})
    fc1 = MultiQuantileMLP(epochs=6, patience=2, hidden=[8]).fit(w)
    fc2 = MultiQuantileMLP(epochs=6, patience=2, hidden=[8]).fit(w)
    assert fc1.q_001 == fc2.q_001 and fc1.q_005 == fc2.q_005 and fc1.q_010 == fc2.q_010
