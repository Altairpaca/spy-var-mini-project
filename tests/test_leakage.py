"""泄漏防护测试：标准化只拟合训练子集、训练行标签对齐、无未来访问。"""

from __future__ import annotations

from spyvar.features import FEATURE_SETS
from spyvar.models.mlp import MultiQuantileMLP
from spyvar.rolling import make_window


def _fit_mlp_on_origin(synth_df, origin, seed, **kw):
    w = make_window(
        synth_df, origin, 400, FEATURE_SETS["F0"], seed=seed,
        model_config={},
    )
    model = MultiQuantileMLP(epochs=8, patience=3, hidden=[8], **kw)
    return model.fit(w)


def test_standardizer_uses_train_rows_only(synth_df):
    """验证集含极端值不影响 mu/sd（若 scaler 用了验证行，mu 会被拉偏）。"""
    from spyvar.features import build_features

    origin = 700
    window = synth_df.iloc[origin - 400 + 1 : origin + 1]
    feats = build_features(window, FEATURE_SETS["F0"])
    w = make_window(synth_df, origin, 400, FEATURE_SETS["F0"], seed=42, model_config={})

    # 篡改验证行（时间上最后 10% 的特征行）为极端值
    n_val = max(1, int(0.1 * (len(feats) - 1)))
    feats_poisoned = feats.copy()
    feats_poisoned.iloc[-(n_val + 1) : -1] = 1e6

    from types import SimpleNamespace

    w2 = SimpleNamespace(
        dates=w.dates,
        returns=w.returns,
        features=feats_poisoned,
        window_start=w.window_start,
        window_end=w.window_end,
        origin_date=w.origin_date,
        target_date=w.target_date,
        seed=42,
        model_config={},
    )
    fc_clean = MultiQuantileMLP(epochs=6, patience=2, hidden=[8]).fit(w)
    fc_poison = MultiQuantileMLP(epochs=6, patience=2, hidden=[8]).fit(w2)
    # 若 scaler 包含验证行，mu 被拉偏 => 预测显著漂移；仅用训练行则基本不变
    for a, b in [(fc_clean.q_005, fc_poison.q_005), (fc_clean.q_010, fc_poison.q_010)]:
        assert abs(a - b) < 0.1 * max(abs(a), abs(b), 1e-6) or (abs(a) < 0.5 and abs(b) < 0.5)


def test_training_label_is_next_day_return(synth_df):
    """M2/M3 的训练对 (x_s, r_{s+1})：窗口最后一行只用于预测。"""
    w = make_window(synth_df, 700, 400, FEATURE_SETS["F0"], seed=42, model_config={})
    feats = w.features
    locs = w.dates.get_indexer(feats.index)
    valid = locs + 1 < len(w.returns)
    y = w.returns[locs[valid] + 1]
    # 除最后一行（origin）外全部有效
    assert valid[:-1].all()
    assert not valid[-1]
    # 抽查：特征行日期 s 的标签 == s+1 的收益
    s_pos = locs[10]
    assert y[10] == w.returns[s_pos + 1]


def test_seed_reproducibility(synth_df):
    """同 seed 两次拟合 => 位级一致的预测。"""
    fc1 = _fit_mlp_on_origin(synth_df, 700, seed=42)
    fc2 = _fit_mlp_on_origin(synth_df, 700, seed=42)
    assert fc1.q_001 == fc2.q_001
    assert fc1.q_005 == fc2.q_005
    assert fc1.q_010 == fc2.q_010


def test_different_seed_gives_different_init(synth_df):
    fc1 = _fit_mlp_on_origin(synth_df, 700, seed=42)
    fc2 = _fit_mlp_on_origin(synth_df, 700, seed=7)
    assert (fc1.q_001, fc1.q_005, fc1.q_010) != (fc2.q_001, fc2.q_005, fc2.q_010)
