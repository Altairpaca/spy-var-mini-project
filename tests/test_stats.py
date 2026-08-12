"""统计检验实现正确性：已知数值 + H0 下 p 值均匀性（蒙特卡洛）。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from spyvar.evaluation.backtests import (
    block_bootstrap_pvalue,
    christoffersen_cc,
    christoffersen_ind,
    dm_test,
    dq_test,
    kupiec_lr,
)


def test_kupiec_hand_computed_case():
    """n=1000, x=10, alpha=0.01 的 LR_uc 手算对照（无违例偏离 1% 的基准）。"""
    alpha, n, x = 0.01, 1000, 10
    pi = x / n
    lr = 2.0 * ((n - x) * np.log((1 - pi) / (1 - alpha)) + x * np.log(pi / alpha))
    got, _ = kupiec_lr(x, n, alpha)
    assert got == pytest.approx(lr, rel=1e-12)
    # x = 期望值 => LR ≈ 0
    lr0, p0 = kupiec_lr(10, 1000, 0.01)
    assert lr0 == pytest.approx(0.0, abs=1e-9)
    assert p0 == pytest.approx(1.0)


def test_kupiec_degenerate_cases():
    lr0, p0 = kupiec_lr(0, 500, 0.05)
    assert lr0 == pytest.approx(-2.0 * 500 * np.log(0.95), rel=1e-12)
    assert 0 < p0 < 1
    lrn, _pn = kupiec_lr(500, 500, 0.05)
    assert lrn == pytest.approx(-2.0 * 500 * np.log(0.05), rel=1e-12)


def test_nw_variance_matches_known_value():
    """Newey-West 方差手算对照（Bartlett 核，lag=1）。"""
    rng = np.random.default_rng(21)
    d = rng.standard_normal(50)
    d = d - d.mean()
    gamma0 = np.dot(d, d) / len(d)
    gamma1 = np.dot(d[1:], d[:-1]) / len(d)
    var = gamma0 + 2.0 * (1.0 - 1.0 / 2.0) * gamma1
    from spyvar.evaluation.backtests import _nw_variance

    assert _nw_variance(d + d.mean(), max_lag=1) == pytest.approx(var, rel=1e-10)


def test_block_bootstrap_centered_statistic():
    """中心化：bootstrap 分布围绕原假设 0（无差异数据下 p 值均匀）。"""
    from spyvar.evaluation.backtests import block_bootstrap_pvalue

    rng = np.random.default_rng(22)
    pvals = []
    for _ in range(30):
        loss = rng.exponential(1.0, 300)
        out = block_bootstrap_pvalue(loss, loss.copy(), B=99, block=20, seed=1)
        pvals.append(out["pvalue"])
    assert np.mean(pvals) > 0.2  # 无差异时不应系统性拒绝


def test_dq_test_edge_cases():
    """DQ 边界：样本不足返回 NaN。"""
    dq, p = dq_test(np.array([1.0, 0.0, 1.0]), 0.05, np.full(3, -1.0), lags=4)
    assert np.isnan(dq) and np.isnan(p)


def test_dm_test_rejects_constant_difference():
    """DM：确定性常数差异应被检测。"""
    rng = np.random.default_rng(23)
    base = rng.exponential(1.0, 400)
    out = dm_test(base, base + 0.5)
    assert out["dm_stat"] < -5
    assert out["pvalue"] < 0.001


def test_kupiec_rejects_perfect_coverage():
    """在 5% 水平违例率 15% 时，LR_uc 应显著。"""
    _, p = kupiec_lr(150, 1000, 0.05)
    assert p < 0.001


def test_christoffersen_ind_known_alternating():
    """完美交替序列应表现出负依赖（LR_ind 较大或至少非零）。"""
    n = 1000
    v = np.zeros(n)
    v[::2] = 1
    lr, p = christoffersen_ind(v)
    assert lr > 3.84 or p < 0.05


def test_christoffersen_ind_constant_series():
    """全 0 / 全 1 序列无依赖信息 => LR=0, p=1。"""
    lr0, p0 = christoffersen_ind(np.zeros(500))
    assert lr0 == 0.0 and p0 == 1.0
    lr1, p1 = christoffersen_ind(np.ones(500))
    assert lr1 == 0.0 and p1 == 1.0


def test_christoffersen_cc_equals_sum():
    v = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0] * 30)
    x = int(v.sum())
    n = len(v)
    lr_uc, _ = kupiec_lr(x, n, 0.05)
    lr_ind, _ = christoffersen_ind(v)
    lr_cc, p_cc = christoffersen_cc(v, x, n, 0.05)
    assert lr_cc == pytest.approx(lr_uc + lr_ind, rel=1e-12)
    assert 0 <= p_cc <= 1


def test_kupiec_pvalue_uniform_under_h0():
    """H0 正确覆盖下 p 值应近似均匀（KS 检验不拒绝）。

    n=600 时 LR_uc 有已知的有限样本正偏（Kupiec 检验小样本失真），
    因此用 n=2000 保证渐近近似成立。
    """
    rng = np.random.default_rng(123)
    pvals = []
    for _ in range(400):
        v = rng.random(2000) < 0.05
        x = int(v.sum())
        _, p = kupiec_lr(x, 2000, 0.05)
        pvals.append(p)
    ks = stats.kstest(pvals, "uniform")
    assert ks.pvalue > 0.01


def test_christoffersen_ind_pvalue_uniform_under_h0():
    """独立违例序列下 LR_ind 的 p 值近似均匀（n=2000）。"""
    rng = np.random.default_rng(456)
    pvals = []
    for _ in range(300):
        v = rng.random(2000) < 0.05
        _, p = christoffersen_ind(v)
        pvals.append(p)
    ks = stats.kstest(pvals, "uniform")
    assert ks.pvalue > 0.01


def test_dq_test_rejects_clustered_violations():
    """聚集违例应被 DQ 检验拒绝。"""
    rng = np.random.default_rng(7)
    n = 1200
    q = np.full(n, -1.5)
    y = rng.standard_normal(n)
    clustered = np.zeros(n)
    for i in range(0, n, 50):
        clustered[i : i + 8] = 1
    v = (y <= q).astype(int)
    v = np.maximum(v, clustered)  # 强行制造聚集
    _dq, p = dq_test(v, 0.05, q, lags=4)
    assert p < 0.01


def test_dq_test_accepts_random_violations():
    rng = np.random.default_rng(8)
    n = 2000
    v = (rng.random(n) < 0.05).astype(int)
    q = np.full(n, -1.645)
    _dq, p = dq_test(v, 0.05, q, lags=4)
    assert p > 0.01


def test_dm_known_direction():
    """模型 A 系统性优于 B => DM 统计量为负且显著。"""
    rng = np.random.default_rng(9)
    loss_a = rng.exponential(1.0, 800) * 0.8
    loss_b = rng.exponential(1.0, 800)
    out = dm_test(loss_a, loss_b)
    assert out["dm_stat"] < -2.5
    assert out["pvalue"] < 0.05


def test_dm_no_difference():
    rng = np.random.default_rng(10)
    loss = rng.exponential(1.0, 800)
    out = dm_test(loss, loss.copy())
    assert out["pvalue"] > 0.05


def test_block_bootstrap_rejects_difference():
    rng = np.random.default_rng(11)
    n = 600
    loss_a = rng.exponential(1.0, n) * 0.5
    loss_b = rng.exponential(1.0, n)
    out = block_bootstrap_pvalue(loss_a, loss_b, B=399, block=30, seed=1)
    assert out["pvalue"] < 0.05


def test_block_bootstrap_accepts_no_difference():
    rng = np.random.default_rng(12)
    loss = rng.exponential(1.0, 600)
    out = block_bootstrap_pvalue(loss, loss.copy(), B=199, block=30, seed=1)
    assert out["pvalue"] > 0.05
