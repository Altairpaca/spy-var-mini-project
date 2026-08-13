"""VaR 回测统计检验：Kupiec / Christoffersen / DQ / DM / block bootstrap。

全部实现带已知数值与蒙特卡洛均匀性单测（tests/test_stats.py），
有限样本行为与功效局限在报告与研究日志中说明。
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.special import xlogy


def kupiec_lr(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec 无条件覆盖检验：LR_uc ~ chi2(1)。

    处理 x=0 / x=n 的退化情形（对数似然取极限值）。
    """
    if n <= 0 or x < 0 or x > n:
        raise ValueError(f"非法计数: x={x}, n={n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1): {alpha}")
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - alpha)
    elif x == n:
        lr = -2.0 * n * np.log(alpha)
    else:
        pi = x / n
        lr = 2.0 * (
            (n - x) * np.log((1.0 - pi) / (1.0 - alpha)) + x * np.log(pi / alpha)
        )
    p = float(stats.chi2.sf(max(lr, 0.0), df=1))
    return float(lr), p


def _transition_counts(v: np.ndarray) -> tuple[int, int, int, int]:
    v = np.asarray(v, dtype=int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    return n00, n01, n10, n11


def christoffersen_ind(violations: np.ndarray) -> tuple[float, float]:
    """Christoffersen 独立性检验：LR_ind ~ chi2(1)。

    任一转移计数分母为 0（全部违例或全部无违例）时无依赖信息可识别，
    按约定返回 LR=0（证据不足，不拒绝独立性）。
    """
    v = np.asarray(violations, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 0.0, 1.0
    n00, n01, n10, n11 = _transition_counts(v)
    if n00 + n01 == 0 or n10 + n11 == 0:
        return 0.0, 1.0
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    lnl_pi = (n00 + n10) * np.log(1.0 - pi) + (n01 + n11) * np.log(pi)
    lnl_pi01 = (
        xlogy(n00, 1.0 - pi0)
        + xlogy(n01, pi0)
        + xlogy(n10, 1.0 - pi1)
        + xlogy(n11, pi1)
    )
    lr = -2.0 * (lnl_pi - lnl_pi01)
    p = float(stats.chi2.sf(max(lr, 0.0), df=1))
    return float(lr), p


def christoffersen_cc(violations: np.ndarray, x: int, n: int, alpha: float) -> tuple[float, float]:
    """Christoffersen 条件覆盖检验：LR_cc = LR_uc + LR_ind ~ chi2(2)。"""
    lr_uc, _ = kupiec_lr(x, n, alpha)
    lr_ind, _ = christoffersen_ind(violations)
    lr = lr_uc + lr_ind
    p = float(stats.chi2.sf(max(lr, 0.0), df=2))
    return float(lr), p


def dq_test(
    violations: np.ndarray,
    alpha: float,
    var_forecast: np.ndarray,
    lags: int = 4,
) -> tuple[float, float]:
    """Engle–Manganelli Dynamic Quantile 检验（附加结果）。

    hit_t = I_t - alpha 对 [1, hit_{t-1..t-lags}, VaR_{t-1}] 回归，
    DQ = n * R^2 ~ chi2(1 + lags + 1)。实现作为附加诊断，
    不替代 Kupiec/Christoffersen。
    """
    v = np.asarray(violations, dtype=float)
    q = np.asarray(var_forecast, dtype=float)
    mask = np.isfinite(v) & np.isfinite(q)
    v, q = v[mask], q[mask]
    n = len(v)
    if n <= lags + 2:
        return float("nan"), float("nan")
    hit = v - alpha
    X = np.column_stack(
        [np.ones(n - lags)]
        + [hit[lags - 1 - k : n - 1 - k] for k in range(lags)]
        + [q[lags - 1 : n - 1]]
    )
    y = hit[lags:]
    res = sm.OLS(y, X).fit()
    dq = n * float(res.rsquared)
    p = float(stats.chi2.sf(max(dq, 0.0), df=X.shape[1]))
    return dq, p


def _nw_variance(d: np.ndarray, max_lag: int | None = None) -> float:
    """Newey-West HAC 方差（Bartlett 核）。"""
    n = len(d)
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    d = d - d.mean()
    var = np.dot(d, d) / n
    for k in range(1, max_lag + 1):
        c = np.dot(d[k:], d[:-k]) / n
        var += 2.0 * (1.0 - k / (max_lag + 1.0)) * c
    return max(var, 1e-16)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    """Diebold–Mariano 检验（HAC 方差，双侧 p 值）。

    损失差 d = loss_a - loss_b（负值表示 a 更优）。
    尾部损失序列矩较弱，报告会给出有限样本警示。
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = len(d)
    if n < 4:
        return {"dm_stat": np.nan, "pvalue": np.nan, "n": n}
    var = _nw_variance(d)
    stat = float(d.mean() / np.sqrt(var / n))
    p = float(stats.norm.sf(abs(stat)) * 2.0)
    return {"dm_stat": stat, "pvalue": p, "n": n}


def block_bootstrap_pvalue(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    B: int = 999,
    block: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """移动块 bootstrap 的损失差显著性（双侧）。

    对配对损失差做 block bootstrap，返回经验 p 值；
    与 DM 互为稳健性对照。
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    n = len(d)
    if block is None:
        block = max(5, int(np.floor(np.sqrt(n))))
    obs = float(d.mean())
    se_obs = float(d.std(ddof=1) / np.sqrt(n))
    if se_obs < 1e-300:
        return {"pvalue": 1.0, "n": n, "block": block, "B": B}
    t_obs = abs(obs / se_obs)
    count = 0
    n_blocks = int(np.ceil(n / block))
    for _ in range(B):
        start = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in start])[:n]
        d_b = d[idx]
        t_b = (float(d_b.mean()) - obs) / (float(d_b.std(ddof=1)) / np.sqrt(n))
        count += abs(t_b) >= t_obs
    return {"pvalue": (count + 1) / (B + 1), "n": n, "block": block, "B": B}
