"""合成 SPY 数据生成器（仅测试用）。

生成带已知条件分位数结构的 GARCH(1,1)-t 收益序列，
rv5/bv 由潜在波动率加噪声构造，保证与 |r| 正相关，
使泄漏/对齐/校准类测试可以在已知真值上验证语义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "omega": 2e-6,
    "alpha": 0.08,
    "beta": 0.90,
    "nu": 6.0,
    "mu": 4e-4,
    "rv_noise": 0.25,
}


def make_synthetic_spy(
    n: int = 2200,
    start: str = "2000-01-04",
    seed: int = 7,
    params: dict | None = None,
) -> pd.DataFrame:
    """生成 n 行合成日度数据，列为 log_ret / rv5 / bv。

    r_{t+1} = mu + sigma_{t+1} * z_{t+1}, z ~ t(nu)（方差已知），
    故真实条件分位数为 mu + sigma_{t+1} * t_ppf(alpha, nu)，
    可用于校准类测试。rv5_t = sigma_t^2 * exp(eps_t)，
    bv_t = sigma_t^2 * exp(eps'_t)（eps 为同分布噪声）。
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)
    n_warm = 500
    total = n + n_warm
    sigma2 = np.empty(total)
    sigma2[0] = p["omega"] / (1 - p["alpha"] - p["beta"])
    z = rng.standard_t(p["nu"], total)
    r = np.empty(total)
    for t in range(total):
        sigma2[t] = max(sigma2[t], 1e-12)
        r[t] = p["mu"] + np.sqrt(sigma2[t]) * z[t]
        if t + 1 < total:
            sigma2[t + 1] = p["omega"] + p["alpha"] * (r[t] - p["mu"]) ** 2 + p["beta"] * sigma2[t]

    r = r[n_warm:]
    sigma2 = sigma2[n_warm:]
    eps_rv = rng.normal(0.0, p["rv_noise"], n)
    eps_bv = rng.normal(0.0, p["rv_noise"], n)
    rv5 = sigma2 * np.exp(eps_rv)
    bv = sigma2 * np.exp(eps_bv)

    dates = pd.bdate_range(start=start, periods=n)
    df = pd.DataFrame({"date": dates, "log_ret": r, "rv5": rv5, "bv": bv})
    return df
