"""M0 —— Rolling Historical Simulation。

滚动窗口内历史收益的经验分位数（numpy 线性插值），
无参数、无拟合、确定性，作为最透明基准。
"""

from __future__ import annotations

import numpy as np

from ..rolling import Model, QuantileForecast, WindowData


class HistoricalSimulation(Model):
    model_id = "M0"

    def fit(self, window: WindowData) -> QuantileForecast:
        qs = np.quantile(window.returns, [0.01, 0.05, 0.10])
        return QuantileForecast(
            q_001=float(qs[0]),
            q_005=float(qs[1]),
            q_010=float(qs[2]),
            fit_status="ok",
            meta={"n_obs": len(window.returns)},
        )
