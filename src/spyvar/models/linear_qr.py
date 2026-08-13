"""M2 —— 线性（HAR 风格）分位数回归。

对 1%/5%/10% 各自拟合独立线性分位数回归（pinball 损失，
statsmodels QuantReg），与 MLP 共享完全相同的特征矩阵，
使 Linear vs MLP 的差异可归因于映射非线性而非信息集。
交叉率作为诊断输出（独立拟合允许交叉）。
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from ..rolling import Model, QuantileForecast, WindowData

_ALPHA_TO_Q = {0.01: "q_001", 0.05: "q_005", 0.10: "q_010"}


class LinearQuantile(Model):
    model_id = "M2"

    def fit(self, window: WindowData) -> QuantileForecast:
        if window.features is None:
            raise ValueError("M2 需要特征矩阵")
        feat = window.features.to_numpy(dtype=float)
        locs = window.dates.get_indexer(window.features.index)
        valid = locs + 1 < len(window.returns)
        if valid.sum() < 50:
            raise ValueError(f"训练样本过少: {int(valid.sum())}")
        X = sm.add_constant(feat[valid])
        yv = window.returns[locs[valid] + 1]
        # 训练行标准化（仅用训练行拟合 mu/sd），线性映射与原始尺度等价
        x_raw = X[:, 1:]
        mu = x_raw.mean(axis=0)
        sd = x_raw.std(axis=0)
        sd[sd == 0] = 1.0
        x_origin = np.concatenate([[1.0], (feat[-1] - mu) / sd])
        X = np.column_stack([np.ones(len(yv)), (x_raw - mu) / sd])
        qs = {}
        status = "ok"
        for alpha, key in _ALPHA_TO_Q.items():
            try:
                res = sm.QuantReg(yv, X).fit(q=alpha, max_iter=10000)
                qs[key] = float(res.predict(x_origin).item())
            except Exception as e:  # noqa: BLE001 —— 记录失败，不静默
                qs[key] = np.nan
                status = f"failed:{type(e).__name__}"
        return QuantileForecast(
            q_001=qs["q_001"],
            q_005=qs["q_005"],
            q_010=qs["q_010"],
            fit_status=status,
            meta={"n_train": int(valid.sum()), "n_features": int(feat.shape[1])},
        )
