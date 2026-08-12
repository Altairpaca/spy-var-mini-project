"""M1/M4 —— GARCH 族条件波动模型。

M1: GARCH(1,1)-Student-t（核心基线）
M4: GJR-GARCH(1,1,1)-Student-t（leverage/downside asymmetry 检验）
M1_gauss: GARCH(1,1)-Normal（诊断参考）

每日按滚动窗口重新拟合，一步预测：
VaR_alpha = mu + t_ppf(alpha, nu) * sigma_{t+1|t}
非收敛、NaN、异常一律写入 fit_status，绝不静默。
"""

from __future__ import annotations

import numpy as np
from arch import arch_model
from scipy import stats

from ..rolling import Model, QuantileForecast, WindowData


class GARCHFamily(Model):
    def __init__(self, model_id: str, vol: str = "GARCH", o: int = 0, dist: str = "students-t"):
        self.model_id = model_id
        self._vol = vol
        self._o = o
        self._dist = "t" if dist == "students-t" else dist

    def fit(self, window: WindowData) -> QuantileForecast:
        y = window.returns
        # 手动缩放：arch 优化器在 0.01 量级收益上数值病态（DataScaleWarning），
        # 而 arch 7.2.0 的 rescale=True 不会把 forecast 换算回原始尺度。
        # 因此缩放 100 倍拟合，预测时换算回原始尺度（保守、可审计）。
        scale = 100.0
        res = arch_model(
            y * scale,
            mean="Constant",
            vol=self._vol,
            p=1,
            o=self._o,
            q=1,
            dist=self._dist,
        ).fit(disp="off", options={"maxiter": 1000})
        if res.convergence_flag != 0:
            status = f"non_convergence:{res.convergence_flag}"
        else:
            status = "ok"
        var1 = float(res.forecast(horizon=1, reindex=False).variance.iloc[0, 0])
        sigma = np.sqrt(max(var1, 1e-16)) / scale
        # 数值合理性：一步波动率远超窗口无条件波动率 => 拟合爆炸，弃用该预测
        window_scale = float(np.sqrt(np.mean(y**2)))
        if sigma > 50.0 * max(window_scale, 1e-12):
            status = "exploded"
        mu = float(res.params["mu"]) / scale
        if self._dist in ("t", "students-t"):
            nu = float(res.params["nu"])
            q = mu + sigma * stats.t.ppf([0.01, 0.05, 0.10], nu)
        else:
            q = mu + sigma * stats.norm.ppf([0.01, 0.05, 0.10])
        if not np.isfinite(q).all() or not np.isfinite(sigma):
            status = "nan_output"
        return QuantileForecast(
            q_001=float(q[0]),
            q_005=float(q[1]),
            q_010=float(q[2]),
            fit_status=status,
            meta={
                "sigma_next": sigma,
                "nu": float(res.params.get("nu", np.nan)),
                "omega": float(res.params.get("omega", np.nan)),
                "alpha": float(res.params.get("alpha[1]", np.nan)),
                "gamma": float(res.params.get("gamma[1]", np.nan)),
                "beta": float(res.params.get("beta[1]", np.nan)),
                "convergence_flag": int(res.convergence_flag),
            },
        )
