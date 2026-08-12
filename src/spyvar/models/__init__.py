"""模型注册表：M0–M5。

模型工厂由配置驱动；所有模型通过统一 Model.fit(window) 接口
输出 QuantileForecast，禁止在 fit 之外接触任何数据。
"""

from __future__ import annotations

from collections.abc import Callable

from ..rolling import Model
from .garch import GARCHFamily
from .gru import GRUQuantile
from .historical import HistoricalSimulation
from .linear_qr import LinearQuantile
from .mlp import MultiQuantileMLP

MODEL_REGISTRY: dict[str, Callable[..., Model]] = {
    "M0": lambda cfg: HistoricalSimulation(),
    "M1": lambda cfg: GARCHFamily(model_id="M1", vol="GARCH", o=0, dist="students-t"),
    "M1_gauss": lambda cfg: GARCHFamily(model_id="M1_gauss", vol="GARCH", o=0, dist="normal"),
    "M2": lambda cfg: LinearQuantile(),
    "M3": lambda cfg: MultiQuantileMLP(**cfg.get("M3", {})),
    "M4": lambda cfg: GARCHFamily(model_id="M4", vol="GARCH", o=1, dist="students-t"),
    "M5": lambda cfg: GRUQuantile(**cfg.get("M5", {})),
}


def make_model(model_id: str, cfg: dict) -> Model:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {model_id}; 可用: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_id](cfg)


__all__ = ["MODEL_REGISTRY", "make_model"]
