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

MODEL_REGISTRY: dict[str, Callable[[dict], Model]] = {
    "M0": lambda p: HistoricalSimulation(),
    "M1": lambda p: GARCHFamily(model_id="M1", vol="GARCH", o=0, dist="students-t"),
    "M1_gauss": lambda p: GARCHFamily(model_id="M1_gauss", vol="GARCH", o=0, dist="normal"),
    "M2": lambda p: LinearQuantile(),
    "M3": lambda p: MultiQuantileMLP(**p),
    "M4": lambda p: GARCHFamily(model_id="M4", vol="GARCH", o=1, dist="students-t"),
    "M5": lambda p: GRUQuantile(**p),
}

# 配置中非模型构造参数的键
_NON_PARAM_KEYS = {"enabled"}


def make_model(model_id: str, cfg: dict) -> Model:
    """按配置构造模型；过滤 enabled 等非构造参数键。"""
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"未知模型 {model_id}; 可用: {sorted(MODEL_REGISTRY)}")
    params = {
        k: v for k, v in cfg.get(model_id, {}).items() if k not in _NON_PARAM_KEYS
    }
    return MODEL_REGISTRY[model_id](params)


__all__ = ["MODEL_REGISTRY", "make_model"]
