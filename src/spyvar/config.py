"""实验配置加载与校验。

配置是唯一权威的实验协议来源；每个实验记录 config 内容哈希，
保证任意报告数字都能追溯到冻结的配置。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置缺失、类型错误或自相矛盾。"""


@dataclass(frozen=True)
class Config:
    """冻结实验配置（加载后不可变）。"""

    raw: dict[str, Any]
    config_path: str
    sha256: str

    @property
    def data_path(self) -> str:
        return str(self.raw["data"]["path"])

    @property
    def data_sha256(self) -> str | None:
        return self.raw.get("data", {}).get("sha256")

    @property
    def development_end(self) -> str:
        return str(self.raw["dates"]["development_end"])

    @property
    def final_test_start(self) -> str:
        return str(self.raw["dates"]["final_test_start"])

    @property
    def tails(self) -> list[float]:
        return [float(t) for t in self.raw["tails"]]

    @property
    def window_candidates(self) -> list[int]:
        return [int(w) for w in self.raw["window"]["candidates"]]

    @property
    def primary_window(self) -> int | None:
        w = self.raw.get("window", {}).get("primary")
        return int(w) if w is not None else None

    @property
    def primary_seed(self) -> int:
        return int(self.raw["seeds"]["primary"])

    @property
    def robustness_seeds(self) -> list[int]:
        return [int(s) for s in self.raw["seeds"]["robustness"]]

    @property
    def feature_sets(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self.raw["features"]["sets"].items()}

    @property
    def max_lag(self) -> int:
        return int(self.raw["features"].get("max_lag", 22))

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.raw.get("models", {}).items()}

    @property
    def workers(self) -> int:
        return int(self.raw.get("parallel", {}).get("workers", 16))

    @property
    def regimes(self) -> dict[str, tuple[str, str]]:
        return {
            k: (str(v[0]), str(v[1]))
            for k, v in self.raw.get("evaluation", {}).get("regimes", {}).items()
        }

    def section(self, name: str) -> dict[str, Any]:
        """返回配置的任意子节（深拷贝）。"""
        return json.loads(json.dumps(self.raw.get(name, {})))

    def json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2, sort_keys=True)


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> Config:
    """加载 YAML 配置并计算内容哈希。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {p}")
    _validate(raw, str(p))
    return Config(raw=raw, config_path=str(p), sha256=content_sha256(p.read_text(encoding="utf-8")))


def _validate(raw: dict[str, Any], path: str) -> None:
    """校验配置必需字段，缺失即报错（宁可失败也不静默缺省）。"""
    for key in ("data", "dates", "tails", "window", "features", "seeds", "models", "parallel"):
        if key not in raw:
            raise ConfigError(f"{path}: 缺少必需节 {key!r}")
    if "path" not in raw["data"]:
        raise ConfigError(f"{path}: data.path 缺失")
    tails = raw["tails"]
    if not (isinstance(tails, list) and all(isinstance(t, (int, float)) for t in tails)):
        raise ConfigError(f"{path}: tails 必须是数值列表")
    if not all(0 < t < 1 for t in tails):
        raise ConfigError(f"{path}: tails 必须在 (0,1) 内")
    if sorted(tails) != tails:
        raise ConfigError(f"{path}: tails 必须升序排列")
    win = raw["window"]["candidates"]
    if not (isinstance(win, list) and all(isinstance(w, int) and w > 0 for w in win)):
        raise ConfigError(f"{path}: window.candidates 必须是正整数列表")
    if not isinstance(raw["seeds"].get("primary"), int):
        raise ConfigError(f"{path}: seeds.primary 必须是整数")
    if "sets" not in raw["features"] or not raw["features"]["sets"]:
        raise ConfigError(f"{path}: features.sets 必须至少定义 F0")
    if not isinstance(raw["parallel"].get("workers"), int) or raw["parallel"]["workers"] < 1:
        raise ConfigError(f"{path}: parallel.workers 必须 >= 1")
