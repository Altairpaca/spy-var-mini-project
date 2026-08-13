"""数据层：原始数据加载、校验、SHA256 冻结。"""

from .loader import DataValidationError, load_spy_data, sha256_file
from .synthetic import make_synthetic_spy

__all__ = ["DataValidationError", "load_spy_data", "make_synthetic_spy", "sha256_file"]
