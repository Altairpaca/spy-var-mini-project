"""pytest 共享夹具：合成 SPY 数据与最小引擎运行。"""

from __future__ import annotations

import pandas as pd
import pytest

from spyvar.data.synthetic import make_synthetic_spy


@pytest.fixture(scope="session")
def synth_df() -> pd.DataFrame:
    """大样本合成数据（测试窗口语义与校准行为）。"""
    df = make_synthetic_spy(n=2500, seed=7)
    df = df.set_index("date", drop=False)
    return df


@pytest.fixture(scope="session")
def synth_df_small() -> pd.DataFrame:
    """小样本合成数据（快速模型级测试）。"""
    df = make_synthetic_spy(n=800, seed=11)
    df = df.set_index("date", drop=False)
    return df
