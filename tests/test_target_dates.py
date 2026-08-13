"""Audit regression tests: target-date partition boundary (P0 fix).

The research partition is defined by the forecast TARGET date:
  development target_date < 2008-01-01
  final target_date >= 2008-01-01
An origin on 2007-12-31 whose target is 2008-01-02 belongs to the final test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.common import forecast_origins
from spyvar.data.loader import load_spy_data

SPLIT = "2008-01-01"


@pytest.fixture(scope="module")
def real_df():
    return load_spy_data("data/raw/spy_data.csv")


def test_origins_filtered_by_target_date(real_df):
    """Origins are selected by the TARGET date, not the origin date."""
    dev = forecast_origins(real_df, "1999-01-01", "2007-12-31", 1500)
    fin = forecast_origins(real_df, SPLIT, "2099-12-31", 1500)
    dev_targets = real_df.index[np.array(dev) + 1]
    fin_targets = real_df.index[np.array(fin) + 1]
    # boundary: no development target >= split, no final target < split
    assert (dev_targets < pd.Timestamp(SPLIT)).all()
    assert (fin_targets >= pd.Timestamp(SPLIT)).all()


def test_partition_has_no_gap_or_overlap(real_df):
    """Dev and final target sets are disjoint and jointly gapless."""
    dev = forecast_origins(real_df, "1999-01-01", "2007-12-31", 1500)
    fin = forecast_origins(real_df, SPLIT, "2099-12-31", 1500)
    dev_t = set(real_df.index[np.array(dev) + 1])
    fin_t = set(real_df.index[np.array(fin) + 1])
    assert dev_t.isdisjoint(fin_t)
    # every target date from the earliest feasible one to the last is covered
    all_positions = np.arange(1500, len(real_df) - 1)
    all_targets = set(real_df.index[all_positions + 1])
    assert dev_t | fin_t == all_targets


def test_cross_boundary_origin_assigned_to_final(real_df):
    """An origin at 2007-12-31 (target 2008-01-02) must be in the final set."""
    fin = forecast_origins(real_df, SPLIT, "2099-12-31", 1500)
    fin_targets = real_df.index[np.array(fin) + 1]
    assert pd.Timestamp("2008-01-02") in fin_targets
    # and its origin (previous trading day) is before the split
    pos = real_df.index.get_loc(pd.Timestamp("2008-01-02")) - 1
    assert real_df.index[pos] < pd.Timestamp(SPLIT)


def test_first_final_target_is_first_trading_day_2008(real_df):
    fin = forecast_origins(real_df, SPLIT, "2099-12-31", 1500)
    fin_targets = real_df.index[np.array(fin) + 1]
    assert fin_targets.min() == pd.Timestamp("2008-01-02")


def test_development_validation_years_target_based(real_df):
    """Development validation panel (2006-2007) is target-date based."""
    dev = forecast_origins(real_df, "2006-01-01", "2007-12-31", 1500)
    targets = real_df.index[np.array(dev) + 1]
    assert (targets >= pd.Timestamp("2006-01-01")).all()
    assert (targets <= pd.Timestamp("2007-12-31")).all()
