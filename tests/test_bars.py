from datetime import date

import numpy as np
import pandas as pd
import pytest

from stockscan.bars import adjust_ohlc, clean_daily_bars, empty_daily_bars
from tests.conftest import make_raw_bars


def test_bars_after_the_cutoff_are_dropped():
    bars = make_raw_bars(sessions=10, end="2026-11-12")
    cleaned = clean_daily_bars(bars, cutoff=date(2026, 11, 11))
    assert cleaned["date"].max() == pd.Timestamp("2026-11-11")
    assert len(cleaned) == 9


def test_rows_with_missing_or_invalid_prices_are_dropped():
    bars = make_raw_bars(sessions=10)
    bars.loc[2, "close"] = np.nan
    bars.loc[4, "low"] = 0.0
    bars.loc[6, "open"] = -1.0
    cleaned = clean_daily_bars(bars, cutoff=date(2026, 11, 11))
    assert len(cleaned) == 7
    assert cleaned[["open", "high", "low", "close"]].notna().all().all()


def test_zero_volume_bars_are_treated_as_no_session():
    bars = make_raw_bars(sessions=10)
    bars.loc[9, "volume"] = 0  # e.g. halted all day
    bars.loc[8, "volume"] = np.nan
    cleaned = clean_daily_bars(bars, cutoff=date(2026, 11, 11))
    assert len(cleaned) == 8
    assert cleaned["date"].max() < pd.Timestamp("2026-11-10")


def test_duplicates_are_removed_and_output_is_sorted():
    bars = make_raw_bars(sessions=5)
    shuffled = pd.concat([bars, bars.iloc[[1]]]).sample(frac=1, random_state=1)
    cleaned = clean_daily_bars(shuffled, cutoff=date(2026, 11, 11))
    assert len(cleaned) == 5
    assert cleaned["date"].is_monotonic_increasing


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="adj_close"):
        clean_daily_bars(make_raw_bars(sessions=3).drop(columns=["adj_close"]), date(2026, 11, 11))


def test_empty_frame_is_handled():
    assert clean_daily_bars(empty_daily_bars(), date(2026, 11, 11)).empty


def test_adjust_ohlc_scales_prices_by_the_dividend_factor():
    bars = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": pd.to_datetime(["2026-11-10", "2026-11-11"]),
            "open": [100.0, 99.0],
            "high": [102.0, 101.0],
            "low": [98.0, 97.0],
            "close": [100.0, 100.0],
            "adj_close": [99.0, 100.0],  # a 1% dividend went ex on the 11th
            "volume": [1e6, 1e6],
        }
    )
    adjusted = adjust_ohlc(bars)
    assert adjusted.loc[0, ["open", "high", "low", "close"]].tolist() == pytest.approx(
        [99.0, 100.98, 97.02, 99.0]
    )
    assert adjusted.loc[1, ["open", "high", "low", "close"]].tolist() == [99.0, 101.0, 97.0, 100.0]
    assert adjusted["raw_close"].tolist() == [100.0, 100.0]
    assert "adj_close" not in adjusted.columns
    assert adjusted["volume"].tolist() == [1e6, 1e6]


def test_adjust_ohlc_without_adj_close_leaves_prices_alone():
    bars = make_raw_bars(sessions=3)
    bars["adj_close"] = np.nan
    adjusted = adjust_ohlc(bars)
    pd.testing.assert_series_equal(adjusted["close"], bars["close"])
