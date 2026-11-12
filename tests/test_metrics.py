"""Unit tests for every metric, using small hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest

from stockscan import metrics
from stockscan.bars import adjust_ohlc
from tests.conftest import make_raw_bars, make_universe_bars


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_pct_diff():
    result = metrics.pct_diff(series([105, 95, 100]), series([100, 100, 0]))
    assert result.iloc[0] == pytest.approx(5.0)
    assert result.iloc[1] == pytest.approx(-5.0)
    assert np.isnan(result.iloc[2])  # division by zero is "unknown", not infinity


class TestRelativeVolume:
    def test_compares_today_with_the_previous_sessions_only(self):
        volume = series([100, 100, 100, 300])
        result = metrics.relative_volume(volume, window=3)
        assert result.iloc[:3].isna().all()  # not enough history yet
        # Baseline is the mean of the three days BEFORE: 100, so 300 is 3x (not 300/150).
        assert result.iloc[3] == pytest.approx(3.0)

    def test_average_volume_excludes_the_current_day(self):
        result = metrics.average_volume(series([10, 20, 30, 1000]), window=3)
        assert result.iloc[3] == pytest.approx(20.0)

    def test_zero_baseline_gives_nan(self):
        result = metrics.relative_volume(series([0, 0, 0, 50]), window=3)
        assert np.isnan(result.iloc[3])


def test_gap_pct_is_open_versus_previous_close():
    result = metrics.gap_pct(open_=series([10, 102, 97]), close=series([100, 100, 99]))
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(2.0)
    assert result.iloc[2] == pytest.approx(-3.0)


def test_day_change_pct_is_close_versus_previous_close():
    result = metrics.day_change_pct(series([100, 110, 99]))
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(-10.0)


class TestTrueRangeAndAtr:
    def test_true_range_uses_the_previous_close_for_gaps(self):
        high = series([10, 12, 15])
        low = series([8, 11, 14])
        close = series([9, 11.5, 14.5])
        result = metrics.true_range(high, low, close)
        assert result.tolist() == pytest.approx(
            [
                2.0,  # first row: high - low
                3.0,  # max(1, |12-9|, |11-9|)
                3.5,  # gap up: max(1, |15-11.5|, |14-11.5|)
            ]
        )

    def test_atr_uses_wilder_smoothing(self):
        # Constant close keeps the true range equal to high - low: 2, 4, 6, 8.
        close = series([100, 100, 100, 100])
        high = close + series([1, 2, 3, 4])
        low = close - series([1, 2, 3, 4])
        result = metrics.atr(high, low, close, window=2)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(3.0)  # seed: mean(2, 4)
        assert result.iloc[2] == pytest.approx((3.0 * 1 + 6) / 2)  # 4.5
        assert result.iloc[3] == pytest.approx((4.5 * 1 + 8) / 2)  # 6.25

    def test_atr_is_nan_with_too_little_history(self):
        bars = adjust_ohlc(make_raw_bars(sessions=10))
        assert metrics.atr(bars["high"], bars["low"], bars["close"], window=14).isna().all()


def test_sma_needs_a_full_window():
    result = metrics.sma(series([1, 2, 3, 4]), window=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)


def test_rolling_high_and_low_use_a_trailing_window():
    high = series([5, 9, 7, 6, 4])
    low = series([4, 8, 6, 5, 1])
    highs = metrics.rolling_high(high, window=3, min_periods=2)
    lows = metrics.rolling_low(low, window=3, min_periods=2)
    assert np.isnan(highs.iloc[0])
    assert highs.tolist()[1:] == [9, 9, 9, 7]  # the 9 leaves the window on the last day
    assert lows.tolist()[1:] == [4, 4, 5, 1]


class TestComputeMetrics:
    @pytest.fixture
    def result(self) -> pd.DataFrame:
        return metrics.compute_metrics(adjust_ohlc(make_raw_bars(sessions=300)))

    def test_adds_every_metric_column_and_keeps_rows(self, result):
        assert len(result) == 300
        assert set(metrics.METRIC_COLUMNS) <= set(result.columns)

    def test_last_row_is_fully_populated_with_enough_history(self, result):
        assert result.iloc[-1][metrics.METRIC_COLUMNS].notna().all()

    def test_values_are_internally_consistent(self, result):
        last = result.iloc[-1]
        previous = result.iloc[-2]
        assert last["day_change_pct"] == pytest.approx(
            (last["close"] / previous["close"] - 1) * 100
        )
        assert last["gap_pct"] == pytest.approx((last["open"] / previous["close"] - 1) * 100)
        assert last["atr_pct"] == pytest.approx(last["atr_14"] / last["close"] * 100)
        assert last["move_atr"] == pytest.approx(last["day_change_pct"] / last["atr_pct"])
        assert last["rel_volume"] == pytest.approx(
            last["volume"] / result["volume"].iloc[-21:-1].mean()
        )
        assert last["sma_20"] == pytest.approx(result["close"].iloc[-20:].mean())
        assert last["sma_50"] == pytest.approx(result["close"].iloc[-50:].mean())
        assert last["high_52w"] == pytest.approx(result["high"].iloc[-252:].max())
        assert last["low_52w"] == pytest.approx(result["low"].iloc[-252:].min())
        assert last["dist_52w_high_pct"] <= 0
        assert last["dist_52w_low_pct"] >= 0

    def test_short_history_gives_nan_instead_of_wrong_numbers(self):
        result = metrics.compute_metrics(adjust_ohlc(make_raw_bars(sessions=15)))
        last = result.iloc[-1]
        assert np.isnan(last["rel_volume"])  # needs 20 prior sessions
        assert np.isnan(last["sma_20"])
        assert np.isnan(last["high_52w"])
        assert not np.isnan(last["atr_14"])  # 14 sessions are enough
        assert not np.isnan(last["gap_pct"])


def test_compute_universe_metrics_keeps_tickers_independent():
    bars = adjust_ohlc(make_universe_bars(["AAA", "BBB"], sessions=80))
    combined = metrics.compute_universe_metrics(bars)
    alone = metrics.compute_metrics(bars[bars["ticker"] == "BBB"].reset_index(drop=True))
    from_combined = combined[combined["ticker"] == "BBB"].reset_index(drop=True)
    pd.testing.assert_frame_equal(from_combined, alone)
    # First row of each ticker has no previous close: it must not borrow the other ticker's.
    assert combined.groupby("ticker").head(1)["gap_pct"].isna().all()


def test_compute_universe_metrics_on_empty_input():
    empty = adjust_ohlc(make_raw_bars(sessions=3)).iloc[0:0]
    result = metrics.compute_universe_metrics(empty)
    assert result.empty
    assert set(metrics.METRIC_COLUMNS) <= set(result.columns)
