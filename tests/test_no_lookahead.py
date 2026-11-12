"""Look-ahead bias guard.

A signal for day T may only use data available at the close of day T. The check: compute
the signals for T, then corrupt or delete every bar after T and compute again. If any
value at or before T changes, future data leaked into the signal.

The last test proves the guard has teeth by running it against a deliberately leaky metric.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from stockscan import metrics
from stockscan.bars import adjust_ohlc
from tests.conftest import make_raw_bars

VALUE_COLUMNS = ["open", "high", "low", "close", "raw_close", "volume"]
AS_OF_POSITIONS = [60, 150, 251, 252, 299, 349]

MetricFn = Callable[[pd.DataFrame], pd.DataFrame]


def corrupt_future(bars: pd.DataFrame, as_of: int, seed: int = 123) -> pd.DataFrame:
    """Replace every value after position ``as_of`` with wildly different numbers."""
    rng = np.random.default_rng(seed)
    corrupted = bars.copy()
    future = corrupted.index > as_of
    for column in VALUE_COLUMNS:
        noise = rng.uniform(0.1, 10.0, int(future.sum()))
        corrupted.loc[future, column] = corrupted.loc[future, column].to_numpy() * noise
    return corrupted


def assert_no_lookahead(compute: MetricFn, bars: pd.DataFrame, as_of: int) -> None:
    baseline = compute(bars).iloc[: as_of + 1]
    with_corrupted_future = compute(corrupt_future(bars, as_of)).iloc[: as_of + 1]
    with_no_future = compute(bars.iloc[: as_of + 1])

    pd.testing.assert_frame_equal(baseline, with_corrupted_future, check_exact=True)
    pd.testing.assert_frame_equal(baseline, with_no_future, check_exact=True)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return adjust_ohlc(make_raw_bars(sessions=400, seed=7)).reset_index(drop=True)


@pytest.mark.parametrize("as_of", AS_OF_POSITIONS)
def test_metrics_for_day_t_ignore_everything_after_day_t(bars, as_of):
    assert_no_lookahead(metrics.compute_metrics, bars, as_of)


def test_every_metric_column_is_covered_by_the_guard(bars):
    # compute_metrics is what the guard exercises; a metric computed elsewhere would escape it.
    produced = set(metrics.compute_metrics(bars).columns) - set(bars.columns)
    assert produced == set(metrics.METRIC_COLUMNS)


@pytest.mark.parametrize(
    "leaky",
    [
        # Tomorrow's close used as today's signal.
        lambda frame: frame.assign(leak=frame["close"].shift(-1)),
        # Centered window: half of it is in the future.
        lambda frame: frame.assign(leak=frame["close"].rolling(5, center=True).mean()),
        # Full-sample statistic: the "all-time high" knows about future highs.
        lambda frame: frame.assign(leak=frame["close"] / frame["high"].max()),
    ],
    ids=["negative-shift", "centered-window", "full-sample-max"],
)
def test_the_guard_catches_a_leaky_metric(bars, leaky):
    with pytest.raises(AssertionError):
        assert_no_lookahead(leaky, bars, as_of=200)
