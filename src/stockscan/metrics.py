"""Daily metrics: small pure functions over pandas Series.

NO LOOK-AHEAD CONTRACT
----------------------
Every function here maps a time-ordered Series to a Series of the same index, and the
value at position T depends only on inputs at positions <= T. Only backward-looking
operations are allowed: ``shift(n)`` with n >= 0 and trailing ``rolling`` windows.
Never use ``shift(-1)``, ``center=True``, full-series statistics (``series.mean()``,
``series.max()``), ``bfill`` or ``interpolate``. ``tests/test_no_lookahead.py`` enforces
this for every metric: add new metrics to ``compute_metrics`` and they are covered.

Inputs are expected to be clean (see ``bars.clean_daily_bars``): sorted by date, no NaNs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REL_VOLUME_WINDOW = 20
ATR_WINDOW = 14
SMA_FAST = 20
SMA_SLOW = 50
RANGE_WINDOW = 252  # sessions in ~52 weeks
RANGE_MIN_PERIODS = 60  # recent listings still get a (shorter) range instead of NaN

METRIC_COLUMNS = [
    "avg_volume_20",
    "rel_volume",
    "gap_pct",
    "day_change_pct",
    "atr_14",
    "atr_pct",
    "move_atr",
    "sma_20",
    "sma_50",
    "close_vs_sma20_pct",
    "close_vs_sma50_pct",
    "high_52w",
    "low_52w",
    "dist_52w_high_pct",
    "dist_52w_low_pct",
]


def pct_diff(value: pd.Series, reference: pd.Series) -> pd.Series:
    """Percentage difference of ``value`` relative to ``reference``: +5.0 means 5% above."""
    return (value / reference.where(reference != 0) - 1.0) * 100.0


def average_volume(volume: pd.Series, window: int = REL_VOLUME_WINDOW) -> pd.Series:
    """Mean volume of the ``window`` sessions BEFORE each day (the day itself is excluded,
    so an unusual day is compared with a baseline it has not already inflated)."""
    return volume.shift(1).rolling(window, min_periods=window).mean()


def relative_volume(volume: pd.Series, window: int = REL_VOLUME_WINDOW) -> pd.Series:
    """Day's volume divided by the average of the previous ``window`` sessions."""
    baseline = average_volume(volume, window)
    return volume / baseline.where(baseline > 0)


def gap_pct(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Open versus the previous session's close, in percent."""
    return pct_diff(open_, close.shift(1))


def day_change_pct(close: pd.Series) -> pd.Series:
    """Close versus the previous session's close, in percent."""
    return pct_diff(close, close.shift(1))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(high-low, |high-prev close|, |low-prev close|). First row falls back to high-low."""
    previous_close = close.shift(1)
    candidates = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    )
    return candidates.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range with Wilder's smoothing.

    Seeded with the simple mean of the first ``window`` true ranges, then
    ``ATR[t] = (ATR[t-1] * (window - 1) + TR[t]) / window``. NaN until ``window`` bars exist.
    """
    ranges = true_range(high, low, close).to_numpy(dtype=float)
    values = np.full(len(ranges), np.nan)
    if len(ranges) >= window:
        values[window - 1] = ranges[:window].mean()
        for i in range(window, len(ranges)):
            values[i] = (values[i - 1] * (window - 1) + ranges[i]) / window
    return pd.Series(values, index=close.index)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the trailing ``window`` sessions (including the day)."""
    return series.rolling(window, min_periods=window).mean()


def rolling_high(
    high: pd.Series, window: int = RANGE_WINDOW, min_periods: int = RANGE_MIN_PERIODS
) -> pd.Series:
    """Highest high of the trailing ``window`` sessions (including the day)."""
    return high.rolling(window, min_periods=min_periods).max()


def rolling_low(
    low: pd.Series, window: int = RANGE_WINDOW, min_periods: int = RANGE_MIN_PERIODS
) -> pd.Series:
    """Lowest low of the trailing ``window`` sessions (including the day)."""
    return low.rolling(window, min_periods=min_periods).min()


def compute_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    """All metrics for ONE ticker.

    ``bars`` must be that ticker's adjusted bars sorted by date, with columns
    open, high, low, close, volume. Returns the input plus ``METRIC_COLUMNS``, one row per
    session, so the same code serves the daily scan (last row) and later backtests (all rows).
    """
    out = bars.copy()
    open_, high, low, close, volume = (
        out["open"],
        out["high"],
        out["low"],
        out["close"],
        out["volume"].astype(float),
    )

    out["avg_volume_20"] = average_volume(volume)
    out["rel_volume"] = relative_volume(volume)
    out["gap_pct"] = gap_pct(open_, close)
    out["day_change_pct"] = day_change_pct(close)
    out["atr_14"] = atr(high, low, close)
    out["atr_pct"] = out["atr_14"] / close * 100.0
    # Signed size of the day's move in units of "a normal day's range".
    out["move_atr"] = out["day_change_pct"] / out["atr_pct"].where(out["atr_pct"] > 0)
    out["sma_20"] = sma(close, SMA_FAST)
    out["sma_50"] = sma(close, SMA_SLOW)
    out["close_vs_sma20_pct"] = pct_diff(close, out["sma_20"])
    out["close_vs_sma50_pct"] = pct_diff(close, out["sma_50"])
    out["high_52w"] = rolling_high(high)
    out["low_52w"] = rolling_low(low)
    out["dist_52w_high_pct"] = pct_diff(close, out["high_52w"])  # <= 0
    out["dist_52w_low_pct"] = pct_diff(close, out["low_52w"])  # >= 0
    return out


def compute_universe_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    """Apply ``compute_metrics`` per ticker to a long frame of adjusted bars."""
    if bars.empty:
        return bars.assign(**{column: pd.Series(dtype="float64") for column in METRIC_COLUMNS})
    ordered = bars.sort_values(["ticker", "date"])
    parts = [compute_metrics(group) for _, group in ordered.groupby("ticker", sort=True)]
    return pd.concat(parts, ignore_index=True)
