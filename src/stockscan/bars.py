"""Daily bar schema, cleaning and price adjustment.

Canonical "raw daily bars" frame (long format, one row per ticker per session):

    ticker (str) | date (datetime64, midnight, no tz = New York session date)
    open | high | low | close | adj_close | volume

Providers return this shape; everything downstream relies on it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

RAW_DAILY_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def empty_daily_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype="str"),
            "date": pd.Series(dtype="datetime64[ns]"),
            **{
                column: pd.Series(dtype="float64")
                for column in [*PRICE_COLUMNS, "adj_close", "volume"]
            },
        }
    )


def clean_daily_bars(bars: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Return only bars that are final and usable.

    * Drops every bar dated after ``cutoff``. This is the look-ahead guard: a bar for a
      session that has not finished yet must never reach a metric.
    * Drops rows with missing or non-positive prices, and rows with no volume (a full-day
      halt shows up as a zero-volume or empty bar; such a ticker simply has no bar that day).
    * Drops duplicate (ticker, date) rows, keeping the last one, and sorts.
    """
    missing = [column for column in RAW_DAILY_COLUMNS if column not in bars.columns]
    if missing:
        raise ValueError(f"daily bars are missing columns: {missing}")

    frame = bars[RAW_DAILY_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame[frame["date"] <= pd.Timestamp(cutoff)]

    prices_ok = frame[PRICE_COLUMNS].notna().all(axis=1) & (frame[PRICE_COLUMNS] > 0).all(axis=1)
    volume_ok = frame["volume"].notna() & (frame["volume"] > 0)
    frame = frame[prices_ok & volume_ok]

    frame = frame.drop_duplicates(subset=["ticker", "date"], keep="last")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def adjust_ohlc(bars: pd.DataFrame) -> pd.DataFrame:
    """Scale OHLC by ``adj_close / close`` so dividends do not look like price drops.

    Yahoo's OHLC and volume are already split-adjusted; ``adj_close`` adds dividends.
    The unadjusted close is kept as ``raw_close`` (the price you actually see on a chart
    for that day) and ``adj_close`` is dropped. A missing ``adj_close`` means "no adjustment".
    """
    frame = bars.copy()
    factor = (frame["adj_close"] / frame["close"]).fillna(1.0)
    frame["raw_close"] = frame["close"]
    for column in PRICE_COLUMNS:
        frame[column] = frame[column] * factor
    return frame.drop(columns=["adj_close"])
