"""Archive of intraday bars for the Phase 2 opening-range backtest.

Yahoo only keeps about 60 days of 5-minute bars, so a small daily job saves them while
they exist. One parquet file per session: ``intraday/5m/YYYY-MM-DD.parquet`` with
columns ticker, ts (bar start, UTC), open, high, low, close, volume.

Which tickers: the always-include list (NVDA, QQQ as benchmark) plus the scan's top-N
of each session in the window AND of the session before it. The previous session matters
because a signal from the close of day T is traded on day T+1: the backtest needs
T+1's intraday bars for T's top-N.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from stockscan.config import IntradayConfig
from stockscan.providers.base import INTRADAY_COLUMNS, PriceProvider
from stockscan.storage import ScanStore
from stockscan.timeutil import NY, last_completed_session_cutoff

logger = logging.getLogger(__name__)

MAX_DAYS = 55  # Yahoo serves roughly the last 60 days of 5-minute bars
PREVIOUS_SESSION_LOOKBACK_DAYS = 4  # reaches the prior session across a long weekend


@dataclass(frozen=True)
class IntradayOutcome:
    tickers: list[str]
    written: list[Path]
    already_saved: list[date]


def select_intraday_tickers(
    store: ScanStore, start: date, end: date, config: IntradayConfig
) -> list[str]:
    """Always-include tickers plus the top-N of every stored scan relevant to the window."""
    tickers = list(config.always_include)
    earliest = start - timedelta(days=PREVIOUS_SESSION_LOOKBACK_DAYS)
    for scan_date in store.scan_dates(earliest, end):
        tickers += store.top_tickers(scan_date, config.top_n)
    return list(dict.fromkeys(tickers))


def split_sessions(bars: pd.DataFrame, cutoff: date) -> dict[date, pd.DataFrame]:
    """Group bars by New York session date, dropping sessions after ``cutoff`` (unfinished)."""
    if bars.empty:
        return {}
    sessions = bars["ts"].dt.tz_convert(NY).dt.date
    return {
        session: group.sort_values(["ticker", "ts"]).reset_index(drop=True)[INTRADAY_COLUMNS]
        for session, group in bars.groupby(sessions)
        if session <= cutoff
    }


def save_intraday(
    *,
    prices: PriceProvider,
    store: ScanStore,
    config: IntradayConfig,
    output_dir: Path,
    now: datetime,
    days: int = 5,
    force: bool = False,
    close_buffer_minutes: int = 20,
) -> IntradayOutcome:
    """Save finished sessions from the last ``days`` calendar days that are not saved yet.

    Looking back a few days makes the job self-healing: a failed run is caught up by the
    next one. Existing files are left alone unless ``force`` is set.
    """
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_DAYS}")
    cutoff = last_completed_session_cutoff(now, close_buffer_minutes)
    start = cutoff - timedelta(days=days - 1)
    directory = output_dir / "intraday" / config.interval

    tickers = select_intraday_tickers(store, start, cutoff, config)
    bars = prices.intraday_bars(tickers, start, cutoff, config.interval)

    written: list[Path] = []
    already_saved: list[date] = []
    for session, frame in sorted(split_sessions(bars, cutoff).items()):
        path = directory / f"{session.isoformat()}.parquet"
        if path.exists() and not force:
            already_saved.append(session)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        written.append(path)
        logger.info(
            "saved %d %s bars for %d tickers to %s",
            len(frame),
            config.interval,
            frame["ticker"].nunique(),
            path,
        )
    return IntradayOutcome(tickers=tickers, written=written, already_saved=already_saved)
