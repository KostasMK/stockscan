"""Price data from Yahoo Finance via the unofficial ``yfinance`` library.

yfinance scrapes endpoints Yahoo does not document, so it can break or be rate-limited
without notice. Everything Yahoo-specific is contained in this file; the rest of the
project sees only ``PriceProvider``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stockscan.bars import RAW_DAILY_COLUMNS, empty_daily_bars
from stockscan.http import RateLimiter
from stockscan.providers.base import INTRADAY_COLUMNS, PriceProvider, ProviderError
from stockscan.timeutil import NY

logger = logging.getLogger(__name__)

_FIELD_NAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

Downloader = Callable[..., pd.DataFrame]


def _default_downloader(**kwargs: Any) -> pd.DataFrame:
    import yfinance  # imported lazily: slow to import and not needed in tests

    return yfinance.download(**kwargs)


def _empty_intraday_bars() -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series(dtype="float64") for column in INTRADAY_COLUMNS})
    frame["ticker"] = frame["ticker"].astype("str")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame


def _per_ticker_frames(raw: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Split yfinance's wide (ticker, field) frame into one tidy frame per ticker."""
    if raw is None or raw.empty:
        return {}
    frames: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for ticker in tickers:
            if ticker in available:
                frames[ticker] = raw[ticker]
    elif len(tickers) == 1:  # older yfinance versions return flat columns for one ticker
        frames[tickers[0]] = raw
    for ticker, frame in frames.items():
        tidy = frame.rename(columns=_FIELD_NAMES)
        tidy = tidy[[column for column in _FIELD_NAMES.values() if column in tidy.columns]]
        frames[ticker] = tidy.dropna(how="all")
    return {ticker: frame for ticker, frame in frames.items() if not frame.empty}


class YFinanceProvider:
    """``PriceProvider`` backed by ``yfinance.download`` (one batched request per chunk)."""

    def __init__(
        self,
        chunk_size: int = 60,
        min_interval_seconds: float = 1.0,
        retries: int = 3,
        backoff_seconds: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        downloader: Downloader = _default_downloader,
    ) -> None:
        self._chunk_size = chunk_size
        self._limiter = RateLimiter(min_interval_seconds, sleep=sleep)
        self._retries = retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._download = downloader

    def _download_chunk(self, tickers: Sequence[str], **kwargs: Any) -> dict[str, pd.DataFrame]:
        """One batched download with retries. An empty answer counts as a failure."""
        problem = "empty response"
        for attempt in range(self._retries + 1):
            self._limiter.wait()
            try:
                raw = self._download(
                    tickers=list(tickers),
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    **kwargs,
                )
            except Exception as error:  # yfinance raises many unrelated exception types
                problem = f"{type(error).__name__}: {error}"
            else:
                frames = _per_ticker_frames(raw, tickers)
                if frames:
                    return frames
                problem = "empty response"
            if attempt < self._retries:
                delay = self._backoff_seconds * (2**attempt)
                logger.warning(
                    "yfinance download of %d tickers failed (%s); retry %d/%d in %.0fs",
                    len(tickers),
                    problem,
                    attempt + 1,
                    self._retries,
                    delay,
                )
                self._sleep(delay)
        logger.error("yfinance gave no data for %d tickers: %s", len(tickers), problem)
        return {}

    def _download_all(self, tickers: Sequence[str], **kwargs: Any) -> dict[str, pd.DataFrame]:
        unique = list(dict.fromkeys(tickers))
        frames: dict[str, pd.DataFrame] = {}
        for offset in range(0, len(unique), self._chunk_size):
            frames.update(
                self._download_chunk(unique[offset : offset + self._chunk_size], **kwargs)
            )
        if unique and not frames:
            raise ProviderError(
                f"yfinance returned no data for any of {len(unique)} tickers "
                "(Yahoo may be rate-limiting or the library may need an update)"
            )
        return frames

    def daily_bars(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        frames = self._download_all(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
            interval="1d",
        )
        parts = []
        for ticker, frame in frames.items():
            index = pd.DatetimeIndex(frame.index)
            if index.tz is not None:
                index = index.tz_convert(NY).tz_localize(None)
            part = frame.reset_index(drop=True)
            part.insert(0, "date", index.normalize())
            part.insert(0, "ticker", ticker)
            if "adj_close" not in part.columns:
                part["adj_close"] = float("nan")
            parts.append(part[RAW_DAILY_COLUMNS])
        if not parts:
            return empty_daily_bars()
        return pd.concat(parts, ignore_index=True)

    def intraday_bars(
        self, tickers: Sequence[str], start: date, end: date, interval: str
    ) -> pd.DataFrame:
        frames = self._download_all(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            prepost=False,
        )
        parts = []
        for ticker, frame in frames.items():
            index = pd.DatetimeIndex(frame.index)
            # Yahoo intraday timestamps are tz-aware; treat a naive index as New York time.
            index = index.tz_localize(NY) if index.tz is None else index
            part = frame.reset_index(drop=True)
            part.insert(0, "ts", index.tz_convert("UTC"))
            part.insert(0, "ticker", ticker)
            parts.append(part[INTRADAY_COLUMNS].dropna(subset=["open", "high", "low", "close"]))
        if not parts:
            return _empty_intraday_bars()
        return pd.concat(parts, ignore_index=True)


class CachedPriceProvider:
    """Parquet disk cache around any ``PriceProvider``.

    Repeated runs within ``max_age`` (re-running a scan, iterating on a report) reuse the
    first download instead of hitting the source again.
    """

    def __init__(
        self, inner: PriceProvider, cache_dir: Path, max_age: timedelta = timedelta(hours=6)
    ) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._max_age = max_age

    def _path(self, kind: str, tickers: Sequence[str], *parts: object) -> Path:
        key = "|".join([kind, ",".join(sorted(set(tickers))), *map(str, parts)])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{kind}_{digest}.parquet"

    def _cached(self, path: Path, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        try:
            age = timedelta(seconds=time.time() - path.stat().st_mtime)
            if age <= self._max_age:
                logger.info("using cached prices from %s", path.name)
                return pd.read_parquet(path)
        except (OSError, ValueError):
            pass  # no cache file, or an unreadable one: fetch again
        frame = fetch()
        if not frame.empty:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame

    def daily_bars(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        path = self._path("daily", tickers, start, end)
        return self._cached(path, lambda: self._inner.daily_bars(tickers, start, end))

    def intraday_bars(
        self, tickers: Sequence[str], start: date, end: date, interval: str
    ) -> pd.DataFrame:
        path = self._path("intraday", tickers, start, end, interval)
        return self._cached(path, lambda: self._inner.intraday_bars(tickers, start, end, interval))
