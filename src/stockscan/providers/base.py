"""Provider interfaces.

The scanner only talks to these protocols, so the unofficial yfinance source (or Alpha
Vantage) can be replaced without touching metrics, scoring or the pipeline. Tests use
in-memory fakes that satisfy the same protocols.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import pandas as pd

INTRADAY_COLUMNS = ["ticker", "ts", "open", "high", "low", "close", "volume"]


class ProviderError(Exception):
    """The provider could not deliver any data for a request."""


class NewsUnavailableError(Exception):
    """News cannot be fetched for the rest of this run (no key, rate limit, plan limits).

    The pipeline reacts by skipping news for the remaining tickers instead of failing.
    """


class PriceProvider(Protocol):
    def daily_bars(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Raw daily bars for sessions ``start``..``end`` inclusive.

        Returns a long frame with ``bars.RAW_DAILY_COLUMNS``. Tickers without data are
        simply absent. Raises ``ProviderError`` only if nothing could be fetched at all.
        """
        ...

    def intraday_bars(
        self, tickers: Sequence[str], start: date, end: date, interval: str
    ) -> pd.DataFrame:
        """Regular-hours intraday bars for New York dates ``start``..``end`` inclusive.

        Returns ``INTRADAY_COLUMNS`` where ``ts`` is the bar's start time, tz-aware UTC.
        """
        ...


@dataclass(frozen=True)
class NewsSentiment:
    ticker: str
    score: float | None
    """Relevance-weighted mean sentiment in [-1, 1]; None when no relevant article exists."""
    label: str | None
    article_count: int
    headline: str | None
    url: str | None
    published_utc: datetime | None


class NewsProvider(Protocol):
    def sentiment(self, ticker: str, start: datetime, end: datetime) -> NewsSentiment:
        """Sentiment from articles published in [start, end] (tz-aware).

        Raises ``NewsUnavailableError`` when no further calls can succeed in this run.
        """
        ...
