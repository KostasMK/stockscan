"""US market calendar: which days are trading sessions and when each one closes.

The scanner used to treat "the latest bar the price source returned" as the latest
session. That cannot tell a holiday from a bar Yahoo has not published yet, so a late or
incomplete answer silently produced a scan of an older session. With a calendar the
expected session is known in advance and a missing bar is an error, not a fallback.

``MarketCalendar`` is the interface; ``ExchangeCalendar`` implements it with the
``exchange_calendars`` library (NYSE rules computed locally, no network). A broker
calendar (Alpaca, Part 3) can implement the same interface.
"""

from __future__ import annotations

import warnings
from datetime import UTC, date, datetime, timedelta
from functools import cache
from typing import Protocol

import pandas as pd
from pandas.errors import Pandas4Warning

from stockscan.timeutil import NY, require_aware

NYSE = "XNYS"


class MarketCalendar(Protocol):
    def is_session(self, day: date) -> bool:
        """True if the exchange trades on this New York date."""
        ...

    def session_close_utc(self, session: date) -> datetime:
        """Close of the regular session in UTC (13:00 New York on half-days)."""
        ...

    def previous_session(self, day: date) -> date:
        """Latest session strictly before ``day``."""
        ...


class ExchangeCalendar:
    """``MarketCalendar`` backed by ``exchange_calendars`` (default: NYSE)."""

    def __init__(self, code: str = NYSE) -> None:
        import exchange_calendars  # imported lazily: it takes about a second

        with warnings.catch_warnings():
            # exchange_calendars 4.x still uses a Series API that pandas 3 deprecates.
            warnings.simplefilter("ignore", Pandas4Warning)
            self._calendar = exchange_calendars.get_calendar(code)

    def is_session(self, day: date) -> bool:
        return bool(self._calendar.is_session(pd.Timestamp(day)))

    def session_close_utc(self, session: date) -> datetime:
        if not self.is_session(session):
            raise ValueError(f"{session.isoformat()} is not a trading session")
        return self._calendar.session_close(pd.Timestamp(session)).to_pydatetime()

    def previous_session(self, day: date) -> date:
        return self._calendar.date_to_session(
            pd.Timestamp(day - timedelta(days=1)), direction="previous"
        ).date()


@cache
def nyse_calendar() -> ExchangeCalendar:
    return ExchangeCalendar(NYSE)


def last_completed_session(
    calendar: MarketCalendar, now: datetime, buffer_minutes: int = 20
) -> date:
    """The latest session whose close plus ``buffer_minutes`` is at or before ``now``.

    This is the session a scan run at ``now`` must produce. It uses the real close time,
    so half-days (13:00 New York) are handled, and it is always a trading day.
    """
    require_aware(now)
    buffer = timedelta(minutes=buffer_minutes)
    candidate = now.astimezone(NY).date()
    if not calendar.is_session(candidate):
        candidate = calendar.previous_session(candidate)
    while calendar.session_close_utc(candidate) + buffer > now.astimezone(UTC):
        candidate = calendar.previous_session(candidate)
    return candidate
