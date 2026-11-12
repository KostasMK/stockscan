"""Time handling.

Rule of the project: compute and store in America/New_York (session dates) and UTC
(timestamps); convert to Europe/Athens only when showing something to the user.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
ATHENS = ZoneInfo("Europe/Athens")
REGULAR_CLOSE = time(16, 0)


def now_utc() -> datetime:
    return datetime.now(UTC)


def _require_aware(moment: datetime) -> None:
    if moment.tzinfo is None:
        raise ValueError("naive datetime: every timestamp in stockscan must be timezone-aware")


def last_completed_session_cutoff(now: datetime, buffer_minutes: int = 20) -> date:
    """Latest New York calendar date whose regular session has certainly finished.

    Any daily bar dated after this is still forming (or does not exist yet) and must be
    discarded: using it would leak intraday data into an end-of-day signal.

    The result is a calendar date, not necessarily a trading day: on a Sunday it returns
    Sunday (or Saturday before 16:00+buffer), and the latest bar on or before it is Friday's.
    """
    _require_aware(now)
    ny_now = now.astimezone(NY)
    trusted_from = datetime.combine(ny_now.date(), REGULAR_CLOSE, tzinfo=NY) + timedelta(
        minutes=buffer_minutes
    )
    return ny_now.date() if ny_now >= trusted_from else ny_now.date() - timedelta(days=1)


def session_close_utc(session: date) -> datetime:
    """16:00 New York on the session date, in UTC. (Ignores the few 13:00 early closes.)"""
    return datetime.combine(session, REGULAR_CLOSE, tzinfo=NY).astimezone(UTC)


def to_athens(moment: datetime) -> datetime:
    _require_aware(moment)
    return moment.astimezone(ATHENS)


def format_athens(moment: datetime) -> str:
    """Human-readable Athens time, e.g. ``2026-11-12 23:30 EET``."""
    return to_athens(moment).strftime("%Y-%m-%d %H:%M %Z")
