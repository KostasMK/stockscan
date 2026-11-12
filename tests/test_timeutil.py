from datetime import UTC, date, datetime

import pytest

from stockscan.timeutil import (
    format_athens,
    last_completed_session_cutoff,
    session_close_utc,
)


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


class TestLastCompletedSessionCutoff:
    def test_during_us_session_returns_previous_day(self):
        # 19:54 UTC on 12 Nov 2026 = 14:54 New York (EST): market still open.
        assert last_completed_session_cutoff(utc(2026, 11, 12, 19, 54)) == date(2026, 11, 11)

    def test_right_after_close_is_still_inside_buffer(self):
        # 21:10 UTC = 16:10 New York: closed, but inside the 20-minute buffer.
        assert last_completed_session_cutoff(utc(2026, 11, 12, 21, 10)) == date(2026, 11, 11)

    def test_scheduled_run_in_winter_uses_today(self):
        # The GitHub Action fires at 21:30 UTC = 16:30 EST.
        assert last_completed_session_cutoff(utc(2026, 11, 12, 21, 30)) == date(2026, 11, 12)

    def test_scheduled_run_in_summer_uses_today(self):
        # 21:30 UTC = 17:30 EDT.
        assert last_completed_session_cutoff(utc(2026, 7, 15, 21, 30)) == date(2026, 7, 15)

    def test_after_utc_midnight_still_refers_to_new_york_date(self):
        # 01:00 UTC on the 13th is 20:00 on the 12th in New York.
        assert last_completed_session_cutoff(utc(2026, 11, 13, 1, 0)) == date(2026, 11, 12)

    def test_buffer_is_configurable(self):
        assert last_completed_session_cutoff(utc(2026, 11, 12, 21, 10), buffer_minutes=5) == date(
            2026, 11, 12
        )

    def test_naive_datetime_is_rejected(self):
        with pytest.raises(ValueError, match="naive"):
            last_completed_session_cutoff(datetime(2026, 11, 12, 21, 30))


def test_session_close_utc_follows_daylight_saving():
    assert session_close_utc(date(2026, 11, 12)) == utc(2026, 11, 12, 21, 0)  # EST
    assert session_close_utc(date(2026, 7, 15)) == utc(2026, 7, 15, 20, 0)  # EDT


def test_format_athens_converts_from_utc():
    assert format_athens(utc(2026, 11, 12, 21, 30)) == "2026-11-12 23:30 EET"
    assert format_athens(utc(2026, 7, 15, 21, 30)) == "2026-07-16 00:30 EEST"
