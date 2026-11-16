"""NYSE calendar and the "which session should this run produce" rule."""

from datetime import UTC, date, datetime

import pytest

from stockscan.market_calendar import last_completed_session, nyse_calendar


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.fixture(scope="module")
def nyse():
    return nyse_calendar()


class TestExchangeCalendar:
    def test_weekdays_weekends_and_holidays(self, nyse):
        assert nyse.is_session(date(2026, 11, 13))  # Friday
        assert not nyse.is_session(date(2026, 11, 14))  # Saturday
        assert not nyse.is_session(date(2026, 11, 26))  # Thanksgiving
        assert not nyse.is_session(date(2026, 12, 25))  # Christmas

    def test_regular_and_half_day_closes(self, nyse):
        assert nyse.session_close_utc(date(2026, 11, 13)) == utc(2026, 11, 13, 21, 0)  # 16:00 EST
        assert nyse.session_close_utc(date(2026, 7, 15)) == utc(2026, 7, 15, 20, 0)  # 16:00 EDT
        assert nyse.session_close_utc(date(2026, 11, 27)) == utc(2026, 11, 27, 18, 0)  # 13:00
        assert nyse.session_close_utc(date(2026, 12, 24)) == utc(2026, 12, 24, 18, 0)

    def test_close_of_a_non_session_is_an_error(self, nyse):
        with pytest.raises(ValueError, match="not a trading session"):
            nyse.session_close_utc(date(2026, 11, 26))

    def test_previous_session_skips_weekends_and_holidays(self, nyse):
        assert nyse.previous_session(date(2026, 11, 16)) == date(2026, 11, 13)  # Mon -> Fri
        assert nyse.previous_session(date(2026, 11, 27)) == date(2026, 11, 25)  # over Thanksgiving


class TestLastCompletedSession:
    def test_during_the_session_it_is_the_previous_session(self, nyse):
        # 14:54 New York on Thursday 12 Nov.
        assert last_completed_session(nyse, utc(2026, 11, 12, 19, 54)) == date(2026, 11, 11)

    def test_inside_the_buffer_after_the_close(self, nyse):
        assert last_completed_session(nyse, utc(2026, 11, 12, 21, 10)) == date(2026, 11, 11)
        assert last_completed_session(nyse, utc(2026, 11, 12, 21, 20)) == date(2026, 11, 12)

    def test_late_cron_after_utc_midnight_still_expects_that_evenings_session(self, nyse):
        # The two runs that went wrong: GitHub started them at 01:59 and 01:32 UTC.
        assert last_completed_session(nyse, utc(2026, 11, 13, 1, 59)) == date(2026, 11, 12)
        assert last_completed_session(nyse, utc(2026, 11, 14, 1, 32)) == date(2026, 11, 13)

    def test_backstop_run_at_0330_utc(self, nyse):
        # Tuesday 03:30 UTC is Monday evening in New York.
        assert last_completed_session(nyse, utc(2026, 11, 17, 3, 30)) == date(2026, 11, 16)
        # Saturday 03:30 UTC still expects Friday's session.
        assert last_completed_session(nyse, utc(2026, 11, 14, 3, 30)) == date(2026, 11, 13)

    def test_weekend_and_monday_morning_expect_friday(self, nyse):
        assert last_completed_session(nyse, utc(2026, 11, 15, 12, 0)) == date(2026, 11, 13)
        assert last_completed_session(nyse, utc(2026, 11, 16, 14, 0)) == date(2026, 11, 13)

    def test_holiday_expects_the_session_before_it(self, nyse):
        # Thanksgiving evening.
        assert last_completed_session(nyse, utc(2026, 11, 26, 22, 0)) == date(2026, 11, 25)

    def test_half_day_is_finished_soon_after_13_00(self, nyse):
        # Black Friday closes at 13:00 New York = 18:00 UTC.
        assert last_completed_session(nyse, utc(2026, 11, 27, 18, 10)) == date(2026, 11, 25)
        assert last_completed_session(nyse, utc(2026, 11, 27, 18, 20)) == date(2026, 11, 27)

    def test_buffer_is_configurable(self, nyse):
        now = utc(2026, 11, 12, 21, 10)
        assert last_completed_session(nyse, now, buffer_minutes=5) == date(2026, 11, 12)

    def test_naive_datetime_is_rejected(self, nyse):
        with pytest.raises(ValueError, match="naive"):
            last_completed_session(nyse, datetime(2026, 11, 12, 21, 30))
