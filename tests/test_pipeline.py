"""End-to-end pipeline tests with in-memory fakes: no network, no real clock."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockscan.config import AppConfig, NewsConfig, ScanConfig, ScoringConfig
from stockscan.http import HttpError
from stockscan.pipeline import (
    DataQualityError,
    attach_news,
    build_scan,
    check_session,
    run_scan,
)
from stockscan.providers.base import NewsSentiment, NewsUnavailableError
from stockscan.storage import ScanStore
from tests.conftest import make_raw_bars, make_universe_bars

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
DAY = date(2026, 11, 11)  # a Wednesday
AFTER_CLOSE = datetime(2026, 11, 11, 21, 30, tzinfo=UTC)  # 16:30 New York
CONFIG = AppConfig(scan=ScanConfig(top_n=3))


class FakePrices:
    """Returns everything it has, like a provider that does not respect ``end``."""

    def __init__(self, bars: pd.DataFrame) -> None:
        self.bars = bars
        self.requests: list[tuple[list[str], date, date]] = []

    def daily_bars(self, tickers, start, end):
        self.requests.append((list(tickers), start, end))
        return self.bars

    def intraday_bars(self, tickers, start, end, interval):
        raise NotImplementedError


class FakeNews:
    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[tuple[str, datetime, datetime]] = []
        self.fail_after = fail_after

    def sentiment(self, ticker, start, end):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise NewsUnavailableError("rate limit reached")
        self.calls.append((ticker, start, end))
        return NewsSentiment(
            ticker, 0.3, "Somewhat-Bullish", 4, f"{ticker} headline", "https://n.test/x", end
        )


class FakeNotifier:
    def __init__(self, error: Exception | None = None) -> None:
        self.messages: list[str] = []
        self.error = error

    def send(self, text: str) -> None:
        if self.error:
            raise self.error
        self.messages.append(text)


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_universe_bars(TICKERS, end="2026-11-11")


@pytest.fixture
def store():
    with ScanStore(None) as opened:
        yield opened


def run(store, tmp_path, bars, **overrides):
    arguments = {
        "config": CONFIG,
        "tickers": TICKERS,
        "prices": FakePrices(bars),
        "store": store,
        "output_dir": tmp_path,
        "now": AFTER_CLOSE,
        "sleep": lambda seconds: None,
    }
    arguments.update(overrides)
    return run_scan(**arguments)


class TestBuildScan:
    def test_one_scored_row_per_ticker_for_the_latest_session(self, bars):
        scan_date, scan, missing = build_scan(bars, TICKERS, DAY, ScoringConfig())
        assert scan_date == DAY
        assert missing == []
        assert sorted(scan["ticker"]) == TICKERS
        assert (scan["scan_date"] == pd.Timestamp(DAY)).all()
        assert scan["rank"].tolist() == [1, 2, 3, 4, 5, 6]
        assert scan["score"].is_monotonic_decreasing
        assert scan["score"].between(0, 100).all()

    def test_ticker_without_a_bar_today_is_reported_not_ranked(self, bars):
        halted = bars[~((bars["ticker"] == "CCC") & (bars["date"] == pd.Timestamp(DAY)))]
        _, scan, missing = build_scan(halted, TICKERS, DAY, ScoringConfig())
        assert missing == ["CCC"]
        assert "CCC" not in set(scan["ticker"])
        assert len(scan) == 5

    def test_ticker_missing_entirely_is_reported(self, bars):
        _, scan, missing = build_scan(bars, [*TICKERS, "GONE"], DAY, ScoringConfig())
        assert missing == ["GONE"]
        assert len(scan) == 6

    def test_tickers_outside_the_universe_are_ignored(self, bars):
        extra = pd.concat([bars, make_raw_bars("ZZZ", end="2026-11-11")])
        _, scan, _ = build_scan(extra, TICKERS, DAY, ScoringConfig())
        assert "ZZZ" not in set(scan["ticker"])

    def test_refuses_to_rank_when_most_of_the_universe_is_missing(self, bars):
        only_two = bars[bars["ticker"].isin(["AAA", "BBB"])]
        with pytest.raises(DataQualityError, match="only 2 of 6"):
            build_scan(only_two, TICKERS, DAY, ScoringConfig(), min_coverage=0.5)

    def test_no_data_at_all(self, bars):
        with pytest.raises(DataQualityError, match="no usable price data"):
            build_scan(bars.iloc[0:0], TICKERS, DAY, ScoringConfig())

    def test_new_listing_with_short_history_is_still_included(self, bars):
        short = pd.concat([bars, make_raw_bars("NEW", sessions=10, end="2026-11-11", seed=99)])
        _, scan, missing = build_scan(short, [*TICKERS, "NEW"], DAY, ScoringConfig())
        assert missing == []
        new = scan[scan["ticker"] == "NEW"].iloc[0]
        assert pd.isna(new["rel_volume"])  # unknown, not invented
        assert new["score"] >= 0


class TestNoLookAheadEndToEnd:
    """The same guarantee as tests/test_no_lookahead.py, checked on the whole scan."""

    def test_scan_for_day_t_is_identical_with_or_without_future_bars(self):
        through_friday = make_universe_bars(TICKERS, sessions=303, end="2026-11-16")
        as_of_wednesday = through_friday[through_friday["date"] <= pd.Timestamp(DAY)]

        _, with_future, _ = build_scan(through_friday, TICKERS, DAY, ScoringConfig())
        _, without_future, _ = build_scan(as_of_wednesday, TICKERS, DAY, ScoringConfig())
        pd.testing.assert_frame_equal(with_future, without_future, check_exact=True)

    def test_run_during_the_session_ignores_the_half_formed_bar(self, store, tmp_path):
        # 14:54 New York on Thursday the 12th: the provider already returns a bar for the 12th.
        during_session = datetime(2026, 11, 12, 19, 54, tzinfo=UTC)
        with_partial_bar = make_universe_bars(TICKERS, sessions=301, end="2026-11-12")
        outcome = run(store, tmp_path, with_partial_bar, now=during_session)
        assert outcome.scan_date == DAY  # Wednesday, the last finished session

        with ScanStore(None) as clean_store:
            finished_only = with_partial_bar[with_partial_bar["date"] <= pd.Timestamp(DAY)]
            expected = run(clean_store, tmp_path / "clean", finished_only, now=during_session)
        pd.testing.assert_frame_equal(outcome.scan, expected.scan)

    def test_as_of_rerun_uses_only_bars_up_to_that_date(self, store, tmp_path, bars):
        monday = date(2026, 11, 9)
        outcome = run(store, tmp_path, bars, as_of=monday)
        assert outcome.scan_date == monday
        _, expected, _ = build_scan(
            bars[bars["date"] <= pd.Timestamp(monday)], TICKERS, monday, ScoringConfig()
        )
        assert outcome.scan["ticker"].tolist() == expected["ticker"].tolist()
        assert outcome.scan["score"].tolist() == pytest.approx(expected["score"].tolist())

    def test_as_of_cannot_be_an_unfinished_session(self, store, tmp_path, bars):
        with pytest.raises(ValueError, match="not a finished session"):
            run(store, tmp_path, bars, as_of=date(2026, 11, 12))


class TestRunScan:
    def test_stores_reports_and_alerts(self, store, tmp_path, bars):
        notifier = FakeNotifier()
        outcome = run(store, tmp_path, bars, notifier=notifier)

        assert outcome.status == "completed"
        assert outcome.scan_date == DAY
        assert store.has_scan(DAY)
        assert len(store.load_scan(DAY)) == 6  # every ticker is stored, not just the top 3
        assert (tmp_path / "reports" / "2026-11-11.md").exists()
        assert (tmp_path / "reports" / "2026-11-11.csv").exists()
        assert (tmp_path / "scans" / "2026-11-11.parquet").exists()
        assert outcome.alert_sent is True
        assert len(notifier.messages) == 1
        assert "Top 3 of 6" in notifier.messages[0]

    def test_requests_enough_history_and_nothing_after_the_cutoff(self, store, tmp_path, bars):
        prices = FakePrices(bars)
        run(store, tmp_path, bars, prices=prices)
        tickers, start, end = prices.requests[0]
        assert tickers == TICKERS
        assert end == DAY
        assert (end - start).days == CONFIG.scan.history_calendar_days

    def test_second_run_for_the_same_session_is_skipped_without_alerting(
        self, store, tmp_path, bars
    ):
        run(store, tmp_path, bars)
        notifier = FakeNotifier()
        outcome = run(store, tmp_path, bars, notifier=notifier)
        assert outcome.status == "skipped"
        assert "already stored" in outcome.message
        assert notifier.messages == []

    def test_weekend_run_does_not_rescan_friday(self, store, tmp_path):
        friday_bars = make_universe_bars(TICKERS, end="2026-11-13")
        friday_evening = datetime(2026, 11, 13, 21, 30, tzinfo=UTC)
        saturday = datetime(2026, 11, 14, 21, 30, tzinfo=UTC)
        assert run(store, tmp_path, friday_bars, now=friday_evening).status == "completed"
        outcome = run(store, tmp_path, friday_bars, now=saturday)
        assert outcome.status == "skipped"
        assert outcome.scan_date == date(2026, 11, 13)

    def test_force_redoes_a_stored_session(self, store, tmp_path, bars):
        run(store, tmp_path, bars)
        notifier = FakeNotifier()
        outcome = run(store, tmp_path, bars, notifier=notifier, force=True)
        assert outcome.status == "completed"
        assert len(store.load_scan(DAY)) == 6
        assert len(notifier.messages) == 1

    def test_stale_price_source_is_rejected(self, store, tmp_path, bars):
        two_weeks_later = datetime(2026, 11, 25, 21, 30, tzinfo=UTC)
        with pytest.raises(DataQualityError, match="No complete daily bars for session 2026-11-25"):
            run(store, tmp_path, bars, now=two_weeks_later)

    def test_missing_tickers_are_listed_in_the_report(self, store, tmp_path, bars):
        outcome = run(store, tmp_path, bars, tickers=[*TICKERS, "GONE"])
        assert outcome.missing == ["GONE"]
        report = (tmp_path / "reports" / "2026-11-11.md").read_text(encoding="utf-8")
        assert "No usable bar for this session (1): GONE" in report

    def test_long_only_filters_the_alert_but_stores_everything(self, store, tmp_path, bars):
        notifier = FakeNotifier()
        outcome = run(store, tmp_path, bars, notifier=notifier, long_only=True)
        stored = store.load_scan(DAY)
        assert len(stored) == 6
        down = stored.loc[stored["direction"] != "up", "ticker"]
        assert len(down) > 0, "fixture should contain at least one stock that closed down"
        for ticker in down:
            assert f". {ticker}</b>" not in notifier.messages[0]
        assert "long-only" in notifier.messages[0]
        assert outcome.status == "completed"

    def test_long_only_default_comes_from_config(self, store, tmp_path, bars):
        notifier = FakeNotifier()
        config = AppConfig(scan=ScanConfig(top_n=3, long_only=True))
        run(store, tmp_path, bars, notifier=notifier, config=config)
        assert "long-only" in notifier.messages[0]

    def test_telegram_failure_is_reported_but_data_is_kept(self, store, tmp_path, bars):
        notifier = FakeNotifier(error=HttpError("POST failed: HTTP 400: chat not found"))
        outcome = run(store, tmp_path, bars, notifier=notifier)
        assert outcome.status == "completed"
        assert outcome.alert_sent is False
        assert "chat not found" in outcome.alert_error
        assert store.has_scan(DAY)
        assert (tmp_path / "reports" / "2026-11-11.md").exists()

    def test_runs_without_news_and_without_telegram(self, store, tmp_path, bars):
        outcome = run(store, tmp_path, bars)
        assert outcome.status == "completed"
        assert outcome.alert_sent is False
        assert outcome.alert_error is None
        assert outcome.scan["news_headline"].isna().all()


class SequencePrices(FakePrices):
    """Returns a different answer on each call (the last one repeats)."""

    def __init__(self, *answers: pd.DataFrame) -> None:
        super().__init__(answers[0])
        self.answers = list(answers)

    def daily_bars(self, tickers, start, end):
        self.requests.append((list(tickers), start, end))
        return self.answers[min(len(self.requests), len(self.answers)) - 1]


def without_session(bars: pd.DataFrame, day: str) -> pd.DataFrame:
    return bars[bars["date"] != pd.Timestamp(day)].reset_index(drop=True)


def with_volume_on(bars: pd.DataFrame, day: str, volume: float) -> pd.DataFrame:
    out = bars.copy()
    out.loc[out["date"] == pd.Timestamp(day), "volume"] = volume
    return out


class TestMissingSessionRegression:
    """Regressions for the scheduled runs of 12 and 13 Nov 2026.

    GitHub started them at 01:59 and 01:32 UTC (Thursday and Friday evening in New York),
    Yahoo's answer had no usable bar for that evening's session, and the old code quietly
    took the latest bar it had: the first run skipped (11 Nov already stored), the second
    scanned 12 Nov instead of 13 Nov.
    """

    THURSDAY_RUN = datetime(2026, 11, 13, 1, 59, tzinfo=UTC)
    FRIDAY_RUN = datetime(2026, 11, 14, 1, 32, tzinfo=UTC)

    def test_thursday_run_fails_instead_of_reporting_wednesday_as_already_stored(
        self, store, tmp_path, bars
    ):
        run(store, tmp_path, bars)  # Wednesday 11 Nov is stored, as it was on the data branch
        sleeps: list[float] = []
        prices = FakePrices(bars)  # Yahoo: nothing for Thursday 12 Nov yet
        notifier = FakeNotifier()
        with pytest.raises(DataQualityError) as error:
            run(
                store,
                tmp_path,
                bars,
                prices=prices,
                notifier=notifier,
                now=self.THURSDAY_RUN,
                sleep=sleeps.append,
            )
        message = str(error.value)
        assert "No complete daily bars for session 2026-11-12 after 3 attempt(s)" in message
        assert "0/6 tickers have a usable bar (no row: 6" in message
        assert "newest bar returned: 2026-11-11" in message
        assert sleeps == [300, 600]  # 5 then 10 minutes between the three attempts
        assert [request[2] for request in prices.requests] == [date(2026, 11, 12)] * 3
        assert notifier.messages == []
        assert store.scan_dates() == [DAY]

    def test_friday_run_fails_instead_of_scanning_thursday(self, store, tmp_path):
        through_thursday = make_universe_bars(TICKERS, end="2026-11-12")
        with pytest.raises(DataQualityError, match="session 2026-11-13"):
            run(store, tmp_path, through_thursday, now=self.FRIDAY_RUN)
        assert store.scan_dates() == []  # Thursday was not scanned in Friday's place
        assert not (tmp_path / "reports" / "2026-11-12.md").exists()

    @pytest.mark.parametrize("volume", [0.0, float("nan")])
    def test_newest_bar_with_no_volume_fails_too(self, store, tmp_path, volume):
        thursday = with_volume_on(
            make_universe_bars(TICKERS, end="2026-11-12"), "2026-11-12", volume
        )
        with pytest.raises(DataQualityError, match="zero or missing volume: 6,"):
            run(store, tmp_path, thursday, now=self.THURSDAY_RUN)
        assert store.scan_dates() == []

    def test_bar_that_appears_on_retry_is_scanned(self, store, tmp_path):
        complete = make_universe_bars(TICKERS, end="2026-11-12")
        prices = SequencePrices(without_session(complete, "2026-11-12"), complete)
        sleeps: list[float] = []
        outcome = run(
            store, tmp_path, complete, prices=prices, now=self.THURSDAY_RUN, sleep=sleeps.append
        )
        assert outcome.status == "completed"
        assert outcome.scan_date == date(2026, 11, 12)
        assert sleeps == [300]
        assert outcome.data_check.startswith("Session 2026-11-12: 6/6 tickers have a usable bar")

    def test_backstop_run_is_a_no_op_without_fetching(self, store, tmp_path):
        thursday = make_universe_bars(TICKERS, end="2026-11-12")
        run(store, tmp_path, thursday, now=self.THURSDAY_RUN)
        prices = FakePrices(thursday)
        backstop = datetime(2026, 11, 13, 3, 30, tzinfo=UTC)
        outcome = run(store, tmp_path, thursday, prices=prices, now=backstop)
        assert outcome.status == "skipped"
        assert outcome.scan_date == date(2026, 11, 12)
        assert prices.requests == []  # no Yahoo call when the session is already stored

    def test_backstop_run_catches_the_session_the_first_run_missed(self, store, tmp_path, bars):
        thursday = make_universe_bars(TICKERS, end="2026-11-12")
        with pytest.raises(DataQualityError):
            run(store, tmp_path, bars, now=self.THURSDAY_RUN)
        backstop = datetime(2026, 11, 13, 3, 30, tzinfo=UTC)
        outcome = run(store, tmp_path, thursday, now=backstop)
        assert outcome.status == "completed"
        assert outcome.scan_date == date(2026, 11, 12)

    def test_historical_rerun_does_not_wait_for_missing_data(self, store, tmp_path, bars):
        sleeps: list[float] = []
        gap = without_session(bars, "2026-11-10")
        with pytest.raises(DataQualityError, match="after 1 attempt"):
            run(store, tmp_path, gap, as_of=date(2026, 11, 10), sleep=sleeps.append)
        assert sleeps == []


class TestMarketCalendarInTheScan:
    def test_holiday_evening_scans_the_session_before(self, store, tmp_path):
        wednesday = make_universe_bars(TICKERS, end="2026-11-25")
        thanksgiving_evening = datetime(2026, 11, 26, 22, 0, tzinfo=UTC)
        outcome = run(store, tmp_path, wednesday, now=thanksgiving_evening)
        assert outcome.scan_date == date(2026, 11, 25)

    def test_half_day_is_scanned_after_13_00_and_news_stops_at_that_close(self, store, tmp_path):
        black_friday = make_universe_bars(TICKERS, end="2026-11-27")
        news = FakeNews()
        outcome = run(
            store,
            tmp_path,
            black_friday,
            news=news,
            now=datetime(2026, 11, 27, 18, 30, tzinfo=UTC),  # 13:30 New York
        )
        assert outcome.scan_date == date(2026, 11, 27)
        assert news.calls[0][2] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)

    @pytest.mark.parametrize("day", [date(2026, 11, 26), date(2026, 11, 8)])
    def test_as_of_must_be_a_trading_session(self, store, tmp_path, bars, day):
        with pytest.raises(ValueError, match="not a trading session"):
            run(store, tmp_path, bars, as_of=day)


class TestCheckSession:
    def test_counts_each_kind_of_problem(self, bars):
        day = "2026-11-11"
        frame = bars.copy()
        on_day = frame["date"] == pd.Timestamp(day)
        frame.loc[on_day & (frame["ticker"] == "AAA"), "volume"] = 0.0
        frame.loc[on_day & (frame["ticker"] == "BBB"), "close"] = float("nan")
        frame = frame[~(on_day & (frame["ticker"] == "CCC"))]
        check = check_session(frame, [*TICKERS, "GONE"], DAY)
        assert (check.tickers, check.usable) == (7, 3)
        assert (check.no_row, check.no_volume, check.bad_prices) == (2, 1, 1)
        assert check.newest_bar == DAY
        assert check.coverage == pytest.approx(3 / 7)

    def test_empty_answer(self, bars):
        check = check_session(bars.iloc[0:0], TICKERS, DAY)
        assert (check.usable, check.no_row, check.newest_bar) == (0, 6, None)
        assert "newest bar returned: none" in check.describe()


class TestNews:
    def test_news_is_fetched_for_the_top_candidates_only(self, store, tmp_path, bars):
        news = FakeNews()
        outcome = run(store, tmp_path, bars, news=news)
        top3 = outcome.scan.sort_values("rank")["ticker"].head(3).tolist()
        assert [call[0] for call in news.calls] == top3  # 3 calls, not 6
        with_news = outcome.scan[outcome.scan["news_headline"].notna()]
        assert sorted(with_news["ticker"]) == sorted(top3)
        assert with_news["news_score"].tolist() == [0.3, 0.3, 0.3]
        assert with_news["news_articles"].tolist() == [4, 4, 4]
        report = (tmp_path / "reports" / "2026-11-11.md").read_text(encoding="utf-8")
        assert f"[{top3[0]} headline](https://n.test/x)" in report

    def test_news_window_ends_at_the_session_close(self, store, tmp_path, bars):
        news = FakeNews()
        run(store, tmp_path, bars, news=news)
        _, start, end = news.calls[0]
        assert end == datetime(2026, 11, 11, 21, 0, tzinfo=UTC)  # 16:00 New York
        assert start == datetime(2026, 11, 10, 21, 0, tzinfo=UTC)  # 24h lookback

    def test_news_outage_does_not_stop_the_scan(self, store, tmp_path, bars):
        news = FakeNews(fail_after=1)
        outcome = run(store, tmp_path, bars, news=news)
        assert outcome.status == "completed"
        assert len(news.calls) == 1  # stopped asking after the failure
        assert outcome.scan["news_headline"].notna().sum() == 1
        assert outcome.notes == ["News sentiment incomplete: rate limit reached"]
        report = (tmp_path / "reports" / "2026-11-11.md").read_text(encoding="utf-8")
        assert "News sentiment incomplete" in report

    def test_news_can_be_disabled_in_config(self, store, tmp_path, bars):
        news = FakeNews()
        config = AppConfig(scan=ScanConfig(top_n=3), news=NewsConfig(enabled=False))
        run(store, tmp_path, bars, news=news, config=config)
        assert news.calls == []

    def test_attach_news_without_a_provider_adds_empty_columns(self, bars):
        _, scan, _ = build_scan(bars, TICKERS, DAY, ScoringConfig())
        with_columns, note = attach_news(scan, ["AAA"], None, AFTER_CLOSE, 24)
        assert note is None
        assert with_columns["news_score"].isna().all()
        assert "news_url" in with_columns.columns
