from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockscan.config import IntradayConfig
from stockscan.intraday import save_intraday, select_intraday_tickers, split_sessions
from stockscan.providers.base import INTRADAY_COLUMNS
from stockscan.storage import ScanStore
from tests.conftest import make_intraday_bars, make_scan

CONFIG = IntradayConfig(top_n=2, always_include=("NVDA", "QQQ"))
AFTER_CLOSE = datetime(2026, 11, 11, 21, 30, tzinfo=UTC)  # Wednesday 16:30 New York


class FakeIntradayPrices:
    def __init__(self, sessions: list[str]) -> None:
        self.sessions = sessions
        self.requests: list[tuple[list[str], date, date, str]] = []

    def daily_bars(self, tickers, start, end):
        raise NotImplementedError

    def intraday_bars(self, tickers, start, end, interval):
        self.requests.append((list(tickers), start, end, interval))
        return make_intraday_bars(list(tickers), self.sessions)


@pytest.fixture
def store():
    with ScanStore(None) as opened:
        yield opened


def save(store, tmp_path, prices, **overrides):
    arguments = {
        "prices": prices,
        "store": store,
        "config": CONFIG,
        "output_dir": tmp_path,
        "now": AFTER_CLOSE,
    }
    arguments.update(overrides)
    return save_intraday(**arguments)


class TestTickerSelection:
    def test_without_scan_history_only_the_always_list_is_used(self, store):
        tickers = select_intraday_tickers(store, date(2026, 11, 11), date(2026, 11, 11), CONFIG)
        assert tickers == ["NVDA", "QQQ"]

    def test_includes_top_n_of_the_day_and_of_the_previous_session(self, store):
        monday = make_scan(["AAA", "BBB", "CCC", "DDD"], end="2026-11-09")
        tuesday = make_scan(["EEE", "FFF", "GGG", "HHH"], end="2026-11-10")
        wednesday = make_scan(["III", "JJJ", "KKK", "LLL"], end="2026-11-11")
        for scan in (monday, tuesday, wednesday):
            store.save_scan(scan)

        tickers = select_intraday_tickers(store, date(2026, 11, 11), date(2026, 11, 11), CONFIG)
        expected_top = [
            *store.top_tickers(date(2026, 11, 9), 2),  # within the 4-day look-back
            *store.top_tickers(date(2026, 11, 10), 2),  # yesterday's signals trade today
            *store.top_tickers(date(2026, 11, 11), 2),  # today's hot stocks
        ]
        assert tickers == ["NVDA", "QQQ", *expected_top]

    def test_no_duplicates_when_a_top_ticker_is_also_always_included(self, store):
        store.save_scan(make_scan(["NVDA", "AAA", "BBB"], end="2026-11-11"))
        config = IntradayConfig(top_n=3, always_include=("NVDA", "QQQ"))
        tickers = select_intraday_tickers(store, date(2026, 11, 11), date(2026, 11, 11), config)
        assert sorted(tickers) == ["AAA", "BBB", "NVDA", "QQQ"]


def test_split_sessions_groups_by_new_york_date_and_drops_unfinished_days():
    bars = make_intraday_bars(["NVDA"], ["2026-11-10", "2026-11-11", "2026-11-12"])
    sessions = split_sessions(bars, cutoff=date(2026, 11, 11))
    assert sorted(sessions) == [date(2026, 11, 10), date(2026, 11, 11)]
    assert all(len(frame) == 6 for frame in sessions.values())
    assert split_sessions(bars.iloc[0:0], cutoff=date(2026, 11, 11)) == {}


class TestSaveIntraday:
    def test_writes_one_parquet_file_per_finished_session(self, store, tmp_path):
        prices = FakeIntradayPrices(["2026-11-10", "2026-11-11"])
        outcome = save(store, tmp_path, prices)

        directory = tmp_path / "intraday" / "5m"
        assert [path.name for path in outcome.written] == [
            "2026-11-10.parquet",
            "2026-11-11.parquet",
        ]
        frame = pd.read_parquet(directory / "2026-11-11.parquet")
        assert list(frame.columns) == INTRADAY_COLUMNS
        assert sorted(frame["ticker"].unique()) == ["NVDA", "QQQ"]
        assert str(frame["ts"].dt.tz) == "UTC"
        assert frame["ts"].min() == pd.Timestamp("2026-11-11 14:30", tz="UTC")  # 09:30 EST

    def test_requests_the_look_back_window_ending_at_the_last_finished_session(
        self, store, tmp_path
    ):
        prices = FakeIntradayPrices(["2026-11-11"])
        save(store, tmp_path, prices, days=5)
        tickers, start, end, interval = prices.requests[0]
        assert (start, end, interval) == (date(2026, 11, 7), date(2026, 11, 11), "5m")
        assert tickers == ["NVDA", "QQQ"]

    def test_existing_files_are_kept_unless_forced(self, store, tmp_path):
        prices = FakeIntradayPrices(["2026-11-10", "2026-11-11"])
        save(store, tmp_path, prices)
        marker = tmp_path / "intraday" / "5m" / "2026-11-10.parquet"
        marker.write_bytes(b"do not touch")

        again = save(store, tmp_path, prices)
        assert again.written == []
        assert again.already_saved == [date(2026, 11, 10), date(2026, 11, 11)]
        assert marker.read_bytes() == b"do not touch"

        forced = save(store, tmp_path, prices, force=True)
        assert len(forced.written) == 2
        assert len(pd.read_parquet(marker)) == 12

    def test_a_missed_day_is_caught_up_by_the_next_run(self, store, tmp_path):
        save(store, tmp_path, FakeIntradayPrices(["2026-11-09"]), now=AFTER_CLOSE)
        # Tuesday's run failed; Wednesday's run sees both days and fills the gap.
        outcome = save(store, tmp_path, FakeIntradayPrices(["2026-11-10", "2026-11-11"]))
        assert [path.name for path in outcome.written] == [
            "2026-11-10.parquet",
            "2026-11-11.parquet",
        ]

    def test_session_in_progress_is_not_saved(self, store, tmp_path):
        during_session = datetime(2026, 11, 12, 19, 54, tzinfo=UTC)
        prices = FakeIntradayPrices(["2026-11-11", "2026-11-12"])
        outcome = save(store, tmp_path, prices, now=during_session)
        assert [path.name for path in outcome.written] == ["2026-11-11.parquet"]

    def test_no_bars_means_nothing_written(self, store, tmp_path):
        outcome = save(store, tmp_path, FakeIntradayPrices([]))
        assert outcome.written == []
        assert not (tmp_path / "intraday").exists()

    @pytest.mark.parametrize("days", [0, 56])
    def test_days_outside_yahoos_window_are_rejected(self, store, tmp_path, days):
        with pytest.raises(ValueError, match="days must be between"):
            save(store, tmp_path, FakeIntradayPrices([]), days=days)
