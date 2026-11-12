"""The yfinance wrapper is tested against frames shaped like real yfinance output
(captured from yfinance 1.8: wide frame, (ticker, field) column MultiIndex)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stockscan.bars import RAW_DAILY_COLUMNS
from stockscan.providers.base import INTRADAY_COLUMNS, ProviderError
from stockscan.providers.yfinance_provider import CachedPriceProvider, YFinanceProvider

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def wide_frame(index: pd.DatetimeIndex, tickers: list[str], empty: tuple[str, ...] = ()):
    columns = pd.MultiIndex.from_product([tickers, FIELDS])
    data = np.arange(len(index) * len(columns), dtype=float).reshape(len(index), -1) + 1
    frame = pd.DataFrame(data, index=index, columns=columns)
    for ticker in empty:
        frame[ticker] = np.nan
    return frame


class FakeDownloader:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def provider(downloader: FakeDownloader, **kwargs) -> YFinanceProvider:
    return YFinanceProvider(
        downloader=downloader, sleep=lambda _: None, min_interval_seconds=0, **kwargs
    )


DAILY_INDEX = pd.DatetimeIndex(["2026-11-10", "2026-11-11"], name="Date")


def test_daily_bars_are_reshaped_to_the_canonical_long_schema():
    downloader = FakeDownloader([wide_frame(DAILY_INDEX, ["AAPL", "NVDA"])])
    bars = provider(downloader).daily_bars(["AAPL", "NVDA"], date(2026, 11, 10), date(2026, 11, 11))

    assert list(bars.columns) == RAW_DAILY_COLUMNS
    assert len(bars) == 4
    aapl_last = bars[(bars["ticker"] == "AAPL") & (bars["date"] == "2026-11-11")].iloc[0]
    # Second row of the wide frame, first six values: 13..18.
    assert [aapl_last[c] for c in ["open", "high", "low", "close", "adj_close", "volume"]] == [
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
    ]


def test_request_asks_for_unadjusted_prices_and_an_exclusive_end_date():
    downloader = FakeDownloader([wide_frame(DAILY_INDEX, ["AAPL"])])
    provider(downloader).daily_bars(["AAPL"], date(2026, 11, 10), date(2026, 11, 11))
    call = downloader.calls[0]
    assert call["start"] == "2026-11-10"
    assert call["end"] == "2026-11-12"  # yfinance's end is exclusive
    assert call["auto_adjust"] is False
    assert call["interval"] == "1d"


def test_tickers_without_data_are_simply_absent():
    downloader = FakeDownloader([wide_frame(DAILY_INDEX, ["AAPL", "GONE"], empty=("GONE",))])
    bars = provider(downloader).daily_bars(["AAPL", "GONE"], date(2026, 11, 10), date(2026, 11, 11))
    assert set(bars["ticker"]) == {"AAPL"}


def test_timezone_aware_daily_index_becomes_new_york_session_dates():
    index = pd.DatetimeIndex(["2026-11-11 05:00"], tz="UTC")  # midnight New York
    downloader = FakeDownloader([wide_frame(index, ["AAPL"])])
    bars = provider(downloader).daily_bars(["AAPL"], date(2026, 11, 11), date(2026, 11, 11))
    assert bars["date"].tolist() == [pd.Timestamp("2026-11-11")]


def test_flat_columns_for_a_single_ticker_are_supported():
    frame = wide_frame(DAILY_INDEX, ["QQQ"])["QQQ"]
    bars = provider(FakeDownloader([frame])).daily_bars(
        ["QQQ"], date(2026, 11, 10), date(2026, 11, 11)
    )
    assert set(bars["ticker"]) == {"QQQ"}
    assert len(bars) == 2


def test_large_universes_are_downloaded_in_chunks():
    tickers = [f"T{i}" for i in range(5)]
    downloader = FakeDownloader(
        [
            wide_frame(DAILY_INDEX, tickers[0:2]),
            wide_frame(DAILY_INDEX, tickers[2:4]),
            wide_frame(DAILY_INDEX, tickers[4:5]),
        ]
    )
    bars = provider(downloader, chunk_size=2).daily_bars(
        tickers, date(2026, 11, 10), date(2026, 11, 11)
    )
    assert [call["tickers"] for call in downloader.calls] == [["T0", "T1"], ["T2", "T3"], ["T4"]]
    assert set(bars["ticker"]) == set(tickers)


def test_errors_and_empty_answers_are_retried():
    downloader = FakeDownloader(
        [RuntimeError("Too Many Requests"), pd.DataFrame(), wide_frame(DAILY_INDEX, ["AAPL"])]
    )
    bars = provider(downloader).daily_bars(["AAPL"], date(2026, 11, 10), date(2026, 11, 11))
    assert len(downloader.calls) == 3
    assert len(bars) == 2


def test_total_failure_raises_provider_error():
    downloader = FakeDownloader([RuntimeError("down")] * 3)
    with pytest.raises(ProviderError, match="no data"):
        provider(downloader, retries=2).daily_bars(["AAPL"], date(2026, 11, 10), date(2026, 11, 11))


def test_one_failed_chunk_does_not_lose_the_others():
    downloader = FakeDownloader([wide_frame(DAILY_INDEX, ["A"]), RuntimeError("down")])
    bars = provider(downloader, chunk_size=1, retries=0).daily_bars(
        ["A", "B"], date(2026, 11, 10), date(2026, 11, 11)
    )
    assert set(bars["ticker"]) == {"A"}


def test_intraday_bars_are_returned_in_utc():
    index = pd.DatetimeIndex(
        ["2026-11-11 09:30", "2026-11-11 09:35"], tz="America/New_York", name="Datetime"
    )
    downloader = FakeDownloader([wide_frame(index, ["NVDA", "QQQ"])])
    bars = provider(downloader).intraday_bars(
        ["NVDA", "QQQ"], date(2026, 11, 11), date(2026, 11, 11), "5m"
    )
    assert list(bars.columns) == INTRADAY_COLUMNS
    assert len(bars) == 4
    assert str(bars["ts"].dt.tz) == "UTC"
    assert bars["ts"].min() == pd.Timestamp("2026-11-11 14:30", tz="UTC")  # 09:30 EST
    assert downloader.calls[0]["interval"] == "5m"
    assert downloader.calls[0]["prepost"] is False


class CountingProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def daily_bars(self, tickers, start, end):
        self.calls += 1
        return self.frame

    def intraday_bars(self, tickers, start, end, interval):
        self.calls += 1
        return self.frame


class TestCachedPriceProvider:
    def test_second_identical_request_is_served_from_disk(self, tmp_path, raw_bars):
        inner = CountingProvider(raw_bars)
        cached = CachedPriceProvider(inner, tmp_path)
        args = (["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        first = cached.daily_bars(*args)
        second = cached.daily_bars(*args)
        assert inner.calls == 1
        pd.testing.assert_frame_equal(first, second, check_dtype=False)

    def test_different_requests_do_not_share_a_cache_entry(self, tmp_path, raw_bars):
        inner = CountingProvider(raw_bars)
        cached = CachedPriceProvider(inner, tmp_path)
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 10))
        cached.daily_bars(["TEST", "X"], date(2026, 1, 1), date(2026, 11, 11))
        assert inner.calls == 3

    def test_stale_cache_is_refetched(self, tmp_path, raw_bars):
        inner = CountingProvider(raw_bars)
        cached = CachedPriceProvider(inner, tmp_path, max_age=timedelta(seconds=-1))
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        assert inner.calls == 2

    def test_empty_results_are_not_cached(self, tmp_path, raw_bars):
        inner = CountingProvider(raw_bars.iloc[0:0])
        cached = CachedPriceProvider(inner, tmp_path)
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        cached.daily_bars(["TEST"], date(2026, 1, 1), date(2026, 11, 11))
        assert inner.calls == 2
