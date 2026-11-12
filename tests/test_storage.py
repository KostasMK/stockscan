from datetime import date

import pandas as pd
import pytest

from stockscan.storage import SCAN_COLUMNS, ScanStore, normalize_scan_frame
from tests.conftest import make_scan

DAY = date(2026, 11, 11)
PREVIOUS_DAY = date(2026, 11, 10)


@pytest.fixture
def store():
    with ScanStore(None) as opened:
        yield opened


@pytest.fixture
def scan() -> pd.DataFrame:
    return make_scan()


def test_normalize_adds_missing_columns_in_schema_order(scan):
    normalized = normalize_scan_frame(scan)
    assert list(normalized.columns) == SCAN_COLUMNS
    assert normalized["news_headline"].isna().all()  # news was never attached
    assert normalized["run_at_utc"].dt.tz is None  # stored as naive UTC


def test_save_and_load_round_trip(store, scan):
    store.save_scan(scan)
    loaded = store.load_scan(DAY)
    assert list(loaded.columns) == SCAN_COLUMNS
    assert loaded["ticker"].tolist() == scan.sort_values("rank")["ticker"].tolist()
    assert loaded["rank"].tolist() == list(range(1, len(scan) + 1))
    by_ticker = scan.set_index("ticker")
    for row in loaded.to_dict("records"):
        assert row["score"] == pytest.approx(by_ticker.loc[row["ticker"], "score"])
        assert row["rel_volume"] == pytest.approx(by_ticker.loc[row["ticker"], "rel_volume"])
        assert row["reason"] == by_ticker.loc[row["ticker"], "reason"]


def test_one_row_per_ticker_per_day_and_resaving_replaces(store, scan):
    store.save_scan(scan)
    changed = scan.copy()
    changed["score"] = 1.0
    store.save_scan(changed.iloc[:3])  # a re-run that produced fewer rows
    loaded = store.load_scan(DAY)
    assert len(loaded) == 3  # the old rows for that date are gone, not merged
    assert loaded["score"].tolist() == [1.0, 1.0, 1.0]


def test_duplicate_rows_are_rejected(store, scan):
    with pytest.raises(ValueError, match="duplicate"):
        store.save_scan(pd.concat([scan, scan.iloc[[0]]]))


def test_history_accumulates_across_days(store, scan):
    store.save_scan(scan)
    store.save_scan(make_scan(end="2026-11-10"))
    assert store.scan_dates() == [PREVIOUS_DAY, DAY]
    assert store.scan_dates(start=DAY) == [DAY]
    assert store.scan_dates(end=PREVIOUS_DAY) == [PREVIOUS_DAY]
    assert store.latest_scan_date() == DAY
    assert store.has_scan(PREVIOUS_DAY)
    assert not store.has_scan(date(2026, 11, 9))
    assert len(store.load_scan(DAY)) == len(scan)


def test_empty_store(store):
    assert store.latest_scan_date() is None
    assert store.scan_dates() == []
    assert store.load_scan(DAY).empty
    assert store.top_tickers(DAY, 3) == []


def test_top_tickers_follow_rank(store, scan):
    store.save_scan(scan)
    assert store.top_tickers(DAY, 2) == scan.sort_values("rank")["ticker"].head(2).tolist()


def test_database_file_persists_between_sessions(tmp_path, scan):
    path = tmp_path / "nested" / "scan.duckdb"
    with ScanStore(path) as first:
        first.save_scan(scan)
    with ScanStore(path) as second:
        assert second.has_scan(DAY)


def test_parquet_export_and_import_rebuild_the_database(tmp_path, store, scan):
    store.save_scan(scan)
    store.save_scan(make_scan(end="2026-11-10"))
    files = [
        store.export_parquet(DAY, tmp_path / "scans" / "2026-11-11.parquet"),
        store.export_parquet(PREVIOUS_DAY, tmp_path / "scans" / "2026-11-10.parquet"),
    ]

    with ScanStore(None) as fresh:
        assert fresh.import_parquet(files) == 2 * len(scan)
        assert fresh.scan_dates() == [PREVIOUS_DAY, DAY]
        pd.testing.assert_frame_equal(fresh.load_scan(DAY), store.load_scan(DAY))
        # Importing again must not duplicate anything.
        fresh.import_parquet(files)
        assert len(fresh.load_scan(DAY)) == len(scan)


def test_import_of_nothing_is_a_no_op(store):
    assert store.import_parquet([]) == 0
