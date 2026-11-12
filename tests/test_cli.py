"""CLI tests: real commands, fake data provider, temporary folders, no network."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
import respx
import typer
from typer.testing import CliRunner

from stockscan import cli
from stockscan.storage import ScanStore
from stockscan.telegram import API_BASE
from stockscan.universe import UniverseDiff
from tests.conftest import make_intraday_bars, make_raw_bars, make_universe_bars

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
AFTER_CLOSE = datetime(2026, 11, 11, 21, 30, tzinfo=UTC)
TOKEN = "999:CLI-SECRET"
runner = CliRunner()


class FakePrices:
    def __init__(self) -> None:
        self.daily = make_universe_bars(TICKERS, end="2026-11-11")

    def daily_bars(self, tickers, start, end):
        if list(tickers) == [cli.FX_TICKER]:
            fx = make_raw_bars(cli.FX_TICKER, sessions=3, end="2026-11-11")
            fx["close"] = 1.10
            fx["volume"] = 0.0  # FX quotes carry no volume
            return fx
        return self.daily

    def intraday_bars(self, tickers, start, end, interval):
        return make_intraday_bars(list(tickers), ["2026-11-10", "2026-11-11"])


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    universe = tmp_path / "universe.csv"
    universe.write_text("ticker,name,sector\n" + "".join(f"{t},{t} Inc,Tech\n" for t in TICKERS))
    config = tmp_path / "config.toml"
    config.write_text(f'[scan]\nuniverse_file = "{universe.as_posix()}"\ntop_n = 3\n')
    monkeypatch.setenv("STOCKSCAN_CONFIG_FILE", str(config))
    monkeypatch.setenv("STOCKSCAN_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("STOCKSCAN_DB_PATH", str(tmp_path / "db" / "scan.duckdb"))
    monkeypatch.setenv("STOCKSCAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_price_provider", lambda env: FakePrices())
    monkeypatch.setattr(cli, "now_utc", lambda: AFTER_CLOSE)
    return tmp_path


def invoke(*args: str):
    return runner.invoke(cli.app, list(args))


def test_cli_exposes_only_research_commands():
    # Scope guard: this tool must never grow order placement or broker features.
    def names(app: typer.Typer) -> set[str]:
        commands = {c.name or c.callback.__name__ for c in app.registered_commands}
        return commands | {group.name for group in app.registered_groups}

    assert names(cli.app) == {"run", "report", "size", "universe", "db", "intraday"}
    assert names(cli.universe_app) == {"refresh"}
    assert names(cli.db_app) == {"import"}
    assert names(cli.intraday_app) == {"save"}


class TestRun:
    def test_full_run_writes_everything_and_prints_the_top(self, workspace):
        result = invoke("run")
        assert result.exit_code == 0, result.output
        assert "Scanned 6 tickers for session 2026-11-11." in result.output
        assert "skipping news sentiment" in result.output
        assert "skipping the alert" in result.output
        assert (workspace / "out" / "reports" / "2026-11-11.md").exists()
        assert (workspace / "out" / "reports" / "2026-11-11.csv").exists()
        assert (workspace / "out" / "scans" / "2026-11-11.parquet").exists()
        with ScanStore(workspace / "db" / "scan.duckdb") as store:
            stored = store.load_scan(date(2026, 11, 11))
        assert len(stored) == 6
        top_ticker = stored["ticker"].iloc[0]
        assert f"  1  {top_ticker}" in result.output

    def test_second_run_is_a_no_op(self, workspace):
        invoke("run")
        result = invoke("run")
        assert result.exit_code == 0
        assert "already stored" in result.output

    @respx.mock
    def test_alert_is_sent_when_telegram_is_configured(self, workspace, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        route = respx.post(f"{API_BASE}/bot{TOKEN}/sendMessage").respond(json={"ok": True})
        result = invoke("run")
        assert result.exit_code == 0, result.output
        assert route.call_count == 1
        assert "Telegram alert sent." in result.output
        assert TOKEN not in result.output

    @respx.mock
    def test_no_telegram_flag_and_historical_reruns_never_alert(self, workspace, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        route = respx.post(f"{API_BASE}/bot{TOKEN}/sendMessage").respond(json={"ok": True})
        assert invoke("run", "--no-telegram").exit_code == 0
        assert invoke("run", "--date", "2026-11-09").exit_code == 0
        assert route.call_count == 0

    @respx.mock
    def test_failed_alert_exits_non_zero_but_keeps_the_data(self, workspace, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        respx.post(f"{API_BASE}/bot{TOKEN}/sendMessage").respond(
            status_code=400, json={"ok": False, "description": "Bad Request: chat not found"}
        )
        result = invoke("run")
        assert result.exit_code == 1
        assert "chat not found" in result.output
        assert "scan report --date 2026-11-11 --send-telegram" in result.output
        assert TOKEN not in result.output
        with ScanStore(workspace / "db" / "scan.duckdb") as store:
            assert store.has_scan(date(2026, 11, 11))

    def test_long_only_flag_filters_the_printed_table(self, workspace):
        result = invoke("run", "--long-only")
        assert result.exit_code == 0
        with ScanStore(workspace / "db" / "scan.duckdb") as store:
            stored = store.load_scan(date(2026, 11, 11))
        assert len(stored) == 6
        for ticker in stored.loc[stored["direction"] != "up", "ticker"]:
            assert f" {ticker} " not in result.output

    def test_bad_date_is_a_clean_error(self, workspace):
        result = invoke("run", "--date", "11/11/2026")
        assert result.exit_code == 1
        assert "YYYY-MM-DD" in result.output

    def test_unfinished_session_is_a_clean_error(self, workspace):
        result = invoke("run", "--date", "2026-11-12")
        assert result.exit_code == 1
        assert "not a finished session" in result.output

    def test_missing_universe_file_is_a_clean_error(self, workspace):
        (workspace / "universe.csv").unlink()
        result = invoke("run")
        assert result.exit_code == 1
        assert "scan universe refresh" in result.output


class TestReport:
    def test_shows_the_latest_stored_session(self, workspace):
        invoke("run")
        result = invoke("report")
        assert result.exit_code == 0, result.output
        assert "# Hot stocks: Wed 11 Nov 2026" in result.output
        assert "showing top 3, all directions" in result.output

    def test_date_top_and_long_only_options(self, workspace):
        invoke("run")
        invoke("run", "--date", "2026-11-09")
        result = invoke("report", "--date", "2026-11-09", "--top", "2", "--long-only")
        assert result.exit_code == 0
        assert "# Hot stocks: Mon 09 Nov 2026" in result.output
        assert "long-only (closed up)" in result.output

    def test_write_option_regenerates_the_files(self, workspace):
        invoke("run")
        markdown = workspace / "out" / "reports" / "2026-11-11.md"
        markdown.unlink()
        assert invoke("report", "--write").exit_code == 0
        assert markdown.exists()

    def test_unknown_date_and_empty_database_are_clean_errors(self, workspace):
        empty = invoke("report")
        assert empty.exit_code == 1
        assert "no scans stored yet" in empty.output
        invoke("run")
        unknown = invoke("report", "--date", "2020-01-02")
        assert unknown.exit_code == 1
        assert "no scan stored for 2020-01-02" in unknown.output

    @respx.mock
    def test_send_telegram_resends_the_alert(self, workspace, monkeypatch):
        invoke("run")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
        route = respx.post(f"{API_BASE}/bot{TOKEN}/sendMessage").respond(json={"ok": True})
        result = invoke("report", "--send-telegram")
        assert result.exit_code == 0, result.output
        assert route.call_count == 1

    def test_send_telegram_without_credentials_fails(self, workspace):
        invoke("run")
        result = invoke("report", "--send-telegram")
        assert result.exit_code == 1
        assert "TELEGRAM_BOT_TOKEN" in result.output


class TestSize:
    BASE = ("size", "--pot", "10000", "--entry", "50", "--stop", "48", "--eurusd", "1.10")

    def test_whole_share_result_with_default_one_percent_risk(self, workspace):
        result = invoke(*self.BASE, "--usd-balance", "--fee-min", "0", "--fee-pct", "0")
        assert result.exit_code == 0, result.output
        assert "risk 1% = EUR 100.00" in result.output
        assert "Whole shares (limit order)" in result.output
        assert "Fractional" not in result.output
        shares_line = next(line for line in result.output.splitlines() if "Shares " in line)
        assert shares_line.split()[-1] == "55"
        assert "EUR 100.00 / $110.00" in result.output  # max loss in both currencies
        assert "1.00%" in result.output

    def test_fractional_option_shows_both_results(self, workspace):
        result = invoke(
            *self.BASE, "--fractional", "--usd-balance", "--entry", "50", "--stop", "47"
        )
        assert result.exit_code == 0, result.output
        assert "Whole shares (limit order)" in result.output
        assert "Fractional (market order)" in result.output

    def test_default_fees_assume_eur_conversion(self, workspace):
        result = invoke(*self.BASE)
        assert "commission max(EUR 1, 0.25%) per side, FX 1% each way (converting EUR)" in (
            result.output
        )
        usd = invoke(*self.BASE, "--usd-balance")
        assert "FX 0% each way (from USD balance)" in usd.output

    def test_warns_above_two_percent(self, workspace):
        result = invoke(*self.BASE, "--risk", "3")
        assert result.exit_code == 0
        assert "above the 2% limit" in result.output

    def test_rate_is_fetched_when_not_given(self, workspace):
        result = invoke("size", "--pot", "10000", "--entry", "50", "--stop", "48")
        assert result.exit_code == 0, result.output
        assert "EUR/USD 1.1000 (Yahoo, 2026-11-11)" in result.output

    def test_stop_above_entry_is_rejected(self, workspace):
        result = invoke("size", "--pot", "1000", "--entry", "50", "--stop", "55", "--eurusd", "1.1")
        assert result.exit_code == 1
        assert "long share positions only" in result.output

    def test_states_that_no_order_is_placed(self, workspace):
        assert "no order is placed" in invoke(*self.BASE).output


def test_universe_refresh_prints_the_diff(workspace, monkeypatch):
    monkeypatch.setattr(
        cli, "refresh_universe", lambda path: UniverseDiff(101, added=["NEW"], removed=["OLD"])
    )
    result = invoke("universe", "refresh")
    assert result.exit_code == 0
    assert "101 tickers" in result.output
    assert "added:   NEW" in result.output
    assert "removed: OLD" in result.output


def test_db_import_rebuilds_history_from_parquet(workspace, monkeypatch):
    invoke("run")
    invoke("run", "--date", "2026-11-10")
    monkeypatch.setenv("STOCKSCAN_DB_PATH", str(workspace / "fresh.duckdb"))
    result = invoke("db", "import", str(workspace / "out" / "scans"))
    assert result.exit_code == 0, result.output
    assert "Imported 12 rows from 2 files" in result.output
    assert "2 sessions (2026-11-10 to 2026-11-11)" in result.output
    with ScanStore(workspace / "fresh.duckdb") as store:
        assert len(store.load_scan(date(2026, 11, 11))) == 6


def test_db_import_with_nothing_to_import(workspace):
    result = invoke("db", "import")
    assert result.exit_code == 0
    assert "nothing to import" in result.output


def test_intraday_save_archives_top_tickers(workspace):
    invoke("run")
    result = invoke("intraday", "save")
    assert result.exit_code == 0, result.output
    with ScanStore(workspace / "db" / "scan.duckdb") as store:
        top3 = store.top_tickers(date(2026, 11, 11), 10)
    assert "NVDA, QQQ" in result.output
    saved = pd.read_parquet(workspace / "out" / "intraday" / "5m" / "2026-11-11.parquet")
    assert set(saved["ticker"]) == {"NVDA", "QQQ", *top3}
    again = invoke("intraday", "save")
    assert "Already saved: 2026-11-10, 2026-11-11" in again.output
