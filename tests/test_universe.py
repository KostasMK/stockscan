from pathlib import Path

import httpx
import pytest
import respx

from stockscan.universe import (
    WIKIPEDIA_URLS,
    UniverseError,
    load_universe,
    normalize_ticker,
    parse_nasdaq100_html,
    refresh_universe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def constituents_html(tickers: list[str]) -> str:
    rows = "".join(
        f"<tr><td>{ticker}</td><td>{ticker} Corp</td><td>US0000</td>"
        f"<td>Technology</td><td>Software</td></tr>"
        for ticker in tickers
    )
    return (
        "<html><body>"
        "<table><tr><th>Year</th><th>Closing level</th></tr>"
        "<tr><td>2025</td><td>1</td></tr></table>"
        "<table><tr><th>Ticker</th><th>Company</th><th>ISIN</th>"
        "<th>ICB Industry[2]</th><th>ICB Subsector[2]</th></tr>"
        f"{rows}</table></body></html>"
    )


def fake_tickers(count: int) -> list[str]:
    return [f"T{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(count)]


def test_committed_universe_file_is_loadable_and_plausible():
    tickers = load_universe(REPO_ROOT / "data" / "universe" / "nasdaq100.csv")
    assert 95 <= len(tickers) <= 110
    assert {"AAPL", "MSFT", "NVDA"} <= set(tickers)
    assert len(tickers) == len(set(tickers))


def test_normalize_ticker():
    assert normalize_ticker(" brk.b ") == "BRK-B"


def test_load_universe_normalises_and_deduplicates(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text(
        "ticker,name\naapl,Apple\nBRK.B,Berkshire\nAAPL,Apple again\n", encoding="utf-8"
    )
    assert load_universe(path) == ["AAPL", "BRK-B"]


def test_load_universe_errors_are_explicit(tmp_path):
    with pytest.raises(UniverseError, match="scan universe refresh"):
        load_universe(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol\nAAPL\n", encoding="utf-8")
    with pytest.raises(UniverseError, match="ticker"):
        load_universe(bad)


def test_parse_picks_the_constituents_table_and_the_broad_sector():
    frame = parse_nasdaq100_html(constituents_html(fake_tickers(101)))
    assert list(frame.columns) == ["ticker", "name", "sector"]
    assert len(frame) == 101
    assert frame["ticker"].is_monotonic_increasing
    assert frame.loc[0, "name"] == "TAA Corp"
    assert set(frame["sector"]) == {"Technology"}  # not the "Subsector" column


def test_parse_rejects_a_table_of_the_wrong_size():
    with pytest.raises(UniverseError, match="layout"):
        parse_nasdaq100_html(constituents_html(fake_tickers(30)))


def test_parse_rejects_garbage_symbols():
    tickers = [*fake_tickers(100), "not a ticker!"]
    with pytest.raises(UniverseError, match="unexpected ticker"):
        parse_nasdaq100_html(constituents_html(tickers))


@respx.mock
def test_refresh_writes_the_csv_and_reports_the_diff(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("ticker,name,sector\nTAA,x,y\nOLD,x,y\n", encoding="utf-8")
    respx.get(WIKIPEDIA_URLS[0]).respond(text=constituents_html(fake_tickers(101)))
    with httpx.Client() as client:
        diff = refresh_universe(path, client)
    assert diff.count == 101
    assert diff.removed == ["OLD"]
    assert "TAA" not in diff.added
    assert len(diff.added) == 100
    assert len(load_universe(path)) == 101


@respx.mock
def test_refresh_falls_back_to_the_second_page(tmp_path):
    respx.get(WIKIPEDIA_URLS[0]).respond(text="<html><body><p>moved</p></body></html>")
    respx.get(WIKIPEDIA_URLS[1]).respond(text=constituents_html(fake_tickers(100)))
    with httpx.Client() as client:
        assert refresh_universe(tmp_path / "universe.csv", client).count == 100


@respx.mock
def test_failed_refresh_leaves_the_existing_file_untouched(tmp_path):
    path = tmp_path / "universe.csv"
    original = "ticker,name,sector\nAAPL,Apple,Tech\n"
    path.write_text(original, encoding="utf-8")
    for url in WIKIPEDIA_URLS:
        respx.get(url).respond(status_code=404)
    with httpx.Client() as client, pytest.raises(UniverseError, match="refresh failed"):
        refresh_universe(path, client)
    assert path.read_text(encoding="utf-8") == original
