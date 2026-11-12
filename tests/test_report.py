from datetime import UTC, date, datetime

import pandas as pd
import pytest

from stockscan.report import (
    fmt_pct,
    fmt_plain_pct,
    fmt_price,
    fmt_ratio,
    render_markdown,
    select_top,
    write_reports,
)
from tests.conftest import make_scan

DAY = date(2026, 11, 11)
GENERATED = datetime(2026, 11, 11, 21, 30, tzinfo=UTC)


@pytest.fixture
def scan() -> pd.DataFrame:
    return make_scan()


def render(scan: pd.DataFrame, **overrides) -> str:
    arguments = {"scan_date": DAY, "generated_at": GENERATED, "top_n": 3, "long_only": False}
    arguments.update(overrides)
    return render_markdown(scan, **arguments)


def test_formatters():
    assert fmt_price(1234.5) == "$1,234.50"
    assert fmt_pct(4.25) == "+4.2%"
    assert fmt_pct(-0.04) == "-0.0%"
    assert fmt_plain_pct(3.44) == "3.4%"
    assert fmt_ratio(3.14) == "3.1x"
    assert fmt_price(None) == fmt_pct(float("nan")) == fmt_ratio(pd.NA) == "n/a"


class TestSelectTop:
    def test_takes_the_best_ranks(self, scan):
        shuffled = scan.sample(frac=1, random_state=3)
        assert select_top(shuffled, 3, long_only=False)["rank"].tolist() == [1, 2, 3]

    def test_long_only_keeps_only_stocks_that_closed_up(self, scan):
        shown = select_top(scan, 10, long_only=True)
        assert len(shown) == (scan["direction"] == "up").sum()
        assert set(shown["direction"]) <= {"up"}
        assert shown["rank"].is_monotonic_increasing


class TestRenderMarkdown:
    def test_header_shows_session_and_athens_times(self, scan):
        text = render(scan)
        assert "# Hot stocks: Wed 11 Nov 2026" in text
        assert "US close 16:00 New York = 23:00 Athens" in text
        assert "Generated: 2026-11-11 23:30 EET (Europe/Athens)" in text
        assert f"Scanned: {len(scan)} tickers, showing top 3, all directions" in text

    def test_table_has_one_row_per_shown_ticker_with_the_key_metrics(self, scan):
        text = render(scan)
        top = scan.sort_values("rank").iloc[0]
        row = next(line for line in text.splitlines() if line.startswith("| 1 |"))
        assert f"**{top['ticker']}**" in row
        assert fmt_price(top["raw_close"]) in row
        assert fmt_ratio(top["rel_volume"]) in row
        assert fmt_pct(top["gap_pct"]) in row
        assert fmt_plain_pct(top["atr_pct"]) in row
        assert f"{top['score']:.1f}" in row
        assert sum(line.startswith("| ") and "**" in line for line in text.splitlines()) == 3

    def test_every_shown_ticker_gets_its_reason_and_full_score_breakdown(self, scan):
        text = render(scan)
        for row in scan.sort_values("rank").head(3).to_dict("records"):
            assert f"### {row['rank']}. {row['ticker']} ({row['direction']})" in text
            assert row["reason"] in text
            assert f"rel. volume {row['c_rel_volume']:.1f}" in text
        assert "- Score " in text
        assert "52w extreme" in text

    def test_long_only_is_stated_and_applied(self, scan):
        text = render(scan, top_n=10, long_only=True)
        assert "long-only (closed up)" in text
        for ticker in scan.loc[scan["direction"] != "up", "ticker"]:
            assert f"**{ticker}**" not in text

    def test_news_and_notes_are_included_when_present(self, scan):
        top_ticker = scan.sort_values("rank")["ticker"].iloc[0]
        with_news = scan.copy()
        with_news["news_score"] = float("nan")
        with_news["news_label"] = None
        with_news["news_articles"] = None
        with_news["news_headline"] = None
        with_news["news_url"] = None
        mask = with_news["ticker"] == top_ticker
        with_news.loc[mask, "news_score"] = 0.21
        with_news.loc[mask, "news_label"] = "Somewhat-Bullish"
        with_news.loc[mask, "news_articles"] = 12
        with_news.loc[mask, "news_headline"] = "Chip maker [update] beats estimates"
        with_news.loc[mask, "news_url"] = "https://news.example.test/a"

        text = render(with_news, missing=["XXX", "YYY"], notes=["News sentiment incomplete: x"])
        assert (
            "- News: Somewhat-Bullish (+0.21, 12 articles): "
            "[Chip maker (update) beats estimates](https://news.example.test/a)"
        ) in text
        assert text.count("- News:") == 1
        assert "No usable bar for this session (2): XXX, YYY" in text
        assert "- News sentiment incomplete: x" in text

    def test_empty_selection_is_reported_plainly(self, scan):
        none_up = scan.assign(direction="down")
        assert "No stocks matched today." in render(none_up, long_only=True)

    def test_report_states_it_is_research_only(self, scan):
        assert "not investment advice" in render(scan)


def test_write_reports_creates_markdown_and_full_csv(tmp_path, scan):
    markdown_path, csv_path = write_reports(
        scan, tmp_path, scan_date=DAY, generated_at=GENERATED, top_n=3, long_only=False
    )
    assert markdown_path == tmp_path / "reports" / "2026-11-11.md"
    assert csv_path == tmp_path / "reports" / "2026-11-11.csv"
    assert markdown_path.read_text(encoding="utf-8").startswith("# Hot stocks")

    table = pd.read_csv(csv_path)
    assert len(table) == len(scan)  # the CSV holds every ticker, not just the top
    assert table["rank"].tolist() == list(range(1, len(scan) + 1))
    assert {"ticker", "score", "rel_volume", "gap_pct", "atr_pct", "reason"} <= set(table.columns)
