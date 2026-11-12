import json
from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from stockscan.http import HttpError
from stockscan.report import fmt_pct, fmt_price, fmt_ratio, select_top
from stockscan.telegram import API_BASE, TelegramNotifier, build_alert, split_message
from tests.conftest import make_scan

DAY = date(2026, 11, 11)
SENT = datetime(2026, 11, 11, 21, 31, tzinfo=UTC)
TOKEN = "123456:SECRET-TOKEN"
SEND_URL = f"{API_BASE}/bot{TOKEN}/sendMessage"


def alert(long_only: bool = False, top_n: int = 3) -> str:
    scan = make_scan()
    return build_alert(
        select_top(scan, top_n, long_only),
        scan_date=DAY,
        sent_at=SENT,
        long_only=long_only,
        scanned=len(scan),
    )


class TestBuildAlert:
    def test_header_uses_athens_time(self):
        text = alert()
        assert "<b>Hot stocks · Wed 11 Nov 2026</b>" in text
        assert "US close 23:00 Athens · sent 23:31 Athens" in text
        assert "Top 3 of 6 · all directions" in text

    def test_each_entry_has_ticker_price_relvol_gap_atr_and_reason(self):
        text = alert()
        for row in select_top(make_scan(), 3, False).to_dict("records"):
            assert f"<b>{row['rank']}. {row['ticker']}</b> {fmt_price(row['raw_close'])}" in text
            assert f"Rel vol {fmt_ratio(row['rel_volume'])}" in text
            assert f"gap {fmt_pct(row['gap_pct'])}" in text
            assert f"<i>{row['reason']}</i>" in text
        assert text.count("ATR ") == 3

    def test_long_only_is_labelled(self):
        assert "long-only" in alert(long_only=True)

    def test_text_is_html_escaped(self):
        scan = make_scan()
        scan["reason"] = "a <b> & c"
        text = build_alert(
            select_top(scan, 1, False), scan_date=DAY, sent_at=SENT, long_only=False, scanned=6
        )
        assert "<i>a &lt;b&gt; &amp; c</i>" in text

    def test_news_line_only_when_there_is_a_score(self):
        scan = make_scan()
        scan["news_score"] = float("nan")
        scan["news_label"] = None
        top = scan["rank"] == 1
        scan.loc[top, "news_score"] = -0.4
        scan.loc[top, "news_label"] = "Bearish"
        text = build_alert(
            select_top(scan, 3, False), scan_date=DAY, sent_at=SENT, long_only=False, scanned=6
        )
        assert text.count("News: ") == 1
        assert "News: Bearish (-0.40)" in text

    def test_empty_selection(self):
        scan = make_scan().iloc[0:0]
        text = build_alert(scan, scan_date=DAY, sent_at=SENT, long_only=True, scanned=6)
        assert "No stocks matched today." in text

    def test_ten_entries_fit_in_one_telegram_message(self):
        tickers = [f"T{i:02d}" for i in range(12)]
        scan = make_scan(tickers)
        text = build_alert(
            select_top(scan, 10, False), scan_date=DAY, sent_at=SENT, long_only=False, scanned=12
        )
        assert len(split_message(text)) == 1


class TestSplitMessage:
    def test_short_text_is_untouched(self):
        assert split_message("hello\n\nworld") == ["hello\n\nworld"]

    def test_long_text_is_split_on_blank_lines_within_the_limit(self):
        blocks = [f"block {i} " + "x" * 40 for i in range(10)]
        parts = split_message("\n\n".join(blocks), limit=120)
        assert all(len(part) <= 120 for part in parts)
        assert "\n\n".join(parts) == "\n\n".join(blocks)  # nothing lost, order kept

    def test_single_oversized_block_is_hard_split(self):
        parts = split_message("y" * 250, limit=100)
        assert [len(part) for part in parts] == [100, 100, 50]


class TestTelegramNotifier:
    @respx.mock
    def test_posts_html_message_to_the_chat(self):
        route = respx.post(SEND_URL).respond(json={"ok": True})
        with httpx.Client() as client:
            TelegramNotifier(TOKEN, "42", client).send("<b>hi</b>")
        body = json.loads(route.calls.last.request.content)
        assert body["chat_id"] == "42"
        assert body["text"] == "<b>hi</b>"
        assert body["parse_mode"] == "HTML"
        assert body["link_preview_options"] == {"is_disabled": True}

    @respx.mock
    def test_long_messages_are_sent_in_parts(self):
        route = respx.post(SEND_URL).respond(json={"ok": True})
        text = "\n\n".join("z" * 3000 for _ in range(3))
        with httpx.Client() as client:
            TelegramNotifier(TOKEN, "42", client).send(text)
        assert route.call_count == 3

    @respx.mock
    def test_failure_raises_without_exposing_the_token(self):
        respx.post(SEND_URL).respond(
            status_code=400, json={"ok": False, "description": "Bad Request: chat not found"}
        )
        with httpx.Client() as client, pytest.raises(HttpError) as raised:
            TelegramNotifier(TOKEN, "42", client).send("hi")
        assert "chat not found" in str(raised.value)
        assert TOKEN not in str(raised.value)
        assert "SECRET" not in str(raised.value)

    @respx.mock
    def test_rate_limit_is_retried(self):
        route = respx.post(SEND_URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "1"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        with httpx.Client() as client:
            TelegramNotifier(TOKEN, "42", client, sleep=lambda _: None).send("hi")
        assert route.call_count == 2
