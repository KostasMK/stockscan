"""Telegram alert: plain HTTPS calls to the Bot API, no bot framework.

This module can only *send text to you*. It does not read messages, accept commands or
trigger anything.
"""

from __future__ import annotations

import html
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Protocol

import httpx
import pandas as pd

from stockscan.http import request_with_retry
from stockscan.report import fmt_pct, fmt_plain_pct, fmt_price, fmt_ratio, fmt_session
from stockscan.timeutil import session_close_utc, to_athens

API_BASE = "https://api.telegram.org"
MESSAGE_LIMIT = 4096
_ARROWS = {"up": "▲", "down": "▼", "flat": "▬"}


class Notifier(Protocol):
    def send(self, text: str) -> None:
        """Deliver a message. Raises ``HttpError`` if it could not be sent."""
        ...


def _entry(row: dict[str, Any]) -> str:
    arrow = _ARROWS.get(str(row["direction"]), "")
    lines = [
        f"<b>{int(row['rank'])}. {html.escape(str(row['ticker']))}</b> "
        f"{fmt_price(row['raw_close'])} {arrow} {fmt_pct(row['day_change_pct'])}",
        f"Rel vol {fmt_ratio(row['rel_volume'])} · gap {fmt_pct(row['gap_pct'])} "
        f"· ATR {fmt_plain_pct(row['atr_pct'])}",
        f"<i>{html.escape(str(row['reason']))}</i>",
    ]
    score = row.get("news_score")
    if score is not None and not pd.isna(score):
        lines.append(f"News: {html.escape(str(row.get('news_label')))} ({score:+.2f})")
    return "\n".join(lines)


def build_alert(
    shown: pd.DataFrame,
    *,
    scan_date: date,
    sent_at: datetime,
    long_only: bool,
    scanned: int,
) -> str:
    """HTML message for the top candidates (``shown`` comes from ``report.select_top``)."""
    close_athens = to_athens(session_close_utc(scan_date)).strftime("%H:%M")
    sent_athens = to_athens(sent_at).strftime("%H:%M")
    mode = "long-only" if long_only else "all directions"
    header = (
        f"<b>Hot stocks · {fmt_session(scan_date)}</b>\n"
        f"US close {close_athens} Athens · sent {sent_athens} Athens\n"
        f"Top {len(shown)} of {scanned} · {mode}"
    )
    if shown.empty:
        return f"{header}\n\nNo stocks matched today."
    blocks = [header, *(_entry(row) for row in shown.to_dict("records"))]
    blocks.append("Research only, not advice.")
    return "\n\n".join(blocks)


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split on blank lines so that every part fits Telegram's message size limit."""
    parts: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        while len(block) > limit:  # a single oversized block: hard split
            if current:
                parts.append(current)
                current = ""
            parts.append(block[:limit])
            block = block[limit:]
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > limit:
            parts.append(current)
            current = block
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        client: httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._token = token
        self._chat_id = chat_id
        self._client = client
        self._sleep = sleep

    def send(self, text: str) -> None:
        for part in split_message(text):
            request_with_retry(
                self._client,
                "POST",
                f"{API_BASE}/bot{self._token}/sendMessage",
                json_body={
                    "chat_id": self._chat_id,
                    "text": part,
                    "parse_mode": "HTML",
                    "link_preview_options": {"is_disabled": True},
                },
                sleep=self._sleep,
                secrets=[self._token],  # the token is part of the URL: keep it out of errors
            )
