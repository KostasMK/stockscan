"""Shared test helpers: deterministic synthetic market data (no network in tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_raw_bars(
    ticker: str = "TEST",
    sessions: int = 300,
    end: str = "2026-11-11",
    seed: int = 0,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Random-walk daily bars in the provider's raw schema, ending on ``end`` (a weekday)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=sessions)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0005, 0.02, sessions)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.008, sessions))
    open_[0] = start_price
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.015, sessions))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.015, sessions))
    volume = rng.integers(1_000_000, 5_000_000, sessions).astype(float)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        }
    )


def make_universe_bars(
    tickers: list[str], sessions: int = 300, end: str = "2026-11-11"
) -> pd.DataFrame:
    frames = [
        make_raw_bars(ticker, sessions=sessions, end=end, seed=index, start_price=50 + 40 * index)
        for index, ticker in enumerate(tickers)
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def raw_bars() -> pd.DataFrame:
    return make_raw_bars()


def make_scan(tickers: list[str] | None = None, end: str = "2026-11-11") -> pd.DataFrame:
    """A realistic scored scan (one row per ticker) built with the real pipeline code."""
    from datetime import date

    from stockscan.config import ScoringConfig
    from stockscan.pipeline import build_scan

    tickers = tickers or ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    bars = make_universe_bars(tickers, end=end)
    _, scan, _ = build_scan(bars, tickers, date.fromisoformat(end), ScoringConfig())
    scan["scoring_version"] = ScoringConfig().version
    scan["run_at_utc"] = pd.Timestamp(f"{end} 21:30", tz="UTC")
    return scan


def make_intraday_bars(tickers: list[str], sessions: list[str], bars_per_session: int = 6):
    """5-minute bars from 09:30 New York for each session, in the provider's schema (UTC)."""
    rows = []
    for session in sessions:
        stamps = pd.date_range(
            f"{session} 09:30", periods=bars_per_session, freq="5min", tz="America/New_York"
        ).tz_convert("UTC")
        for offset, ticker in enumerate(tickers):
            for step, stamp in enumerate(stamps):
                price = 100.0 + 10 * offset + step
                rows.append(
                    {
                        "ticker": ticker,
                        "ts": stamp,
                        "open": price,
                        "high": price + 1,
                        "low": price - 1,
                        "close": price + 0.5,
                        "volume": 1000.0 + step,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    """Tests must never pick up real keys from the environment or a developer's `.env`.

    An environment variable (even a blank one) takes priority over the `.env` file, and
    blank means "not configured", so this guarantees no test can call a real API or send
    a real Telegram message.
    """
    for name in ("ALPHAVANTAGE_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.setenv(name, "")
