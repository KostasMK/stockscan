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
