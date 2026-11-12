"""The ticker universe: a versioned CSV plus a refresh from Wikipedia's Nasdaq-100 page."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

from stockscan.http import USER_AGENT, HttpError, request_with_retry

# The constituents table has moved between these pages over the years: try both.
WIKIPEDIA_URLS = (
    "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    "https://en.wikipedia.org/wiki/Nasdaq-100",
)
UNIVERSE_COLUMNS = ["ticker", "name", "sector"]
# The index holds ~100 companies, a few with two share classes. Anything far outside
# this range means the page layout changed and we parsed the wrong table.
MIN_CONSTITUENTS = 95
MAX_CONSTITUENTS = 110
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,5}(-[A-Z])?$")


class UniverseError(Exception):
    """The universe could not be loaded or refreshed."""


@dataclass(frozen=True)
class UniverseDiff:
    count: int
    added: list[str]
    removed: list[str]


def normalize_ticker(raw: str) -> str:
    """Upper-case and use Yahoo's class-share separator (``BRK.B`` -> ``BRK-B``)."""
    return raw.strip().upper().replace(".", "-")


def load_universe(path: Path) -> list[str]:
    """Tickers from the universe CSV, in file order, without duplicates."""
    if not path.exists():
        raise UniverseError(f"universe file not found: {path} (run `scan universe refresh`)")
    frame = pd.read_csv(path, dtype=str)
    if "ticker" not in frame.columns:
        raise UniverseError(f"{path} has no 'ticker' column")
    tickers = [normalize_ticker(ticker) for ticker in frame["ticker"].dropna()]
    unique = list(dict.fromkeys(ticker for ticker in tickers if ticker))
    if not unique:
        raise UniverseError(f"{path} contains no tickers")
    return unique


def _find_column(columns: list[str], *needles: str) -> str | None:
    for needle in needles:
        for column in columns:
            if needle in column.lower():
                return column
    return None


def parse_nasdaq100_html(html: str) -> pd.DataFrame:
    """Extract the constituents table (ticker, name, sector) from the Wikipedia page."""
    try:
        # flavor is pinned: the default falls back to html5lib (not installed) on odd pages.
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError as error:  # "No tables found"
        raise UniverseError("no tables found on the page") from error

    for table in tables:
        columns = [
            " ".join(str(part) for part in column) if isinstance(column, tuple) else str(column)
            for column in table.columns
        ]
        table = table.set_axis(columns, axis=1)
        ticker_column = _find_column(columns, "ticker", "symbol")
        if ticker_column is None or not MIN_CONSTITUENTS <= len(table) <= MAX_CONSTITUENTS:
            continue
        name_column = _find_column(columns, "company", "security", "name")
        # Prefer the broad classification ("ICB Industry", "GICS Sector") over sub-levels.
        broad = [column for column in columns if "sub" not in column.lower()]
        sector_column = _find_column(broad, "sector", "industry")
        result = pd.DataFrame(
            {
                "ticker": table[ticker_column].astype(str).map(normalize_ticker),
                "name": table[name_column].astype(str).str.strip() if name_column else "",
                "sector": table[sector_column].astype(str).str.strip() if sector_column else "",
            }
        )
        invalid = [ticker for ticker in result["ticker"] if not _TICKER_PATTERN.match(ticker)]
        if invalid:
            raise UniverseError(f"unexpected ticker symbols in constituents table: {invalid[:5]}")
        result = result.drop_duplicates(subset="ticker").sort_values("ticker")
        return result.reset_index(drop=True)[UNIVERSE_COLUMNS]

    raise UniverseError(
        f"no constituents table with {MIN_CONSTITUENTS}-{MAX_CONSTITUENTS} rows found; "
        "the page layout may have changed"
    )


def fetch_nasdaq100(client: httpx.Client) -> pd.DataFrame:
    """Try each known page until one yields a valid constituents table."""
    problems: list[str] = []
    for url in WIKIPEDIA_URLS:
        try:
            return parse_nasdaq100_html(request_with_retry(client, "GET", url).text)
        except (UniverseError, HttpError) as error:
            problems.append(f"{url}: {error}")
    raise UniverseError("universe refresh failed. " + " | ".join(problems))


def refresh_universe(path: Path, client: httpx.Client | None = None) -> UniverseDiff:
    """Download the current constituents, overwrite the CSV and report what changed.

    The file is only written after the download parsed and validated, so a failed
    refresh never damages the existing universe.
    """
    if client is None:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
        ) as owned:
            fresh = fetch_nasdaq100(owned)
    else:
        fresh = fetch_nasdaq100(client)

    previous = set(load_universe(path)) if path.exists() else set()
    current = set(fresh["ticker"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(path, index=False, lineterminator="\n")
    return UniverseDiff(
        count=len(fresh), added=sorted(current - previous), removed=sorted(previous - current)
    )
