"""The daily scan: fetch -> clean -> metrics -> score -> news -> store -> report -> alert.

All collaborators (price provider, news provider, notifier, store, clock) are passed in,
so the whole pipeline runs in tests with fakes and no network.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from stockscan.bars import adjust_ohlc, clean_daily_bars
from stockscan.config import AppConfig, ScoringConfig
from stockscan.http import HttpError
from stockscan.metrics import compute_universe_metrics
from stockscan.providers.base import NewsProvider, NewsUnavailableError, PriceProvider
from stockscan.report import select_top, write_reports
from stockscan.scoring import score_universe
from stockscan.storage import ScanStore
from stockscan.telegram import Notifier, build_alert
from stockscan.timeutil import last_completed_session_cutoff, session_close_utc

logger = logging.getLogger(__name__)

MAX_STALE_DAYS = 5  # longest plausible gap between sessions (long weekend) plus a margin
NEWS_COLUMNS = ["news_score", "news_label", "news_articles", "news_headline", "news_url"]


class DataQualityError(Exception):
    """The price data is too incomplete or too old to produce a trustworthy scan."""


@dataclass(frozen=True)
class ScanOutcome:
    status: str
    """"completed" or "skipped" (that session is already stored)."""
    scan_date: date
    message: str
    scan: pd.DataFrame | None = None
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    report_paths: tuple[Path, ...] = ()
    alert_sent: bool = False
    alert_error: str | None = None


def build_scan(
    raw_bars: pd.DataFrame,
    tickers: Sequence[str],
    cutoff: date,
    scoring: ScoringConfig,
    min_coverage: float = 0.5,
) -> tuple[date, pd.DataFrame, list[str]]:
    """Turn raw daily bars into one scored row per ticker for the latest finished session.

    Pure function. ``cutoff`` is the look-ahead boundary: bars dated after it are
    discarded before anything is computed, so the result for a session never depends on
    what the provider happens to know about later days.

    Returns (scan date, scored rows, tickers of the universe with no bar for that date).
    """
    clean = clean_daily_bars(raw_bars, cutoff)
    clean = clean[clean["ticker"].isin(set(tickers))]
    if clean.empty:
        raise DataQualityError(f"no usable price data on or before {cutoff.isoformat()}")

    session = clean["date"].max()
    scan_date = session.date()
    metrics = compute_universe_metrics(adjust_ohlc(clean))
    today = metrics[metrics["date"] == session]

    present = set(today["ticker"])
    missing = [ticker for ticker in tickers if ticker not in present]
    coverage = len(present) / len(set(tickers))
    if coverage < min_coverage:
        raise DataQualityError(
            f"only {len(present)} of {len(set(tickers))} tickers have a bar for "
            f"{scan_date.isoformat()}; refusing to rank on partial data"
        )

    scored = score_universe(today, scoring).rename(columns={"date": "scan_date"})
    return scan_date, scored, missing


def attach_news(
    scan: pd.DataFrame,
    tickers: Sequence[str],
    news: NewsProvider | None,
    scan_date: date,
    lookback_hours: int,
) -> tuple[pd.DataFrame, str | None]:
    """Add news columns for ``tickers`` (the few top candidates). Never raises.

    The article window ends at the session close, so a stored day-T row contains only
    news that existed at the close of day T. Returns the frame and an optional note
    explaining why news is incomplete.
    """
    out = scan.copy()
    out["news_score"] = float("nan")
    out["news_label"] = None
    out["news_articles"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["news_headline"] = None
    out["news_url"] = None
    if news is None or not tickers:
        return out, None

    end = session_close_utc(scan_date)
    start = end - timedelta(hours=lookback_hours)
    note: str | None = None
    for ticker in tickers:
        try:
            result = news.sentiment(ticker, start, end)
        except NewsUnavailableError as error:
            note = f"News sentiment incomplete: {error}"
            logger.warning(note)
            break
        mask = out["ticker"] == ticker
        out.loc[mask, "news_score"] = result.score if result.score is not None else float("nan")
        out.loc[mask, "news_label"] = result.label
        out.loc[mask, "news_articles"] = result.article_count
        out.loc[mask, "news_headline"] = result.headline
        out.loc[mask, "news_url"] = result.url
    return out, note


def run_scan(
    *,
    config: AppConfig,
    tickers: Sequence[str],
    prices: PriceProvider,
    store: ScanStore,
    output_dir: Path,
    now: datetime,
    news: NewsProvider | None = None,
    notifier: Notifier | None = None,
    as_of: date | None = None,
    force: bool = False,
    long_only: bool | None = None,
) -> ScanOutcome:
    """Run one scan end to end.

    ``as_of`` re-runs a past session using only bars up to that date. ``force`` overwrites
    a session that is already stored. ``long_only`` overrides the config default and only
    affects what is shown (report, alert, news lookups): every ticker is always stored.
    """
    long_only = config.scan.long_only if long_only is None else long_only
    cutoff = last_completed_session_cutoff(now, config.scan.close_buffer_minutes)
    if as_of is not None:
        if as_of > cutoff:
            raise ValueError(
                f"--date {as_of.isoformat()} is not a finished session yet "
                f"(latest finished: {cutoff.isoformat()})"
            )
        cutoff = as_of

    start = cutoff - timedelta(days=config.scan.history_calendar_days)
    raw = prices.daily_bars(tickers, start, cutoff)
    scan_date, scan, missing = build_scan(
        raw, tickers, cutoff, config.scoring, config.scan.min_coverage
    )
    if (cutoff - scan_date).days > MAX_STALE_DAYS:
        raise DataQualityError(
            f"latest bar is from {scan_date.isoformat()}, {(cutoff - scan_date).days} days "
            f"before {cutoff.isoformat()}: the price source looks stale"
        )
    if missing:
        logger.warning("no usable bar on %s for: %s", scan_date, ", ".join(missing))

    if store.has_scan(scan_date) and not force:
        return ScanOutcome(
            status="skipped",
            scan_date=scan_date,
            message=(
                f"Session {scan_date.isoformat()} is already stored; nothing to do "
                "(use --force to redo it)."
            ),
        )

    shown = select_top(scan, config.scan.top_n, long_only)
    news_provider = news if config.news.enabled else None
    scan, news_note = attach_news(
        scan, list(shown["ticker"]), news_provider, scan_date, config.news.lookback_hours
    )
    notes = [news_note] if news_note else []
    scan["scoring_version"] = config.scoring.version
    scan["run_at_utc"] = now

    # Store first: if the report or the alert fails, the data is still safe.
    store.save_scan(scan)
    stored = store.load_scan(scan_date)
    parquet_path = store.export_parquet(scan_date, output_dir / "scans" / f"{scan_date}.parquet")
    markdown_path, csv_path = write_reports(
        stored,
        output_dir,
        scan_date=scan_date,
        generated_at=now,
        top_n=config.scan.top_n,
        long_only=long_only,
        missing=missing,
        notes=notes,
    )

    alert_sent, alert_error = False, None
    if notifier is not None:
        text = build_alert(
            select_top(stored, config.scan.top_n, long_only),
            scan_date=scan_date,
            sent_at=now,
            long_only=long_only,
            scanned=len(stored),
        )
        try:
            notifier.send(text)
            alert_sent = True
        except HttpError as error:
            alert_error = str(error)
            logger.error("Telegram alert failed: %s", alert_error)

    return ScanOutcome(
        status="completed",
        scan_date=scan_date,
        message=f"Scanned {len(stored)} tickers for session {scan_date.isoformat()}.",
        scan=stored,
        missing=missing,
        notes=notes,
        report_paths=(markdown_path, csv_path, parquet_path),
        alert_sent=alert_sent,
        alert_error=alert_error,
    )
