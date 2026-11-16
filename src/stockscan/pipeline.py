"""The daily scan: fetch -> clean -> metrics -> score -> news -> store -> report -> alert.

All collaborators (price provider, news provider, notifier, store, clock) are passed in,
so the whole pipeline runs in tests with fakes and no network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from stockscan.bars import PRICE_COLUMNS, adjust_ohlc, clean_daily_bars
from stockscan.config import AppConfig, ScoringConfig
from stockscan.http import HttpError
from stockscan.market_calendar import MarketCalendar, last_completed_session, nyse_calendar
from stockscan.metrics import compute_universe_metrics
from stockscan.providers.base import NewsProvider, NewsUnavailableError, PriceProvider
from stockscan.report import select_top, write_reports
from stockscan.scoring import score_universe
from stockscan.storage import ScanStore
from stockscan.telegram import Notifier, build_alert

logger = logging.getLogger(__name__)

NEWS_COLUMNS = ["news_score", "news_label", "news_articles", "news_headline", "news_url"]


class DataQualityError(Exception):
    """The price data is too incomplete or too old to produce a trustworthy scan."""


@dataclass(frozen=True)
class SessionCheck:
    """How complete the raw daily bars are for one session (diagnostics for the log)."""

    session: date
    tickers: int
    usable: int
    no_row: int
    no_volume: int
    bad_prices: int
    newest_bar: date | None

    @property
    def coverage(self) -> float:
        return self.usable / self.tickers if self.tickers else 0.0

    def describe(self) -> str:
        newest = self.newest_bar.isoformat() if self.newest_bar else "none"
        return (
            f"Session {self.session.isoformat()}: {self.usable}/{self.tickers} tickers have a "
            f"usable bar (no row: {self.no_row}, zero or missing volume: {self.no_volume}, "
            f"missing or non-positive prices: {self.bad_prices}); newest bar returned: {newest}."
        )


def check_session(raw_bars: pd.DataFrame, tickers: Sequence[str], session: date) -> SessionCheck:
    """Classify every universe ticker's bar for ``session``. Pure function.

    Uses the same validity rules as ``clean_daily_bars``, so "usable" here means the bar
    will survive cleaning.
    """
    universe = set(tickers)
    frame = raw_bars[raw_bars["ticker"].isin(universe)]
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    newest = dates.max() if len(dates) else None
    rows = frame[dates == pd.Timestamp(session)].drop_duplicates("ticker", keep="last")
    prices_ok = rows[PRICE_COLUMNS].notna().all(axis=1) & (rows[PRICE_COLUMNS] > 0).all(axis=1)
    volume_ok = rows["volume"].notna() & (rows["volume"] > 0)
    return SessionCheck(
        session=session,
        tickers=len(universe),
        usable=int((prices_ok & volume_ok).sum()),
        no_row=len(universe) - len(rows),
        no_volume=int((prices_ok & ~volume_ok).sum()),
        bad_prices=int((~prices_ok).sum()),
        newest_bar=None if newest is None or pd.isna(newest) else newest.date(),
    )


@dataclass(frozen=True)
class ScanOutcome:
    status: str
    """"completed" or "skipped" (that session is already stored)."""
    scan_date: date
    message: str
    data_check: str | None = None
    """Completeness of the price data for the session (``SessionCheck.describe``)."""
    scan: pd.DataFrame | None = None
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    report_paths: tuple[Path, ...] = ()
    alert_sent: bool = False
    alert_error: str | None = None


def build_scan(
    raw_bars: pd.DataFrame,
    tickers: Sequence[str],
    session: date,
    scoring: ScoringConfig,
    min_coverage: float = 0.5,
) -> tuple[date, pd.DataFrame, list[str]]:
    """Turn raw daily bars into one scored row per ticker for ``session``.

    Pure function. ``session`` is also the look-ahead boundary: bars dated after it are
    discarded before anything is computed, so the result for a session never depends on
    what the provider happens to know about later days.

    The scan is always for exactly ``session``. If too few tickers have a bar that day the
    function raises instead of falling back to an older session: an older "latest bar"
    usually means the source has not published the session yet, not that it was a holiday
    (holidays are excluded by the market calendar before this is called).

    Returns (scan date, scored rows, tickers of the universe with no bar for that date).
    """
    clean = clean_daily_bars(raw_bars, session)
    clean = clean[clean["ticker"].isin(set(tickers))]
    if clean.empty:
        raise DataQualityError(f"no usable price data on or before {session.isoformat()}")

    scan_date = session
    metrics = compute_universe_metrics(adjust_ohlc(clean))
    today = metrics[metrics["date"] == pd.Timestamp(session)]

    present = set(today["ticker"])
    missing = [ticker for ticker in tickers if ticker not in present]
    coverage = len(present) / len(set(tickers))
    if coverage < min_coverage:
        raise DataQualityError(
            f"only {len(present)} of {len(set(tickers))} tickers have a bar for "
            f"{scan_date.isoformat()}; refusing to rank on partial data. "
            + check_session(raw_bars, tickers, session).describe()
        )

    scored = score_universe(today, scoring).rename(columns={"date": "scan_date"})
    return scan_date, scored, missing


def attach_news(
    scan: pd.DataFrame,
    tickers: Sequence[str],
    news: NewsProvider | None,
    close_utc: datetime,
    lookback_hours: int,
) -> tuple[pd.DataFrame, str | None]:
    """Add news columns for ``tickers`` (the few top candidates). Never raises.

    The article window ends at the session close (``close_utc``, 13:00 New York on
    half-days), so a stored day-T row contains only news that existed at the close of
    day T. Returns the frame and an optional note explaining why news is incomplete.
    """
    out = scan.copy()
    out["news_score"] = float("nan")
    out["news_label"] = None
    out["news_articles"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["news_headline"] = None
    out["news_url"] = None
    if news is None or not tickers:
        return out, None

    end = close_utc
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


def fetch_session_bars(
    *,
    prices: PriceProvider,
    tickers: Sequence[str],
    start: date,
    session: date,
    min_coverage: float,
    attempts: int,
    retry_minutes: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[pd.DataFrame, SessionCheck]:
    """Fetch daily bars until ``session`` is covered, retrying with backoff.

    Yahoo sometimes serves the newest session late (seen on GitHub runners in the hours
    after the close). Retrying gives it time; if it is still missing after ``attempts``
    tries, this raises ``DataQualityError`` rather than letting the scan use older data.
    """
    check: SessionCheck | None = None
    for attempt in range(attempts):
        raw = prices.daily_bars(tickers, start, session)
        check = check_session(raw, tickers, session)
        if check.coverage >= min_coverage:
            return raw, check
        if attempt < attempts - 1:
            delay_minutes = retry_minutes * 2**attempt
            logger.warning(
                "%s Retry %d/%d in %.0f min.",
                check.describe(),
                attempt + 1,
                attempts - 1,
                delay_minutes,
            )
            sleep(delay_minutes * 60)
    assert check is not None  # attempts >= 1
    raise DataQualityError(
        f"No complete daily bars for session {session.isoformat()} after {attempts} "
        f"attempt(s). {check.describe()} Not scanning an older session instead; a later "
        "run will pick this session up once the data is there."
    )


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
    calendar: MarketCalendar | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ScanOutcome:
    """Run one scan end to end.

    The session comes from the market calendar: the latest trading day whose close (plus
    ``scan.close_buffer_minutes``) has passed. ``as_of`` re-runs a past session using only
    bars up to that date. ``force`` overwrites a session that is already stored.
    ``long_only`` overrides the config default and only affects what is shown (report,
    alert, news lookups): every ticker is always stored.
    """
    long_only = config.scan.long_only if long_only is None else long_only
    calendar = calendar or nyse_calendar()
    latest = last_completed_session(calendar, now, config.scan.close_buffer_minutes)
    session, attempts = latest, config.scan.session_fetch_attempts
    if as_of is not None:
        if not calendar.is_session(as_of):
            raise ValueError(
                f"--date {as_of.isoformat()} is not a trading session "
                "(weekend or US market holiday)"
            )
        if as_of > latest:
            raise ValueError(
                f"--date {as_of.isoformat()} is not a finished session yet "
                f"(latest finished: {latest.isoformat()})"
            )
        session, attempts = as_of, 1  # history does not appear by waiting

    if store.has_scan(session) and not force:
        return ScanOutcome(
            status="skipped",
            scan_date=session,
            message=(
                f"Session {session.isoformat()} is already stored; nothing to do "
                "(use --force to redo it)."
            ),
        )

    raw, check = fetch_session_bars(
        prices=prices,
        tickers=tickers,
        start=session - timedelta(days=config.scan.history_calendar_days),
        session=session,
        min_coverage=config.scan.min_coverage,
        attempts=attempts,
        retry_minutes=config.scan.session_retry_minutes,
        sleep=sleep,
    )
    scan_date, scan, missing = build_scan(
        raw, tickers, session, config.scoring, config.scan.min_coverage
    )
    if missing:
        logger.warning("no usable bar on %s for: %s", scan_date, ", ".join(missing))

    shown = select_top(scan, config.scan.top_n, long_only)
    news_provider = news if config.news.enabled else None
    scan, news_note = attach_news(
        scan,
        list(shown["ticker"]),
        news_provider,
        calendar.session_close_utc(scan_date),
        config.news.lookback_hours,
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
        data_check=check.describe(),
        scan=stored,
        missing=missing,
        notes=notes,
        report_paths=(markdown_path, csv_path, parquet_path),
        alert_sent=alert_sent,
        alert_error=alert_error,
    )
