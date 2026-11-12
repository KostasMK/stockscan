"""Command line interface: ``scan run | report | size | universe | db | intraday``.

The CLI only wires real collaborators together and prints results; all logic lives in
the other modules. There is deliberately no command that places, prepares or sends an
order anywhere.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NoReturn

import httpx
import typer
from pydantic import ValidationError

from stockscan.config import AppConfig, EnvSettings, load_config
from stockscan.http import USER_AGENT, DailyBudget, HttpError, JsonDiskCache, RateLimiter
from stockscan.intraday import MAX_DAYS, save_intraday
from stockscan.pipeline import DataQualityError, ScanOutcome, run_scan
from stockscan.providers.alphavantage import AlphaVantageNews
from stockscan.providers.base import NewsProvider, PriceProvider, ProviderError
from stockscan.providers.yfinance_provider import CachedPriceProvider, YFinanceProvider
from stockscan.report import (
    fmt_pct,
    fmt_plain_pct,
    fmt_price,
    fmt_ratio,
    render_markdown,
    select_top,
    write_reports,
)
from stockscan.sizing import FeeModel, PositionSize, size_position
from stockscan.storage import ScanStore
from stockscan.telegram import Notifier, TelegramNotifier, build_alert
from stockscan.timeutil import format_athens, now_utc
from stockscan.universe import UniverseError, load_universe, refresh_universe

app = typer.Typer(
    help="Daily hot-stocks scanner and research helper. Research and alerts only: no orders.",
    no_args_is_help=True,
    add_completion=False,
)
universe_app = typer.Typer(help="Manage the ticker universe.", no_args_is_help=True)
db_app = typer.Typer(help="Local DuckDB history.", no_args_is_help=True)
intraday_app = typer.Typer(help="Intraday bar archive for backtests.", no_args_is_help=True)
app.add_typer(universe_app, name="universe")
app.add_typer(db_app, name="db")
app.add_typer(intraday_app, name="intraday")

FX_TICKER = "EURUSD=X"


def _fail(message: str) -> NoReturn:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _warn(message: str) -> None:
    typer.secho(message, fg=typer.colors.YELLOW, err=True)


def _load() -> tuple[EnvSettings, AppConfig]:
    try:
        env = EnvSettings()
        return env, load_config(env.config_file)
    except (FileNotFoundError, ValidationError) as error:
        _fail(str(error))


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        _fail(f"'{value}' is not a date in YYYY-MM-DD format")


def _http_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)


def _price_provider(env: EnvSettings) -> PriceProvider:
    return CachedPriceProvider(YFinanceProvider(), env.cache_dir / "prices")


def _news_provider(
    env: EnvSettings, config: AppConfig, client: httpx.Client
) -> NewsProvider | None:
    if env.alphavantage_api_key is None:
        return None
    cache_dir = env.cache_dir / "alphavantage"
    return AlphaVantageNews(
        api_key=env.alphavantage_api_key.get_secret_value(),
        client=client,
        cache=JsonDiskCache(cache_dir),
        budget=DailyBudget(cache_dir / "budget.json", config.news.max_calls_per_day),
        limiter=RateLimiter(config.news.min_interval_seconds),
        article_limit=config.news.article_limit,
        min_relevance=config.news.min_relevance,
    )


def _notifier(env: EnvSettings, client: httpx.Client) -> Notifier | None:
    if env.telegram_bot_token is None or env.telegram_chat_id is None:
        return None
    return TelegramNotifier(env.telegram_bot_token.get_secret_value(), env.telegram_chat_id, client)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logging."),
) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # Windows consoles default to a legacy code page
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpx logs full request URLs at INFO, and those contain the bot token / API key.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# --- scan run -------------------------------------------------------------------


def _print_top(outcome: ScanOutcome, top_n: int, long_only: bool) -> None:
    if outcome.scan is None:
        return
    shown = select_top(outcome.scan, top_n, long_only)
    typer.echo("")
    typer.echo(f"{'#':>3}  {'Ticker':<6} {'Price':>10} {'Day':>7} {'RelVol':>6} {'Gap':>7} "
               f"{'ATR':>5} {'Score':>5}  Reason")  # fmt: skip
    for row in shown.to_dict("records"):
        typer.echo(
            f"{int(row['rank']):>3}  {row['ticker']:<6} {fmt_price(row['raw_close']):>10} "
            f"{fmt_pct(row['day_change_pct']):>7} {fmt_ratio(row['rel_volume']):>6} "
            f"{fmt_pct(row['gap_pct']):>7} {fmt_plain_pct(row['atr_pct']):>5} "
            f"{row['score']:>5.1f}  {row['reason']}"
        )
    typer.echo("")


@app.command()
def run(
    as_of: str | None = typer.Option(
        None,
        "--date",
        help="Re-run a past session (YYYY-MM-DD) using only data up to that date. "
        "Never sends a Telegram alert.",
    ),
    force: bool = typer.Option(False, "--force", help="Redo a session that is already stored."),
    long_only: bool | None = typer.Option(
        None,
        "--long-only/--all-directions",
        help="Show only stocks that closed up in the report and alert "
        "(default: scan.long_only in the config). Everything is stored either way.",
    ),
    no_telegram: bool = typer.Option(False, "--no-telegram", help="Do not send the alert."),
    no_news: bool = typer.Option(False, "--no-news", help="Skip Alpha Vantage sentiment."),
) -> None:
    """Fetch data, compute metrics, store them, write the report and send the alert."""
    env, config = _load()
    session = _parse_date(as_of)
    show_long_only = config.scan.long_only if long_only is None else long_only

    try:
        tickers = load_universe(config.scan.universe_file)
    except UniverseError as error:
        _fail(str(error))

    with _http_client() as client, ScanStore(env.db_path) as store:
        news = None if no_news else _news_provider(env, config, client)
        if news is None and not no_news and config.news.enabled:
            _warn("ALPHAVANTAGE_API_KEY is not set: skipping news sentiment.")
        notifier = None if (no_telegram or session is not None) else _notifier(env, client)
        if notifier is None and not no_telegram and session is None:
            _warn("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set: skipping the alert.")

        try:
            outcome = run_scan(
                config=config,
                tickers=tickers,
                prices=_price_provider(env),
                store=store,
                output_dir=env.output_dir,
                now=now_utc(),
                news=news,
                notifier=notifier,
                as_of=session,
                force=force,
                long_only=long_only,
            )
        except (ProviderError, DataQualityError, ValueError) as error:
            _fail(str(error))

    typer.echo(outcome.message)
    if outcome.status == "skipped":
        return
    _print_top(outcome, config.scan.top_n, show_long_only)
    for note in outcome.notes:
        _warn(note)
    if outcome.missing:
        _warn(f"No usable bar for: {', '.join(outcome.missing)}")
    for path in outcome.report_paths:
        typer.echo(f"Wrote {path}")
    if outcome.alert_sent:
        typer.echo("Telegram alert sent.")
    if outcome.alert_error:
        _fail(
            f"results are stored, but the Telegram alert failed: {outcome.alert_error}\n"
            f"Resend with: scan report --date {outcome.scan_date.isoformat()} --send-telegram"
        )


# --- scan report ----------------------------------------------------------------


@app.command()
def report(
    session: str | None = typer.Option(
        None, "--date", help="Session to report (YYYY-MM-DD). Default: latest stored."
    ),
    long_only: bool | None = typer.Option(
        None, "--long-only/--all-directions", help="Show only stocks that closed up."
    ),
    top: int | None = typer.Option(None, "--top", min=1, help="Rows to show (default: config)."),
    write: bool = typer.Option(False, "--write", help="Also (re)write the report files."),
    send_telegram: bool = typer.Option(
        False, "--send-telegram", help="Also send the alert for this session."
    ),
) -> None:
    """Show the report for a stored session (nothing is fetched or recomputed)."""
    env, config = _load()
    top_n = top or config.scan.top_n
    show_long_only = config.scan.long_only if long_only is None else long_only

    with ScanStore(env.db_path) as store:
        scan_date = _parse_date(session) or store.latest_scan_date()
        if scan_date is None:
            _fail("no scans stored yet: run `scan run` or `scan db import` first")
        scan = store.load_scan(scan_date)
    if scan.empty:
        _fail(
            f"no scan stored for {scan_date.isoformat()}: run `scan run --date "
            f"{scan_date.isoformat()}` or `scan db import`"
        )

    generated_at = now_utc()
    options = {
        "scan_date": scan_date,
        "generated_at": generated_at,
        "top_n": top_n,
        "long_only": show_long_only,
    }
    typer.echo(render_markdown(scan, **options))
    if write:
        for path in write_reports(scan, env.output_dir, **options):
            typer.echo(f"Wrote {path}", err=True)

    if send_telegram:
        with _http_client() as client:
            notifier = _notifier(env, client)
            if notifier is None:
                _fail("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set")
            text = build_alert(
                select_top(scan, top_n, show_long_only),
                scan_date=scan_date,
                sent_at=generated_at,
                long_only=show_long_only,
                scanned=len(scan),
            )
            try:
                notifier.send(text)
            except HttpError as error:
                _fail(f"Telegram alert failed: {error}")
        typer.echo("Telegram alert sent.", err=True)


# --- scan size ------------------------------------------------------------------


def _fetch_eurusd(env: EnvSettings) -> tuple[float, date]:
    """Latest daily EUR/USD close from the price provider (used when --eurusd is omitted)."""
    today = now_utc().date()
    try:
        bars = _price_provider(env).daily_bars([FX_TICKER], today - timedelta(days=10), today)
    except ProviderError as error:
        _fail(f"could not fetch EUR/USD ({error}); pass the rate with --eurusd")
    quotes = bars.dropna(subset=["close"]).sort_values("date")
    if quotes.empty:
        _fail("could not fetch EUR/USD; pass the rate with --eurusd")
    last = quotes.iloc[-1]
    return float(last["close"]), last["date"].date()


def _size_rows(result: PositionSize) -> list[str]:
    shares = f"{result.shares:,.4f}".rstrip("0").rstrip(".") if result.fractional else (
        f"{int(result.shares):,}"
    )  # fmt: skip
    return [
        shares,
        f"${result.position_usd:,.2f} / EUR {result.position_eur:,.2f}",
        f"${result.total_cost_usd:,.2f} / EUR {result.total_cost_eur:,.2f}",
        f"EUR {result.fees_eur:,.2f}",
        f"EUR {result.max_loss_eur:,.2f} / ${result.max_loss_usd:,.2f}",
        f"{result.risk_pct_of_pot:.2f}%",
        {"risk": "risk budget", "pot": "pot size (no leverage)", "none": "-"}[result.limited_by],
    ]


@app.command()
def size(
    pot: float = typer.Option(..., "--pot", help="Trading pot in EUR."),
    entry: float = typer.Option(..., "--entry", help="Entry price in USD."),
    stop: float = typer.Option(..., "--stop", help="Stop-loss price in USD (below entry)."),
    risk: float | None = typer.Option(
        None, "--risk", help="Risk per trade as % of the pot (default: config, 1%)."
    ),
    eurusd: float | None = typer.Option(
        None, "--eurusd", help="USD per 1 EUR. Fetched from Yahoo if omitted."
    ),
    convert_eur: bool | None = typer.Option(
        None,
        "--convert-eur/--usd-balance",
        help="Whether the trade converts EUR<->USD (FX fee applies) or uses a USD balance.",
    ),
    fee_min: float | None = typer.Option(
        None, "--fee-min", help="Minimum commission per side in EUR (default: config)."
    ),
    fee_pct: float | None = typer.Option(
        None, "--fee-pct", help="Commission per side as % of the trade (default: config)."
    ),
    fx_fee_pct: float | None = typer.Option(
        None, "--fx-fee-pct", help="FX fee % each way; overrides --convert-eur/--usd-balance."
    ),
    fractional: bool = typer.Option(
        False, "--fractional", help="Also show the fractional-share size (market orders)."
    ),
) -> None:
    """Position size for a long share trade: whole shares, cost, max loss and risk %.

    This is a calculator. It does not place, prepare or send any order.
    """
    env, config = _load()
    sizing = config.sizing
    converting = sizing.assume_eur_conversion if convert_eur is None else convert_eur
    default_fx = sizing.fx_fee_pct_eur_conversion if converting else sizing.fx_fee_pct_usd_balance
    fees = FeeModel(
        commission_min_eur=sizing.commission_min_eur if fee_min is None else fee_min,
        commission_pct=sizing.commission_pct if fee_pct is None else fee_pct,
        fx_fee_pct=default_fx if fx_fee_pct is None else fx_fee_pct,
    )
    risk_pct = sizing.risk_pct if risk is None else risk

    rate_note = "as given"
    if eurusd is None:
        eurusd, rate_date = _fetch_eurusd(env)
        rate_note = f"Yahoo, {rate_date.isoformat()}"

    def compute(use_fractional: bool) -> PositionSize:
        return size_position(
            pot_eur=pot,
            entry=entry,
            stop=stop,
            eurusd=eurusd,
            fees=fees,
            risk_pct=risk_pct,
            fractional=use_fractional,
            fractional_decimals=sizing.fractional_decimals,
            max_risk_pct_warning=sizing.max_risk_pct_warning,
        )

    try:
        results = [compute(False)] + ([compute(True)] if fractional else [])
    except ValueError as error:
        _fail(str(error))

    whole = results[0]
    funding = "converting EUR" if converting else "from USD balance"
    typer.echo("Position size (long, shares only; calculator, no order is placed)")
    typer.echo(
        f"  Pot EUR {pot:,.2f} | risk {risk_pct:g}% = EUR {whole.risk_budget_eur:,.2f} | "
        f"entry ${entry:,.2f} | stop ${stop:,.2f} | EUR/USD {eurusd:.4f} ({rate_note})"
    )
    typer.echo(
        f"  Fees: commission max(EUR {fees.commission_min_eur:g}, {fees.commission_pct:g}%) "
        f"per side, FX {fees.fx_fee_pct:g}% each way ({funding})"
    )
    typer.echo("")

    labels = [
        "Shares",
        "Position",
        "Total cost incl. entry fees",
        "Fees (entry + exit at stop)",
        "Max loss at stop",
        "Risk as % of pot",
        "Limited by",
    ]
    columns = [_size_rows(result) for result in results]
    headers = ["Whole shares (limit order)", "Fractional (market order)"][: len(results)]
    widths = [max(len(header), *(len(cell) for cell in column)) for header, column in
              zip(headers, columns, strict=True)]  # fmt: skip
    typer.echo(
        "  " + " " * 29 + "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    )
    for index, label in enumerate(labels):
        cells = "  ".join(
            column[index].ljust(width) for column, width in zip(columns, widths, strict=True)
        )
        typer.echo(f"  {label:<29}{cells}".rstrip())
    typer.echo("")

    seen: list[str] = []
    for result in results:
        kind = "fractional" if result.fractional else "whole"
        for warning in result.warnings:
            # Warnings that apply to every variant are printed once, without a prefix.
            shared = all(warning in other.warnings for other in results)
            text = warning if shared else f"[{kind}] {warning}"
            if text not in seen:
                seen.append(text)
                _warn(f"  ! {text}")
    typer.echo(
        "  Note: a stop is not a guarantee. A gap through the stop price loses more than shown."
    )


# --- scan universe / db / intraday ----------------------------------------------


@universe_app.command("refresh")
def universe_refresh() -> None:
    """Download the current Nasdaq-100 constituents and update the universe CSV."""
    _, config = _load()
    try:
        diff = refresh_universe(config.scan.universe_file)
    except UniverseError as error:
        _fail(str(error))
    typer.echo(f"{config.scan.universe_file}: {diff.count} tickers")
    typer.echo(f"  added:   {', '.join(diff.added) or '-'}")
    typer.echo(f"  removed: {', '.join(diff.removed) or '-'}")
    if diff.added or diff.removed:
        typer.echo("Review the diff and commit the file to version the change.")


@db_app.command("import")
def db_import(
    directory: Path | None = typer.Argument(
        None,
        help="Folder with scan parquet files (default: <output dir>/scans). "
        "For history from GitHub Actions use: data-branch/scans",
    ),
) -> None:
    """Load exported daily scans (parquet) into the local DuckDB database."""
    env, _ = _load()
    folder = directory or env.output_dir / "scans"
    files = sorted(folder.glob("*.parquet")) if folder.is_dir() else []
    if not files:
        typer.echo(f"No parquet files in {folder}: nothing to import.")
        return
    with ScanStore(env.db_path) as store:
        rows = store.import_parquet(files)
        dates = store.scan_dates()
    typer.echo(
        f"Imported {rows} rows from {len(files)} files into {env.db_path}. "
        f"History now covers {len(dates)} sessions ({dates[0]} to {dates[-1]})."
    )


@intraday_app.command("save")
def intraday_save(
    days: int = typer.Option(
        5, "--days", min=1, max=MAX_DAYS, help="Calendar days to look back for unsaved sessions."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite sessions already saved."),
) -> None:
    """Save 5-minute bars for the scan's top tickers (plus NVDA, QQQ) for later backtests."""
    env, config = _load()
    with ScanStore(env.db_path) as store:
        try:
            outcome = save_intraday(
                prices=_price_provider(env),
                store=store,
                config=config.intraday,
                output_dir=env.output_dir,
                now=now_utc(),
                days=days,
                force=force,
                close_buffer_minutes=config.scan.close_buffer_minutes,
            )
        except (ProviderError, ValueError) as error:
            _fail(str(error))
    typer.echo(f"Tickers ({len(outcome.tickers)}): {', '.join(outcome.tickers)}")
    for path in outcome.written:
        typer.echo(f"Wrote {path}")
    if outcome.already_saved:
        typer.echo(f"Already saved: {', '.join(day.isoformat() for day in outcome.already_saved)}")
    if not outcome.written and not outcome.already_saved:
        typer.echo("No finished sessions with intraday bars in that window.")
    typer.echo(f"Done at {format_athens(now_utc())}.")
