# stockscan: guide for AI assistants and contributors

Read this file first. It is meant to let a new session continue without the owner
re-explaining anything. Keep it updated when behaviour, structure or status changes.

## Purpose

Personal quant research project. The owner lives in the Europe/Athens timezone and
trades US shares (no leverage, small pot) through a retail broker. Goals, in
order: (1) learn quant/data engineering and have a strong portfolio project, (2) test
ideas on data before trading them, (3) maybe find small edges. Risk rules matter more
than returns.

Phase 1 is a daily "hot stocks" scanner for the Nasdaq-100 with a Telegram alert, a
stored history, and a position-sizing calculator. Phases 2 to 4 are in `ROADMAP.md`.

## Hard rules (non-negotiable)

1. **Research and alerting only.** Never add broker integration, order placement, order
   "preparation", or any auto-trade feature. `tests/test_cli.py` pins the command list.
2. **No look-ahead bias.** A signal for day T uses only data available at the close of
   day T. New metrics go into `metrics.compute_metrics` so `tests/test_no_lookahead.py`
   covers them automatically. Never use `shift(-n)`, `center=True`, full-series
   statistics, `bfill` or `interpolate` in signal code.
3. **Secrets never enter the repo.** Keys live in `.env` (gitignored) or GitHub Actions
   secrets. Never log a URL that contains a token (see "Gotchas").
4. **Every external API call is rate-limited and cached**, and Alpha Vantage is never
   called in a loop over the universe (top candidates only).
5. Type hints everywhere, small pure functions for calculations, a unit test for every
   metric, `ruff` clean, tunables in `config/config.toml` rather than in code.

## How to run and test

`uv` manages Python 3.12 and the virtualenv. Run everything from the repository root
(config and universe paths are relative to it).

```bash
uv sync                      # install
uv run pytest                # 277 tests, no network, ~20 s
uv run ruff check . && uv run ruff format --check .
uv run scan run              # live scan (works without keys: news/Telegram are skipped)
uv run scan --help
```

On the owner's Windows machine `uv` and `gh` are in `%USERPROFILE%\.local\bin`; a fresh
shell may need that on PATH. Commit in small logical steps; CI (`.github/workflows/ci.yml`)
runs ruff and pytest on every push to `main`.

## Architecture

```
src/stockscan/
  cli.py          typer app: run | report | size | universe refresh | db import | intraday save
  config.py       EnvSettings (.env: secrets, paths) + AppConfig (config/config.toml: tunables)
  timeutil.py     NY/UTC/Athens; last_completed_session_cutoff = the look-ahead boundary
  universe.py     load CSV; refresh from Wikipedia with validation and diff
  http.py         RateLimiter, request_with_retry (redacts secrets), JsonDiskCache, DailyBudget
  providers/
    base.py               PriceProvider / NewsProvider protocols, NewsSentiment, errors
    yfinance_provider.py  YFinanceProvider (batch download, retries) + CachedPriceProvider
    alphavantage.py       AlphaVantageNews + summarize_news (pure)
  bars.py         raw daily bar schema, clean_daily_bars (cutoff, halts, bad rows), adjust_ohlc
  metrics.py      one pure function per metric + compute_metrics / compute_universe_metrics
  scoring.py      components in [0,1] -> weighted score, contributions, direction, reason, rank
  pipeline.py     build_scan (pure) and run_scan (orchestration); attach_news
  storage.py      ScanStore: DuckDB table daily_scan, parquet export/import
  report.py       Markdown + CSV, select_top (long-only filter), number formatting
  telegram.py     build_alert (HTML), TelegramNotifier (plain Bot API calls)
  sizing.py       FeeModel, size_position (whole/fractional), warnings
  intraday.py     5-minute bar archive for Phase 2
config/config.toml            score weights/caps, news limits, fee defaults, intraday settings
data/universe/nasdaq100.csv   versioned universe (ticker, name, sector)
tests/                        one file per module; conftest.py builds synthetic data
```

Data flow of `scan run` (`pipeline.run_scan`):

```
universe CSV -> PriceProvider.daily_bars (420 calendar days)
  -> clean_daily_bars(cutoff)      drop bars after the last finished session, halts, bad rows
  -> adjust_ohlc                   dividend-adjust OHLC, keep raw_close for display
  -> compute_universe_metrics      all sessions, per ticker
  -> rows of the scan date -> score_universe -> rank
  -> skip if that date is already stored (unless --force)
  -> attach_news for the shown top-N only
  -> ScanStore.save_scan -> export parquet -> write_reports -> Telegram alert
```

All collaborators are injected, so tests run the full pipeline with fakes.

### Storage model

- `daily_scan` (DuckDB): primary key `(scan_date, ticker)`, every ticker every day, with
  prices, all metrics, score contributions `c_*`, `reason`, `direction`, news columns,
  `scoring_version` (hash of weights and caps) and `run_at_utc`. Schema: `storage.SCAN_SCHEMA`.
- The DuckDB file (`data/stockscan.duckdb`, gitignored) is a rebuildable cache. The
  source of truth is the orphan **`data` branch**:
  `scans/YYYY-MM-DD.parquet`, `reports/YYYY-MM-DD.{md,csv}`, `intraday/5m/YYYY-MM-DD.parquet`.
- The daily Action checks the `data` branch out into `data-branch/` (gitignored on
  `main`), imports `scans/` into a fresh DuckDB, runs with `STOCKSCAN_OUTPUT_DIR=data-branch`
  and commits the new files there. Locally the default output folder is `out/`.
- Adding a column: append it to `SCAN_SCHEMA` (old parquet files import with NULLs) and
  add an `ALTER TABLE` path for existing local databases, or tell the owner to delete the
  local file and run `scan db import`.

## Conventions and decisions

- **Dates and times:** session dates are New York dates stored as naive midnight
  timestamps (`date` / `scan_date`); instants are UTC; Athens only for display
  (`timeutil.format_athens`). Naive `datetime`s are rejected.
- **Prices:** metrics use dividend-adjusted OHLC (`open/high/low/close`); `raw_close` is
  the unadjusted close and is what reports and alerts show. Yahoo OHLC and volume are
  already split-adjusted.
- **Relative volume** compares today with the mean of the *previous* 20 sessions
  (today excluded). **ATR** is Wilder's, seeded with a simple mean. **52-week range**
  uses 252 sessions with at least 60 (newer listings get a shorter range, not NaN).
- **Score** is direction-neutral; `direction` is up/down/flat from the day change.
  `--long-only` (or `scan.long_only`) filters the report, the alert and the news lookups
  only. Storage always keeps everything.
- **Scan date** is the latest bar on or before the cutoff, so there is no holiday
  calendar. A stored date is skipped, which makes weekend/holiday runs no-ops.
  `scan run --date` re-runs history and never sends Telegram.
- **News window** is the 24 hours ending at 16:00 New York of the scan date. Articles
  after the close are ignored even if the API returns them.
- **Failure policy:** missing tickers are reported and skipped; coverage below 50% or a
  latest bar older than 5 days aborts the run (`DataQualityError`); news problems add a
  note; a Telegram failure exits non-zero *after* results are stored.
- **Sizing:** long only (stop must be below entry). Fees: commission
  `max(min_eur, pct)` per side plus FX fee both ways. Defaults model Revolut and assume
  EUR conversion (1% FX); the owner will adjust `[sizing]` once they know their plan's fees.
- **Tests:** no network. HTTP is mocked with `respx`; yfinance through an injected
  downloader; an autouse fixture blanks all secrets so a real `.env` can never leak into
  tests. `filterwarnings = error` is on.
- **Commits:** small and logical, message explains why. This repo's local git config
  uses the owner's GitHub noreply address so the personal email stays out of a public repo.

## Gotchas (learned the hard way)

- **yfinance returns today's half-formed bar** during the US session. The cutoff in
  `clean_daily_bars` is what protects against it; do not remove it. Its `end` argument is
  exclusive; columns are a `(ticker, field)` MultiIndex; failed tickers come back as
  all-NaN columns, not errors.
- **Yahoo may throttle or block** datacenter IPs (GitHub runners) without warning. The
  provider retries with backoff and the run fails loudly if nothing comes back. If this
  becomes frequent, implement another `PriceProvider`; nothing else needs to change.
- **Alpha Vantage reports errors with HTTP 200** and an `Information` / `Note` /
  `Error Message` field. A probe with a different key answered "This is a premium
  endpoint" for NEWS_SENTIMENT, so the owner's free key may not include it. The code
  handles that (news is skipped with a note), but it is **unverified with the owner's key**.
  Alpha Vantage as a backup *price* source is not implemented: 25 calls/day cannot cover
  101 tickers.
- **httpx logs full URLs at INFO**, and the Telegram token and the Alpha Vantage key are
  part of the URL. `cli.main` forces the httpx logger to WARNING and
  `request_with_retry` redacts secrets from errors. Keep both.
- **DuckDB 2.x** refuses new work on a connection while a result is partly fetched
  (`fetchone()` leaves it open). Use `ScanStore._run` (drains with `fetchall`) or `.df()`.
- **pandas 3:** `groupby(...).first()` skips NaN (use `head(1)`); datetime columns are not
  always nanosecond resolution; `DataFrame.round` warns on datetime columns;
  `read_html` needs `flavor="lxml"` pinned or it tries to import html5lib.
- **Wikipedia moved the constituents table** to `List_of_NASDAQ-100_companies`; the
  refresh tries both pages and validates size and symbols before overwriting the CSV.
- **`astral-sh/setup-uv` has no moving major tag**: pin an exact version (`@v10.3.3`).
- **GitHub cron** can start late or occasionally skip a run, and GitHub disables
  scheduled workflows in repositories with no activity for 60 days. The intraday job
  looks back 5 days to heal short gaps.
- **Historical re-runs are not point-in-time for adjustments:** `scan run --date X` uses
  Yahoo's current dividend adjustments, so values can differ slightly from the row stored
  on day X. The stored row is the record; do not overwrite history casually (`--force`).
- **Survivorship:** the universe CSV is today's index. For backtests use the tickers
  present in `daily_scan` on each date.
- **Half-days:** the news window assumes a 16:00 close; on the few 13:00 closes it
  includes three extra hours. Harmless for alerts, worth knowing for analysis.
- **Windows:** consoles default to a legacy code page; `cli.main` reconfigures stdout to
  UTF-8. `.gitattributes` forces LF so `ruff format --check` agrees between Windows and CI.

## Current status

Phase 1 is implemented and tested (277 tests passing, ruff clean), and a live
scan against Yahoo works end to end without keys.

Open items:

- [ ] **Live check with real keys.** The owner is creating a Telegram bot and an Alpha
      Vantage key. Then: put them in `.env`, run `uv run scan run --force`, confirm the
      Telegram message arrives and whether NEWS_SENTIMENT works on the free tier. Add the
      same three values as GitHub Actions secrets (`gh secret set ...`, see README).
- [ ] Owner to confirm Revolut fees and adjust `[sizing]` in `config/config.toml`.
- [ ] Watch the first week of scheduled runs (Actions tab, `data` branch) for Yahoo
      throttling on GitHub runners.

Next: Phase 1.5 acceptance checks, then Phase 2 (see `ROADMAP.md`).
