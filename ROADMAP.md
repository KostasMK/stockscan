# Roadmap

Phase 1 (daily scanner, alerting, position sizing, intraday archive) is built. Its status
and open items are tracked in `CLAUDE.md` under "Current status".

Rules that apply to every phase:

- **Research and alerting only.** No broker integration, no order placement, no
  "auto-trade" or "one-click trade" features. Not now, not later.
- **No look-ahead bias.** Every new signal, strategy or statistic gets a test that fails
  if future data can influence a past decision.
- Tests for every calculation, type hints throughout, `ruff` clean, tunables in
  `config/config.toml`.
- Honest numbers: fees and slippage always on, sample sizes always shown.

---

## Phase 1.5: keep the data flowing (ongoing, no new code expected)

The Phase 2 backtest needs months of intraday history, and Yahoo only serves about the
last 60 days of 5-minute bars. The daily GitHub Action already archives them
(`scan intraday save`, see `src/stockscan/intraday.py`).

Acceptance criteria:

- [ ] The `data` branch gains one `intraday/5m/YYYY-MM-DD.parquet` and one
      `scans/YYYY-MM-DD.parquet` per trading day with no gaps over 4 consecutive weeks.
      (A missed day inside the 5-day look-back heals itself; a longer outage is a gap.)
- [ ] One live end-to-end run with real Telegram and Alpha Vantage keys has been checked
      by eye (see "Current status" in `CLAUDE.md`).
- [ ] Optional one-off: `uv run scan intraday save --days 55` on a local machine to
      backfill NVDA and QQQ for the last ~8 weeks, committed to the `data` branch.

---

## Phase 2: backtester, first strategy = 30-minute opening-range breakout

Goal: test an idea on data before any money is involved, and learn how backtests lie.

### Scope

1. **Engine.** Start with a small custom, bar-by-bar engine (easier to audit for
   look-ahead than a vectorised one, and the learning goal matters). `vectorbt` may be
   added later as a cross-check, not as the first implementation.
   - Decisions at bar *t* use bars up to and including *t* only; fills happen at the
     **next** bar's open (or at the stop/target price if that bar trades through it).
   - If a bar touches both stop and target, assume the stop was hit first (pessimistic).
2. **Costs.** Reuse `sizing.FeeModel` for commissions and FX. Add slippage as a
   configurable number of basis points per side, applied against the trade.
3. **Position sizing.** Reuse `sizing.size_position` (1% risk, whole shares, no leverage).
4. **Strategy: ORB-30, long only.**
   - Universe for day T+1 = top-N of the scan at the close of day T (already stored in
     `daily_scan`), optionally filtered to `direction = 'up'`.
   - Opening range = high/low of 09:30 to 10:00 New York (first six 5-minute bars).
   - Entry: first 5-minute close above the range high after 10:00, filled next bar open.
   - Stop: range low. Target: a configurable multiple of risk (R). Exit at 15:55 if
     neither is hit. One trade per ticker per day.
5. **Walk-forward testing.** Rolling windows (for example 3 months train, 1 month test).
   Parameters are chosen on the train window only; reported results are the
   concatenated test windows only.
6. **Benchmark.** Buy-and-hold QQQ over the same dates with the same starting pot,
   using the same fee model for its single entry.

### Acceptance criteria

- [ ] `uv run scan backtest orb --from YYYY-MM-DD --to YYYY-MM-DD` runs from the
      archived parquet files only (no network) and writes a Markdown report plus a
      trades parquet file.
- [ ] Report shows, for strategy and QQQ benchmark: total return, max drawdown, number
      of trades, win rate, average win, average loss, expectancy per trade (in R and in
      EUR), exposure, and total fees + slippage paid.
- [ ] Fees and slippage are on by default and cannot be silently disabled; a run with
      zero costs is labelled as such in the report.
- [ ] Look-ahead test for the engine: corrupting every bar after time *t* does not
      change any trade decided at or before *t*.
- [ ] Known-answer tests on synthetic bars: breakout then target, breakout then stop,
      no breakout, stop and target in the same bar, end-of-day exit.
- [ ] Walk-forward report clearly separates in-sample from out-of-sample results.
- [ ] The report prints the sample size and refuses to call anything an "edge" below a
      configurable minimum number of out-of-sample trades (default 100).

### Known traps to handle

- **Survivorship bias:** the universe CSV is today's Nasdaq-100. For backtests, use the
  membership recorded in `daily_scan` on each date (it is point-in-time from the day the
  scanner started), not the current CSV.
- **Data coverage:** intraday bars exist only for the tickers archived each day (top-N
  of that day and the previous session, NVDA, QQQ). The backtest must fail loudly, not
  skip silently, when bars for a required ticker/day are missing.
- **Adjusted prices:** intraday bars are unadjusted; daily metrics are dividend-adjusted.
  Do not mix them inside one calculation.
- Half-days (13:00 close) and the first/last bars of the session need explicit handling.

---

## Phase 3: paper-trading log

Goal: record every signal when it happens and what actually followed, so that live
behaviour can be compared with the backtest.

### Scope

- `scan paper signal`: after the daily scan, write the next day's planned trades
  (ticker, entry rule, stop, target, size from `size_position`, pot at that time) to an
  append-only `paper_signals` table. Signals are immutable once written.
- `scan paper resolve`: on later days, fill in the outcome from archived intraday bars
  using the Phase 2 engine: entry time/price, exit time/price, exit reason, P&L in R and
  EUR including fees and slippage. Unfilled signals are resolved as "no entry".
- Both tables are exported to the `data` branch like the scans; the daily Action runs
  both commands.
- Optional Telegram line: yesterday's signals and how they resolved.

### Acceptance criteria

- [ ] Every signal has exactly one outcome row or is still open; none are lost or duplicated.
- [ ] Signals cannot be edited or deleted through the tool (append-only, enforced by test).
- [ ] A signal is written before the session it refers to opens (timestamp check in test).
- [ ] `scan paper stats` prints win rate, average win/loss, expectancy, max drawdown and
      trade count, side by side with the Phase 2 backtest for the same period.
- [ ] Still no orders: the log records what *would* have happened. No broker connection.

---

## Phase 4: web dashboard

Goal: one place to look at the watchlist, the journal and the statistics.

### Scope

- Read-only web UI over the DuckDB/parquet history. Suggested stack: Streamlit for the
  first version (fast to build, Python only); FastAPI + a small frontend if it outgrows that.
- Pages:
  - **Watchlist:** latest scan with filters (direction, min score, min relative volume),
    score breakdown per ticker, small price chart.
  - **Trade journal:** paper signals and outcomes from Phase 3, with free-text notes.
    A manual "real trade" entry form is allowed (typed in by hand after the fact).
  - **Stats:** win rate, average win/loss, expectancy, max drawdown, equity curve versus
    QQQ, results split by `scoring_version`.
- Runs locally by default (`uv run scan dashboard`). If it is ever hosted, it must be
  behind authentication and contain no secrets.

### Acceptance criteria

- [ ] All three pages work from a fresh clone after `scan db import`.
- [ ] Statistics shown in the dashboard match `scan paper stats` exactly (shared code,
      tested once).
- [ ] Times are displayed in Europe/Athens; stored values stay in New York dates / UTC.
- [ ] No button, link or API anywhere that sends an order or talks to a broker.

---

## Ideas parked for later (not committed)

- More universes (S&P 500) and a liquidity filter.
- Earnings-date flag in the scan (many "hot" days are earnings reactions).
- A separate after-hours headline field (today, news after the close is deliberately
  excluded from a day's record).
- Replace or back up yfinance with a paid/official daily data source.
