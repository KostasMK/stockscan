# stockscan

A personal daily "hot stocks" scanner and quant research tool for US equities.

Every evening after the US close it scans the Nasdaq-100, ranks the stocks that behaved
unusually (volume, gaps, big moves, 52-week extremes), stores the numbers for later
analysis, writes a report and sends a Telegram alert. It also includes a position-sizing
calculator that puts risk rules before returns.

**This is a research and alerting tool only. It never places orders and has no broker
integration, by design.** Nothing here is investment advice.

```
  #  Ticker      Price     Day RelVol     Gap   ATR Score  Reason
  1  CRWV       $60.77  +24.7%   2.3x  +14.8% 10.2%  58.2  +24.7% on the day (2.4 ATR); gap +14.8%; volume 2.3x average
  2  SNDK      $977.69  +13.7%   1.5x   +7.9% 11.0%  43.7  gap +7.9%; +13.7% on the day (1.2 ATR); -16.3% vs 20d SMA
  3  MPWR    $1,478.69   +8.9%   0.9x   +5.5%  6.2%  42.1  gap +5.5%; +8.9% on the day (1.4 ATR); +18.6% vs 20d SMA
```

## What it does

| Step | Detail |
|---|---|
| Universe | Nasdaq-100 tickers in a versioned CSV (`data/universe/nasdaq100.csv`), refreshable from Wikipedia |
| Data | Daily OHLCV from Yahoo Finance via `yfinance`, behind a swappable interface |
| Metrics | Relative volume, gap %, day change %, ATR(14) and ATR %, close vs 20/50-day SMA, distance from 52-week high/low |
| Ranking | Transparent weighted score, weights in `config/config.toml`, every component shown |
| News | Alpha Vantage sentiment and top headline for the top 10 only (cached, capped) |
| Storage | DuckDB, one row per ticker per day; history kept as parquet on the `data` branch |
| Output | Markdown + CSV report, Telegram message with the top 10 |
| Schedule | GitHub Actions at 21:30 UTC on weekdays, or `uv run scan run` by hand |

Times are shown in Europe/Athens; everything is computed and stored in New York session
dates and UTC timestamps.

## Setup from scratch

You need [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/). uv installs
Python 3.12 for the project by itself.

```bash
# 1. Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
#    macOS / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code and install dependencies
git clone https://github.com/KostasMK/stockscan.git
cd stockscan
uv sync

# 3. Create your local secrets file (it is gitignored)
cp .env.example .env        # PowerShell: Copy-Item .env.example .env

# 4. First run. Works without any keys: news and Telegram are skipped with a warning.
uv run scan run
```

Results land in `out/reports/` (Markdown and CSV) and in `data/stockscan.duckdb`.

### Telegram setup

1. In Telegram, open a chat with **@BotFather** and send `/newbot`.
2. Choose a display name, then a username ending in `bot` (for example `kostas_scan_bot`).
3. BotFather replies with a token like `123456789:AAH...`. Put it in `.env` as
   `TELEGRAM_BOT_TOKEN`. Treat it like a password.
4. Open a chat with your new bot and press **Start** (or send it any message). A bot
   cannot message you until you have written to it first.
5. Get your chat ID: open this URL in a browser, with your token in place of `<TOKEN>`:

   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```

   Find `"chat":{"id":123456789,...}` in the response. That number is `TELEGRAM_CHAT_ID`.
   If the result is empty, send the bot another message and reload.
6. Test it: `uv run scan report --send-telegram` (after at least one `scan run`).

### Alpha Vantage setup (optional)

Get a free key at <https://www.alphavantage.co/support/#api-key> and put it in `.env` as
`ALPHAVANTAGE_API_KEY`. The free tier allows roughly 25 calls per day, so the scanner
asks for news on the top 10 only, caches every answer under `.cache/` and enforces its
own daily cap (`news.max_calls_per_day`). If Alpha Vantage refuses a call (rate limit, or
the endpoint is not included in your plan), the scan carries on without news and says so
in the report.

## Commands

```bash
uv run scan run                       # today's scan: fetch, compute, store, report, alert
uv run scan run --long-only           # report/alert show only stocks that closed up
uv run scan run --date 2026-11-09     # re-run a past session (uses only data up to that day)
uv run scan run --force               # redo a session that is already stored

uv run scan report                    # print the latest stored report
uv run scan report --date 2026-11-11 --top 5 --long-only
uv run scan report --send-telegram    # resend the alert for a stored session

uv run scan size --pot 1500 --entry 173.73 --stop 166 --fractional

uv run scan universe refresh          # update the Nasdaq-100 CSV, prints added/removed
uv run scan db import data-branch/scans   # load history exported by GitHub Actions
uv run scan intraday save             # archive 5-minute bars for later backtests
```

`scan run` does nothing if the latest finished session is already stored, so running it
on a weekend or holiday never sends a duplicate alert. If you run it while the US market
is open, it scans the previous session: today's half-formed bar is ignored.

## The score

```
score = 100 * sum(weight * component) / sum(weights)
```

Each component is a metric squashed into the range 0 to 1:

| Component | Metric | 0 at | 1 at (cap) | Default weight |
|---|---|---|---|---|
| `rel_volume` | volume / average of the previous 20 sessions | 1x | 5x | 0.30 |
| `move` | \|day change\| in ATRs | 0 | 3 ATR | 0.25 |
| `gap` | \|open vs previous close\| | 0% | 5% | 0.20 |
| `extreme` | closeness to the 52-week high or low | 10% away | at the extreme | 0.15 |
| `stretch` | \|close vs 20-day SMA\| | 0% | 10% | 0.10 |

The score is direction-neutral: it measures how unusual the day was, and a `direction`
label (up/down) tells you which way. Use `--long-only`, or set `scan.long_only = true` in
the config, to see only stocks that closed up. Every stored row keeps its five
contributions (`c_*` columns, adding up to the score) and a one-line reason, and a
`scoring_version` hash so history can be split by the rules that produced it.

Weights and caps live in [`config/config.toml`](config/config.toml). Change them there,
never in code.

## Position sizing

```
$ uv run scan size --pot 1500 --entry 173.73 --stop 166 --fractional

  Pot EUR 1,500.00 | risk 1% = EUR 15.00 | entry $173.73 | stop $166.00 | EUR/USD 1.1059
  Fees: commission max(EUR 1, 0.25%) per side, FX 1% each way (converting EUR)

                               Whole shares (limit order)  Fractional (market order)
  Shares                       1                           1.292
  Position                     $173.73 / EUR 157.09        $224.46 / EUR 202.96
  Total cost incl. entry fees  $176.57 / EUR 159.66        $227.81 / EUR 205.99
  Fees (entry + exit at stop)  EUR 5.07                    EUR 5.97
  Max loss at stop             EUR 12.06 / $13.34          EUR 15.00 / $16.59
  Risk as % of pot             0.80%                       1.00%

  ! Fees (EUR 5.07) are 42% of the maximum loss: this trade is expensive for its size.
```

- Default risk is 1% of the pot; anything above 2% prints a warning.
- Max loss includes commissions and FX fees on the way in and out.
- No leverage: the position can never cost more than the pot.
- `--usd-balance` assumes you trade from USD you already hold (no FX fee);
  `--convert-eur` (the default) applies the FX fee both ways.
- Fee defaults are in `[sizing]` in the config; `--fee-min`, `--fee-pct` and
  `--fx-fee-pct` override them for one calculation.
- `--eurusd` sets the rate; without it the latest Yahoo quote is used.

It is a calculator. It does not create or send orders.

## Automation with GitHub Actions

`.github/workflows/daily-scan.yml` runs at 21:30 UTC on weekdays (after the US close in
both winter and summer time). Each run:

1. rebuilds a DuckDB database from the history on the `data` branch,
2. runs the scan and sends the alert,
3. archives 5-minute bars for the top tickers plus NVDA and QQQ,
4. commits `scans/`, `reports/` and `intraday/` to the `data` branch.

One-time setup:

```bash
gh secret set TELEGRAM_BOT_TOKEN     # each command prompts for the value
gh secret set TELEGRAM_CHAT_ID
gh secret set ALPHAVANTAGE_API_KEY
```

Secrets live only in GitHub Actions secrets and your local `.env`. They are never
committed. You can also start a run by hand: **Actions > Daily scan > Run workflow**.

To analyse the accumulated history on your own machine:

```bash
git fetch origin data
git worktree add data-branch data        # first time only
git -C data-branch pull                  # afterwards
uv run scan db import data-branch/scans
```

```python
import duckdb

con = duckdb.connect("data/stockscan.duckdb", read_only=True)
con.sql("""
    SELECT scan_date, ticker, direction, score, rel_volume, gap_pct
    FROM daily_scan
    WHERE rank <= 3
    ORDER BY scan_date DESC, rank
""").show()
```

## No look-ahead bias

A signal for day T may only use data available at the close of day T. This is enforced
in three places:

- `timeutil.last_completed_session_cutoff` decides which session is finished; every bar
  after it is dropped before any calculation.
- Every metric uses only backward-looking operations (`shift` with a positive lag,
  trailing windows).
- `tests/test_no_lookahead.py` recomputes all metrics after corrupting or deleting every
  bar after T and requires identical results. It also proves the check fails for three
  deliberately leaky metrics. `tests/test_pipeline.py` repeats the check on the whole scan.

News follows the same rule: only articles published before the session close count.

## Development

```bash
uv run pytest            # all tests (no network needed)
uv run ruff check .      # lint
uv run ruff format .     # format
```

Project layout, conventions and gotchas for contributors (human or AI) are in
[`CLAUDE.md`](CLAUDE.md). Planned work is in [`ROADMAP.md`](ROADMAP.md).
