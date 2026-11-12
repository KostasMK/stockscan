# stockscan data branch

Generated data only. This branch is written by the "Daily scan" GitHub Action in the
`main` branch and has no code. Do not edit it by hand.

| Path | Content |
|---|---|
| `scans/YYYY-MM-DD.parquet` | One row per ticker for that session: prices, metrics, score, reason, news |
| `reports/YYYY-MM-DD.md` / `.csv` | The daily report (top candidates) and the full table |
| `intraday/5m/YYYY-MM-DD.parquet` | 5-minute bars (UTC) for the top tickers, NVDA and QQQ |

Session dates are New York dates. To load the history into a local DuckDB database, see
"Automation with GitHub Actions" in the README on `main`.
