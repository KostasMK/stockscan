"""Daily report: a Markdown summary of the top candidates and a CSV of every ticker."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stockscan.scoring import COMPONENT_LABELS, COMPONENTS
from stockscan.timeutil import format_athens, session_close_utc, to_athens

MISSING = "n/a"
DISCLAIMER = "Research output for my own learning, not investment advice. No orders are placed."


def _is_missing(value: Any) -> bool:
    return value is None or bool(pd.isna(value))


def fmt_price(value: Any) -> str:
    return MISSING if _is_missing(value) else f"${value:,.2f}"


def fmt_pct(value: Any, decimals: int = 1) -> str:
    """Signed percentage, e.g. ``+4.2%``."""
    return MISSING if _is_missing(value) else f"{value:+.{decimals}f}%"


def fmt_plain_pct(value: Any, decimals: int = 1) -> str:
    """Unsigned percentage for quantities that are never negative, e.g. ATR%."""
    return MISSING if _is_missing(value) else f"{value:.{decimals}f}%"


def fmt_ratio(value: Any) -> str:
    return MISSING if _is_missing(value) else f"{value:.1f}x"


def fmt_session(scan_date: date) -> str:
    return scan_date.strftime("%a %d %b %Y")


def select_top(scan: pd.DataFrame, top_n: int, long_only: bool) -> pd.DataFrame:
    """The rows to show: best rank first, optionally only stocks that closed up."""
    ordered = scan.sort_values("rank")
    if long_only:
        ordered = ordered[ordered["direction"] == "up"]
    return ordered.head(top_n).reset_index(drop=True)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows]
    return lines


def _news_line(row: dict[str, Any]) -> str | None:
    if _is_missing(row.get("news_headline")) and _is_missing(row.get("news_score")):
        return None
    parts = []
    if not _is_missing(row.get("news_score")):
        articles = row.get("news_articles")
        count = "" if _is_missing(articles) else f", {int(articles)} articles"
        parts.append(f"{row.get('news_label')} ({row['news_score']:+.2f}{count})")
    if not _is_missing(row.get("news_headline")):
        title = str(row["news_headline"]).replace("[", "(").replace("]", ")")
        url = row.get("news_url")
        parts.append(title if _is_missing(url) else f"[{title}]({url})")
    return "- News: " + ": ".join(parts)


def render_markdown(
    scan: pd.DataFrame,
    *,
    scan_date: date,
    generated_at: datetime,
    top_n: int,
    long_only: bool,
    missing: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """Markdown report for one scan. ``scan`` holds every ticker of that date."""
    shown = select_top(scan, top_n, long_only)
    close_athens = to_athens(session_close_utc(scan_date)).strftime("%H:%M")
    mode = "long-only (closed up)" if long_only else "all directions"
    version = (
        scan["scoring_version"].iloc[0] if "scoring_version" in scan and len(scan) else MISSING
    )

    lines = [
        f"# Hot stocks: {fmt_session(scan_date)}",
        "",
        f"- Session: {scan_date.isoformat()}, US close 16:00 New York = {close_athens} Athens",
        f"- Generated: {format_athens(generated_at)} (Europe/Athens)",
        f"- Scanned: {len(scan)} tickers, showing top {len(shown)}, {mode}",
        f"- Scoring version: `{version}` (weights and caps: `config/config.toml`)",
        "",
    ]

    if shown.empty:
        lines += ["No stocks matched today.", ""]
    else:
        headers = [
            "Rank",
            "Ticker",
            "Dir",
            "Price",
            "Day %",
            "Rel vol",
            "Gap %",
            "ATR %",
            "vs SMA20",
            "vs SMA50",
            "From 52w high",
            "From 52w low",
            "Score",
        ]
        rows = [
            [
                str(int(row["rank"])),
                f"**{row['ticker']}**",
                str(row["direction"]),
                fmt_price(row["raw_close"]),
                fmt_pct(row["day_change_pct"]),
                fmt_ratio(row["rel_volume"]),
                fmt_pct(row["gap_pct"]),
                fmt_plain_pct(row["atr_pct"]),
                fmt_pct(row["close_vs_sma20_pct"]),
                fmt_pct(row["close_vs_sma50_pct"]),
                fmt_pct(row["dist_52w_high_pct"]),
                fmt_pct(row["dist_52w_low_pct"]),
                f"{row['score']:.1f}",
            ]
            for row in shown.to_dict("records")
        ]
        lines += _table(headers, rows)
        lines += ["", "## Why they ranked", ""]
        for row in shown.to_dict("records"):
            breakdown = " + ".join(
                f"{COMPONENT_LABELS[name]} {row[f'c_{name}']:.1f}" for name in COMPONENTS
            )
            lines.append(f"### {int(row['rank'])}. {row['ticker']} ({row['direction']})")
            lines.append("")
            lines.append(f"{row['reason']}")
            lines.append("")
            lines.append(f"- Score {row['score']:.1f} = {breakdown}")
            news = _news_line(row)
            if news:
                lines.append(news)
            lines.append("")

    extra = list(notes)
    if missing:
        extra.append(f"No usable bar for this session ({len(missing)}): {', '.join(missing)}")
    if extra:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in extra]
        lines.append("")

    lines += ["---", "", DISCLAIMER, ""]
    return "\n".join(lines)


def write_reports(
    scan: pd.DataFrame,
    output_dir: Path,
    *,
    scan_date: date,
    generated_at: datetime,
    top_n: int,
    long_only: bool,
    missing: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Write ``reports/YYYY-MM-DD.md`` (top candidates) and ``.csv`` (every ticker)."""
    directory = output_dir / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"{scan_date.isoformat()}.md"
    csv_path = directory / f"{scan_date.isoformat()}.csv"

    markdown = render_markdown(
        scan,
        scan_date=scan_date,
        generated_at=generated_at,
        top_n=top_n,
        long_only=long_only,
        missing=missing,
        notes=notes,
    )
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    table = scan.sort_values("rank").copy()
    float_columns = table.select_dtypes("float").columns
    table[float_columns] = table[float_columns].round(4)
    table.to_csv(csv_path, index=False, lineterminator="\n")
    return markdown_path, csv_path
