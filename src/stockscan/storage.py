"""DuckDB storage: one row per ticker per scan date, kept forever for later analysis.

The local database file is a cache of the truth, not the truth itself: the daily job
exports each scan to ``scans/YYYY-MM-DD.parquet`` on the ``data`` branch, and
``import_parquet`` rebuilds or updates a local database from those files.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date
from pathlib import Path
from types import TracebackType
from typing import Self

import duckdb
import pandas as pd

from stockscan.metrics import METRIC_COLUMNS
from stockscan.scoring import CONTRIBUTION_COLUMNS

TABLE = "daily_scan"

# Column order and SQL types of the scan table. Add new columns at the end: old parquet
# files then import with NULLs for them.
SCAN_SCHEMA: list[tuple[str, str]] = [
    ("scan_date", "DATE"),
    ("ticker", "VARCHAR"),
    ("rank", "INTEGER"),
    ("score", "DOUBLE"),
    ("direction", "VARCHAR"),
    ("reason", "VARCHAR"),
    ("raw_close", "DOUBLE"),  # unadjusted close: the price shown to the user
    ("open", "DOUBLE"),  # OHLC below are dividend-adjusted (what the metrics used)
    ("high", "DOUBLE"),
    ("low", "DOUBLE"),
    ("close", "DOUBLE"),
    ("volume", "DOUBLE"),
    *[(column, "DOUBLE") for column in METRIC_COLUMNS],
    *[(column, "DOUBLE") for column in CONTRIBUTION_COLUMNS],
    ("news_score", "DOUBLE"),
    ("news_label", "VARCHAR"),
    ("news_articles", "INTEGER"),
    ("news_headline", "VARCHAR"),
    ("news_url", "VARCHAR"),
    ("scoring_version", "VARCHAR"),
    ("run_at_utc", "TIMESTAMP"),
]
SCAN_COLUMNS = [name for name, _ in SCAN_SCHEMA]


def normalize_scan_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with exactly ``SCAN_COLUMNS`` (missing ones as nulls) and plain dtypes."""
    out = pd.DataFrame(index=frame.index)
    for name, sql_type in SCAN_SCHEMA:
        column = frame[name] if name in frame.columns else pd.Series(pd.NA, index=frame.index)
        if sql_type == "DOUBLE":
            out[name] = pd.to_numeric(column, errors="coerce").astype("float64")
        elif sql_type == "INTEGER":
            out[name] = pd.to_numeric(column, errors="coerce").astype("Int64")
        elif sql_type == "VARCHAR":
            out[name] = column.astype("string")
        elif sql_type == "DATE":
            out[name] = pd.to_datetime(column).dt.normalize()
        else:  # TIMESTAMP, stored as naive UTC
            stamps = pd.to_datetime(column, utc=True)
            out[name] = stamps.dt.tz_convert(UTC).dt.tz_localize(None)
    return out.reset_index(drop=True)


class ScanStore:
    """Thin wrapper around a DuckDB file holding the ``daily_scan`` table."""

    def __init__(self, db_path: Path | None) -> None:
        """Open (and create if needed) the database. ``None`` gives an in-memory database."""
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path) if db_path is not None else ":memory:")
        columns = ", ".join(f'"{name}" {sql_type}' for name, sql_type in SCAN_SCHEMA)
        self._run(
            f'CREATE TABLE IF NOT EXISTS {TABLE} ({columns}, PRIMARY KEY ("scan_date", "ticker"))'
        )

    def _run(self, sql: str, parameters: Sequence[object] | None = None) -> list[tuple]:
        """Execute a statement and drain its result.

        DuckDB refuses further work on a connection while a result is only partly
        fetched, so every statement goes through here (or ``.df()``), never ``fetchone``.
        """
        return self._con.execute(sql, parameters or []).fetchall()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- writing ---------------------------------------------------------------

    def save_scan(self, frame: pd.DataFrame) -> None:
        """Insert a scan, replacing any rows already stored for the same date(s)."""
        staged = normalize_scan_frame(frame)
        if staged.empty:
            return
        if staged.duplicated(subset=["scan_date", "ticker"]).any():
            raise ValueError("scan frame has duplicate (scan_date, ticker) rows")
        casts = ", ".join(f'CAST("{name}" AS {sql_type})' for name, sql_type in SCAN_SCHEMA)
        self._con.register("staged_scan", staged)
        try:
            self._run("BEGIN")
            self._run(
                f'DELETE FROM {TABLE} WHERE "scan_date" IN '
                '(SELECT DISTINCT CAST("scan_date" AS DATE) FROM staged_scan)'
            )
            self._run(f"INSERT INTO {TABLE} SELECT {casts} FROM staged_scan")
            self._run("COMMIT")
        except Exception:
            self._run("ROLLBACK")
            raise
        finally:
            self._con.unregister("staged_scan")

    def import_parquet(self, files: Sequence[Path]) -> int:
        """Load exported scan files, replacing rows that already exist. Returns rows read."""
        if not files:
            return 0
        listing = [str(path) for path in files]
        self._run(
            f"INSERT OR REPLACE INTO {TABLE} BY NAME "
            "SELECT * FROM read_parquet(?, union_by_name = true)",
            [listing],
        )
        return int(self._run("SELECT count(*) FROM read_parquet(?)", [listing])[0][0])

    def export_parquet(self, scan_date: date, path: Path) -> Path:
        """Write one scan date to a parquet file (the unit stored on the data branch)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path).replace("'", "''")
        self._run(
            f"COPY (SELECT * FROM {TABLE} WHERE \"scan_date\" = DATE '{scan_date.isoformat()}' "
            f"ORDER BY \"rank\") TO '{target}' (FORMAT PARQUET)"
        )
        return path

    # -- reading ---------------------------------------------------------------

    def has_scan(self, scan_date: date) -> bool:
        rows = self._run(f'SELECT count(*) FROM {TABLE} WHERE "scan_date" = ?', [scan_date])
        return rows[0][0] > 0

    def load_scan(self, scan_date: date) -> pd.DataFrame:
        """All rows for one scan date, best rank first. Empty frame if there are none."""
        return self._con.execute(
            f'SELECT * FROM {TABLE} WHERE "scan_date" = ? ORDER BY "rank"', [scan_date]
        ).df()

    def scan_dates(self, start: date | None = None, end: date | None = None) -> list[date]:
        """Stored scan dates in ascending order, optionally limited to [start, end]."""
        rows = self._run(
            f'SELECT DISTINCT "scan_date" FROM {TABLE} '
            'WHERE "scan_date" >= coalesce(?, "scan_date") '
            'AND "scan_date" <= coalesce(?, "scan_date") ORDER BY 1',
            [start, end],
        )
        return [row[0] for row in rows]

    def latest_scan_date(self) -> date | None:
        dates = self.scan_dates()
        return dates[-1] if dates else None

    def top_tickers(self, scan_date: date, count: int) -> list[str]:
        rows = self._run(
            f'SELECT "ticker" FROM {TABLE} WHERE "scan_date" = ? ORDER BY "rank" LIMIT ?',
            [scan_date, count],
        )
        return [row[0] for row in rows]
