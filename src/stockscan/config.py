"""Configuration.

Two sources, kept separate on purpose:

* ``EnvSettings``: secrets and machine-specific paths, read from the environment / ``.env``.
* ``AppConfig``: research tunables (score weights, fees, ...) read from a versioned TOML file,
  so a change in behaviour always shows up in git history.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic import model_validator as pydantic_model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Secrets and paths. Every field is optional so the tool degrades gracefully."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    alphavantage_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None

    config_file: Path = Field(
        Path("config/config.toml"),
        validation_alias=AliasChoices("config_file", "STOCKSCAN_CONFIG_FILE"),
    )
    output_dir: Path = Field(
        Path("out"), validation_alias=AliasChoices("output_dir", "STOCKSCAN_OUTPUT_DIR")
    )
    db_path: Path = Field(
        Path("data/stockscan.duckdb"),
        validation_alias=AliasChoices("db_path", "STOCKSCAN_DB_PATH"),
    )
    cache_dir: Path = Field(
        Path(".cache"), validation_alias=AliasChoices("cache_dir", "STOCKSCAN_CACHE_DIR")
    )

    @field_validator(
        "alphavantage_api_key", "telegram_bot_token", "telegram_chat_id", mode="before"
    )
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        # `.env.example` ships blank values and GitHub Actions passes unset secrets as "".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def telegram_configured(self) -> bool:
        return self.telegram_bot_token is not None and self.telegram_chat_id is not None


class _Strict(BaseModel):
    # Unknown keys are almost always typos in config.toml: fail loudly.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScanConfig(_Strict):
    universe_file: Path = Path("data/universe/nasdaq100.csv")
    history_calendar_days: int = Field(420, ge=30)
    """Calendar days of daily history to fetch (252 sessions are needed for 52-week ranges)."""
    close_buffer_minutes: int = Field(20, ge=0)
    """Minutes after 16:00 New York before today's bar is trusted as final."""
    min_coverage: float = Field(0.5, gt=0, le=1)
    """Abort if fewer than this share of the universe has a bar for the scan date."""
    top_n: int = Field(10, ge=1)
    long_only: bool = False
    """Show only stocks that closed up in the report and alert. Storage always keeps everything."""


class ScoringWeights(_Strict):
    rel_volume: float = Field(0.30, ge=0)
    gap: float = Field(0.20, ge=0)
    move: float = Field(0.25, ge=0)
    extreme: float = Field(0.15, ge=0)
    stretch: float = Field(0.10, ge=0)

    @pydantic_model_validator(mode="after")
    def _not_all_zero(self) -> Self:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one scoring weight must be positive")
        return self


class ScoringCaps(_Strict):
    """Value at which each component saturates at 1.0."""

    rel_volume: float = Field(5.0, gt=1)
    gap_pct: float = Field(5.0, gt=0)
    move_atr: float = Field(3.0, gt=0)
    extreme_pct: float = Field(10.0, gt=0)
    stretch_pct: float = Field(10.0, gt=0)


class ScoringConfig(_Strict):
    weights: ScoringWeights = ScoringWeights()
    caps: ScoringCaps = ScoringCaps()

    @property
    def version(self) -> str:
        """Short hash stored with every row, so history can be split by scoring rules."""
        payload = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:8]


class NewsConfig(_Strict):
    enabled: bool = True
    max_calls_per_day: int = Field(20, ge=0)
    min_interval_seconds: float = Field(1.5, ge=0)
    lookback_hours: int = Field(24, ge=1)
    article_limit: int = Field(50, ge=1, le=1000)
    min_relevance: float = Field(0.2, ge=0, le=1)


class SizingConfig(_Strict):
    risk_pct: float = Field(1.0, gt=0)
    max_risk_pct_warning: float = Field(2.0, gt=0)
    commission_min_eur: float = Field(1.0, ge=0)
    commission_pct: float = Field(0.25, ge=0)
    fx_fee_pct_usd_balance: float = Field(0.0, ge=0)
    fx_fee_pct_eur_conversion: float = Field(1.0, ge=0)
    assume_eur_conversion: bool = True
    """Default funding assumption when neither --convert-eur nor --usd-balance is passed."""
    fractional_decimals: int = Field(4, ge=0, le=8)


class IntradayConfig(_Strict):
    interval: str = "5m"
    top_n: int = Field(10, ge=0)
    always_include: tuple[str, ...] = ("NVDA", "QQQ")


class AppConfig(_Strict):
    scan: ScanConfig = ScanConfig()
    scoring: ScoringConfig = ScoringConfig()
    news: NewsConfig = NewsConfig()
    sizing: SizingConfig = SizingConfig()
    intraday: IntradayConfig = IntradayConfig()


def load_config(path: Path) -> AppConfig:
    """Load and validate the TOML config. Missing sections fall back to defaults."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run from the repository root or set "
            "STOCKSCAN_CONFIG_FILE."
        )
    with path.open("rb") as handle:
        return AppConfig.model_validate(tomllib.load(handle))
