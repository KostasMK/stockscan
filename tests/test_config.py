from pathlib import Path

import pytest
from pydantic import ValidationError

from stockscan.config import AppConfig, EnvSettings, ScoringConfig, ScoringWeights, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_config_file_is_valid_and_matches_code_defaults():
    # If this fails, config/config.toml and the defaults in config.py have drifted apart.
    assert load_config(REPO_ROOT / "config" / "config.toml") == AppConfig()


def test_missing_config_file_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="STOCKSCAN_CONFIG_FILE"):
        load_config(tmp_path / "nope.toml")


def test_partial_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[scoring.weights]\nrel_volume = 0.9\n", encoding="utf-8")
    config = load_config(path)
    assert config.scoring.weights.rel_volume == 0.9
    assert config.scoring.weights.gap == AppConfig().scoring.weights.gap
    assert config.scan.top_n == 10


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[scoring.weights]\nrel_volumee = 0.9\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_all_zero_weights_are_rejected():
    with pytest.raises(ValidationError):
        ScoringWeights(rel_volume=0, gap=0, move=0, extreme=0, stretch=0)


def test_scoring_version_changes_with_the_weights():
    default = ScoringConfig()
    tweaked = ScoringConfig(weights=ScoringWeights(rel_volume=0.5))
    assert default.version == ScoringConfig().version
    assert default.version != tweaked.version


def test_blank_secrets_count_as_missing(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "  ")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    settings = EnvSettings(_env_file=None)
    assert settings.alphavantage_api_key is None
    assert settings.telegram_configured is False


def test_secrets_are_read_from_the_environment_and_not_printed(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setenv("STOCKSCAN_OUTPUT_DIR", "data-branch")
    settings = EnvSettings(_env_file=None)
    assert settings.telegram_configured is True
    assert settings.output_dir == Path("data-branch")
    assert "123:abc" not in repr(settings)
