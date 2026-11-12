"""Shared plumbing for external APIs: rate limiting, retries, disk cache, call budgets.

Every external call in the project goes through these helpers so that politeness
(spacing, caps) and resilience (retries, caching) live in one place.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "stockscan/0.1 (personal research tool)"
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0


class HttpError(Exception):
    """A request failed for good. The message never contains secrets."""


class BudgetExhaustedError(Exception):
    """The self-imposed daily call cap for an API has been reached."""


def redact(text: str, secrets: Iterable[str]) -> str:
    """Remove secrets (API keys, bot tokens) from text before it is logged or raised."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class RateLimiter:
    """Enforces a minimum gap between consecutive calls."""

    def __init__(
        self,
        min_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        """Block until the next call is allowed, then mark it as made."""
        if self._last_call is not None:
            remaining = self._min_interval - (self._clock() - self._last_call)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()


class JsonDiskCache:
    """One JSON file per key. Good enough for a few dozen API responses a day."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def _path(self, key: str) -> Path:
        return self._directory / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def get(self, key: str, max_age: timedelta | None = None) -> Any | None:
        path = self._path(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            saved_at = datetime.fromisoformat(entry["saved_at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None  # missing or corrupt cache entries are simply a miss
        if max_age is not None and datetime.now(UTC) - saved_at > max_age:
            return None
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        entry = {"saved_at": datetime.now(UTC).isoformat(), "value": value}
        self._path(key).write_text(json.dumps(entry), encoding="utf-8")


class DailyBudget:
    """Persistent per-UTC-day call counter with a hard cap."""

    def __init__(
        self,
        path: Path,
        max_calls: int,
        today: Callable[[], date] = lambda: datetime.now(UTC).date(),
    ) -> None:
        self._path = path
        self._max_calls = max_calls
        self._today = today

    def _used(self) -> int:
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if not isinstance(state, dict) or state.get("date") != self._today().isoformat():
            return 0
        return int(state.get("calls", 0))

    @property
    def remaining(self) -> int:
        return max(0, self._max_calls - self._used())

    def spend(self) -> None:
        """Record one call, or raise ``BudgetExhaustedError`` if none are left today."""
        used = self._used()
        if used >= self._max_calls:
            raise BudgetExhaustedError(f"daily budget of {self._max_calls} calls is used up")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {"date": self._today().isoformat(), "calls": used + 1}
        self._path.write_text(json.dumps(state), encoding="utf-8")


def _retry_delay(response: httpx.Response | None, attempt: int, backoff_seconds: float) -> float:
    if response is not None:
        try:
            return min(float(response.headers.get("Retry-After", "")), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    return backoff_seconds * (2**attempt)


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    json_body: Mapping[str, Any] | None = None,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    secrets: Iterable[str] = (),
) -> httpx.Response:
    """Send a request, retrying network errors, 429 and 5xx with exponential backoff.

    Raises ``HttpError`` (with secrets redacted) once retries are exhausted or on a
    non-retryable 4xx. httpx's own exceptions embed the URL, which may carry a key or
    token, so they are never allowed to escape.
    """
    secrets = tuple(secrets)
    failure = "no attempt made"
    for attempt in range(retries + 1):
        response: httpx.Response | None = None
        try:
            response = client.request(method, url, params=params, json=json_body)
        except httpx.TransportError as error:
            failure = f"{type(error).__name__}: {error}"
        else:
            if response.status_code < 400:
                return response
            failure = f"HTTP {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRY_STATUS_CODES:
                break
        if attempt < retries:
            delay = _retry_delay(response, attempt, backoff_seconds)
            logger.warning(
                "%s %s failed (%s); retry %d/%d in %.1fs",
                method,
                redact(url, secrets),
                redact(failure, secrets),
                attempt + 1,
                retries,
                delay,
            )
            sleep(delay)
    raise HttpError(redact(f"{method} {url} failed: {failure}", secrets))
