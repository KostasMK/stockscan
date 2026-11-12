from datetime import date, timedelta

import httpx
import pytest
import respx

from stockscan.http import (
    BudgetExhaustedError,
    DailyBudget,
    HttpError,
    JsonDiskCache,
    RateLimiter,
    redact,
    request_with_retry,
)

URL = "https://api.example.test/data"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestRateLimiter:
    def test_first_call_does_not_wait(self):
        clock = FakeClock()
        RateLimiter(2.0, clock=clock, sleep=clock.sleep).wait()
        assert clock.sleeps == []

    def test_back_to_back_calls_are_spaced_out(self):
        clock = FakeClock()
        limiter = RateLimiter(2.0, clock=clock, sleep=clock.sleep)
        limiter.wait()
        clock.now += 0.5
        limiter.wait()
        assert clock.sleeps == [pytest.approx(1.5)]

    def test_no_wait_when_enough_time_has_passed(self):
        clock = FakeClock()
        limiter = RateLimiter(2.0, clock=clock, sleep=clock.sleep)
        limiter.wait()
        clock.now += 5
        limiter.wait()
        assert clock.sleeps == []


class TestJsonDiskCache:
    def test_round_trip(self, tmp_path):
        cache = JsonDiskCache(tmp_path / "cache")
        assert cache.get("news_AAPL_2026-11-11") is None
        cache.set("news_AAPL_2026-11-11", {"feed": [1, 2]})
        assert cache.get("news_AAPL_2026-11-11") == {"feed": [1, 2]}

    def test_expired_entries_are_a_miss(self, tmp_path):
        cache = JsonDiskCache(tmp_path)
        cache.set("key", 1)
        assert cache.get("key", max_age=timedelta(hours=1)) == 1
        assert cache.get("key", max_age=timedelta(seconds=-1)) is None

    def test_corrupt_entries_are_a_miss(self, tmp_path):
        cache = JsonDiskCache(tmp_path)
        cache.set("key", 1)
        next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
        assert cache.get("key") is None

    def test_keys_cannot_escape_the_cache_directory(self, tmp_path):
        cache = JsonDiskCache(tmp_path / "cache")
        cache.set("../../evil", 1)
        assert [path.parent for path in tmp_path.rglob("*.json")] == [tmp_path / "cache"]


class TestDailyBudget:
    def test_spending_up_to_the_cap_then_refusing(self, tmp_path):
        budget = DailyBudget(
            tmp_path / "budget.json", max_calls=2, today=lambda: date(2026, 11, 12)
        )
        assert budget.remaining == 2
        budget.spend()
        budget.spend()
        assert budget.remaining == 0
        with pytest.raises(BudgetExhaustedError):
            budget.spend()

    def test_count_survives_restarts_and_resets_next_day(self, tmp_path):
        path = tmp_path / "budget.json"
        DailyBudget(path, max_calls=2, today=lambda: date(2026, 11, 12)).spend()
        assert DailyBudget(path, max_calls=2, today=lambda: date(2026, 11, 12)).remaining == 1
        assert DailyBudget(path, max_calls=2, today=lambda: date(2026, 11, 13)).remaining == 2


def test_redact_removes_every_secret():
    assert redact("GET /bot123:abc/send?apikey=KEY", ["123:abc", "KEY", ""]) == (
        "GET /bot***/send?apikey=***"
    )


class TestRequestWithRetry:
    @respx.mock
    def test_success_returns_the_response(self):
        respx.get(URL).respond(json={"ok": True})
        with httpx.Client() as client:
            assert request_with_retry(client, "GET", URL).json() == {"ok": True}

    @respx.mock
    def test_retries_server_errors_with_exponential_backoff(self):
        route = respx.get(URL).mock(
            side_effect=[httpx.Response(503), httpx.Response(500), httpx.Response(200, json={})]
        )
        sleeps: list[float] = []
        with httpx.Client() as client:
            response = request_with_retry(
                client, "GET", URL, backoff_seconds=1.0, sleep=sleeps.append
            )
        assert response.status_code == 200
        assert route.call_count == 3
        assert sleeps == [1.0, 2.0]

    @respx.mock
    def test_retries_network_errors(self):
        route = respx.get(URL).mock(
            side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={})]
        )
        with httpx.Client() as client:
            request_with_retry(client, "GET", URL, sleep=lambda _: None)
        assert route.call_count == 2

    @respx.mock
    def test_honours_retry_after_header(self):
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={}),
            ]
        )
        sleeps: list[float] = []
        with httpx.Client() as client:
            request_with_retry(client, "GET", URL, sleep=sleeps.append)
        assert sleeps == [7.0]

    @respx.mock
    def test_client_errors_are_not_retried(self):
        route = respx.get(URL).respond(status_code=404)
        with httpx.Client() as client, pytest.raises(HttpError, match="HTTP 404"):
            request_with_retry(client, "GET", URL, sleep=lambda _: None)
        assert route.call_count == 1

    @respx.mock
    def test_gives_up_after_the_last_retry_without_leaking_secrets(self):
        route = respx.get(f"{URL}/botTOKEN").respond(status_code=500, text="apikey=KEY rejected")
        with httpx.Client() as client, pytest.raises(HttpError) as raised:
            request_with_retry(
                client,
                "GET",
                f"{URL}/botTOKEN",
                retries=2,
                sleep=lambda _: None,
                secrets=["TOKEN", "KEY"],
            )
        assert route.call_count == 3
        assert "TOKEN" not in str(raised.value)
        assert "KEY" not in str(raised.value)
