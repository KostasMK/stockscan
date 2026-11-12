"""News sentiment from Alpha Vantage's NEWS_SENTIMENT endpoint.

The free tier allows only a couple of dozen calls per day, so this provider:

* is called for the top candidates only, never for the whole universe;
* caches every response on disk (a re-run on the same day costs zero calls);
* enforces its own daily budget and spacing between calls;
* turns "rate limited / not on your plan" answers into ``NewsUnavailableError`` and
  remembers it, so the remaining tickers do not waste calls.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from stockscan.http import (
    BudgetExhaustedError,
    DailyBudget,
    HttpError,
    JsonDiskCache,
    RateLimiter,
    request_with_retry,
)
from stockscan.providers.base import NewsSentiment, NewsUnavailableError

logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"
_API_TIME_FORMAT = "%Y%m%dT%H%M"
_PUBLISHED_FORMAT = "%Y%m%dT%H%M%S"


def sentiment_label(score: float) -> str:
    """Alpha Vantage's own thresholds for turning a score into a label."""
    if score <= -0.35:
        return "Bearish"
    if score <= -0.15:
        return "Somewhat-Bearish"
    if score < 0.15:
        return "Neutral"
    if score < 0.35:
        return "Somewhat-Bullish"
    return "Bullish"


def _parse_published(raw: object) -> datetime | None:
    try:
        return datetime.strptime(str(raw), _PUBLISHED_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _as_float(raw: object) -> float | None:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def summarize_news(
    ticker: str,
    payload: dict[str, Any],
    start: datetime,
    end: datetime,
    min_relevance: float,
) -> NewsSentiment:
    """Reduce a NEWS_SENTIMENT response to one score and one headline for ``ticker``.

    Pure function. Only articles published inside [start, end] count, whatever the API
    returned: with ``end`` set to the session close this is what keeps post-close news
    out of a day-T record. The score is the relevance-weighted mean of the per-ticker
    sentiment scores; the headline is the most relevant article (newest on ties).
    """
    relevant: list[tuple[float, float, datetime, dict[str, Any]]] = []
    feed = payload.get("feed")
    for article in feed if isinstance(feed, list) else []:
        published = _parse_published(article.get("time_published"))
        if published is None or not start <= published <= end:
            continue
        for entry in article.get("ticker_sentiment") or []:
            if str(entry.get("ticker", "")).upper() != ticker.upper():
                continue
            relevance = _as_float(entry.get("relevance_score"))
            score = _as_float(entry.get("ticker_sentiment_score"))
            if relevance is not None and score is not None and relevance >= min_relevance:
                relevant.append((relevance, score, published, article))
            break

    total_relevance = sum(relevance for relevance, *_ in relevant)
    if not relevant or total_relevance <= 0:
        return NewsSentiment(ticker, None, None, 0, None, None, None)

    score = sum(relevance * value for relevance, value, *_ in relevant) / total_relevance
    _, _, published, top = max(relevant, key=lambda item: (item[0], item[2]))
    return NewsSentiment(
        ticker=ticker,
        score=round(score, 4),
        label=sentiment_label(score),
        article_count=len(relevant),
        headline=str(top.get("title") or "").strip() or None,
        url=str(top.get("url") or "").strip() or None,
        published_utc=published,
    )


class AlphaVantageNews:
    """``NewsProvider`` backed by Alpha Vantage, with cache, budget and rate limit."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client,
        cache: JsonDiskCache,
        budget: DailyBudget,
        limiter: RateLimiter,
        article_limit: int = 50,
        min_relevance: float = 0.2,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._cache = cache
        self._budget = budget
        self._limiter = limiter
        self._article_limit = article_limit
        self._min_relevance = min_relevance
        self._unavailable_reason: str | None = None

    def sentiment(self, ticker: str, start: datetime, end: datetime) -> NewsSentiment:
        start_utc, end_utc = start.astimezone(UTC), end.astimezone(UTC)
        key = f"news_{ticker}_{start_utc:{_API_TIME_FORMAT}}_{end_utc:{_API_TIME_FORMAT}}"
        payload = self._cache.get(key)
        if payload is None:
            payload = self._fetch(ticker, start_utc, end_utc)
            self._cache.set(key, payload)
        return summarize_news(ticker, payload, start_utc, end_utc, self._min_relevance)

    def _give_up(self, reason: str) -> NewsUnavailableError:
        self._unavailable_reason = reason
        return NewsUnavailableError(reason)

    def _fetch(self, ticker: str, start: datetime, end: datetime) -> dict[str, Any]:
        if self._unavailable_reason is not None:
            raise NewsUnavailableError(self._unavailable_reason)
        try:
            self._budget.spend()
        except BudgetExhaustedError as error:
            raise self._give_up(f"Alpha Vantage {error}") from error

        self._limiter.wait()
        try:
            response = request_with_retry(
                self._client,
                "GET",
                BASE_URL,
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "time_from": start.strftime(_API_TIME_FORMAT),
                    "time_to": end.strftime(_API_TIME_FORMAT),
                    "sort": "LATEST",
                    "limit": self._article_limit,
                    "apikey": self._api_key,
                },
                retries=2,
                secrets=[self._api_key],
            )
            payload = response.json()
        except (HttpError, ValueError) as error:
            raise self._give_up(f"Alpha Vantage request failed: {error}") from error
        if not isinstance(payload, dict):
            raise self._give_up("Alpha Vantage returned an unexpected response")

        # Alpha Vantage reports problems with HTTP 200 and a message instead of data.
        message = next(
            (str(payload[k]) for k in ("Information", "Note", "Error Message") if k in payload),
            None,
        )
        if message is None:
            return payload
        if "no articles" in message.lower():
            return {"feed": []}
        raise self._give_up(f"Alpha Vantage: {message.replace(self._api_key, '***')[:200]}")
