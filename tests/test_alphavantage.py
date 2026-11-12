from datetime import UTC, date, datetime

import httpx
import pytest
import respx

from stockscan.http import DailyBudget, JsonDiskCache, RateLimiter
from stockscan.providers.alphavantage import (
    BASE_URL,
    AlphaVantageNews,
    sentiment_label,
    summarize_news,
)
from stockscan.providers.base import NewsUnavailableError

START = datetime(2026, 11, 10, 21, 0, tzinfo=UTC)
END = datetime(2026, 11, 11, 21, 0, tzinfo=UTC)  # the session close
API_KEY = "SECRETKEY"


def article(title: str, published: str, ticker: str, relevance: float, score: float) -> dict:
    return {
        "title": title,
        "url": f"https://news.example.test/{title.replace(' ', '-')}",
        "time_published": published,
        "overall_sentiment_score": 0.0,
        "ticker_sentiment": [
            {
                "ticker": "OTHER",
                "relevance_score": "0.9",
                "ticker_sentiment_score": "-0.9",
                "ticker_sentiment_label": "Bearish",
            },
            {
                "ticker": ticker,
                "relevance_score": str(relevance),
                "ticker_sentiment_score": str(score),
                "ticker_sentiment_label": "n/a",
            },
        ],
    }


PAYLOAD = {
    "items": "4",
    "feed": [
        article("NVDA beats estimates", "20261111T150000", "NVDA", 0.8, 0.5),
        article("Chip stocks mixed", "20261111T120000", "NVDA", 0.4, -0.1),
        article("Passing mention", "20261111T130000", "NVDA", 0.05, -0.9),  # below min relevance
        article("After-hours shock", "20261111T213000", "NVDA", 0.9, -0.8),  # after the close
    ],
}


class TestSummarizeNews:
    def test_relevance_weighted_score_and_most_relevant_headline(self):
        result = summarize_news("NVDA", PAYLOAD, START, END, min_relevance=0.2)
        assert result.article_count == 2
        assert result.score == pytest.approx((0.8 * 0.5 + 0.4 * -0.1) / 1.2, abs=1e-4)
        assert result.label == "Somewhat-Bullish"
        assert result.headline == "NVDA beats estimates"
        assert result.url == "https://news.example.test/NVDA-beats-estimates"
        assert result.published_utc == datetime(2026, 11, 11, 15, 0, tzinfo=UTC)

    def test_articles_published_after_the_close_are_ignored(self):
        # No look-ahead: the after-hours article must not affect a day-T record,
        # even though the API returned it.
        result = summarize_news("NVDA", PAYLOAD, START, END, min_relevance=0.2)
        assert result.headline != "After-hours shock"
        late_only = {"feed": [PAYLOAD["feed"][3]]}
        assert summarize_news("NVDA", late_only, START, END, 0.2).score is None

    def test_no_relevant_articles_gives_an_empty_result(self):
        result = summarize_news("AAPL", PAYLOAD, START, END, min_relevance=0.2)
        assert (result.score, result.label, result.article_count, result.headline) == (
            None,
            None,
            0,
            None,
        )

    @pytest.mark.parametrize(
        "payload",
        [{}, {"feed": None}, {"feed": [{"title": "broken"}]}, {"feed": [{"time_published": "x"}]}],
    )
    def test_malformed_payloads_do_not_crash(self, payload):
        assert summarize_news("NVDA", payload, START, END, 0.2).article_count == 0


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (-0.5, "Bearish"),
        (-0.2, "Somewhat-Bearish"),
        (0.0, "Neutral"),
        (0.2, "Somewhat-Bullish"),
        (0.5, "Bullish"),
    ],
)
def test_sentiment_label(score, label):
    assert sentiment_label(score) == label


@pytest.fixture
def make_news(tmp_path):
    clients: list[httpx.Client] = []

    def factory(max_calls: int = 5) -> AlphaVantageNews:
        client = httpx.Client()
        clients.append(client)
        return AlphaVantageNews(
            api_key=API_KEY,
            client=client,
            cache=JsonDiskCache(tmp_path / "cache"),
            budget=DailyBudget(
                tmp_path / "budget.json", max_calls, today=lambda: date(2026, 11, 12)
            ),
            limiter=RateLimiter(0),
        )

    yield factory
    for client in clients:
        client.close()


class TestAlphaVantageNews:
    @respx.mock
    def test_fetches_with_the_exact_time_window(self, make_news):
        route = respx.get(BASE_URL).respond(json=PAYLOAD)
        result = make_news().sentiment("NVDA", START, END)
        assert result.headline == "NVDA beats estimates"
        params = route.calls.last.request.url.params
        assert params["function"] == "NEWS_SENTIMENT"
        assert params["tickers"] == "NVDA"
        assert params["time_from"] == "20261110T2100"
        assert params["time_to"] == "20261111T2100"

    @respx.mock
    def test_repeat_calls_are_served_from_cache_even_by_a_new_instance(self, make_news):
        route = respx.get(BASE_URL).respond(json=PAYLOAD)
        make_news().sentiment("NVDA", START, END)
        make_news().sentiment("NVDA", START, END)
        assert route.call_count == 1

    @respx.mock
    def test_daily_budget_stops_further_calls(self, make_news):
        route = respx.get(BASE_URL).respond(json=PAYLOAD)
        news = make_news(max_calls=1)
        news.sentiment("NVDA", START, END)
        with pytest.raises(NewsUnavailableError, match="budget"):
            news.sentiment("AAPL", START, END)
        assert route.call_count == 1

    @respx.mock
    @pytest.mark.parametrize(
        "body",
        [
            {"Information": "Our standard API rate limit is 25 requests per day."},
            {"Information": "This is a premium endpoint. Subscribe to unlock."},
            {"Note": "Thank you for using Alpha Vantage! Please slow down."},
            {"Error Message": "Invalid API call."},
        ],
    )
    def test_api_messages_disable_news_for_the_rest_of_the_run(self, make_news, body):
        route = respx.get(BASE_URL).respond(json=body)
        news = make_news()
        with pytest.raises(NewsUnavailableError, match="Alpha Vantage"):
            news.sentiment("NVDA", START, END)
        with pytest.raises(NewsUnavailableError):
            news.sentiment("AAPL", START, END)
        assert route.call_count == 1  # the second ticker did not waste a call

    @respx.mock
    def test_no_articles_message_is_an_empty_result_not_an_outage(self, make_news):
        respx.get(BASE_URL).respond(json={"Information": "No articles found. Adjust the range."})
        news = make_news()
        assert news.sentiment("NVDA", START, END).article_count == 0
        respx.get(BASE_URL).respond(json=PAYLOAD)
        assert news.sentiment("AAPL", START, END).article_count == 0  # still usable

    @respx.mock
    def test_http_failures_never_leak_the_api_key(self, make_news, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        respx.get(BASE_URL).respond(status_code=500, text=f"bad key {API_KEY}")
        with pytest.raises(NewsUnavailableError) as raised:
            make_news().sentiment("NVDA", START, END)
        assert API_KEY not in str(raised.value)
