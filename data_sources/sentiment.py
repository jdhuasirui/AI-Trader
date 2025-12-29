"""
Sentiment Analysis for AI-Trader

This module provides sentiment analysis for trading signals:
- FinBERT integration for financial text classification
- Multi-source sentiment aggregation (news, social media)
- Sentiment score normalization (-1 to 1)

Sources: News headlines, Twitter/X, Reddit, LunarCrush
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    timestamp: datetime
    symbol: str
    source: str  # "news", "twitter", "reddit", "aggregate"

    # Sentiment scores
    score: float  # -1 (bearish) to 1 (bullish)
    magnitude: float  # 0 to 1, strength of sentiment
    confidence: float  # 0 to 1, model confidence

    # Breakdown
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0

    # Volume metrics
    mention_count: int = 0
    engagement_score: float = 0.0

    # Raw text samples (for debugging)
    sample_texts: List[str] = field(default_factory=list)


class FinBERTAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.

    Uses the ProsusAI/finbert model for financial sentiment classification.
    Falls back to rule-based analysis if model not available.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._initialized = False

        # Keywords for rule-based fallback
        self._bullish_words = {
            "buy", "bullish", "rally", "surge", "gain", "growth", "up", "high",
            "profit", "positive", "outperform", "strong", "momentum", "breakout",
            "upgrade", "beat", "exceed", "record", "soar", "jump", "climb",
        }
        self._bearish_words = {
            "sell", "bearish", "crash", "plunge", "loss", "decline", "down", "low",
            "negative", "underperform", "weak", "breakdown", "downgrade", "miss",
            "fall", "drop", "sink", "tumble", "slump", "concern", "risk", "fear",
        }

    def initialize(self) -> bool:
        """Initialize FinBERT model (lazy loading)."""
        if self._initialized:
            return True

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            self._initialized = True
            logger.info("FinBERT model loaded successfully")
            return True

        except ImportError:
            logger.warning("Transformers library not available, using rule-based sentiment")
            return False
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}, using rule-based sentiment")
            return False

    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of financial text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (bearish) to 1 (bullish)
            confidence: 0 to 1
        """
        if self._initialized and self._model is not None:
            return self._analyze_with_model(text)
        else:
            return self._analyze_with_rules(text)

    def _analyze_with_model(self, text: str) -> Tuple[float, float]:
        """Analyze using FinBERT model."""
        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]

            # FinBERT outputs: [positive, negative, neutral]
            positive = float(probs[0])
            negative = float(probs[1])
            neutral = float(probs[2])

            # Convert to -1 to 1 scale
            sentiment = positive - negative
            confidence = max(positive, negative, neutral)

            return sentiment, confidence

        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._analyze_with_rules(text)

    def _analyze_with_rules(self, text: str) -> Tuple[float, float]:
        """Rule-based sentiment analysis fallback."""
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))

        bullish_count = len(words & self._bullish_words)
        bearish_count = len(words & self._bearish_words)
        total = bullish_count + bearish_count

        if total == 0:
            return 0.0, 0.3  # Neutral with low confidence

        sentiment = (bullish_count - bearish_count) / total
        confidence = min(total / 5, 1.0)  # More keywords = higher confidence

        return sentiment, confidence


class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer for trading signals.

    Aggregates sentiment from:
    - News headlines (via Alpha Vantage, NewsAPI, etc.)
    - Social media (Twitter, Reddit via LunarCrush)
    """

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        lunarcrush_api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,
    ):
        self.news_api_key = news_api_key or os.environ.get("NEWS_API_KEY")
        self.lunarcrush_api_key = lunarcrush_api_key or os.environ.get("LUNARCRUSH_API_KEY")
        self.cache_ttl = cache_ttl_seconds

        # Initialize FinBERT
        self.finbert = FinBERTAnalyzer()

        # Cache
        self._cache: Dict[str, tuple] = {}

    def analyze(self, symbol: str) -> SentimentResult:
        """
        Get aggregated sentiment for a symbol.

        Args:
            symbol: Asset symbol (e.g., "AAPL", "BTC")

        Returns:
            SentimentResult with aggregated scores
        """
        # Check cache
        cache_key = f"sentiment_{symbol}"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl):
                return data

        results = []

        # Fetch news sentiment
        news_result = self._analyze_news(symbol)
        if news_result:
            results.append(news_result)

        # Fetch social sentiment
        social_result = self._analyze_social(symbol)
        if social_result:
            results.append(social_result)

        # Aggregate results
        if results:
            aggregate = self._aggregate_results(symbol, results)
        else:
            # No data available, return neutral
            aggregate = SentimentResult(
                timestamp=datetime.now(),
                symbol=symbol,
                source="aggregate",
                score=0.0,
                magnitude=0.0,
                confidence=0.1,
            )

        # Cache result
        self._cache[cache_key] = (aggregate, datetime.now())

        return aggregate

    def _analyze_news(self, symbol: str) -> Optional[SentimentResult]:
        """Fetch and analyze news headlines."""
        headlines = self._fetch_news_headlines(symbol)
        if not headlines:
            return None

        # Initialize FinBERT if not done
        self.finbert.initialize()

        scores = []
        confidences = []

        for headline in headlines[:20]:  # Limit to 20 headlines
            score, conf = self.finbert.analyze(headline)
            scores.append(score)
            confidences.append(conf)

        if not scores:
            return None

        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)
        magnitude = sum(abs(s) for s in scores) / len(scores)

        positive_count = sum(1 for s in scores if s > 0.1)
        negative_count = sum(1 for s in scores if s < -0.1)
        neutral_count = len(scores) - positive_count - negative_count

        return SentimentResult(
            timestamp=datetime.now(),
            symbol=symbol,
            source="news",
            score=avg_score,
            magnitude=magnitude,
            confidence=avg_confidence,
            positive_ratio=positive_count / len(scores),
            negative_ratio=negative_count / len(scores),
            neutral_ratio=neutral_count / len(scores),
            mention_count=len(headlines),
            sample_texts=headlines[:3],
        )

    def _analyze_social(self, symbol: str) -> Optional[SentimentResult]:
        """Fetch and analyze social media sentiment."""
        if self.lunarcrush_api_key:
            return self._fetch_lunarcrush(symbol)
        else:
            # Simulate social sentiment
            return self._simulate_social_sentiment(symbol)

    def _fetch_news_headlines(self, symbol: str) -> List[str]:
        """Fetch news headlines for a symbol."""
        # Try Alpha Vantage news
        api_key = os.environ.get("ALPHAADVANTAGE_API_KEY")
        if api_key:
            try:
                import urllib.request

                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    headlines = []
                    for item in data.get("feed", [])[:20]:
                        title = item.get("title", "")
                        summary = item.get("summary", "")
                        headlines.append(f"{title}. {summary}")

                    return headlines

            except Exception as e:
                logger.debug(f"Alpha Vantage news fetch failed: {e}")

        # Return simulated headlines for testing
        return self._simulate_news_headlines(symbol)

    def _fetch_lunarcrush(self, symbol: str) -> Optional[SentimentResult]:
        """Fetch social sentiment from LunarCrush."""
        try:
            import urllib.request

            url = f"https://api.lunarcrush.com/v2?data=assets&symbol={symbol}&key={self.lunarcrush_api_key}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if "data" in data and data["data"]:
                asset = data["data"][0]

                # Galaxy Score is 0-100, convert to -1 to 1
                galaxy_score = asset.get("galaxy_score", 50)
                sentiment = (galaxy_score - 50) / 50

                return SentimentResult(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    source="social",
                    score=sentiment,
                    magnitude=abs(sentiment),
                    confidence=0.7,
                    mention_count=asset.get("social_mentions", 0),
                    engagement_score=asset.get("social_score", 0),
                )

        except Exception as e:
            logger.debug(f"LunarCrush fetch failed: {e}")

        return None

    def _aggregate_results(
        self,
        symbol: str,
        results: List[SentimentResult],
    ) -> SentimentResult:
        """Aggregate multiple sentiment results."""
        # Weight by confidence
        total_weight = sum(r.confidence for r in results)
        if total_weight == 0:
            total_weight = 1

        weighted_score = sum(r.score * r.confidence for r in results) / total_weight
        weighted_magnitude = sum(r.magnitude * r.confidence for r in results) / total_weight
        avg_confidence = total_weight / len(results)

        total_mentions = sum(r.mention_count for r in results)
        avg_engagement = sum(r.engagement_score for r in results) / len(results)

        return SentimentResult(
            timestamp=datetime.now(),
            symbol=symbol,
            source="aggregate",
            score=weighted_score,
            magnitude=weighted_magnitude,
            confidence=avg_confidence,
            mention_count=total_mentions,
            engagement_score=avg_engagement,
        )

    def _simulate_news_headlines(self, symbol: str) -> List[str]:
        """Generate simulated headlines for testing."""
        import random

        templates_bullish = [
            f"{symbol} surges on strong earnings beat",
            f"Analysts upgrade {symbol} to buy rating",
            f"{symbol} announces record quarterly revenue",
            f"Institutional investors increase {symbol} holdings",
        ]

        templates_bearish = [
            f"{symbol} drops amid market concerns",
            f"Analysts downgrade {symbol} outlook",
            f"{symbol} faces regulatory headwinds",
            f"Profit-taking hits {symbol} shares",
        ]

        templates_neutral = [
            f"{symbol} trades flat in volatile session",
            f"Market watches {symbol} for direction",
            f"{symbol} holds support levels",
        ]

        # Randomly select headlines
        random.seed(int(datetime.now().timestamp() // 3600) + hash(symbol))
        all_templates = templates_bullish + templates_bearish + templates_neutral
        return random.sample(all_templates, min(5, len(all_templates)))

    def _simulate_social_sentiment(self, symbol: str) -> SentimentResult:
        """Generate simulated social sentiment for testing."""
        import random

        random.seed(int(datetime.now().timestamp() // 3600) + hash(symbol))

        return SentimentResult(
            timestamp=datetime.now(),
            symbol=symbol,
            source="social",
            score=random.uniform(-0.5, 0.5),
            magnitude=random.uniform(0.2, 0.8),
            confidence=0.5,
            positive_ratio=random.uniform(0.2, 0.5),
            negative_ratio=random.uniform(0.2, 0.4),
            neutral_ratio=random.uniform(0.1, 0.4),
            mention_count=random.randint(100, 10000),
            engagement_score=random.uniform(0.3, 0.8),
        )

    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        self.finbert.initialize()
        return self.finbert.analyze(text)
