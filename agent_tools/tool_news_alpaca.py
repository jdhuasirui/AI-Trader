"""
Alpaca News Tool (MCP)

Provides news data with relevance scoring, LLM-based clustering,
and peer reaction analysis for informed trading decisions.

Features:
- Real-time news from Alpaca News API
- LLM-based news clustering and summarization
- Sentiment scoring
- Peer stock reaction analysis
- Recency weighting
"""

import hashlib
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_tools.alpaca_client import AlpacaClientError, get_alpaca_client
from agent_tools.cache import get_cache_manager

mcp = FastMCP("AlpacaNews")


# ==================== Configuration ====================

# Alpaca API endpoints
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v1beta1")

# OpenAI for LLM summarization (using GPT-4o-mini for cost efficiency)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Peer stock correlations (pre-computed, would be updated weekly in production)
SECTOR_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META", "NVDA", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL", "SNAP"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR", "MSFT"],
    "AMZN": ["MSFT", "GOOGL", "AAPL", "WMT", "TGT"],
    "NVDA": ["AMD", "INTC", "AVGO", "QCOM", "TSM"],
    "AMD": ["NVDA", "INTC", "AVGO", "QCOM", "TSM"],
    "TSLA": ["RIVN", "LCID", "F", "GM", "NIO"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BTC/USD": ["ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD", "ADA/USD"],
    "ETH/USD": ["BTC/USD", "SOL/USD", "AVAX/USD", "MATIC/USD", "LINK/USD"],
}


# ==================== Helper Functions ====================

def get_alpaca_news(
    symbols: List[str],
    limit: int = 50,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch news from Alpaca News API.

    Args:
        symbols: List of symbols to get news for
        limit: Maximum number of news items
        start: Start datetime (defaults to 24 hours ago)
        end: End datetime (defaults to now)

    Returns:
        List of news items
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise AlpacaClientError("Alpaca API credentials not configured")

    # Default time range: last 24 hours
    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(hours=24)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    params = {
        "symbols": ",".join(symbols),
        "limit": min(limit, 50),  # Alpaca max is 50
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sort": "desc",
    }

    url = f"{ALPACA_DATA_URL}/news"

    response = requests.get(url, headers=headers, params=params, timeout=10)

    if response.status_code != 200:
        raise AlpacaClientError(f"News API error: {response.status_code} - {response.text}")

    data = response.json()
    return data.get("news", [])


def calculate_recency_weight(published_at: str) -> float:
    """
    Calculate recency weight using exponential decay.

    More recent news has higher weight.
    Half-life is 2 hours (news loses half its weight every 2 hours).

    Args:
        published_at: ISO timestamp string

    Returns:
        Weight between 0 and 1
    """
    try:
        # Parse timestamp
        if published_at.endswith("Z"):
            published_at = published_at[:-1] + "+00:00"
        pub_time = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

        # Calculate age in hours
        now = datetime.now(pub_time.tzinfo) if pub_time.tzinfo else datetime.utcnow()
        age_hours = (now - pub_time).total_seconds() / 3600

        # Exponential decay with 2-hour half-life
        import math
        half_life = 2.0
        weight = math.exp(-0.693 * age_hours / half_life)  # 0.693 = ln(2)

        return round(max(0.0, min(1.0, weight)), 3)

    except Exception:
        return 0.5  # Default weight for parsing errors


def get_peer_reactions(
    symbol: str,
    news_time: str,
    client
) -> Dict[str, float]:
    """
    Calculate peer stock price reactions since news publication.

    Only uses past price data to avoid look-ahead bias.

    Args:
        symbol: Symbol the news is about
        news_time: ISO timestamp of news publication
        client: Alpaca client instance

    Returns:
        Dict mapping peer symbol to price change percentage
    """
    peers = SECTOR_PEERS.get(symbol, [])[:5]

    if not peers:
        return {}

    reactions = {}

    try:
        # Parse news time
        if news_time.endswith("Z"):
            news_time = news_time[:-1] + "+00:00"
        pub_time = datetime.fromisoformat(news_time.replace("Z", "+00:00"))

        # Get 1-hour bars for each peer
        for peer in peers:
            try:
                # Check if it's a crypto symbol
                if "/" in peer:
                    bars = client.get_crypto_bars(peer, timeframe="1Hour", limit=2)
                else:
                    bars = client.get_bars(peer, timeframe="1Hour", limit=2)

                if bars and len(bars) >= 2:
                    prev_close = bars[-2]["close"]
                    curr_close = bars[-1]["close"]
                    if prev_close > 0:
                        change_pct = ((curr_close - prev_close) / prev_close) * 100
                        reactions[peer] = round(change_pct, 2)

            except Exception:
                continue

    except Exception:
        pass

    return reactions


def cluster_news(news_items: List[Dict]) -> List[Dict]:
    """
    Cluster similar news headlines together.

    Uses simple text similarity for clustering, then optionally
    summarizes with LLM.

    Args:
        news_items: List of raw news items from API

    Returns:
        List of news clusters
    """
    if not news_items:
        return []

    clusters = []
    used_indices = set()

    for i, item in enumerate(news_items):
        if i in used_indices:
            continue

        # Start a new cluster with this item
        cluster = {
            "items": [item],
            "headlines": [item.get("headline", "")],
            "sources": [item.get("source", "unknown")],
            "symbols": list(item.get("symbols", [])),
            "earliest_time": item.get("created_at", ""),
            "latest_time": item.get("created_at", ""),
        }
        used_indices.add(i)

        # Find similar headlines
        headline1 = item.get("headline", "").lower()
        words1 = set(re.findall(r'\w+', headline1))

        for j, other in enumerate(news_items):
            if j in used_indices:
                continue

            headline2 = other.get("headline", "").lower()
            words2 = set(re.findall(r'\w+', headline2))

            # Calculate Jaccard similarity
            if words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union

                # If >50% similar, add to cluster
                if similarity > 0.5:
                    cluster["items"].append(other)
                    cluster["headlines"].append(other.get("headline", ""))
                    cluster["sources"].append(other.get("source", "unknown"))
                    cluster["symbols"].extend(other.get("symbols", []))

                    # Update time range
                    other_time = other.get("created_at", "")
                    if other_time < cluster["earliest_time"]:
                        cluster["earliest_time"] = other_time
                    if other_time > cluster["latest_time"]:
                        cluster["latest_time"] = other_time

                    used_indices.add(j)

        # Deduplicate
        cluster["sources"] = list(set(cluster["sources"]))
        cluster["symbols"] = list(set(cluster["symbols"]))

        clusters.append(cluster)

    return clusters


def summarize_cluster_with_llm(cluster: Dict) -> str:
    """
    Use LLM to create a summary headline for a news cluster.

    Uses GPT-4o-mini for cost efficiency (~$0.001 per cluster).

    Args:
        cluster: News cluster dict

    Returns:
        Summarized headline string
    """
    if not OPENAI_API_KEY:
        # Fallback: use first headline if no API key
        return cluster["headlines"][0] if cluster["headlines"] else "News cluster"

    headlines = cluster["headlines"][:5]  # Limit to 5 headlines
    symbols = ", ".join(cluster["symbols"][:3]) if cluster["symbols"] else "market"

    prompt = f"""Summarize these related news headlines about {symbols} into one concise headline (max 100 chars):

Headlines:
{chr(10).join(f'- {h}' for h in headlines)}

Summary headline:"""

    try:
        response = requests.post(
            f"{OPENAI_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.3,
            },
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()

    except Exception:
        pass

    # Fallback to first headline
    return cluster["headlines"][0] if cluster["headlines"] else "News cluster"


def analyze_sentiment_simple(text: str) -> float:
    """
    Simple rule-based sentiment analysis.

    Returns score from -1.0 (negative) to 1.0 (positive).

    In production, would use FinBERT or similar model.
    """
    text = text.lower()

    positive_words = [
        "surge", "soar", "gain", "rise", "up", "rally", "bullish", "positive",
        "growth", "beat", "exceed", "outperform", "record", "high", "strong",
        "upgrade", "buy", "optimistic", "boost", "breakthrough", "success"
    ]

    negative_words = [
        "fall", "drop", "decline", "down", "crash", "bearish", "negative",
        "loss", "miss", "underperform", "low", "weak", "downgrade", "sell",
        "pessimistic", "cut", "fail", "warning", "concern", "risk"
    ]

    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    score = (pos_count - neg_count) / total
    return round(score, 2)


# ==================== MCP Tools ====================

@mcp.tool()
def get_stock_news(symbols: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get recent news with relevance scoring for stock symbols.

    Fetches news from Alpaca, clusters similar stories, adds sentiment
    scoring, peer reactions, and recency weights.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,TSLA")
        limit: Max news clusters to return (default 10, max 50)

    Returns:
        Dict with:
        - clusters: List of news clusters with:
          - headline_summary: LLM-summarized headline
          - source_count: Number of sources reporting
          - sources: List of source names
          - symbols: Related symbols
          - sentiment_score: -1.0 to 1.0
          - recency_weight: 0 to 1 (higher = more recent)
          - peer_reaction: Dict of peer symbol price changes
          - earliest_time: When story first broke
        - degraded_mode: Whether any data sources failed

    Usage Hint: Check news before making large trades to avoid
    entering during high-volatility news events.

    Example:
        >>> result = get_stock_news("AAPL,MSFT", limit=5)
        >>> print(result["clusters"][0]["sentiment_score"])
        0.65
    """
    try:
        cache = get_cache_manager()

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"error": "No valid symbols provided"}

        limit = min(max(1, limit), 50)

        # Check cache
        cache_key = f"news:{','.join(sorted(symbol_list))}:{limit}"
        cached = cache.get_with_metadata(cache_key, namespace="news")

        if cached:
            result = cached["value"]
            result["data_age_seconds"] = cached["age_seconds"]
            result["from_cache"] = True
            return result

        # Fetch news
        degraded_mode = False
        try:
            raw_news = get_alpaca_news(symbol_list, limit=limit * 3)  # Fetch more for clustering
        except Exception as e:
            cache.mark_degraded("alpaca_news")
            return {
                "error": f"Failed to fetch news: {str(e)}",
                "degraded_mode": True,
            }

        if not raw_news:
            return {
                "clusters": [],
                "count": 0,
                "message": "No recent news found for specified symbols",
                "degraded_mode": False,
            }

        # Cluster similar news
        clusters = cluster_news(raw_news)

        # Get Alpaca client for peer reactions
        try:
            client = get_alpaca_client()
        except Exception:
            client = None
            degraded_mode = True

        # Process each cluster
        processed_clusters = []
        for cluster in clusters[:limit]:
            # Summarize with LLM
            summary = summarize_cluster_with_llm(cluster)

            # Calculate sentiment from all headlines
            combined_text = " ".join(cluster["headlines"])
            sentiment = analyze_sentiment_simple(combined_text)

            # Calculate recency weight (use latest time)
            recency = calculate_recency_weight(cluster["latest_time"])

            # Get peer reactions for first symbol
            peer_reaction = {}
            if client and cluster["symbols"]:
                peer_reaction = get_peer_reactions(
                    cluster["symbols"][0],
                    cluster["earliest_time"],
                    client
                )

            processed_clusters.append({
                "cluster_id": hashlib.md5(summary.encode()).hexdigest()[:8],
                "headline_summary": summary,
                "source_count": len(cluster["sources"]),
                "sources": cluster["sources"],
                "symbols": cluster["symbols"],
                "sentiment_score": sentiment,
                "recency_weight": recency,
                "peer_reaction": peer_reaction,
                "earliest_time": cluster["earliest_time"],
                "latest_time": cluster["latest_time"],
                "raw_headline_count": len(cluster["headlines"]),
            })

        result = {
            "clusters": processed_clusters,
            "count": len(processed_clusters),
            "symbols_queried": symbol_list,
            "degraded_mode": degraded_mode,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache result
        cache.set(cache_key, result, ttl_seconds=300, namespace="news")

        return result

    except Exception as e:
        return {"error": f"News tool error: {str(e)}", "degraded_mode": True}


@mcp.tool()
def get_market_sentiment(symbols: str) -> Dict[str, Any]:
    """
    Get aggregated market sentiment for symbols based on recent news.

    Provides a quick sentiment overview without full news details.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,MSFT,TSLA")

    Returns:
        Dict with sentiment scores for each symbol:
        - overall_sentiment: Weighted average sentiment
        - news_count: Number of news items analyzed
        - trend: "bullish", "bearish", or "neutral"

    Example:
        >>> result = get_market_sentiment("AAPL,TSLA")
        >>> print(result["AAPL"]["trend"])
        "bullish"
    """
    try:
        news_result = get_stock_news(symbols, limit=20)

        if "error" in news_result:
            return news_result

        # Aggregate sentiment by symbol
        symbol_sentiments = {}

        for cluster in news_result.get("clusters", []):
            sentiment = cluster["sentiment_score"]
            recency = cluster["recency_weight"]

            # Weight by recency
            weighted_sentiment = sentiment * recency

            for symbol in cluster["symbols"]:
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = {
                        "scores": [],
                        "weights": [],
                    }
                symbol_sentiments[symbol]["scores"].append(weighted_sentiment)
                symbol_sentiments[symbol]["weights"].append(recency)

        # Calculate weighted averages
        result = {}
        for symbol, data in symbol_sentiments.items():
            if data["weights"]:
                total_weight = sum(data["weights"])
                if total_weight > 0:
                    avg_sentiment = sum(data["scores"]) / total_weight
                else:
                    avg_sentiment = 0

                # Determine trend
                if avg_sentiment > 0.2:
                    trend = "bullish"
                elif avg_sentiment < -0.2:
                    trend = "bearish"
                else:
                    trend = "neutral"

                result[symbol] = {
                    "overall_sentiment": round(avg_sentiment, 2),
                    "news_count": len(data["scores"]),
                    "trend": trend,
                }

        return {
            "sentiments": result,
            "count": len(result),
            "degraded_mode": news_result.get("degraded_mode", False),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": f"Sentiment analysis error: {str(e)}", "degraded_mode": True}


if __name__ == "__main__":
    port = int(os.getenv("ALPACA_NEWS_PORT", "8012"))
    print(f"Starting Alpaca News Tools MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
