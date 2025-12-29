"""
Trade Flow Analysis Tool (MCP)

Provides trade-level data analysis including:
- Recent trades with venue classification
- Block trade detection (0.05% of ADV)
- Lit vs dark pool volume breakdown
- Buy/sell pressure estimation
- VWAP calculation

Features:
- ADV-based institutional activity detection
- Dark pool volume tracking
- Real-time trade flow analysis
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_tools.alpaca_client import AlpacaClientError, get_alpaca_client
from agent_tools.cache import get_cache_manager

mcp = FastMCP("TradeFlow")


# ==================== Configuration ====================

ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v2")

# Block trade threshold: 0.05% of ADV
BLOCK_THRESHOLD_PCT = 0.0005

# ADV calculation period (days)
ADV_LOOKBACK_DAYS = 20

# Dark pool exchange identifiers
DARK_POOL_EXCHANGES = {
    "D",   # FINRA ADF (dark pool)
    "B",   # NASDAQ OMX BX (often dark liquidity)
    "Z",   # BATS BZX (has dark orders)
}


# ==================== Helper Functions ====================

def get_recent_trades_from_api(
    symbol: str,
    limit: int = 1000,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch recent trades from Alpaca Trades API.

    Args:
        symbol: Stock symbol
        limit: Maximum trades to fetch
        start: Start datetime
        end: End datetime

    Returns:
        List of trade records
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise AlpacaClientError("Alpaca API credentials not configured")

    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(minutes=30)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": min(limit, 10000),
    }

    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/trades"

    response = requests.get(url, headers=headers, params=params, timeout=15)

    if response.status_code != 200:
        raise AlpacaClientError(f"Trades API error: {response.status_code} - {response.text}")

    data = response.json()
    return data.get("trades", [])


def calculate_adv(symbol: str, client) -> Optional[float]:
    """
    Calculate 20-day Average Daily Volume.

    Args:
        symbol: Stock symbol
        client: Alpaca client

    Returns:
        Average daily volume or None if unavailable
    """
    cache = get_cache_manager()

    # Check cache first (ADV cached for 24 hours)
    cache_key = f"adv:{symbol}"
    cached = cache.get(cache_key, namespace="adv")
    if cached is not None:
        return cached

    try:
        bars = client.get_bars(symbol, timeframe="1Day", limit=ADV_LOOKBACK_DAYS)

        if not bars:
            return None

        volumes = [bar["volume"] for bar in bars if bar.get("volume")]
        if not volumes:
            return None

        adv = sum(volumes) / len(volumes)

        # Cache for 24 hours
        cache.set(cache_key, adv, ttl_seconds=86400, namespace="adv")

        return adv

    except Exception:
        return None


def classify_venue(exchange: str) -> str:
    """Classify trade venue as lit or dark"""
    if exchange in DARK_POOL_EXCHANGES:
        return "dark"
    return "lit"


def estimate_trade_direction(trade: Dict) -> str:
    """
    Estimate if trade was buyer or seller initiated.

    Uses simple heuristic based on trade conditions.
    More accurate methods would require quote data.
    """
    conditions = trade.get("conditions", [])

    # @: Regular sale
    # F: Intermarket sweep (usually aggressive)
    # I: Odd lot (often retail)

    if "F" in conditions:
        return "aggressive"
    elif "I" in conditions:
        return "retail"
    else:
        return "unknown"


def calculate_vwap(trades: List[Dict]) -> Optional[float]:
    """Calculate VWAP from trade list"""
    if not trades:
        return None

    total_value = 0
    total_volume = 0

    for trade in trades:
        price = trade.get("price", 0)
        size = trade.get("size", 0)
        if price > 0 and size > 0:
            total_value += price * size
            total_volume += size

    if total_volume == 0:
        return None

    return round(total_value / total_volume, 4)


# ==================== MCP Tools ====================

@mcp.tool()
def get_recent_trades(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """
    Get recent trades for a symbol with venue classification.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        limit: Number of trades to return (default 100, max 1000)

    Returns:
        Dict with:
        - trades: List of recent trades with:
          - timestamp: Trade time
          - price: Execution price
          - size: Trade size
          - exchange: Exchange code
          - venue_type: "lit" or "dark"
          - conditions: Trade conditions
        - count: Number of trades
        - vwap: Volume-weighted average price

    Example:
        >>> result = get_recent_trades("AAPL", 50)
        >>> print(result["trades"][0])
        {"timestamp": "...", "price": 185.50, "size": 100, "venue_type": "lit"}
    """
    try:
        limit = min(max(1, limit), 1000)

        # Fetch trades
        raw_trades = get_recent_trades_from_api(symbol, limit=limit)

        if not raw_trades:
            return {
                "symbol": symbol,
                "trades": [],
                "count": 0,
                "message": "No recent trades found",
            }

        # Process trades
        processed_trades = []
        for trade in raw_trades[:limit]:
            exchange = trade.get("x", "")
            venue_type = classify_venue(exchange)

            processed_trades.append({
                "timestamp": trade.get("t"),
                "price": trade.get("p"),
                "size": trade.get("s"),
                "exchange": exchange,
                "venue_type": venue_type,
                "conditions": trade.get("c", []),
                "direction_hint": estimate_trade_direction(trade),
            })

        # Calculate VWAP
        vwap = calculate_vwap([{"price": t.get("p"), "size": t.get("s")} for t in raw_trades])

        return {
            "symbol": symbol,
            "trades": processed_trades,
            "count": len(processed_trades),
            "vwap": vwap,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Failed to get trades: {str(e)}", "symbol": symbol}


@mcp.tool()
def analyze_trade_flow(symbol: str, minutes: int = 5) -> Dict[str, Any]:
    """
    Analyze recent trade flow with institutional activity detection.

    Detects large block trades (>0.05% of ADV) and breaks down
    volume by lit vs dark pool venues.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        minutes: Analysis window in minutes (default 5, max 60)

    Returns:
        Dict with:
        - total_volume: Total shares traded
        - lit_volume: Volume on lit exchanges
        - dark_pool_volume: Volume on dark pools
        - dark_pool_pct: Dark pool percentage
        - avg_trade_size: Average trade size
        - large_block_count: Trades exceeding 0.05% ADV
        - large_blocks: Details of large block trades
        - buy_sell_ratio: Estimated buy/sell pressure (0-1)
        - vwap: Volume-weighted average price

    Usage Hint: Check before large orders to assess
    institutional activity and market depth.

    Example:
        >>> result = analyze_trade_flow("AAPL", 10)
        >>> print(result["dark_pool_pct"])
        18.5
    """
    try:
        cache = get_cache_manager()
        minutes = min(max(1, minutes), 60)

        # Check cache (1-minute TTL for trade flow)
        cache_key = f"flow:{symbol}:{minutes}"
        cached = cache.get_with_metadata(cache_key, namespace="trades")

        if cached:
            result = cached["value"]
            result["data_age_seconds"] = cached["age_seconds"]
            result["from_cache"] = True
            return result

        # Fetch trades for the time window
        end = datetime.utcnow()
        start = end - timedelta(minutes=minutes)

        raw_trades = get_recent_trades_from_api(symbol, limit=10000, start=start, end=end)

        if not raw_trades:
            return {
                "symbol": symbol,
                "total_volume": 0,
                "message": "No trades in specified window",
                "degraded_mode": False,
            }

        # Get ADV for block detection
        client = get_alpaca_client()
        adv = calculate_adv(symbol, client)
        block_threshold = adv * BLOCK_THRESHOLD_PCT if adv else None

        # Analyze trades
        total_volume = 0
        lit_volume = 0
        dark_volume = 0
        trade_sizes = []
        large_blocks = []

        price_volume_sum = 0  # For VWAP

        for trade in raw_trades:
            size = trade.get("s", 0)
            price = trade.get("p", 0)
            exchange = trade.get("x", "")

            total_volume += size
            trade_sizes.append(size)
            price_volume_sum += price * size

            venue_type = classify_venue(exchange)
            if venue_type == "dark":
                dark_volume += size
            else:
                lit_volume += size

            # Check for large block
            if block_threshold and size >= block_threshold:
                large_blocks.append({
                    "size": size,
                    "price": price,
                    "venue": exchange,
                    "venue_type": venue_type,
                    "timestamp": trade.get("t"),
                    "pct_of_adv": round((size / adv) * 100, 4) if adv else None,
                })

        # Calculate metrics
        avg_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0
        dark_pool_pct = (dark_volume / total_volume * 100) if total_volume > 0 else 0
        vwap = price_volume_sum / total_volume if total_volume > 0 else None

        # Estimate buy/sell pressure (simplified)
        # In production, would compare to quote data
        # Here we use trade size distribution as proxy
        median_size = sorted(trade_sizes)[len(trade_sizes) // 2] if trade_sizes else 0
        large_trade_volume = sum(s for s in trade_sizes if s > median_size * 2)
        buy_sell_ratio = 0.5  # Default neutral

        if total_volume > 0:
            # Large trades often indicate institutional direction
            large_pct = large_trade_volume / total_volume
            # Slight bias based on momentum would be added here
            buy_sell_ratio = 0.5 + (large_pct * 0.1)

        result = {
            "symbol": symbol,
            "analysis_window_minutes": minutes,
            "total_volume": total_volume,
            "lit_volume": lit_volume,
            "dark_pool_volume": dark_volume,
            "dark_pool_pct": round(dark_pool_pct, 1),
            "trade_count": len(raw_trades),
            "avg_trade_size": round(avg_trade_size, 1),
            "large_block_count": len(large_blocks),
            "large_blocks": large_blocks[:10],  # Top 10 blocks
            "buy_sell_ratio": round(buy_sell_ratio, 2),
            "vwap": round(vwap, 4) if vwap else None,
            "adv_20day": round(adv, 0) if adv else None,
            "block_threshold": round(block_threshold, 0) if block_threshold else None,
            "data_age_seconds": 0,
            "from_cache": False,
            "degraded_mode": False,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache for 1 minute
        cache.set(cache_key, result, ttl_seconds=60, namespace="trades")

        return result

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol, "degraded_mode": True}
    except Exception as e:
        return {"error": f"Trade flow analysis failed: {str(e)}", "symbol": symbol, "degraded_mode": True}


@mcp.tool()
def get_volume_profile(symbol: str, hours: int = 4) -> Dict[str, Any]:
    """
    Get volume profile broken down by price level.

    Useful for identifying support/resistance based on volume.

    Args:
        symbol: Stock symbol
        hours: Lookback period in hours (default 4, max 8)

    Returns:
        Dict with volume at each price level
    """
    try:
        hours = min(max(1, hours), 8)

        end = datetime.utcnow()
        start = end - timedelta(hours=hours)

        raw_trades = get_recent_trades_from_api(symbol, limit=10000, start=start, end=end)

        if not raw_trades:
            return {
                "symbol": symbol,
                "profile": {},
                "message": "No trades in specified window",
            }

        # Build volume profile (group by price level)
        profile = {}

        for trade in raw_trades:
            price = trade.get("p", 0)
            size = trade.get("s", 0)

            # Round to nearest 0.10 for grouping
            price_level = round(price, 1)

            if price_level not in profile:
                profile[price_level] = {
                    "volume": 0,
                    "trade_count": 0,
                    "avg_size": 0,
                }

            profile[price_level]["volume"] += size
            profile[price_level]["trade_count"] += 1

        # Calculate avg size per level
        for level in profile:
            profile[level]["avg_size"] = round(
                profile[level]["volume"] / profile[level]["trade_count"], 1
            )

        # Sort by volume (descending)
        sorted_levels = sorted(
            profile.items(),
            key=lambda x: x[1]["volume"],
            reverse=True
        )

        # Find POC (Point of Control - highest volume price)
        poc = sorted_levels[0][0] if sorted_levels else None

        return {
            "symbol": symbol,
            "hours": hours,
            "profile": dict(sorted_levels[:20]),  # Top 20 levels
            "poc_price": poc,
            "total_volume": sum(p[1]["volume"] for p in sorted_levels),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": f"Volume profile failed: {str(e)}", "symbol": symbol}


if __name__ == "__main__":
    port = int(os.getenv("TRADE_FLOW_PORT", "8013"))
    print(f"Starting Trade Flow Analysis MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
