"""
Alpaca Price Data Tool (MCP)

Provides real-time and historical price data through Alpaca Data API.
Replaces the local price data tool for live trading.

Features:
- Real-time quotes (15-min delayed on free tier)
- Historical OHLCV bars
- Market snapshots with 5-minute caching
- Intraday data
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_tools.alpaca_client import (
    AlpacaClientError,
    get_alpaca_client,
)
from agent_tools.cache import get_cache_manager

mcp = FastMCP("AlpacaPrices")


# ==================== Symbol Normalization ====================

def normalize_crypto_symbol(symbol: str) -> str:
    """
    Normalize crypto symbol to Alpaca's required format (XXX/USD).

    Alpaca requires crypto symbols in format "BTC/USD", not "BTCUSD".
    This function handles common variations:
    - BTCUSD -> BTC/USD
    - btc/usd -> BTC/USD
    - BTC-USD -> BTC/USD
    - BTC -> BTC/USD (assumes USD pair)

    Args:
        symbol: Crypto symbol in any common format

    Returns:
        Normalized symbol in format "XXX/USD"
    """
    symbol = symbol.upper().strip()

    # Already in correct format
    if "/" in symbol:
        return symbol

    # Handle BTC-USD format (must check before BTCUSD format)
    if "-" in symbol:
        return symbol.replace("-", "/")

    # Handle BTCUSD format (no separator)
    if symbol.endswith("USD") and len(symbol) > 3:
        base = symbol[:-3]
        return f"{base}/USD"

    # Handle bare symbol like "BTC" - assume USD pair
    if len(symbol) <= 5 and symbol.isalpha():
        return f"{symbol}/USD"

    return symbol


# ==================== Common Stock Lists ====================

NASDAQ_100_SYMBOLS = [
    "NVDA", "MSFT", "AAPL", "GOOG", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "NFLX",
    "PLTR", "COST", "ASML", "AMD", "CSCO", "AZN", "TMUS", "MU", "LIN", "PEP",
    "SHOP", "APP", "INTU", "AMAT", "LRCX", "PDD", "QCOM", "ARM", "INTC", "BKNG",
    "AMGN", "TXN", "ISRG", "GILD", "KLAC", "PANW", "ADBE", "HON", "CRWD", "CEG",
    "ADI", "ADP", "DASH", "CMCSA", "VRTX", "MELI", "SBUX", "CDNS", "ORLY", "SNPS",
]

TOP_20_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "COST", "PEP", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "AMGN", "SBUX", "PYPL",
]


# ==================== MCP Tools ====================

@mcp.tool()
def get_price(symbol: str) -> Dict[str, Any]:
    """
    Get real-time quote for a stock symbol.

    Returns the current bid/ask prices and sizes from Alpaca's data feed.
    Note: With free IEX data, prices may be delayed ~15 minutes.

    Args:
        symbol: Stock symbol (e.g., "AAPL", "MSFT", "NVDA")

    Returns:
        Dict with:
        - symbol: Stock symbol
        - bid_price: Current bid price
        - ask_price: Current ask price
        - bid_size: Bid size
        - ask_size: Ask size
        - mid_price: Mid-point price
        - spread: Bid-ask spread
        - timestamp: Quote timestamp

    Example:
        >>> result = get_price("AAPL")
        >>> print(result)
        {
            "symbol": "AAPL",
            "bid_price": 185.45,
            "ask_price": 185.50,
            "mid_price": 185.475,
            "spread": 0.05,
            "timestamp": "2025-01-15T10:30:00Z"
        }
    """
    try:
        client = get_alpaca_client()
        quote = client.get_quote(symbol)

        bid = quote["bid_price"]
        ask = quote["ask_price"]

        return {
            "symbol": symbol,
            "bid_price": bid,
            "ask_price": ask,
            "bid_size": quote["bid_size"],
            "ask_size": quote["ask_size"],
            "mid_price": (bid + ask) / 2 if bid > 0 and ask > 0 else None,
            "spread": ask - bid if bid > 0 and ask > 0 else None,
            "timestamp": quote["timestamp"],
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Failed to get price for {symbol}: {str(e)}", "symbol": symbol}


@mcp.tool()
def get_prices(symbols: str) -> Dict[str, Any]:
    """
    Get real-time quotes for multiple stock symbols.

    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,NVDA")

    Returns:
        Dict mapping symbol to quote data

    Example:
        >>> result = get_prices("AAPL,MSFT,NVDA")
        >>> print(result["AAPL"]["bid_price"])
        185.45
    """
    try:
        # Parse comma-separated symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"error": "No valid symbols provided"}

        if len(symbol_list) > 50:
            return {"error": "Maximum 50 symbols allowed per request"}

        client = get_alpaca_client()
        quotes = client.get_quotes(symbol_list)

        result = {}
        for symbol, quote in quotes.items():
            bid = quote["bid_price"]
            ask = quote["ask_price"]
            result[symbol] = {
                "bid_price": bid,
                "ask_price": ask,
                "mid_price": (bid + ask) / 2 if bid > 0 and ask > 0 else None,
                "spread": ask - bid if bid > 0 and ask > 0 else None,
                "timestamp": quote["timestamp"],
            }

        return {"quotes": result, "count": len(result)}

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to get prices: {str(e)}"}


@mcp.tool()
def get_price_history(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Get historical daily price data for a stock.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        days: Number of days of history (default 30, max 365)

    Returns:
        Dict with:
        - symbol: Stock symbol
        - bars: List of OHLCV bars
        - count: Number of bars

    Example:
        >>> result = get_price_history("AAPL", 7)
        >>> print(result["bars"][0])
        {
            "date": "2025-01-10",
            "open": 183.50,
            "high": 186.00,
            "low": 182.00,
            "close": 185.45,
            "volume": 45000000
        }
    """
    try:
        if days <= 0:
            return {"error": "Days must be positive"}
        if days > 365:
            days = 365

        client = get_alpaca_client()
        bars = client.get_bars(symbol, timeframe="1Day", limit=days)

        if not bars:
            return {
                "error": f"No historical data found for {symbol}",
                "symbol": symbol,
            }

        # Format bars for easier consumption
        formatted_bars = []
        for bar in bars:
            formatted_bars.append({
                "date": bar["timestamp"].split("T")[0],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            })

        return {
            "symbol": symbol,
            "bars": formatted_bars,
            "count": len(formatted_bars),
            "start_date": formatted_bars[0]["date"] if formatted_bars else None,
            "end_date": formatted_bars[-1]["date"] if formatted_bars else None,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Failed to get history for {symbol}: {str(e)}", "symbol": symbol}


@mcp.tool()
def get_crypto_price_history(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Get historical daily price data for a cryptocurrency.

    Args:
        symbol: Crypto pair (e.g., "BTC/USD", "ETH/USD", "BTCUSD" - auto-normalized)
        days: Number of days of history (default 30, max 365)

    Returns:
        Dict with:
        - symbol: Crypto symbol
        - bars: List of OHLCV bars
        - count: Number of bars

    Example:
        >>> result = get_crypto_price_history("BTC/USD", 7)
        >>> print(result["bars"][0])
        {
            "date": "2025-01-10",
            "open": 95000.50,
            "high": 96500.00,
            "low": 94000.00,
            "close": 96200.45,
            "volume": 15000.5
        }
    """
    try:
        if days <= 0:
            return {"error": "Days must be positive"}
        if days > 365:
            days = 365

        # Normalize symbol format (BTCUSD -> BTC/USD)
        normalized_symbol = normalize_crypto_symbol(symbol)

        client = get_alpaca_client()
        bars = client.get_crypto_bars(normalized_symbol, timeframe="1Day", limit=days)

        if not bars:
            return {
                "error": f"No historical data found for {normalized_symbol}",
                "symbol": normalized_symbol,
                "original_symbol": symbol,
            }

        # Format bars for easier consumption
        formatted_bars = []
        for bar in bars:
            formatted_bars.append({
                "date": bar["timestamp"].split("T")[0] if "T" in bar["timestamp"] else bar["timestamp"][:10],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
            })

        result = {
            "symbol": normalized_symbol,
            "bars": formatted_bars,
            "count": len(formatted_bars),
            "start_date": formatted_bars[0]["date"] if formatted_bars else None,
            "end_date": formatted_bars[-1]["date"] if formatted_bars else None,
        }

        # Include original symbol if it was normalized
        if normalized_symbol != symbol.upper():
            result["original_symbol"] = symbol
            result["note"] = f"Symbol normalized from '{symbol}' to '{normalized_symbol}'"

        return result

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Failed to get crypto history for {symbol}: {str(e)}", "symbol": symbol}


@mcp.tool()
def get_market_status() -> Dict[str, Any]:
    """
    Get current market status (open/closed) and trading hours.

    Returns:
        Dict with:
        - is_open: Whether market is currently open
        - current_time: Current time
        - next_open: Next market open time (if closed)
        - next_close: Next market close time (if open)

    Example:
        >>> result = get_market_status()
        >>> print(result)
        {
            "is_open": true,
            "current_time": "2025-01-15T10:30:00-05:00",
            "next_close": "2025-01-15T16:00:00-05:00"
        }
    """
    try:
        client = get_alpaca_client()
        clock = client.get_clock()

        return {
            "is_open": clock["is_open"],
            "current_time": clock["timestamp"],
            "next_open": clock["next_open"] if not clock["is_open"] else None,
            "next_close": clock["next_close"] if clock["is_open"] else None,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to get market status: {str(e)}"}


@mcp.tool()
def get_top_stocks_prices() -> Dict[str, Any]:
    """
    Get real-time quotes for top 20 most traded stocks.

    Useful for getting a quick overview of market activity.

    Returns:
        Dict with quotes for AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, etc.
    """
    try:
        client = get_alpaca_client()
        quotes = client.get_quotes(TOP_20_SYMBOLS)

        result = {}
        for symbol in TOP_20_SYMBOLS:
            if symbol in quotes:
                quote = quotes[symbol]
                bid = quote["bid_price"]
                ask = quote["ask_price"]
                result[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else None,
                }

        return {
            "quotes": result,
            "count": len(result),
            "market_is_open": client.is_market_open(),
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to get top stocks: {str(e)}"}


@mcp.tool()
def get_intraday_bars(symbol: str, timeframe: str = "1Hour", limit: int = 24) -> Dict[str, Any]:
    """
    Get intraday price bars for a stock.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        timeframe: Bar timeframe - "1Min", "5Min", "15Min", "1Hour" (default "1Hour")
        limit: Number of bars to return (default 24, max 1000)

    Returns:
        Dict with intraday OHLCV bars

    Example:
        >>> result = get_intraday_bars("AAPL", "15Min", 16)
        >>> print(result["bars"][0])
        {
            "timestamp": "2025-01-15T10:00:00Z",
            "open": 185.00,
            "high": 185.50,
            "low": 184.80,
            "close": 185.30,
            "volume": 1500000
        }
    """
    try:
        if limit <= 0:
            return {"error": "Limit must be positive"}
        if limit > 1000:
            limit = 1000

        valid_timeframes = ["1Min", "5Min", "15Min", "1Hour"]
        if timeframe not in valid_timeframes:
            return {
                "error": f"Invalid timeframe. Use one of: {valid_timeframes}",
                "symbol": symbol,
            }

        client = get_alpaca_client()
        bars = client.get_bars(symbol, timeframe=timeframe, limit=limit)

        if not bars:
            return {
                "error": f"No intraday data found for {symbol}",
                "symbol": symbol,
            }

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars,
            "count": len(bars),
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Failed to get intraday bars for {symbol}: {str(e)}", "symbol": symbol}


@mcp.tool()
def compare_prices(symbols: str) -> Dict[str, Any]:
    """
    Compare current prices and calculate percentage changes from previous close.

    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,NVDA")

    Returns:
        Dict with price comparison including daily change percentages
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"error": "No valid symbols provided"}

        if len(symbol_list) > 20:
            return {"error": "Maximum 20 symbols for comparison"}

        client = get_alpaca_client()

        # Get current quotes
        quotes = client.get_quotes(symbol_list)

        # Get yesterday's close for each symbol
        result = {}
        for symbol in symbol_list:
            if symbol not in quotes:
                result[symbol] = {"error": "Quote not available"}
                continue

            quote = quotes[symbol]
            current_price = (quote["bid_price"] + quote["ask_price"]) / 2

            # Get yesterday's bar for close price
            try:
                bars = client.get_bars(symbol, timeframe="1Day", limit=2)
                if bars and len(bars) >= 1:
                    prev_close = bars[-1]["close"]  # Most recent complete bar
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

                    result[symbol] = {
                        "current_price": round(current_price, 2),
                        "prev_close": round(prev_close, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                    }
                else:
                    result[symbol] = {
                        "current_price": round(current_price, 2),
                        "prev_close": None,
                        "change": None,
                        "change_pct": None,
                    }
            except Exception:
                result[symbol] = {
                    "current_price": round(current_price, 2),
                    "prev_close": None,
                    "change": None,
                    "change_pct": None,
                }

        return {
            "comparison": result,
            "count": len(result),
            "timestamp": datetime.now().isoformat(),
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to compare prices: {str(e)}"}


@mcp.tool()
def get_market_snapshot(symbols: str) -> Dict[str, Any]:
    """
    Get complete market snapshot for symbols with consolidated data.

    Returns a comprehensive view of market state in a single API call,
    reducing the need for multiple separate requests. Data is cached
    for 5 minutes to optimize API usage.

    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,NVDA")
                 Supports both stocks and crypto (e.g., "AAPL,BTC/USD")

    Returns:
        Dict with complete market state for each symbol:
        - latest_quote: Current bid/ask prices
        - daily_bar: Today's OHLCV with VWAP
        - prev_bar: Previous day's bar
        - metrics: Calculated metrics (change %, spread, momentum)
        - data_age_seconds: How old the cached data is
        - degraded_mode: Whether any data sources failed

    Usage Hint: Use before trading to get complete picture
    in single call instead of multiple API requests.

    Example:
        >>> result = get_market_snapshot("AAPL,MSFT")
        >>> print(result["AAPL"]["metrics"]["change_pct"])
        1.25
    """
    try:
        cache = get_cache_manager()

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"error": "No valid symbols provided"}

        if len(symbol_list) > 50:
            return {"error": "Maximum 50 symbols allowed per request"}

        # Check cache for each symbol
        result = {}
        uncached_symbols = []
        degraded_mode = False

        for symbol in symbol_list:
            cache_key = f"snapshot:{symbol}"
            cached = cache.get_with_metadata(cache_key, namespace="snapshot")

            if cached:
                result[symbol] = cached["value"]
                result[symbol]["data_age_seconds"] = cached["age_seconds"]
                result[symbol]["from_cache"] = True
            else:
                uncached_symbols.append(symbol)

        # Fetch uncached data
        if uncached_symbols:
            client = get_alpaca_client()

            # Separate crypto and stock symbols
            crypto_symbols = [s for s in uncached_symbols if client._is_crypto_symbol(s)]
            stock_symbols = [s for s in uncached_symbols if not client._is_crypto_symbol(s)]

            # Fetch stock data
            if stock_symbols:
                try:
                    quotes = client.get_quotes(stock_symbols)

                    for symbol in stock_symbols:
                        snapshot = _build_stock_snapshot(client, symbol, quotes.get(symbol))
                        if snapshot:
                            result[symbol] = snapshot
                            # Cache the result
                            cache.set(f"snapshot:{symbol}", snapshot, ttl_seconds=300, namespace="snapshot")
                except Exception as e:
                    degraded_mode = True
                    cache.mark_degraded("alpaca_stock_data")
                    for symbol in stock_symbols:
                        result[symbol] = {
                            "error": str(e),
                            "degraded_mode": True,
                        }

            # Fetch crypto data
            if crypto_symbols:
                try:
                    for symbol in crypto_symbols:
                        normalized = normalize_crypto_symbol(symbol)
                        quote = client.get_quote(normalized)
                        snapshot = _build_crypto_snapshot(client, normalized, quote)
                        if snapshot:
                            result[symbol] = snapshot
                            cache.set(f"snapshot:{symbol}", snapshot, ttl_seconds=300, namespace="snapshot")
                except Exception as e:
                    degraded_mode = True
                    cache.mark_degraded("alpaca_crypto_data")
                    for symbol in crypto_symbols:
                        result[symbol] = {
                            "error": str(e),
                            "degraded_mode": True,
                        }

        # Check overall degraded status
        degraded_sources = cache.get_degraded_sources()
        if degraded_sources:
            degraded_mode = True

        return {
            "snapshots": result,
            "count": len(result),
            "degraded_mode": degraded_mode,
            "degraded_sources": list(degraded_sources.keys()) if degraded_sources else [],
            "timestamp": datetime.now().isoformat(),
            "hint": "Consider smaller positions when degraded_mode is true" if degraded_mode else None,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "degraded_mode": True}
    except Exception as e:
        return {"error": f"Failed to get snapshot: {str(e)}", "degraded_mode": True}


def _build_stock_snapshot(client, symbol: str, quote: Optional[Dict]) -> Optional[Dict]:
    """Build snapshot for a stock symbol"""
    try:
        if not quote:
            quote = client.get_quote(symbol)

        bid = quote.get("bid_price", 0)
        ask = quote.get("ask_price", 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        # Get daily bars (today and previous)
        bars = client.get_bars(symbol, timeframe="1Day", limit=2)

        prev_bar = None
        daily_bar = None
        if bars:
            if len(bars) >= 2:
                prev_bar = bars[-2]
                daily_bar = bars[-1]
            elif len(bars) == 1:
                daily_bar = bars[-1]

        # Calculate metrics
        change = 0
        change_pct = 0
        if prev_bar and mid > 0:
            prev_close = prev_bar["close"]
            change = mid - prev_close
            change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        # Calculate momentum (price change over last N bars)
        momentum = 0
        if bars and len(bars) >= 2:
            first_close = bars[0]["close"]
            last_close = bars[-1]["close"]
            if first_close > 0:
                momentum = ((last_close - first_close) / first_close) * 100

        return {
            "symbol": symbol,
            "type": "stock",
            "latest_quote": {
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 2),
                "spread": round(ask - bid, 4) if bid > 0 and ask > 0 else None,
                "timestamp": quote.get("timestamp"),
            },
            "daily_bar": {
                "open": daily_bar["open"] if daily_bar else None,
                "high": daily_bar["high"] if daily_bar else None,
                "low": daily_bar["low"] if daily_bar else None,
                "close": daily_bar["close"] if daily_bar else None,
                "volume": daily_bar["volume"] if daily_bar else None,
                "vwap": daily_bar.get("vwap") if daily_bar else None,
            } if daily_bar else None,
            "prev_bar": {
                "open": prev_bar["open"],
                "high": prev_bar["high"],
                "low": prev_bar["low"],
                "close": prev_bar["close"],
                "volume": prev_bar["volume"],
            } if prev_bar else None,
            "metrics": {
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "momentum": round(momentum, 2),
            },
            "data_age_seconds": 0,
            "from_cache": False,
            "degraded_mode": False,
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "degraded_mode": True,
        }


def _build_crypto_snapshot(client, symbol: str, quote: Optional[Dict]) -> Optional[Dict]:
    """Build snapshot for a crypto symbol"""
    try:
        if not quote:
            quote = client.get_quote(symbol)

        bid = quote.get("bid_price", 0)
        ask = quote.get("ask_price", 0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        # Get daily bars
        bars = client.get_crypto_bars(symbol, timeframe="1Day", limit=2)

        prev_bar = None
        daily_bar = None
        if bars:
            if len(bars) >= 2:
                prev_bar = bars[-2]
                daily_bar = bars[-1]
            elif len(bars) == 1:
                daily_bar = bars[-1]

        # Calculate metrics
        change = 0
        change_pct = 0
        if prev_bar and mid > 0:
            prev_close = prev_bar["close"]
            change = mid - prev_close
            change_pct = (change / prev_close) * 100 if prev_close > 0 else 0

        return {
            "symbol": symbol,
            "type": "crypto",
            "latest_quote": {
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 2),
                "spread": round(ask - bid, 4) if bid > 0 and ask > 0 else None,
                "timestamp": quote.get("timestamp"),
            },
            "daily_bar": {
                "open": daily_bar["open"] if daily_bar else None,
                "high": daily_bar["high"] if daily_bar else None,
                "low": daily_bar["low"] if daily_bar else None,
                "close": daily_bar["close"] if daily_bar else None,
                "volume": daily_bar["volume"] if daily_bar else None,
                "vwap": daily_bar.get("vwap") if daily_bar else None,
            } if daily_bar else None,
            "prev_bar": {
                "open": prev_bar["open"],
                "high": prev_bar["high"],
                "low": prev_bar["low"],
                "close": prev_bar["close"],
                "volume": prev_bar["volume"],
            } if prev_bar else None,
            "metrics": {
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            },
            "data_age_seconds": 0,
            "from_cache": False,
            "degraded_mode": False,
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "degraded_mode": True,
        }


if __name__ == "__main__":
    port = int(os.getenv("ALPACA_PRICE_PORT", "8010"))
    print(f"Starting Alpaca Price Tools MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
