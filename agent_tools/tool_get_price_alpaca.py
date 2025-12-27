"""
Alpaca Price Data Tool (MCP)

Provides real-time and historical price data through Alpaca Data API.
Replaces the local price data tool for live trading.
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

mcp = FastMCP("AlpacaPrices")


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
        symbol: Crypto pair (e.g., "BTC/USD", "ETH/USD")
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

        client = get_alpaca_client()
        bars = client.get_crypto_bars(symbol, timeframe="1Day", limit=days)

        if not bars:
            return {
                "error": f"No historical data found for {symbol}",
                "symbol": symbol,
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


if __name__ == "__main__":
    port = int(os.getenv("ALPACA_PRICE_PORT", "8010"))
    print(f"Starting Alpaca Price Tools MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
