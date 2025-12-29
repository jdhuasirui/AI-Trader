"""
Corporate Actions Tool (MCP)

Provides corporate action data including:
- Dividends (with ex-date risk adjustment)
- Stock splits
- Other corporate events

Features:
- Dynamic position limit adjustment by dividend yield
- Ex-date proximity warnings
- Risk-adjusted trading recommendations
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

mcp = FastMCP("CorporateActions")


# ==================== Configuration ====================

ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets/v1beta1")

# Ex-date risk adjustment by dividend yield
# Format: (min_yield, max_yield, reduction_pct, days_window)
DIVIDEND_RISK_RULES = [
    (0.00, 0.01, 0.00, 0),    # < 1%: No change
    (0.01, 0.03, 0.25, 1),    # 1-3%: 25% reduction, 1 day before/after
    (0.03, 0.05, 0.50, 1),    # 3-5%: 50% reduction, 1 day before/after
    (0.05, 1.00, 0.75, 2),    # > 5%: 75% reduction, 2 days before/after
]


# ==================== Helper Functions ====================

def get_corporate_actions_from_api(
    symbol: str,
    action_types: List[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch corporate actions from Alpaca API.

    Note: Alpaca's corporate actions API may require specific plan.
    This implementation provides a fallback using public data.

    Args:
        symbol: Stock symbol
        action_types: Types of actions to fetch
        start: Start date
        end: End date

    Returns:
        List of corporate action records
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise AlpacaClientError("Alpaca API credentials not configured")

    if end is None:
        end = datetime.utcnow() + timedelta(days=30)
    if start is None:
        start = datetime.utcnow() - timedelta(days=30)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    params = {
        "symbols": symbol,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
    }

    if action_types:
        params["types"] = ",".join(action_types)

    # Try the corporate actions endpoint
    url = f"{ALPACA_DATA_URL}/corporate-actions"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("corporate_actions", {}).get(symbol, [])
        elif response.status_code == 403:
            # Endpoint not available on current plan
            return []
        else:
            return []

    except Exception:
        return []


def get_dividend_data_fallback(symbol: str) -> List[Dict]:
    """
    Fallback: Estimate dividend data from price history.

    Uses price drops on known dividend dates to estimate yield.
    """
    cache = get_cache_manager()

    # Check cache
    cache_key = f"dividends_fallback:{symbol}"
    cached = cache.get(cache_key, namespace="corporate_actions")
    if cached is not None:
        return cached

    try:
        client = get_alpaca_client()

        # Get recent daily bars
        bars = client.get_bars(symbol, timeframe="1Day", limit=90)

        if not bars:
            return []

        # Look for significant price drops that might indicate ex-dividend
        # This is a rough estimate - proper API would be better
        dividends = []

        for i in range(1, len(bars)):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]

            # Calculate gap
            prev_close = prev_bar["close"]
            curr_open = curr_bar["open"]

            if prev_close > 0:
                gap_pct = (curr_open - prev_close) / prev_close * 100

                # If gap down between -0.5% and -5%, might be dividend
                if -5 < gap_pct < -0.5:
                    # Estimate dividend amount
                    dividend_estimate = prev_close - curr_open

                    if dividend_estimate > 0:
                        dividends.append({
                            "type": "dividend",
                            "ex_date": curr_bar["timestamp"].split("T")[0],
                            "amount": round(dividend_estimate, 4),
                            "estimated": True,
                            "source": "price_gap_detection",
                        })

        # Only keep most recent dividend (avoid false positives)
        dividends = dividends[-1:] if dividends else []

        # Cache for 1 hour
        cache.set(cache_key, dividends, ttl_seconds=3600, namespace="corporate_actions")

        return dividends

    except Exception:
        return []


def calculate_dividend_yield(symbol: str, dividend_amount: float) -> Optional[float]:
    """Calculate annual dividend yield based on current price"""
    try:
        client = get_alpaca_client()
        quote = client.get_quote(symbol)

        bid = quote.get("bid_price", 0)
        ask = quote.get("ask_price", 0)
        price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

        if price <= 0:
            return None

        # Annualize (assume quarterly dividend)
        annual_dividend = dividend_amount * 4
        yield_pct = annual_dividend / price

        return round(yield_pct, 4)

    except Exception:
        return None


def get_risk_adjustment(dividend_yield: float, ex_date: str) -> Dict[str, Any]:
    """
    Calculate position limit adjustment based on dividend yield and ex-date proximity.

    Args:
        dividend_yield: Annual dividend yield (0.05 = 5%)
        ex_date: Ex-dividend date string

    Returns:
        Risk adjustment recommendation
    """
    # Find matching rule
    reduction_pct = 0.0
    days_window = 0

    for min_y, max_y, reduction, window in DIVIDEND_RISK_RULES:
        if min_y <= dividend_yield < max_y:
            reduction_pct = reduction
            days_window = window
            break

    # Calculate days until ex-date
    try:
        ex_datetime = datetime.strptime(ex_date, "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        days_until = (ex_datetime - today).days
    except Exception:
        days_until = None

    # Determine if we're in the risk window
    in_risk_window = False
    if days_until is not None and days_window > 0:
        in_risk_window = -days_window <= days_until <= days_window

    return {
        "dividend_yield": round(dividend_yield * 100, 2),  # As percentage
        "position_limit_reduction": round(reduction_pct * 100, 0),  # As percentage
        "risk_window_days": days_window,
        "days_until_ex_date": days_until,
        "in_risk_window": in_risk_window,
        "recommendation": _get_recommendation(reduction_pct, in_risk_window, days_until),
    }


def _get_recommendation(reduction_pct: float, in_risk_window: bool, days_until: Optional[int]) -> str:
    """Generate human-readable recommendation"""
    if reduction_pct == 0:
        return "Normal trading - low dividend yield"

    if days_until is None:
        return "Unable to determine ex-date proximity"

    if in_risk_window:
        if reduction_pct >= 0.75:
            return f"HIGH RISK: Reduce position limit by 75%. Ex-date in {days_until} days."
        elif reduction_pct >= 0.50:
            return f"ELEVATED RISK: Reduce position limit by 50%. Ex-date in {days_until} days."
        else:
            return f"MODERATE RISK: Reduce position limit by 25%. Ex-date in {days_until} days."

    if days_until > 0:
        return f"Ex-date in {days_until} days. Monitor as date approaches."
    else:
        return "Ex-date has passed. Normal trading can resume."


# ==================== MCP Tools ====================

@mcp.tool()
def get_corporate_actions(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Get upcoming and recent corporate actions for a symbol.

    Returns dividends, splits, and other actions with risk
    adjustment recommendations based on ex-date proximity.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        days: Lookback/forward days (default 30)

    Returns:
        Dict with:
        - dividends: List of dividend records with:
          - ex_date: Ex-dividend date
          - amount: Dividend per share
          - dividend_yield: Annualized yield
          - risk_adjustment: Position limit recommendation
        - splits: List of stock split records
        - other_actions: Other corporate events
        - has_upcoming_action: Boolean flag
        - highest_risk_level: "low", "moderate", "elevated", or "high"

    Usage Hint: Check before opening new positions to
    avoid unexpected dividend-related price adjustments.

    Example:
        >>> result = get_corporate_actions("AAPL")
        >>> print(result["dividends"][0]["risk_adjustment"]["recommendation"])
        "MODERATE RISK: Reduce position limit by 25%..."
    """
    try:
        cache = get_cache_manager()
        days = min(max(7, days), 90)

        # Check cache
        cache_key = f"corp_actions:{symbol}:{days}"
        cached = cache.get_with_metadata(cache_key, namespace="corporate_actions")

        if cached:
            result = cached["value"]
            result["data_age_seconds"] = cached["age_seconds"]
            result["from_cache"] = True
            return result

        # Fetch corporate actions
        start = datetime.utcnow() - timedelta(days=days)
        end = datetime.utcnow() + timedelta(days=days)

        raw_actions = get_corporate_actions_from_api(symbol, start=start, end=end)

        # If API returns nothing, try fallback
        if not raw_actions:
            raw_actions = get_dividend_data_fallback(symbol)

        # Process actions
        dividends = []
        splits = []
        other_actions = []
        highest_risk = "low"

        for action in raw_actions:
            action_type = action.get("type", "").lower()

            if action_type == "dividend" or "dividend" in action_type:
                amount = action.get("amount", action.get("cash", 0))
                ex_date = action.get("ex_date", action.get("ex-date", ""))

                if not ex_date:
                    continue

                # Calculate yield
                div_yield = calculate_dividend_yield(symbol, amount)

                # Get risk adjustment
                risk_adj = get_risk_adjustment(div_yield or 0, ex_date)

                # Update highest risk
                if risk_adj["position_limit_reduction"] >= 75:
                    highest_risk = "high"
                elif risk_adj["position_limit_reduction"] >= 50 and highest_risk != "high":
                    highest_risk = "elevated"
                elif risk_adj["position_limit_reduction"] >= 25 and highest_risk not in ["high", "elevated"]:
                    highest_risk = "moderate"

                dividends.append({
                    "ex_date": ex_date,
                    "amount": amount,
                    "dividend_yield": div_yield,
                    "record_date": action.get("record_date"),
                    "pay_date": action.get("pay_date"),
                    "estimated": action.get("estimated", False),
                    "risk_adjustment": risk_adj,
                })

            elif action_type == "split":
                splits.append({
                    "ex_date": action.get("ex_date"),
                    "old_rate": action.get("old_rate"),
                    "new_rate": action.get("new_rate"),
                    "ratio": action.get("ratio"),
                })

            else:
                other_actions.append({
                    "type": action_type,
                    "date": action.get("date") or action.get("ex_date"),
                    "details": action,
                })

        # Check for upcoming actions
        today = datetime.now().date()
        has_upcoming = False

        for div in dividends:
            try:
                ex_date = datetime.strptime(div["ex_date"], "%Y-%m-%d").date()
                if ex_date >= today:
                    has_upcoming = True
                    break
            except Exception:
                continue

        result = {
            "symbol": symbol,
            "dividends": dividends,
            "splits": splits,
            "other_actions": other_actions,
            "has_upcoming_action": has_upcoming,
            "highest_risk_level": highest_risk,
            "analysis_window_days": days,
            "data_age_seconds": 0,
            "from_cache": False,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache for 1 hour
        cache.set(cache_key, result, ttl_seconds=3600, namespace="corporate_actions")

        return result

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Corporate actions failed: {str(e)}", "symbol": symbol}


@mcp.tool()
def check_ex_date_risk(symbols: str) -> Dict[str, Any]:
    """
    Quick check for ex-date risk across multiple symbols.

    Provides a summary of which symbols have upcoming dividends
    that might affect trading decisions.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,MSFT,KO")

    Returns:
        Dict with risk assessment for each symbol:
        - risk_level: "low", "moderate", "elevated", or "high"
        - has_upcoming_dividend: Boolean
        - days_until_ex_date: Days until next ex-date (if any)

    Example:
        >>> result = check_ex_date_risk("AAPL,KO,T")
        >>> print(result["KO"]["risk_level"])
        "elevated"
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        if not symbol_list:
            return {"error": "No valid symbols provided"}

        if len(symbol_list) > 20:
            return {"error": "Maximum 20 symbols allowed"}

        results = {}

        for symbol in symbol_list:
            try:
                actions = get_corporate_actions(symbol, days=14)

                if "error" in actions:
                    results[symbol] = {
                        "error": actions["error"],
                        "risk_level": "unknown",
                    }
                    continue

                # Find nearest upcoming dividend
                nearest_ex_date = None
                nearest_days = None

                today = datetime.now().date()

                for div in actions.get("dividends", []):
                    try:
                        ex_date = datetime.strptime(div["ex_date"], "%Y-%m-%d").date()
                        days_until = (ex_date - today).days

                        if days_until >= -1:  # Include yesterday (just passed)
                            if nearest_days is None or days_until < nearest_days:
                                nearest_days = days_until
                                nearest_ex_date = div["ex_date"]
                    except Exception:
                        continue

                results[symbol] = {
                    "risk_level": actions.get("highest_risk_level", "low"),
                    "has_upcoming_dividend": actions.get("has_upcoming_action", False),
                    "nearest_ex_date": nearest_ex_date,
                    "days_until_ex_date": nearest_days,
                }

            except Exception as e:
                results[symbol] = {
                    "error": str(e),
                    "risk_level": "unknown",
                }

        # Overall summary
        high_risk_count = sum(
            1 for r in results.values()
            if r.get("risk_level") in ["high", "elevated"]
        )

        return {
            "risk_summary": results,
            "count": len(results),
            "high_risk_count": high_risk_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": f"Risk check failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("CORPORATE_ACTIONS_PORT", "8014"))
    print(f"Starting Corporate Actions MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
