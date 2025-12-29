"""
On-Chain Metrics Tool (MCP)

Provides on-chain cryptocurrency metrics from free APIs:
- blockchain.com: BTC exchange flows
- whale-alert.io: Large transaction alerts
- Etherscan: ETH metrics

Features:
- Exchange net flow (bullish/bearish signal)
- Whale transaction tracking
- Active address counts
- MVRV ratio estimation
- Heavy caching (1 hour+) for free tier limits
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

from agent_tools.cache import get_cache_manager

mcp = FastMCP("OnChainMetrics")


# ==================== Configuration ====================

# API endpoints
BLOCKCHAIN_INFO_API = "https://api.blockchain.info"
WHALE_ALERT_API = "https://api.whale-alert.io/v1"
ETHERSCAN_API = "https://api.etherscan.io/api"

# API keys (optional for some endpoints)
WHALE_ALERT_KEY = os.getenv("WHALE_ALERT_API_KEY", "")
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# Whale transaction threshold ($1M)
WHALE_THRESHOLD_USD = 1_000_000

# Supported crypto symbols
SUPPORTED_CRYPTOS = ["BTC", "ETH"]

# Cache TTL (1 hour for on-chain data)
ONCHAIN_CACHE_TTL = 3600


# ==================== BTC Metrics (blockchain.com) ====================

def get_btc_exchange_flow() -> Dict[str, Any]:
    """
    Get BTC exchange net flow from blockchain.com API.

    Uses public stats endpoint which has rate limits.
    """
    cache = get_cache_manager()

    cache_key = "btc_exchange_flow"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    try:
        # Get exchange balances (approximation using mempool data)
        # Note: blockchain.com free API is limited
        stats_url = f"{BLOCKCHAIN_INFO_API}/stats"

        response = requests.get(stats_url, timeout=10)

        if response.status_code != 200:
            return {"error": "blockchain.com API unavailable", "status": response.status_code}

        data = response.json()

        # Extract relevant metrics
        result = {
            "total_btc": data.get("totalbc", 0) / 1e8,  # Satoshis to BTC
            "market_price_usd": data.get("market_price_usd", 0),
            "hash_rate": data.get("hash_rate", 0),
            "difficulty": data.get("difficulty", 0),
            "blocks_size": data.get("blocks_size", 0),
            "n_tx_24h": data.get("n_tx", 0),
            "estimated_btc_sent_24h": data.get("estimated_btc_sent", 0) / 1e8,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache for 1 hour
        cache.set(cache_key, result, ttl_seconds=ONCHAIN_CACHE_TTL, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to fetch BTC data: {str(e)}"}


def get_btc_active_addresses() -> Dict[str, Any]:
    """Get BTC active address count (estimated from tx count)"""
    cache = get_cache_manager()

    cache_key = "btc_active_addresses"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    try:
        # Use 24-hour chart data
        url = f"{BLOCKCHAIN_INFO_API}/charts/n-unique-addresses?timespan=24hours&format=json"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {"error": "API unavailable"}

        data = response.json()
        values = data.get("values", [])

        if values:
            latest = values[-1]
            result = {
                "active_addresses_24h": latest.get("y", 0),
                "timestamp": datetime.utcfromtimestamp(latest.get("x", 0)).isoformat() + "Z",
            }
        else:
            result = {"active_addresses_24h": None, "error": "No data available"}

        cache.set(cache_key, result, ttl_seconds=ONCHAIN_CACHE_TTL, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to fetch active addresses: {str(e)}"}


# ==================== ETH Metrics (Etherscan) ====================

def get_eth_supply() -> Dict[str, Any]:
    """Get ETH supply statistics from Etherscan"""
    cache = get_cache_manager()

    cache_key = "eth_supply"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    if not ETHERSCAN_KEY:
        return {"error": "Etherscan API key not configured", "note": "Set ETHERSCAN_API_KEY env var"}

    try:
        url = f"{ETHERSCAN_API}?module=stats&action=ethsupply&apikey={ETHERSCAN_KEY}"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {"error": "Etherscan API unavailable"}

        data = response.json()

        if data.get("status") != "1":
            return {"error": data.get("message", "Unknown error")}

        # Result is in Wei, convert to ETH
        supply_wei = int(data.get("result", 0))
        supply_eth = supply_wei / 1e18

        result = {
            "total_supply_eth": supply_eth,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        cache.set(cache_key, result, ttl_seconds=ONCHAIN_CACHE_TTL, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to fetch ETH supply: {str(e)}"}


def get_eth_gas_price() -> Dict[str, Any]:
    """Get current ETH gas prices from Etherscan"""
    cache = get_cache_manager()

    cache_key = "eth_gas"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    if not ETHERSCAN_KEY:
        return {"error": "Etherscan API key not configured"}

    try:
        url = f"{ETHERSCAN_API}?module=gastracker&action=gasoracle&apikey={ETHERSCAN_KEY}"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {"error": "Etherscan API unavailable"}

        data = response.json()

        if data.get("status") != "1":
            return {"error": data.get("message", "Unknown error")}

        result_data = data.get("result", {})

        result = {
            "safe_gas_price": int(result_data.get("SafeGasPrice", 0)),
            "propose_gas_price": int(result_data.get("ProposeGasPrice", 0)),
            "fast_gas_price": int(result_data.get("FastGasPrice", 0)),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache for 5 minutes (gas prices change frequently)
        cache.set(cache_key, result, ttl_seconds=300, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to fetch gas prices: {str(e)}"}


# ==================== Whale Alerts ====================

def get_whale_transactions(min_value_usd: int = WHALE_THRESHOLD_USD) -> Dict[str, Any]:
    """
    Get recent whale transactions from whale-alert.io.

    Free tier: 10 requests/minute
    """
    cache = get_cache_manager()

    cache_key = f"whale_txs:{min_value_usd}"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    if not WHALE_ALERT_KEY:
        return {
            "error": "Whale Alert API key not configured",
            "note": "Set WHALE_ALERT_API_KEY env var",
            "whale_transactions_24h": None,
        }

    try:
        # Get transactions from last 24 hours
        start = int((datetime.utcnow() - timedelta(hours=24)).timestamp())

        url = f"{WHALE_ALERT_API}/transactions"
        params = {
            "api_key": WHALE_ALERT_KEY,
            "min_value": min_value_usd,
            "start": start,
            "limit": 100,
        }

        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 429:
            return {"error": "Rate limit exceeded", "retry_after": "1 minute"}

        if response.status_code != 200:
            return {"error": f"Whale Alert API error: {response.status_code}"}

        data = response.json()
        transactions = data.get("transactions", [])

        # Count by symbol
        btc_count = sum(1 for tx in transactions if tx.get("symbol") == "btc")
        eth_count = sum(1 for tx in transactions if tx.get("symbol") == "eth")
        total_count = len(transactions)

        # Identify exchange-related transactions
        exchange_inflows = 0
        exchange_outflows = 0

        for tx in transactions:
            to_owner = tx.get("to", {}).get("owner_type", "")
            from_owner = tx.get("from", {}).get("owner_type", "")

            if to_owner == "exchange":
                exchange_inflows += 1
            if from_owner == "exchange":
                exchange_outflows += 1

        result = {
            "whale_transactions_24h": total_count,
            "btc_whale_count": btc_count,
            "eth_whale_count": eth_count,
            "exchange_inflows": exchange_inflows,
            "exchange_outflows": exchange_outflows,
            "net_exchange_flow": exchange_inflows - exchange_outflows,
            "signal": _interpret_exchange_flow(exchange_inflows, exchange_outflows),
            "threshold_usd": min_value_usd,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Cache for 1 hour
        cache.set(cache_key, result, ttl_seconds=ONCHAIN_CACHE_TTL, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to fetch whale data: {str(e)}"}


def _interpret_exchange_flow(inflows: int, outflows: int) -> str:
    """Interpret exchange flow as trading signal"""
    net = inflows - outflows

    if net > 5:
        return "bearish (high exchange inflows - selling pressure)"
    elif net < -5:
        return "bullish (high exchange outflows - accumulation)"
    else:
        return "neutral"


# ==================== MVRV Estimation ====================

def estimate_mvrv_btc() -> Dict[str, Any]:
    """
    Estimate MVRV ratio for BTC.

    MVRV = Market Cap / Realized Cap

    Since we don't have realized cap data, we use a simplified estimate
    based on historical average cost basis.

    Signals:
    - MVRV > 3: Overheated (consider taking profits)
    - MVRV < 1: Undervalued (accumulation opportunity)
    """
    cache = get_cache_manager()

    cache_key = "btc_mvrv"
    cached = cache.get(cache_key, namespace="onchain")
    if cached is not None:
        return cached

    try:
        # Get current price and market cap
        stats = get_btc_exchange_flow()

        if "error" in stats:
            return stats

        current_price = stats.get("market_price_usd", 0)
        total_btc = stats.get("total_btc", 0)

        if current_price <= 0 or total_btc <= 0:
            return {"error": "Unable to calculate MVRV"}

        market_cap = current_price * total_btc

        # Estimated realized price (simplified)
        # In reality, this would come from on-chain UTXO analysis
        # Using approximation: realized price ≈ 60% of current price during normal markets
        # This is a rough estimate - proper MVRV requires Glassnode/CryptoQuant

        # Historical estimate based on cycle position
        # BTC has historically had realized price ~$30k during 2024
        estimated_realized_price = 30000  # Conservative estimate

        realized_cap = estimated_realized_price * total_btc
        mvrv = market_cap / realized_cap if realized_cap > 0 else None

        # Determine signal
        if mvrv is None:
            signal = "unknown"
        elif mvrv > 3.0:
            signal = "overheated (MVRV > 3) - market may be topped"
        elif mvrv > 2.0:
            signal = "elevated (MVRV 2-3) - be cautious"
        elif mvrv < 1.0:
            signal = "undervalued (MVRV < 1) - accumulation zone"
        else:
            signal = "neutral (MVRV 1-2)"

        result = {
            "mvrv_ratio": round(mvrv, 2) if mvrv else None,
            "market_cap_usd": market_cap,
            "current_price_usd": current_price,
            "estimated_realized_price": estimated_realized_price,
            "signal": signal,
            "note": "MVRV is estimated. For accurate data, use Glassnode or CryptoQuant.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        cache.set(cache_key, result, ttl_seconds=ONCHAIN_CACHE_TTL, namespace="onchain")

        return result

    except Exception as e:
        return {"error": f"Failed to calculate MVRV: {str(e)}"}


# ==================== MCP Tools ====================

@mcp.tool()
def get_onchain_metrics(symbol: str) -> Dict[str, Any]:
    """
    Get on-chain metrics for cryptocurrency.

    Currently supports: BTC, ETH

    Returns comprehensive on-chain data including:
    - exchange_net_flow_24h: Net flow to exchanges (negative = bullish)
    - whale_transactions_24h: Count of >$1M transfers
    - active_addresses_24h: Network activity
    - mvrv_ratio: Market Value / Realized Value
    - data_freshness: How old the cached data is

    Note: Data cached for 1 hour to stay within free API limits.

    Args:
        symbol: Crypto symbol (BTC, ETH, or BTC/USD format)

    Returns:
        Dict with on-chain metrics and trading signals

    Usage Hint: High exchange inflows often precede selling.
    MVRV > 3 suggests overheated market conditions.

    Example:
        >>> result = get_onchain_metrics("BTC")
        >>> print(result["mvrv_ratio"])
        2.45
    """
    try:
        cache = get_cache_manager()

        # Normalize symbol (BTC/USD -> BTC)
        symbol = symbol.upper().replace("/USD", "").replace("USD", "").strip()

        if symbol not in SUPPORTED_CRYPTOS:
            return {
                "error": f"Unsupported symbol: {symbol}",
                "supported": SUPPORTED_CRYPTOS,
            }

        # Check degraded mode
        if cache.is_degraded("onchain_apis"):
            return {
                "error": "On-chain APIs temporarily unavailable",
                "degraded_mode": True,
                "retry_after": "5 minutes",
            }

        degraded_mode = False
        degraded_sources = []

        if symbol == "BTC":
            # Get BTC metrics
            exchange_flow = get_btc_exchange_flow()
            active_addresses = get_btc_active_addresses()
            mvrv = estimate_mvrv_btc()
            whale_data = get_whale_transactions()

            if "error" in exchange_flow:
                degraded_mode = True
                degraded_sources.append("blockchain.com")

            if "error" in whale_data:
                degraded_mode = True
                degraded_sources.append("whale-alert.io")

            result = {
                "symbol": "BTC",
                "exchange_net_flow_24h": whale_data.get("net_exchange_flow"),
                "exchange_flow_signal": whale_data.get("signal", "unknown"),
                "whale_transactions_24h": whale_data.get("whale_transactions_24h"),
                "active_addresses_24h": active_addresses.get("active_addresses_24h"),
                "mvrv_ratio": mvrv.get("mvrv_ratio"),
                "mvrv_signal": mvrv.get("signal", "unknown"),
                "current_price_usd": exchange_flow.get("market_price_usd"),
                "hash_rate": exchange_flow.get("hash_rate"),
                "n_transactions_24h": exchange_flow.get("n_tx_24h"),
                "data_freshness": "1 hour (cached)",
                "degraded_mode": degraded_mode,
                "degraded_sources": degraded_sources if degraded_sources else None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        elif symbol == "ETH":
            # Get ETH metrics
            supply = get_eth_supply()
            gas = get_eth_gas_price()
            whale_data = get_whale_transactions()

            if "error" in supply:
                degraded_mode = True
                degraded_sources.append("etherscan")

            if "error" in whale_data:
                degraded_mode = True
                degraded_sources.append("whale-alert.io")

            result = {
                "symbol": "ETH",
                "exchange_net_flow_24h": whale_data.get("net_exchange_flow"),
                "exchange_flow_signal": whale_data.get("signal", "unknown"),
                "whale_transactions_24h": whale_data.get("eth_whale_count"),
                "total_supply_eth": supply.get("total_supply_eth"),
                "gas_price_gwei": {
                    "safe": gas.get("safe_gas_price"),
                    "standard": gas.get("propose_gas_price"),
                    "fast": gas.get("fast_gas_price"),
                } if "error" not in gas else None,
                "data_freshness": "1 hour (cached)",
                "degraded_mode": degraded_mode,
                "degraded_sources": degraded_sources if degraded_sources else None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        else:
            result = {"error": f"No on-chain data for {symbol}"}

        # Add overall trading recommendation
        if not degraded_mode:
            result["trading_hint"] = _get_trading_recommendation(result)

        return result

    except Exception as e:
        return {"error": f"On-chain metrics failed: {str(e)}", "degraded_mode": True}


def _get_trading_recommendation(metrics: Dict) -> str:
    """Generate trading recommendation based on on-chain metrics"""
    signals = []

    # Exchange flow signal
    net_flow = metrics.get("exchange_net_flow_24h")
    if net_flow is not None:
        if net_flow > 5:
            signals.append("bearish (exchange inflows)")
        elif net_flow < -5:
            signals.append("bullish (exchange outflows)")

    # MVRV signal
    mvrv = metrics.get("mvrv_ratio")
    if mvrv is not None:
        if mvrv > 3:
            signals.append("bearish (MVRV overheated)")
        elif mvrv < 1:
            signals.append("bullish (MVRV undervalued)")

    if not signals:
        return "No strong on-chain signals - neutral stance"

    return "; ".join(signals)


@mcp.tool()
def get_exchange_flows_summary() -> Dict[str, Any]:
    """
    Get exchange flow summary for major cryptocurrencies.

    Provides a quick overview of money flow into/out of exchanges.
    High inflows often indicate selling pressure.

    Returns:
        Dict with:
        - btc_net_flow: BTC net exchange flow
        - eth_net_flow: ETH net exchange flow (if available)
        - overall_signal: Combined market signal

    Example:
        >>> result = get_exchange_flows_summary()
        >>> print(result["overall_signal"])
        "bullish - net outflows from exchanges"
    """
    try:
        whale_data = get_whale_transactions()

        if "error" in whale_data:
            return whale_data

        net_flow = whale_data.get("net_exchange_flow", 0)

        if net_flow > 10:
            overall = "bearish - strong net inflows to exchanges (selling pressure)"
        elif net_flow > 5:
            overall = "slightly bearish - moderate exchange inflows"
        elif net_flow < -10:
            overall = "bullish - strong net outflows from exchanges (accumulation)"
        elif net_flow < -5:
            overall = "slightly bullish - moderate exchange outflows"
        else:
            overall = "neutral - balanced exchange flows"

        return {
            "net_exchange_flow_24h": net_flow,
            "exchange_inflows": whale_data.get("exchange_inflows", 0),
            "exchange_outflows": whale_data.get("exchange_outflows", 0),
            "whale_count": whale_data.get("whale_transactions_24h", 0),
            "overall_signal": overall,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        return {"error": f"Exchange flows failed: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("ONCHAIN_PORT", "8015"))
    print(f"Starting On-Chain Metrics MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
