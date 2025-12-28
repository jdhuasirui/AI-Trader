"""
Real-time Trading Dashboard - Multi-Account Comparison

Flask-based dashboard for monitoring multiple AI trading agents.
Shows portfolio performance comparison across Claude, ChatGPT, Gemini, and Grok.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template
from flask_cors import CORS

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import Alpaca client for real-time data
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca SDK not available")

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = project_root / "data" / "agent_data_crypto"
REFRESH_INTERVAL = 10  # seconds

# Account configurations
ACCOUNTS = {
    "claude-opus-4.5": {
        "display_name": "Claude Opus 4.5",
        "api_key_env": "ALPACA_API_KEY_CLAUDE",
        "secret_key_env": "ALPACA_SECRET_KEY_CLAUDE",
        "paper_env": "ALPACA_PAPER_CLAUDE",
        "color": "#8B5CF6",  # Purple
        "signature": "claude-opus-4.5-crypto",
    },
    "chatgpt-5.2": {
        "display_name": "ChatGPT 5.2",
        "api_key_env": "ALPACA_API_KEY_CHATGPT",
        "secret_key_env": "ALPACA_SECRET_KEY_CHATGPT",
        "paper_env": "ALPACA_PAPER_CHATGPT",
        "color": "#10B981",  # Green
        "signature": "chatgpt-5.2-crypto",
    },
    "gemini-3.0-pro": {
        "display_name": "Gemini 3.0 Pro",
        "api_key_env": "ALPACA_API_KEY_GEMINI",
        "secret_key_env": "ALPACA_SECRET_KEY_GEMINI",
        "paper_env": "ALPACA_PAPER_GEMINI",
        "color": "#3B82F6",  # Blue
        "signature": "gemini-3.0-pro-crypto",
    },
    "grok-4.2": {
        "display_name": "Grok 4.2",
        "api_key_env": "ALPACA_API_KEY_GROK",
        "secret_key_env": "ALPACA_SECRET_KEY_GROK",
        "paper_env": "ALPACA_PAPER_GROK",
        "color": "#EF4444",  # Red
        "signature": "grok-4.2-crypto",
    },
}


def get_alpaca_client(api_key: str, secret_key: str, paper: bool = True):
    """Create an Alpaca trading client."""
    if not ALPACA_AVAILABLE:
        return None
    try:
        return TradingClient(api_key, secret_key, paper=paper)
    except Exception as e:
        print(f"Error creating Alpaca client: {e}")
        return None


def get_account_data(account_id: str, config: Dict) -> Dict[str, Any]:
    """Get real-time data for a single account."""
    api_key = os.getenv(config["api_key_env"], "")
    secret_key = os.getenv(config["secret_key_env"], "")
    paper = os.getenv(config["paper_env"], "true").lower() == "true"

    if not api_key or not secret_key:
        return {
            "id": account_id,
            "name": config["display_name"],
            "status": "not_configured",
            "error": "API keys not configured",
            "color": config["color"],
        }

    client = get_alpaca_client(api_key, secret_key, paper)
    if not client:
        return {
            "id": account_id,
            "name": config["display_name"],
            "status": "error",
            "error": "Could not create Alpaca client",
            "color": config["color"],
        }

    try:
        # Get account info
        account = client.get_account()

        # Get positions
        positions_raw = client.get_all_positions()
        positions = []
        total_position_value = 0
        total_unrealized_pl = 0

        for pos in positions_raw:
            qty = float(pos.qty)
            current_price = float(pos.current_price)
            avg_price = float(pos.avg_entry_price)
            market_value = float(pos.market_value)
            unrealized_pl = float(pos.unrealized_pl)
            unrealized_plpc = float(pos.unrealized_plpc) * 100

            total_position_value += market_value
            total_unrealized_pl += unrealized_pl

            positions.append({
                "symbol": pos.symbol,
                "qty": qty,
                "avg_price": avg_price,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            })

        # Sort by market value
        positions.sort(key=lambda x: x["market_value"], reverse=True)

        portfolio_value = float(account.portfolio_value)
        cash = float(account.cash)
        equity = float(account.equity)
        last_equity = float(account.last_equity)
        daily_pnl = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity) * 100 if last_equity > 0 else 0

        return {
            "id": account_id,
            "name": config["display_name"],
            "status": "active",
            "color": config["color"],
            "signature": config["signature"],
            "portfolio_value": portfolio_value,
            "cash": cash,
            "equity": equity,
            "last_equity": last_equity,
            "buying_power": float(account.buying_power),
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "total_unrealized_pl": total_unrealized_pl,
            "positions": positions,
            "position_count": len(positions),
            "last_updated": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "id": account_id,
            "name": config["display_name"],
            "status": "error",
            "error": str(e),
            "color": config["color"],
        }


def get_all_accounts_data() -> Dict[str, Any]:
    """Get data for all configured accounts."""
    accounts_data = []
    total_portfolio_value = 0
    total_daily_pnl = 0
    total_unrealized_pl = 0
    active_count = 0

    for account_id, config in ACCOUNTS.items():
        data = get_account_data(account_id, config)
        accounts_data.append(data)

        if data.get("status") == "active":
            active_count += 1
            total_portfolio_value += data.get("portfolio_value", 0)
            total_daily_pnl += data.get("daily_pnl", 0)
            total_unrealized_pl += data.get("total_unrealized_pl", 0)

    # Sort by daily P&L (best first)
    accounts_data.sort(key=lambda x: x.get("daily_pnl", float('-inf')), reverse=True)

    # Add ranking
    for i, acc in enumerate(accounts_data):
        if acc.get("status") == "active":
            acc["rank"] = i + 1

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "accounts": accounts_data,
        "summary": {
            "total_portfolio_value": total_portfolio_value,
            "total_daily_pnl": total_daily_pnl,
            "total_daily_pnl_pct": (total_daily_pnl / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0,
            "total_unrealized_pl": total_unrealized_pl,
            "active_accounts": active_count,
            "total_accounts": len(ACCOUNTS),
        },
        "refresh_interval": REFRESH_INTERVAL,
    }


def load_log_entries(signature: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Load recent log entries for a model."""
    log_dir = DATA_DIR / signature / "log"
    if not log_dir.exists():
        return []

    entries = []
    try:
        for date_dir in sorted(log_dir.iterdir(), reverse=True):
            if date_dir.is_dir():
                for log_file in sorted(date_dir.glob("*.jsonl"), reverse=True):
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                try:
                                    entry = json.loads(line)
                                    entries.append(entry)
                                    if len(entries) >= limit:
                                        break
                                except:
                                    pass
                    if len(entries) >= limit:
                        break
            if len(entries) >= limit:
                break
        return entries[:limit]
    except Exception as e:
        print(f"Error loading logs for {signature}: {e}")
        return []


def load_performance_history(signature: str) -> List[Dict[str, Any]]:
    """Load performance history from summary.json for a model."""
    summary_file = DATA_DIR / signature / "summary.json"
    if not summary_file.exists():
        return []

    try:
        with open(summary_file, 'r') as f:
            summaries = json.load(f)

        if not summaries:
            return []

        # Always use 100000 as the baseline for consistent comparison
        INITIAL_CASH = 100000

        history = []
        for entry in summaries:
            timestamp = entry.get("timestamp")
            end_equity = entry.get("end_equity", INITIAL_CASH)
            cumulative_return = ((end_equity / INITIAL_CASH) - 1) * 100

            history.append({
                "timestamp": timestamp,
                "equity": end_equity,
                "pnl_pct": cumulative_return,
            })

        return history
    except Exception as e:
        print(f"Error loading performance history for {signature}: {e}")
        return []


def get_market_benchmark(start_time: datetime) -> List[Dict[str, Any]]:
    """Get BTC+ETH average performance as market benchmark."""
    if not ALPACA_AVAILABLE:
        return []

    # Use any available API key for market data
    api_key = None
    secret_key = None
    for config in ACCOUNTS.values():
        api_key = os.getenv(config["api_key_env"], "")
        secret_key = os.getenv(config["secret_key_env"], "")
        if api_key and secret_key:
            break

    if not api_key or not secret_key:
        return []

    try:
        client = CryptoHistoricalDataClient(api_key, secret_key)

        request = CryptoBarsRequest(
            symbol_or_symbols=["BTC/USD", "ETH/USD"],
            timeframe=TimeFrame.Hour,
            start=start_time,
        )

        bars = client.get_crypto_bars(request)

        # Process bars into time series
        btc_prices = {}
        eth_prices = {}

        # BarSet uses .data property to access the dictionary
        bars_data = bars.data if hasattr(bars, 'data') else {}

        if "BTC/USD" in bars_data:
            for bar in bars_data["BTC/USD"]:
                # Convert UTC to local time (remove timezone info for consistency)
                local_ts = bar.timestamp.astimezone().replace(tzinfo=None)
                ts = local_ts.isoformat()
                btc_prices[ts] = bar.close

        if "ETH/USD" in bars_data:
            for bar in bars_data["ETH/USD"]:
                # Convert UTC to local time (remove timezone info for consistency)
                local_ts = bar.timestamp.astimezone().replace(tzinfo=None)
                ts = local_ts.isoformat()
                eth_prices[ts] = bar.close

        # Calculate average returns
        all_timestamps = sorted(set(btc_prices.keys()) & set(eth_prices.keys()))
        if not all_timestamps:
            return []

        initial_btc = btc_prices.get(all_timestamps[0], 1)
        initial_eth = eth_prices.get(all_timestamps[0], 1)

        benchmark = []
        for ts in all_timestamps:
            btc_return = ((btc_prices.get(ts, initial_btc) / initial_btc) - 1) * 100
            eth_return = ((eth_prices.get(ts, initial_eth) / initial_eth) - 1) * 100
            avg_return = (btc_return + eth_return) / 2

            benchmark.append({
                "timestamp": ts,
                "pnl_pct": avg_return,
            })

        return benchmark
    except Exception as e:
        print(f"Error fetching market benchmark: {e}")
        return []


def get_performance_data() -> Dict[str, Any]:
    """Get performance history for all accounts and market benchmark."""
    accounts_history = {}
    earliest_time = datetime.now()
    current_time = datetime.now().isoformat()
    INITIAL_CASH = 100000

    for account_id, config in ACCOUNTS.items():
        signature = config["signature"]
        history = load_performance_history(signature)

        # Get current equity from Alpaca to add real-time data point
        current_equity = None
        if ALPACA_AVAILABLE:
            api_key = os.getenv(config["api_key_env"], "")
            secret_key = os.getenv(config["secret_key_env"], "")
            paper = os.getenv(config["paper_env"], "true").lower() == "true"
            if api_key and secret_key:
                try:
                    client = TradingClient(api_key, secret_key, paper=paper)
                    account = client.get_account()
                    current_equity = float(account.equity)
                except Exception as e:
                    print(f"Error getting current equity for {account_id}: {e}")

        if history:
            # Add current real-time data point if we have it
            if current_equity is not None:
                last_timestamp = history[-1]["timestamp"] if history else None
                current_pnl_pct = ((current_equity / INITIAL_CASH) - 1) * 100
                history.append({
                    "timestamp": current_time,
                    "equity": current_equity,
                    "pnl_pct": current_pnl_pct,
                })

            accounts_history[account_id] = {
                "name": config["display_name"],
                "color": config["color"],
                "data": history,
            }

            # Find earliest timestamp
            for entry in history:
                try:
                    ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                    if ts < earliest_time:
                        earliest_time = ts
                except:
                    pass

    # Get market benchmark starting from earliest data point
    benchmark = get_market_benchmark(earliest_time - timedelta(hours=1))

    return {
        "accounts": accounts_history,
        "benchmark": benchmark,
        "timestamp": current_time,
    }


def get_order_history(account_id: str, config: Dict, limit: int = 100) -> List[Dict[str, Any]]:
    """Get order history for a single account from Alpaca API."""
    api_key = os.getenv(config["api_key_env"], "")
    secret_key = os.getenv(config["secret_key_env"], "")
    paper = os.getenv(config["paper_env"], "true").lower() == "true"

    if not api_key or not secret_key or not ALPACA_AVAILABLE:
        return []

    try:
        client = TradingClient(api_key, secret_key, paper=paper)
        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=limit
        )
        orders = client.get_orders(request)

        order_list = []
        for order in orders:
            # Parse order data
            filled_qty = float(order.filled_qty) if order.filled_qty else 0
            filled_avg_price = float(order.filled_avg_price) if order.filled_avg_price else 0
            notional = filled_qty * filled_avg_price if filled_avg_price > 0 else 0

            order_list.append({
                "id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side.value) if hasattr(order.side, 'value') else str(order.side),
                "qty": float(order.qty) if order.qty else 0,
                "filled_qty": filled_qty,
                "filled_avg_price": filled_avg_price,
                "notional": notional,
                "type": str(order.type.value) if hasattr(order.type, 'value') else str(order.type),
                "status": str(order.status.value) if hasattr(order.status, 'value') else str(order.status),
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "created_at": order.created_at.isoformat() if order.created_at else None,
            })

        # Sort by filled_at descending (most recent first)
        order_list.sort(key=lambda x: x.get("filled_at") or "", reverse=True)
        return order_list

    except Exception as e:
        print(f"Error fetching order history for {account_id}: {e}")
        return []


def get_all_order_history() -> Dict[str, Any]:
    """Get order history for all configured accounts."""
    all_orders = {}

    for account_id, config in ACCOUNTS.items():
        orders = get_order_history(account_id, config, limit=50)
        all_orders[account_id] = {
            "name": config["display_name"],
            "color": config["color"],
            "orders": orders,
        }

    return {
        "accounts": all_orders,
        "timestamp": datetime.now().isoformat(),
    }


# Routes
@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', refresh_interval=REFRESH_INTERVAL)


@app.route('/api/data')
def api_data():
    """API endpoint for all accounts data."""
    return jsonify(get_all_accounts_data())


@app.route('/api/account/<account_id>')
def api_account(account_id):
    """API endpoint for specific account data."""
    if account_id not in ACCOUNTS:
        return jsonify({"status": "error", "error": "Account not found"}), 404

    data = get_account_data(account_id, ACCOUNTS[account_id])
    return jsonify(data)


@app.route('/api/account/<account_id>/logs')
def api_account_logs(account_id):
    """API endpoint for account logs."""
    if account_id not in ACCOUNTS:
        return jsonify({"status": "error", "error": "Account not found"}), 404

    signature = ACCOUNTS[account_id]["signature"]
    logs = load_log_entries(signature, limit=50)
    return jsonify({"account": account_id, "logs": logs})


@app.route('/api/performance/history')
def api_performance_history():
    """API endpoint for performance history data."""
    return jsonify(get_performance_data())


@app.route('/api/orders')
def api_orders():
    """API endpoint for all accounts order history."""
    return jsonify(get_all_order_history())


@app.route('/api/account/<account_id>/orders')
def api_account_orders(account_id):
    """API endpoint for specific account order history."""
    if account_id not in ACCOUNTS:
        return jsonify({"status": "error", "error": "Account not found"}), 404

    orders = get_order_history(account_id, ACCOUNTS[account_id], limit=100)
    return jsonify({
        "account": account_id,
        "name": ACCOUNTS[account_id]["display_name"],
        "orders": orders,
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


if __name__ == '__main__':
    port = int(os.environ.get('DASHBOARD_PORT', 8888))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

    print(f"Starting Multi-Account Trading Dashboard on http://localhost:{port}")
    print(f"Configured accounts: {list(ACCOUNTS.keys())}")
    print(f"Refresh interval: {REFRESH_INTERVAL} seconds")

    app.run(host='0.0.0.0', port=port, debug=debug)
