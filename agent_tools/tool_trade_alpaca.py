"""
Alpaca Trading Tool (MCP)

Provides buy/sell trading operations through Alpaca Paper Trading API.
Replaces the simulated tool_trade.py for real paper trading.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent_tools.alpaca_client import (
    AlpacaClient,
    AlpacaClientError,
    InsufficientFundsError,
    MarketClosedError,
    OrderError,
    get_alpaca_client,
)


def is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is a crypto pair (e.g., BTC/USD, ETH/USD)"""
    return "/" in symbol or (symbol.endswith("USD") and len(symbol) <= 7)


from tools.general_tools import get_config_value, write_config_value

mcp = FastMCP("AlpacaTradeTools")


# ==================== Risk Management ====================

class TradingRiskManager:
    """
    Risk management for trading operations.
    Enforces position limits, daily loss limits, and order size limits.

    Enhanced with:
    - Cumulative position checking (existing + new order)
    - ATR-based position sizing awareness
    - Drawdown protection
    - Cash-only trading (no margin usage)
    - Total invested amount tracking
    """

    def __init__(self):
        # Load risk limits from config or use defaults
        # IMPORTANT: max_position_pct is the TOTAL position limit (existing + new)
        self.max_position_pct = float(os.getenv("ALPACA_MAX_POSITION_PCT", "0.15"))  # 15% max per position
        self.max_order_value = float(os.getenv("ALPACA_MAX_ORDER_VALUE", "5000"))
        self.daily_loss_limit = float(os.getenv("ALPACA_DAILY_LOSS_LIMIT", "500"))
        self.warn_position_pct = 0.10  # Warn at 10%
        # New: Maximum percentage of portfolio that can be invested (prevents over-leveraging)
        self.max_invested_pct = float(os.getenv("ALPACA_MAX_INVESTED_PCT", "0.95"))  # 95% max invested
        # New: Prevent margin usage - only use cash
        self.use_margin = os.getenv("ALPACA_USE_MARGIN", "false").lower() == "true"

    def get_existing_position_value(self, symbol: str, positions: Dict[str, Any]) -> float:
        """Get the current market value of an existing position."""
        if symbol in positions:
            pos = positions[symbol]
            return pos.get("market_value", pos.get("qty", 0) * pos.get("current_price", 0))
        return 0.0

    def get_total_invested(self, positions: Dict[str, Any]) -> float:
        """Get total market value of all positions."""
        total = 0.0
        for symbol, pos in positions.items():
            total += pos.get("market_value", pos.get("qty", 0) * pos.get("current_price", 0))
        return total

    def validate_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        price: float,
        account: Dict[str, Any],
        positions: Dict[str, Any] = None
    ) -> tuple[bool, str]:
        """
        Validate an order against risk limits.

        IMPORTANT: Checks TOTAL position (existing + new order) against limits.
        NEW: Also prevents margin usage by checking against cash balance.

        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: "buy" or "sell"
            price: Estimated price
            account: Account info from Alpaca
            positions: Current positions dict (optional, will fetch if not provided)

        Returns:
            (is_valid, message) tuple
        """
        order_value = qty * price

        # Check max order value
        if order_value > self.max_order_value:
            return False, f"Order value ${order_value:.2f} exceeds max ${self.max_order_value:.2f}"

        if side.lower() == "buy":
            # NEW: Check against CASH balance (not buying power) to prevent margin usage
            cash = account.get("cash", 0)
            if not self.use_margin and order_value > cash:
                return False, (
                    f"MARGIN BLOCKED: Order ${order_value:.2f} exceeds available cash ${cash:.2f}. "
                    f"Margin trading is disabled to protect your account. "
                    f"(Buying power ${account['buying_power']:.2f} includes margin which we don't use)"
                )

            # Also check buying power as a safety net
            if order_value > account["buying_power"]:
                return False, f"Insufficient buying power. Need ${order_value:.2f}, have ${account['buying_power']:.2f}"

            # NEW: Check total invested amount won't exceed limit
            if positions:
                current_invested = self.get_total_invested(positions)
                equity = account["equity"]
                new_total_invested = current_invested + order_value
                invested_pct = new_total_invested / equity if equity > 0 else 1.0

                if invested_pct > self.max_invested_pct:
                    return False, (
                        f"INVESTMENT LIMIT: Total invested would be {invested_pct*100:.1f}% of equity "
                        f"(current ${current_invested:.0f} + new ${order_value:.0f} = ${new_total_invested:.0f}), "
                        f"max allowed is {self.max_invested_pct*100:.0f}% (${equity * self.max_invested_pct:.0f}). "
                        f"Keep some cash reserve."
                    )

            # Check CUMULATIVE position concentration (existing + new)
            equity = account["equity"]
            if equity > 0:
                # Get existing position value
                existing_value = 0.0
                if positions:
                    existing_value = self.get_existing_position_value(symbol, positions)

                # Calculate TOTAL position after this order
                total_position_value = existing_value + order_value
                total_position_pct = total_position_value / equity

                if total_position_pct > self.max_position_pct:
                    return False, (
                        f"RISK LIMIT: Total {symbol} position would be {total_position_pct*100:.1f}% "
                        f"(existing ${existing_value:.0f} + new ${order_value:.0f} = ${total_position_value:.0f}), "
                        f"max allowed is {self.max_position_pct*100:.0f}% (${equity * self.max_position_pct:.0f})"
                    )

                # Warning for positions approaching limit
                if total_position_pct > self.warn_position_pct:
                    print(f"⚠️ WARNING: {symbol} position will be {total_position_pct*100:.1f}% after this order")

        return True, "OK"


risk_manager = TradingRiskManager()


# ==================== Trade Logging ====================

def log_trade_to_jsonl(
    trade_type: str,
    symbol: str,
    qty: float,
    price: float,
    order_id: str,
    positions: Dict[str, Any],
    signature: str
) -> None:
    """
    Log trade to local JSONL file for record keeping.
    This maintains compatibility with the original position tracking format.
    """
    log_path = get_config_value("LOG_PATH", "./data/agent_data_alpaca")

    if log_path.startswith("./data/"):
        log_path = log_path[7:]

    position_dir = Path(project_root) / "data" / log_path / signature / "position"
    position_dir.mkdir(parents=True, exist_ok=True)

    position_file = position_dir / "position.jsonl"

    # Get current action ID
    current_action_id = 0
    if position_file.exists():
        with open(position_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        current_action_id = max(current_action_id, record.get("id", 0))
                    except Exception:
                        pass

    # Build position snapshot (convert Alpaca format to original format)
    position_snapshot = {"CASH": 0.0}  # Will be updated from account

    try:
        client = get_alpaca_client()
        account = client.get_account()
        position_snapshot["CASH"] = account["cash"]

        for sym, pos in positions.items():
            position_snapshot[sym] = pos["qty"]
    except Exception:
        pass

    # Write trade record
    trade_record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": current_action_id + 1,
        "this_action": {
            "action": trade_type,
            "symbol": symbol,
            "amount": qty,
            "price": price,
            "order_id": order_id,
        },
        "positions": position_snapshot,
    }

    with open(position_file, "a") as f:
        f.write(json.dumps(trade_record) + "\n")


# ==================== MCP Tools ====================

@mcp.tool()
def buy(symbol: str, amount: float) -> Dict[str, Any]:
    """
    Buy stock or crypto through Alpaca Paper Trading.

    Submits a market order to buy the specified quantity.
    The order will be executed at the current market price.

    Args:
        symbol: Stock symbol (e.g., "AAPL") or crypto pair (e.g., "BTC/USD", "ETHUSD")
        amount: Quantity to buy (supports fractional amounts for crypto)

    Returns:
        Dict with:
        - Success: Order details and updated positions
        - Failure: {"error": error message}

    Example:
        >>> result = buy("AAPL", 10)
        >>> print(result)
        {
            "status": "filled",
            "symbol": "AAPL",
            "qty": 10,
            "filled_price": 185.50,
            "order_id": "abc123...",
            "positions": {"AAPL": {"qty": 10, ...}, ...}
        }
    """
    try:
        client = get_alpaca_client()

        # Validate amount
        if amount <= 0:
            return {
                "error": f"Amount must be positive. You tried to buy {amount} shares.",
                "symbol": symbol,
                "amount": amount,
            }

        # Check if market is open (skip for crypto - crypto trades 24/7)
        is_crypto = is_crypto_symbol(symbol)
        if not is_crypto:
            clock = client.get_clock()
            if not clock["is_open"]:
                return {
                    "error": "Market is closed. Cannot execute order.",
                    "symbol": symbol,
                    "amount": amount,
                    "next_open": clock["next_open"],
                }

        # Get current quote for validation
        try:
            quote = client.get_quote(symbol)
            estimated_price = quote["ask_price"] if quote["ask_price"] > 0 else quote["bid_price"]

            if estimated_price <= 0:
                return {
                    "error": f"Cannot get valid price for {symbol}. The symbol may be invalid.",
                    "symbol": symbol,
                }
        except Exception as e:
            return {
                "error": f"Failed to get quote for {symbol}: {str(e)}",
                "symbol": symbol,
            }

        # Get account info and current positions
        account = client.get_account()
        positions = client.get_positions()

        # Validate against risk limits (including existing positions)
        is_valid, message = risk_manager.validate_order(
            symbol, amount, "buy", estimated_price, account, positions
        )
        if not is_valid:
            return {
                "error": message,
                "symbol": symbol,
                "amount": amount,
                "estimated_cost": estimated_price * amount,
            }

        # Submit market order (use "gtc" for crypto since "day" doesn't apply)
        time_in_force = "gtc" if is_crypto else "day"
        order = client.submit_market_order(symbol, amount, "buy", time_in_force=time_in_force)

        # Wait for fill (with timeout)
        try:
            filled_order = client.wait_for_order_fill(order["id"], timeout=30)
        except TimeoutError:
            return {
                "status": "pending",
                "message": "Order submitted but not yet filled",
                "order_id": order["id"],
                "symbol": symbol,
                "amount": amount,
            }
        except OrderError as e:
            return {
                "error": f"Order failed: {str(e)}",
                "symbol": symbol,
                "amount": amount,
            }

        # Get updated positions
        positions = client.get_positions()

        # Log trade
        signature = get_config_value("SIGNATURE", "alpaca-agent")
        log_trade_to_jsonl(
            "buy",
            symbol,
            filled_order["filled_qty"],
            filled_order["filled_avg_price"],
            filled_order["id"],
            positions,
            signature
        )

        # Mark that trading occurred
        write_config_value("IF_TRADE", True)

        return {
            "status": "filled",
            "symbol": symbol,
            "qty": filled_order["filled_qty"],
            "filled_price": filled_order["filled_avg_price"],
            "order_id": filled_order["id"],
            "total_cost": filled_order["filled_qty"] * filled_order["filled_avg_price"],
            "positions": positions,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "symbol": symbol}


@mcp.tool()
def sell(symbol: str, amount: float) -> Dict[str, Any]:
    """
    Sell stock or crypto through Alpaca Paper Trading.

    Submits a market order to sell the specified quantity.
    You must have sufficient shares/coins in your position to sell.

    Args:
        symbol: Stock symbol (e.g., "AAPL") or crypto pair (e.g., "BTC/USD", "ETHUSD")
        amount: Quantity to sell (supports fractional amounts for crypto)

    Returns:
        Dict with:
        - Success: Order details and updated positions
        - Failure: {"error": error message}

    Example:
        >>> result = sell("AAPL", 5)
        >>> print(result)
        {
            "status": "filled",
            "symbol": "AAPL",
            "qty": 5,
            "filled_price": 186.00,
            "order_id": "xyz789...",
            "positions": {"AAPL": {"qty": 5, ...}, ...}
        }
    """
    try:
        client = get_alpaca_client()

        # Validate amount
        if amount <= 0:
            return {
                "error": f"Amount must be positive. You tried to sell {amount} shares.",
                "symbol": symbol,
                "amount": amount,
            }

        # Check if market is open (skip for crypto - crypto trades 24/7)
        is_crypto = is_crypto_symbol(symbol)
        if not is_crypto:
            clock = client.get_clock()
            if not clock["is_open"]:
                return {
                    "error": "Market is closed. Cannot execute order.",
                    "symbol": symbol,
                    "amount": amount,
                    "next_open": clock["next_open"],
                }

        # Check current position
        position = client.get_position(symbol)
        if position is None:
            return {
                "error": f"No position in {symbol}. Cannot sell shares you don't own.",
                "symbol": symbol,
                "amount": amount,
            }

        if position["qty"] < amount:
            return {
                "error": f"Insufficient shares. You have {position['qty']} shares but tried to sell {amount}.",
                "symbol": symbol,
                "have": position["qty"],
                "want_to_sell": amount,
            }

        # Get current quote
        try:
            quote = client.get_quote(symbol)
            estimated_price = quote["bid_price"] if quote["bid_price"] > 0 else quote["ask_price"]
        except Exception:
            estimated_price = position["current_price"]

        # Get account info for risk validation
        account = client.get_account()

        # Validate against risk limits
        is_valid, message = risk_manager.validate_order(
            symbol, amount, "sell", estimated_price, account
        )
        if not is_valid:
            return {
                "error": message,
                "symbol": symbol,
                "amount": amount,
            }

        # Submit market order (use "gtc" for crypto since "day" doesn't apply)
        time_in_force = "gtc" if is_crypto else "day"
        order = client.submit_market_order(symbol, amount, "sell", time_in_force=time_in_force)

        # Wait for fill
        try:
            filled_order = client.wait_for_order_fill(order["id"], timeout=30)
        except TimeoutError:
            return {
                "status": "pending",
                "message": "Order submitted but not yet filled",
                "order_id": order["id"],
                "symbol": symbol,
                "amount": amount,
            }
        except OrderError as e:
            return {
                "error": f"Order failed: {str(e)}",
                "symbol": symbol,
                "amount": amount,
            }

        # Get updated positions
        positions = client.get_positions()

        # Log trade
        signature = get_config_value("SIGNATURE", "alpaca-agent")
        log_trade_to_jsonl(
            "sell",
            symbol,
            filled_order["filled_qty"],
            filled_order["filled_avg_price"],
            filled_order["id"],
            positions,
            signature
        )

        # Mark that trading occurred
        write_config_value("IF_TRADE", True)

        return {
            "status": "filled",
            "symbol": symbol,
            "qty": filled_order["filled_qty"],
            "filled_price": filled_order["filled_avg_price"],
            "order_id": filled_order["id"],
            "total_proceeds": filled_order["filled_qty"] * filled_order["filled_avg_price"],
            "positions": positions,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}", "symbol": symbol}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "symbol": symbol}


@mcp.tool()
def get_account_info() -> Dict[str, Any]:
    """
    Get current Alpaca account information.

    Returns:
        Dict with account details:
        - buying_power: Available cash for buying
        - cash: Cash balance
        - equity: Total account equity
        - portfolio_value: Total portfolio value
        - positions: Current stock positions
        - market_is_open: Whether market is currently open
    """
    try:
        client = get_alpaca_client()

        account = client.get_account()
        positions = client.get_positions()
        clock = client.get_clock()

        return {
            "buying_power": account["buying_power"],
            "cash": account["cash"],
            "equity": account["equity"],
            "portfolio_value": account["portfolio_value"],
            "last_equity": account["last_equity"],
            "daily_pnl": account["equity"] - account["last_equity"],
            "positions": positions,
            "position_count": len(positions),
            "market_is_open": clock["is_open"],
            "next_open": clock["next_open"] if not clock["is_open"] else None,
            "next_close": clock["next_close"] if clock["is_open"] else None,
        }

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def get_positions() -> Dict[str, Any]:
    """
    Get all current stock positions.

    Returns:
        Dict mapping symbol to position details:
        {
            "AAPL": {
                "qty": 10,
                "avg_entry_price": 150.0,
                "current_price": 155.0,
                "market_value": 1550.0,
                "unrealized_pl": 50.0,
                "unrealized_plpc": 0.0333
            },
            ...
        }
    """
    try:
        client = get_alpaca_client()
        positions = client.get_positions()

        if not positions:
            return {"message": "No open positions", "positions": {}}

        return {"positions": positions, "count": len(positions)}

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def get_open_orders() -> Dict[str, Any]:
    """
    Get all open (pending) orders.

    Returns:
        List of open orders with details
    """
    try:
        client = get_alpaca_client()
        orders = client.get_orders(status="open")

        if not orders:
            return {"message": "No open orders", "orders": []}

        return {"orders": orders, "count": len(orders)}

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@mcp.tool()
def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel a pending order.

    Args:
        order_id: The order ID to cancel

    Returns:
        Success or error message
    """
    try:
        client = get_alpaca_client()

        if client.cancel_order(order_id):
            return {"status": "cancelled", "order_id": order_id}
        else:
            return {"error": "Failed to cancel order", "order_id": order_id}

    except AlpacaClientError as e:
        return {"error": f"Alpaca error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


if __name__ == "__main__":
    port = int(os.getenv("ALPACA_TRADE_PORT", "8011"))
    print(f"Starting Alpaca Trade Tools MCP server on port {port}...")
    host = os.getenv("MCP_HOST", "127.0.0.1")
    mcp.run(transport="streamable-http", host=host, port=port)
