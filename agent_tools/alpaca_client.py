"""
Alpaca API Client Wrapper

Provides a unified interface for Alpaca Trading and Data APIs.
Supports both Paper Trading and Live Trading modes.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest, CryptoLatestQuoteRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaClientError(Exception):
    """Base exception for Alpaca client errors"""
    pass


class OrderError(AlpacaClientError):
    """Exception for order-related errors"""
    pass


class MarketClosedError(AlpacaClientError):
    """Exception when market is closed"""
    pass


class InsufficientFundsError(AlpacaClientError):
    """Exception for insufficient buying power"""
    pass


class AlpacaClient:
    """
    Alpaca API Client Wrapper

    Encapsulates trading and data operations for the Alpaca API.
    Designed for Paper Trading but supports Live Trading as well.

    Usage:
        client = AlpacaClient()  # Uses environment variables
        client = AlpacaClient(api_key="...", secret_key="...", paper=True)

        # Get account info
        account = client.get_account()

        # Get quote
        quote = client.get_quote("AAPL")

        # Submit order
        order = client.submit_market_order("AAPL", 10, "buy")
    """

    _instance: Optional["AlpacaClient"] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None
    ):
        """
        Initialize Alpaca client.

        Args:
            api_key: Alpaca API key. Defaults to ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. Defaults to ALPACA_SECRET_KEY env var.
            paper: Use paper trading. Defaults to ALPACA_PAPER env var or True.
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if paper is None:
            paper_env = os.getenv("ALPACA_PAPER", "true").lower()
            self.paper = paper_env in ("true", "1", "yes")
        else:
            self.paper = paper

        if not self.api_key or not self.secret_key:
            raise AlpacaClientError(
                "Alpaca API credentials not found. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Initialize clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        # Crypto data client
        self.crypto_client = CryptoHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        # Cache for rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto pair (e.g., BTC/USD, ETH/USD)"""
        return "/" in symbol or symbol.endswith("USD") and len(symbol) <= 7

    def _rate_limit(self):
        """Simple rate limiting to avoid API throttling"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    # ==================== Account Operations ====================

    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary with account details:
            - id: Account ID
            - status: Account status (ACTIVE, etc.)
            - currency: Account currency (USD)
            - buying_power: Available buying power
            - cash: Cash balance
            - portfolio_value: Total portfolio value
            - equity: Account equity
            - last_equity: Previous day equity
            - daytrade_count: Pattern day trade count
        """
        self._rate_limit()
        account = self.trading_client.get_account()

        return {
            "id": str(account.id),
            "status": str(account.status),
            "currency": account.currency,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "daytrade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbol to position details:
            {
                "AAPL": {
                    "qty": 10,
                    "avg_entry_price": 150.0,
                    "market_value": 1550.0,
                    "current_price": 155.0,
                    "unrealized_pl": 50.0,
                    "unrealized_plpc": 0.0333,
                    "side": "long"
                },
                ...
            }
        """
        self._rate_limit()
        positions = self.trading_client.get_all_positions()

        result = {}
        for pos in positions:
            result[pos.symbol] = {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": str(pos.side),
            }

        return result

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position details or None if no position
        """
        self._rate_limit()
        try:
            pos = self.trading_client.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "side": str(pos.side),
            }
        except Exception:
            return None

    # ==================== Market Clock ====================

    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock status.

        Returns:
            Dictionary with clock info:
            - is_open: Whether market is currently open
            - timestamp: Current timestamp
            - next_open: Next market open time
            - next_close: Next market close time
        """
        self._rate_limit()
        clock = self.trading_client.get_clock()

        return {
            "is_open": clock.is_open,
            "timestamp": str(clock.timestamp),
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.get_clock()["is_open"]

    # ==================== Price Data ====================

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest quote for a symbol (stock or crypto).

        Args:
            symbol: Stock symbol (AAPL) or crypto pair (BTC/USD)

        Returns:
            Dictionary with quote data:
            - symbol: Symbol
            - bid_price: Current bid price
            - bid_size: Bid size
            - ask_price: Current ask price
            - ask_size: Ask size
            - timestamp: Quote timestamp
        """
        self._rate_limit()

        if self._is_crypto_symbol(symbol):
            # Crypto quote
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.crypto_client.get_crypto_latest_quote(request)
            quote = quotes[symbol]
            return {
                "symbol": symbol,
                "bid_price": float(quote.bid_price) if quote.bid_price else 0.0,
                "bid_size": float(quote.bid_size) if quote.bid_size else 0.0,
                "ask_price": float(quote.ask_price) if quote.ask_price else 0.0,
                "ask_size": float(quote.ask_size) if quote.ask_size else 0.0,
                "timestamp": str(quote.timestamp),
                "is_crypto": True,
            }
        else:
            # Stock quote
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]
            return {
                "symbol": symbol,
                "bid_price": float(quote.bid_price) if quote.bid_price else 0.0,
                "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                "ask_price": float(quote.ask_price) if quote.ask_price else 0.0,
                "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                "timestamp": str(quote.timestamp),
                "is_crypto": False,
            }

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get latest quotes for multiple symbols (stocks or crypto).

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbol to quote data
        """
        self._rate_limit()

        # Separate crypto and stock symbols
        crypto_symbols = [s for s in symbols if self._is_crypto_symbol(s)]
        stock_symbols = [s for s in symbols if not self._is_crypto_symbol(s)]

        result = {}

        # Get crypto quotes
        if crypto_symbols:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=crypto_symbols)
            quotes = self.crypto_client.get_crypto_latest_quote(request)
            for symbol, quote in quotes.items():
                result[symbol] = {
                    "symbol": symbol,
                    "bid_price": float(quote.bid_price) if quote.bid_price else 0.0,
                    "bid_size": float(quote.bid_size) if quote.bid_size else 0.0,
                    "ask_price": float(quote.ask_price) if quote.ask_price else 0.0,
                    "ask_size": float(quote.ask_size) if quote.ask_size else 0.0,
                    "timestamp": str(quote.timestamp),
                    "is_crypto": True,
                }

        # Get stock quotes
        if stock_symbols:
            request = StockLatestQuoteRequest(symbol_or_symbols=stock_symbols)
            quotes = self.data_client.get_stock_latest_quote(request)
            for symbol, quote in quotes.items():
                result[symbol] = {
                    "symbol": symbol,
                    "bid_price": float(quote.bid_price) if quote.bid_price else 0.0,
                    "bid_size": int(quote.bid_size) if quote.bid_size else 0,
                    "ask_price": float(quote.ask_price) if quote.ask_price else 0.0,
                    "ask_size": int(quote.ask_size) if quote.ask_size else 0,
                    "timestamp": str(quote.timestamp),
                    "is_crypto": False,
                }

        return result

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            start: Start datetime (defaults to limit days ago)
            end: End datetime (defaults to now)
            limit: Maximum number of bars

        Returns:
            List of bar dictionaries with OHLCV data
        """
        self._rate_limit()

        # Map timeframe string to TimeFrame enum
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=limit)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit
        )

        bars = self.data_client.get_stock_bars(request)

        result = []
        if symbol in bars:
            for bar in bars[symbol]:
                result.append({
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                })

        return result

    def get_crypto_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get historical bars for a crypto symbol.

        Args:
            symbol: Crypto pair (e.g., BTC/USD, ETH/USD)
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            start: Start datetime (defaults to limit days ago)
            end: End datetime (defaults to now)
            limit: Maximum number of bars

        Returns:
            List of bar dictionaries with OHLCV data
        """
        self._rate_limit()

        # Map timeframe string to TimeFrame enum
        tf_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "Min"),
            "15Min": TimeFrame(15, "Min"),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=limit)

        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit
        )

        bars = self.crypto_client.get_crypto_bars(request)

        result = []
        # Handle BarSet data structure
        bars_data = bars.data if hasattr(bars, 'data') else bars
        if symbol in bars_data:
            for bar in bars_data[symbol]:
                result.append({
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.vwap) if bar.vwap else None,
                })

        return result

    # ==================== Order Operations ====================

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        Submit a market order.

        Args:
            symbol: Stock symbol
            qty: Order quantity (can be fractional)
            side: "buy" or "sell"
            time_in_force: "day", "gtc", "ioc", "fok"

        Returns:
            Order details dictionary
        """
        self._rate_limit()

        # Map side
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Map time in force
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif
        )

        order = self.trading_client.submit_order(request)

        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        Submit a limit order.

        Args:
            symbol: Stock symbol
            qty: Order quantity
            side: "buy" or "sell"
            limit_price: Limit price
            time_in_force: "day", "gtc", "ioc", "fok"

        Returns:
            Order details dictionary
        """
        self._rate_limit()

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            limit_price=limit_price
        )

        order = self.trading_client.submit_order(request)

        return self._order_to_dict(order)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order details dictionary
        """
        self._rate_limit()
        order = self.trading_client.get_order_by_id(order_id)
        return self._order_to_dict(order)

    def get_orders(
        self,
        status: str = "open",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get orders.

        Args:
            status: "open", "closed", or "all"
            limit: Maximum number of orders

        Returns:
            List of order dictionaries
        """
        self._rate_limit()

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        request = GetOrdersRequest(
            status=status_map.get(status.lower(), QueryOrderStatus.OPEN),
            limit=limit
        )

        orders = self.trading_client.get_orders(request)

        return [self._order_to_dict(order) for order in orders]

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        self._rate_limit()
        try:
            self.trading_client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        self._rate_limit()
        cancelled = self.trading_client.cancel_orders()
        return len(cancelled) if cancelled else 0

    def wait_for_order_fill(
        self,
        order_id: str,
        timeout: float = 30.0,
        poll_interval: float = 0.5
    ) -> Dict[str, Any]:
        """
        Wait for an order to be filled.

        Args:
            order_id: Order ID to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Filled order details

        Raises:
            OrderError: If order is cancelled or rejected
            TimeoutError: If order not filled within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            order = self.get_order(order_id)
            status = order["status"]

            if status == "filled":
                return order
            elif status in ["canceled", "cancelled", "rejected", "expired"]:
                raise OrderError(f"Order {status}: {order}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Order not filled within {timeout} seconds")

    def _order_to_dict(self, order) -> Dict[str, Any]:
        """Convert order object to dictionary"""
        return {
            "id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": order.symbol,
            "side": str(order.side),
            "type": str(order.type),
            "status": str(order.status).lower().replace("orderstatus.", ""),
            "qty": float(order.qty) if order.qty else 0,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "time_in_force": str(order.time_in_force),
            "created_at": str(order.created_at),
            "updated_at": str(order.updated_at),
            "submitted_at": str(order.submitted_at),
            "filled_at": str(order.filled_at) if order.filled_at else None,
        }


# Singleton accessor
_client_instance: Optional[AlpacaClient] = None


def get_alpaca_client() -> AlpacaClient:
    """
    Get singleton Alpaca client instance.

    Returns:
        AlpacaClient instance
    """
    global _client_instance

    if _client_instance is None:
        _client_instance = AlpacaClient()

    return _client_instance


def reset_alpaca_client():
    """Reset singleton client (useful for testing)"""
    global _client_instance
    _client_instance = None


if __name__ == "__main__":
    # Test the client
    try:
        client = get_alpaca_client()

        print("Testing Alpaca Client...")
        print("=" * 50)

        # Test account
        account = client.get_account()
        print(f"Account Status: {account['status']}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Cash: ${account['cash']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")

        # Test clock
        clock = client.get_clock()
        print(f"\nMarket Open: {clock['is_open']}")
        print(f"Next Open: {clock['next_open']}")

        # Test quote
        quote = client.get_quote("AAPL")
        print(f"\nAAPL Quote:")
        print(f"  Bid: ${quote['bid_price']:.2f} x {quote['bid_size']}")
        print(f"  Ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")

        # Test positions
        positions = client.get_positions()
        print(f"\nPositions: {len(positions)} symbols")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")

        print("\n" + "=" * 50)
        print("All tests passed!")

    except AlpacaClientError as e:
        print(f"Alpaca Client Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
