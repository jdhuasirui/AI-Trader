"""
Execution Algorithms for AI-Trader

This module implements smart order execution strategies:
- TWAP (Time-Weighted Average Price): Split large orders over time
- Limit-first: Try limit order, fallback to market after timeout
- Fee optimization: Prefer maker orders (0.1% vs 0.2% taker)
- Slippage modeling for backtest simulation
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from .data_structures import (
    OrderIntent,
    ExecutionReport,
    MarketState,
    OrderType,
    OrderStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for execution algorithms."""

    # TWAP settings
    twap_slices: int = 5  # Number of order slices
    twap_interval_seconds: int = 60  # Time between slices
    twap_randomize_size: bool = True  # Add randomness to slice sizes
    twap_size_variance: float = 0.2  # ±20% variance in slice sizes

    # Limit-first settings
    limit_timeout_seconds: int = 10  # Wait for limit fill before fallback
    limit_offset_bps: int = 5  # Offset from mid in basis points (0.05%)
    aggressive_offset_bps: int = -5  # Cross the spread slightly

    # Enhanced Slippage model (for simulation)
    # Based on: slippage = k * (order_qty / period_volume) + spread/2 + delay_impact
    slippage_coefficient: float = 0.1  # k in slippage = k * (qty / volume)
    min_slippage_bps: float = 1.0  # Minimum 0.01% slippage
    max_slippage_bps: float = 50.0  # Maximum 0.5% slippage

    # Order book depth simulation
    orderbook_depth_factor: float = 0.5  # Impact multiplier for thin order books
    typical_spread_bps: float = 2.0  # Typical bid-ask spread (0.02%)
    crypto_spread_bps: float = 5.0  # Crypto spread (0.05%)

    # Latency simulation
    base_latency_ms: float = 50.0  # Baseline execution latency
    latency_std_ms: float = 20.0  # Standard deviation of latency

    # Queue position simulation (for limit orders)
    queue_position_probability: float = 0.7  # Probability of being in front of queue
    queue_decay_rate: float = 0.1  # Rate at which queue position decays

    # Fee rates
    maker_fee_bps: float = 10.0  # 0.1% maker fee
    taker_fee_bps: float = 20.0  # 0.2% taker fee

    # Partial fill simulation
    partial_fill_probability: float = 0.1  # 10% chance of partial fill
    min_fill_ratio: float = 0.5  # Minimum 50% fill in partial scenarios


@dataclass
class SliceOrder:
    """A single slice in a multi-slice execution."""
    sequence: int
    quantity: float
    order_intent: OrderIntent
    scheduled_time: datetime
    executed: bool = False
    execution_report: Optional[ExecutionReport] = None


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    @abstractmethod
    def execute(
        self,
        order: OrderIntent,
        market_state: MarketState,
        submit_order: Callable[[OrderIntent], ExecutionReport],
    ) -> List[ExecutionReport]:
        """
        Execute an order using this algorithm.

        Args:
            order: The order to execute
            market_state: Current market data
            submit_order: Callback to submit individual orders

        Returns:
            List of execution reports for all fills
        """
        pass

    def calculate_slippage(
        self,
        order_qty: float,
        daily_volume: float,
        is_buy: bool,
    ) -> float:
        """
        Calculate expected slippage for an order.

        Uses linear impact model: slippage = k * (qty / volume)
        """
        if daily_volume <= 0:
            return self.config.max_slippage_bps / 10000

        volume_ratio = order_qty / daily_volume
        slippage_bps = self.config.slippage_coefficient * volume_ratio * 10000

        # Clamp to bounds
        slippage_bps = max(self.config.min_slippage_bps, min(slippage_bps, self.config.max_slippage_bps))

        # Convert to decimal (negative for sells, positive for buys)
        slippage = slippage_bps / 10000
        return slippage if is_buy else -slippage


class MarketOrderExecution(ExecutionAlgorithm):
    """Simple market order execution with slippage simulation."""

    def execute(
        self,
        order: OrderIntent,
        market_state: MarketState,
        submit_order: Callable[[OrderIntent], ExecutionReport],
    ) -> List[ExecutionReport]:
        """Execute as a single market order."""

        # Calculate slippage
        volume = market_state.daily_volume_avg_20d or market_state.volume
        slippage = self.calculate_slippage(
            order.quantity,
            volume,
            order.side == "BUY",
        )

        # Simulate fill price
        fill_price = market_state.close * (1 + slippage)

        # Calculate commission
        commission = order.quantity * fill_price * (self.config.taker_fee_bps / 10000)

        # Create execution report
        report = ExecutionReport(
            order_id=f"sim_{order.client_order_id}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            ordered_qty=order.quantity,
            filled_qty=order.quantity,
            remaining_qty=0,
            avg_fill_price=fill_price,
            slippage=slippage * 100,  # As percentage
            commission=commission,
            total_cost=order.quantity * fill_price + commission,
            filled_at=datetime.now(),
        )

        return [report]


class TWAPExecution(ExecutionAlgorithm):
    """
    Time-Weighted Average Price execution.

    Splits a large order into smaller slices executed over time
    to minimize market impact.
    """

    def execute(
        self,
        order: OrderIntent,
        market_state: MarketState,
        submit_order: Callable[[OrderIntent], ExecutionReport],
    ) -> List[ExecutionReport]:
        """Execute using TWAP strategy."""
        reports = []

        # Calculate slice sizes
        slices = self._create_slices(order)

        # Execute each slice
        for slice_order in slices:
            # Wait until scheduled time (in real execution)
            # For simulation, we execute immediately

            # Submit the slice order
            try:
                report = self._execute_slice(slice_order, market_state, submit_order)
                reports.append(report)
                slice_order.executed = True
                slice_order.execution_report = report
            except Exception as e:
                logger.error(f"TWAP slice {slice_order.sequence} failed: {e}")
                # Continue with remaining slices

        return reports

    def _create_slices(self, order: OrderIntent) -> List[SliceOrder]:
        """Create order slices with optional randomization."""
        slices = []
        remaining_qty = order.quantity
        base_slice_qty = order.quantity / self.config.twap_slices

        now = datetime.now()

        for i in range(self.config.twap_slices):
            # Calculate slice quantity
            if i == self.config.twap_slices - 1:
                # Last slice gets remaining quantity
                slice_qty = remaining_qty
            else:
                if self.config.twap_randomize_size:
                    # Add randomness
                    variance = base_slice_qty * self.config.twap_size_variance
                    slice_qty = base_slice_qty + random.uniform(-variance, variance)
                    slice_qty = max(0.01, min(slice_qty, remaining_qty))
                else:
                    slice_qty = base_slice_qty

            remaining_qty -= slice_qty

            # Create slice order intent
            slice_intent = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.LIMIT if i < self.config.twap_slices - 1 else OrderType.MARKET,
                time_in_force="IOC",  # Immediate or Cancel
                source_signal_id=order.source_signal_id,
            )

            scheduled_time = now + timedelta(seconds=i * self.config.twap_interval_seconds)

            slices.append(SliceOrder(
                sequence=i + 1,
                quantity=slice_qty,
                order_intent=slice_intent,
                scheduled_time=scheduled_time,
            ))

        return slices

    def _execute_slice(
        self,
        slice_order: SliceOrder,
        market_state: MarketState,
        submit_order: Callable[[OrderIntent], ExecutionReport],
    ) -> ExecutionReport:
        """Execute a single slice."""
        # For simulation, calculate slippage based on slice size
        volume = market_state.daily_volume_avg_20d or market_state.volume
        slippage = self.calculate_slippage(
            slice_order.quantity,
            volume,
            slice_order.order_intent.side == "BUY",
        )

        # Reduced slippage for smaller orders
        slippage *= 0.8  # TWAP typically reduces impact by ~20%

        fill_price = market_state.close * (1 + slippage)

        # Use maker fee for limit orders
        fee_bps = (
            self.config.maker_fee_bps
            if slice_order.order_intent.order_type == OrderType.LIMIT
            else self.config.taker_fee_bps
        )
        commission = slice_order.quantity * fill_price * (fee_bps / 10000)

        return ExecutionReport(
            order_id=f"twap_{slice_order.sequence}_{slice_order.order_intent.client_order_id}",
            client_order_id=slice_order.order_intent.client_order_id,
            symbol=slice_order.order_intent.symbol,
            side=slice_order.order_intent.side,
            order_type=slice_order.order_intent.order_type,
            status=OrderStatus.FILLED,
            ordered_qty=slice_order.quantity,
            filled_qty=slice_order.quantity,
            remaining_qty=0,
            avg_fill_price=fill_price,
            slippage=slippage * 100,
            commission=commission,
            total_cost=slice_order.quantity * fill_price + commission,
            filled_at=datetime.now(),
        )


class LimitFirstExecution(ExecutionAlgorithm):
    """
    Limit-first execution strategy.

    Attempts to fill via limit order first (lower fees),
    then falls back to market order if timeout reached.
    """

    def execute(
        self,
        order: OrderIntent,
        market_state: MarketState,
        submit_order: Callable[[OrderIntent], ExecutionReport],
    ) -> List[ExecutionReport]:
        """Execute using limit-first strategy."""

        # Calculate limit price with offset
        offset_multiplier = self.config.limit_offset_bps / 10000
        if order.side == "BUY":
            # Bid below current price
            limit_price = market_state.close * (1 - offset_multiplier)
        else:
            # Ask above current price
            limit_price = market_state.close * (1 + offset_multiplier)

        # Create limit order
        limit_order = OrderIntent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            time_in_force="GTC",
            source_signal_id=order.source_signal_id,
        )

        # Simulate limit order execution
        # In real trading, this would wait for the timeout
        filled, limit_report = self._simulate_limit_fill(limit_order, market_state)

        if filled:
            return [limit_report]

        # Fallback to market order for unfilled portion
        remaining_qty = order.quantity - (limit_report.filled_qty if limit_report else 0)

        if remaining_qty > 0:
            market_order = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                quantity=remaining_qty,
                order_type=OrderType.MARKET,
                source_signal_id=order.source_signal_id,
            )

            market_exec = MarketOrderExecution(self.config)
            market_reports = market_exec.execute(market_order, market_state, submit_order)

            reports = []
            if limit_report and limit_report.filled_qty > 0:
                reports.append(limit_report)
            reports.extend(market_reports)
            return reports

        return [limit_report] if limit_report else []

    def _simulate_limit_fill(
        self,
        order: OrderIntent,
        market_state: MarketState,
    ) -> Tuple[bool, Optional[ExecutionReport]]:
        """
        Simulate limit order fill probability.

        Uses a simple model based on price distance from market.
        """
        if order.limit_price is None:
            return False, None

        # Calculate distance from current price
        price_distance = abs(order.limit_price - market_state.close) / market_state.close

        # Fill probability decreases with distance
        # At 0% distance: ~90% fill
        # At 0.5% distance: ~50% fill
        # At 1% distance: ~20% fill
        fill_probability = max(0.1, 0.9 - price_distance * 100)

        # Simulate partial fill possibility
        if random.random() < self.config.partial_fill_probability:
            fill_ratio = random.uniform(self.config.min_fill_ratio, 1.0)
        else:
            fill_ratio = 1.0 if random.random() < fill_probability else 0.0

        filled_qty = order.quantity * fill_ratio

        if filled_qty <= 0:
            return False, None

        # No slippage for limit orders (filled at limit price)
        commission = filled_qty * order.limit_price * (self.config.maker_fee_bps / 10000)

        report = ExecutionReport(
            order_id=f"limit_{order.client_order_id}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED if fill_ratio >= 1.0 else OrderStatus.PARTIAL,
            ordered_qty=order.quantity,
            filled_qty=filled_qty,
            remaining_qty=order.quantity - filled_qty,
            avg_fill_price=order.limit_price,
            slippage=0,  # No slippage for limit orders
            commission=commission,
            total_cost=filled_qty * order.limit_price + commission,
            filled_at=datetime.now(),
        )

        return fill_ratio >= 1.0, report


class SlippageSimulator:
    """
    Enhanced Slippage simulation for backtesting.

    Models realistic execution costs including:
    - Market impact (linear + square root model)
    - Order book depth simulation
    - Bid-ask spread (asset-specific)
    - Latency simulation
    - Queue position for limit orders
    - Fill probability based on order size
    - Partial fills
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    def simulate_fill(
        self,
        order: OrderIntent,
        market_state: MarketState,
    ) -> ExecutionReport:
        """
        Simulate order execution with realistic fills.

        Uses enhanced slippage model:
        slippage = k * (qty / volume) + spread/2 + orderbook_impact + latency_impact

        Args:
            order: Order to execute
            market_state: Current market data

        Returns:
            Simulated execution report
        """
        volume = market_state.daily_volume_avg_20d or market_state.volume
        is_crypto = self._is_crypto(order.symbol)

        # Calculate multi-component slippage
        slippage_components = self._calculate_enhanced_slippage(
            order.quantity, volume, order.side == "BUY", market_state, is_crypto
        )
        total_slippage = slippage_components["total"]

        # Simulate execution latency
        latency_ms = self._simulate_latency()

        # Simulate fill price
        if order.order_type == OrderType.LIMIT and order.limit_price:
            # For limit orders, use queue simulation
            fill_price, filled_qty = self._simulate_limit_execution_with_queue(
                order, market_state, total_slippage
            )
        else:
            # Market order
            fill_price = market_state.close * (1 + total_slippage)
            filled_qty = order.quantity

        # Simulate partial fill based on order size relative to volume
        filled_qty = self._simulate_partial_fill(filled_qty, order.quantity, volume)

        # Calculate commission
        fee_bps = (
            self.config.maker_fee_bps if order.order_type == OrderType.LIMIT
            else self.config.taker_fee_bps
        )
        commission = filled_qty * fill_price * (fee_bps / 10000)

        return ExecutionReport(
            order_id=f"sim_{order.client_order_id}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            status=OrderStatus.FILLED if filled_qty >= order.quantity else OrderStatus.PARTIAL,
            ordered_qty=order.quantity,
            filled_qty=filled_qty,
            remaining_qty=order.quantity - filled_qty,
            avg_fill_price=fill_price,
            slippage=total_slippage * 100,
            commission=commission,
            total_cost=filled_qty * fill_price + commission,
            filled_at=datetime.now() + timedelta(milliseconds=latency_ms),
        )

    def _calculate_enhanced_slippage(
        self,
        quantity: float,
        volume: float,
        is_buy: bool,
        market_state: MarketState,
        is_crypto: bool = False,
    ) -> Dict[str, float]:
        """
        Calculate slippage using enhanced multi-factor model.

        Components:
        1. Market impact: k * (qty / volume)
        2. Spread cost: spread / 2
        3. Orderbook depth impact: depth_factor * sqrt(qty / volume)
        4. Volatility adjustment: higher vol = higher slippage

        Returns:
            Dict with individual components and total slippage
        """
        if volume <= 0:
            return {
                "market_impact": self.config.max_slippage_bps / 10000,
                "spread": 0,
                "orderbook_impact": 0,
                "volatility_adj": 0,
                "total": self.config.max_slippage_bps / 10000,
            }

        volume_ratio = quantity / volume

        # 1. Linear market impact
        market_impact_bps = self.config.slippage_coefficient * volume_ratio * 10000

        # 2. Spread cost (half spread for crossing)
        spread_bps = self.config.crypto_spread_bps if is_crypto else self.config.typical_spread_bps
        spread_cost_bps = spread_bps / 2

        # 3. Orderbook depth impact (square root model for large orders)
        orderbook_impact_bps = 0
        if volume_ratio > 0.001:  # Only for orders > 0.1% of volume
            orderbook_impact_bps = self.config.orderbook_depth_factor * (volume_ratio ** 0.5) * 10000

        # 4. Volatility adjustment
        volatility_adj_bps = 0
        if market_state.volatility_20d:
            # Higher volatility = more slippage
            vol_multiplier = max(1.0, market_state.volatility_20d / 20)  # 20% is baseline
            volatility_adj_bps = (market_impact_bps + orderbook_impact_bps) * (vol_multiplier - 1) * 0.5

        # Total slippage
        total_bps = market_impact_bps + spread_cost_bps + orderbook_impact_bps + volatility_adj_bps

        # Clamp
        total_bps = max(self.config.min_slippage_bps, min(total_bps, self.config.max_slippage_bps))

        # Direction adjustment
        direction = 1 if is_buy else -1

        return {
            "market_impact": market_impact_bps / 10000 * direction,
            "spread": spread_cost_bps / 10000 * direction,
            "orderbook_impact": orderbook_impact_bps / 10000 * direction,
            "volatility_adj": volatility_adj_bps / 10000 * direction,
            "total": total_bps / 10000 * direction,
        }

    def _simulate_latency(self) -> float:
        """Simulate execution latency in milliseconds."""
        base = self.config.base_latency_ms
        std = self.config.latency_std_ms
        # Use log-normal distribution for realistic latency
        latency = random.gauss(base, std)
        return max(1, latency)  # Minimum 1ms

    def _simulate_partial_fill(
        self,
        requested_qty: float,
        order_qty: float,
        volume: float,
    ) -> float:
        """Simulate partial fills based on order size."""
        volume_ratio = order_qty / volume if volume > 0 else 1

        # Large orders more likely to be partially filled
        if volume_ratio > 0.05:  # Order > 5% of volume
            # High probability of partial fill
            if random.random() < 0.5:
                return requested_qty * random.uniform(0.7, 0.95)
        elif volume_ratio > 0.01:  # Order > 1% of volume
            if random.random() < 0.2:
                return requested_qty * random.uniform(0.85, 0.98)
        elif random.random() < self.config.partial_fill_probability:
            return requested_qty * random.uniform(self.config.min_fill_ratio, 1.0)

        return requested_qty

    def _simulate_limit_execution_with_queue(
        self,
        order: OrderIntent,
        market_state: MarketState,
        slippage: float,
    ) -> Tuple[float, float]:
        """
        Simulate limit order execution with queue position modeling.

        Models:
        1. Whether price reaches limit
        2. Queue position probability
        3. Partial fills based on queue position
        """
        limit_price = order.limit_price

        if order.side == "BUY":
            # Buy limit: only fill if market drops to limit
            if limit_price >= market_state.low:
                # Price touched our level - calculate fill probability
                price_distance = (limit_price - market_state.low) / market_state.low
                fill_prob = min(1.0, self.config.queue_position_probability + price_distance * 10)

                if random.random() < fill_prob:
                    # Calculate fill quantity based on how deep price went
                    depth_ratio = (limit_price - market_state.low) / (limit_price - market_state.close + 0.001)
                    fill_ratio = min(1.0, 0.5 + depth_ratio * 0.5)
                    return limit_price, order.quantity * fill_ratio
                else:
                    # Queued but not filled
                    return limit_price, 0
            else:
                return 0, 0
        else:
            # Sell limit: only fill if market rises to limit
            if limit_price <= market_state.high:
                price_distance = (market_state.high - limit_price) / market_state.high
                fill_prob = min(1.0, self.config.queue_position_probability + price_distance * 10)

                if random.random() < fill_prob:
                    depth_ratio = (market_state.high - limit_price) / (market_state.close - limit_price + 0.001)
                    fill_ratio = min(1.0, 0.5 + depth_ratio * 0.5)
                    return limit_price, order.quantity * fill_ratio
                else:
                    return limit_price, 0
            else:
                return 0, 0

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is cryptocurrency."""
        crypto_suffixes = ["USDT", "USD", "BTC", "ETH", "USDC"]
        return any(symbol.endswith(suffix) for suffix in crypto_suffixes) or \
               symbol.startswith("BTC") or symbol.startswith("ETH") or symbol.startswith("SOL")


def get_execution_algorithm(
    name: str,
    config: Optional[ExecutionConfig] = None,
) -> ExecutionAlgorithm:
    """Factory function to get execution algorithm by name."""
    algorithms = {
        "market": MarketOrderExecution,
        "twap": TWAPExecution,
        "limit_first": LimitFirstExecution,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown execution algorithm: {name}. Available: {list(algorithms.keys())}")

    return algorithms[name](config)
