"""
Risk Engine for AI-Trader

This module implements institutional-grade risk management with:
- Hard constraints (pre-trade validation)
- Soft limits with alerts
- Circuit breakers (daily loss limits)
- Position sizing (ATR-based)
- Kill switch for emergency stops

Priority: ★★★★★ (Highest - "保命"措施)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from .data_structures import (
    OrderIntent,
    Portfolio,
    Position,
    MarketState,
    Signal,
    RiskAction,
    RiskViolation,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk engine parameters."""

    # Position limits (as fraction of portfolio)
    max_single_position: float = 0.10  # 10% max single position
    max_sector_exposure: float = 0.30  # 30% max sector exposure
    max_total_leverage: float = 1.0  # No leverage initially
    min_cash_buffer: float = 0.05  # 5% minimum cash

    # Soft limits (warnings)
    concentration_warning: float = 0.25  # Warn at 25% single asset
    correlation_threshold: float = 0.8  # Reject if >0.8 correlation

    # Circuit breakers (daily loss limits)
    daily_loss_reduce_size: float = -0.02  # -2% -> reduce sizes 50%
    daily_loss_halt_new: float = -0.05  # -5% -> halt new positions
    daily_loss_force_liquidate: float = -0.10  # -10% -> force liquidate to 50% cash

    # ATR-Based Position Sizing
    # Formula: Position_Size = (Account_Equity × risk_per_trade) / (ATR × atr_multiplier)
    default_risk_per_trade: float = 0.01  # 1% risk per trade
    atr_multiplier: float = 2.0  # ATR multiplier for stop loss
    atr_period: int = 14  # ATR calculation period

    # Drawdown-based position scaling (回撤降级规则)
    drawdown_reduce_25pct: float = -0.05  # -5% drawdown -> reduce by 25%
    drawdown_reduce_50pct: float = -0.10  # -10% drawdown -> reduce by 50%
    drawdown_halt_new: float = -0.15  # -15% drawdown -> halt new positions
    drawdown_stop_all: float = -0.20  # -20% drawdown -> stop all trading

    # Fractional Kelly parameters
    kelly_fraction: float = 0.25  # Use 1/4 Kelly (conservative)
    max_kelly_position: float = 0.20  # Never exceed 20% from Kelly

    # A-shares specific
    min_lot_size: int = 100  # A-shares trade in 100-share lots

    # Crypto specific
    max_crypto_position: float = 0.15  # 15% max for volatile crypto

    # Circuit breaker cooldown (seconds)
    cooldown_period: int = 3600  # 1 hour cooldown after circuit breaker


@dataclass
class RiskState:
    """Current state of risk metrics."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Daily PnL tracking
    day_start_value: float = 0.0
    current_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    # Current risk action
    current_action: RiskAction = RiskAction.NORMAL

    # Violations log
    violations: List[RiskViolation] = field(default_factory=list)

    # Kill switch state
    kill_switch_active: bool = False
    kill_switch_activated_at: Optional[datetime] = None

    # Circuit breaker state
    circuit_breaker_triggered_at: Optional[datetime] = None
    reduced_size_factor: float = 1.0  # Multiplier for position sizes

    # Circuit breaker level (0=normal, 1=reduce size, 2=halt new, 3=force liquidate)
    circuit_breaker_level: int = 0

    # Trading halted flag
    is_halted: bool = False

    # Position tracking
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> position value
    largest_position_pct: float = 0.0  # Largest position as % of portfolio


class RiskEngine:
    """
    Risk management engine for trading system.

    Validates orders, monitors portfolio risk, and enforces circuit breakers.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.state = RiskState()
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._sector_mapping: Dict[str, str] = {}  # symbol -> sector

    def get_state(self) -> RiskState:
        """
        Get the current risk state.

        Returns:
            RiskState with current risk metrics and status.
        """
        # Update derived fields
        self.state.is_halted = (
            self.state.kill_switch_active or
            self.state.current_action in (RiskAction.KILL_SWITCH, RiskAction.FORCE_LIQUIDATE)
        )

        # Map risk action to circuit breaker level
        action_to_level = {
            RiskAction.NORMAL: 0,
            RiskAction.REDUCE_SIZE: 1,
            RiskAction.HALT_NEW: 2,
            RiskAction.FORCE_LIQUIDATE: 3,
            RiskAction.KILL_SWITCH: 4,
        }
        self.state.circuit_breaker_level = action_to_level.get(self.state.current_action, 0)

        return self.state

    def update_portfolio_state(self, portfolio: Portfolio) -> None:
        """
        Update risk state with current portfolio information.

        Args:
            portfolio: Current portfolio state
        """
        self.state.current_value = portfolio.total_value
        self.state.timestamp = datetime.now()

        # Calculate daily P&L if we have a start value
        if self.state.day_start_value > 0:
            self.state.daily_pnl = portfolio.total_value - self.state.day_start_value
            self.state.daily_pnl_pct = self.state.daily_pnl / self.state.day_start_value
        else:
            # Initialize day start value
            self.state.day_start_value = portfolio.total_value

        # Update position tracking
        self.state.positions = {}
        max_position_pct = 0.0

        for symbol, position in portfolio.positions.items():
            position_value = position.market_value
            self.state.positions[symbol] = position_value

            if portfolio.total_value > 0:
                position_pct = abs(position_value) / portfolio.total_value
                if position_pct > max_position_pct:
                    max_position_pct = position_pct

        self.state.largest_position_pct = max_position_pct

        # Check circuit breakers based on daily P&L
        self.state.current_action = self.check_circuit_breakers(self.state.daily_pnl_pct)

    def validate_order(
        self,
        order: OrderIntent,
        portfolio: Portfolio,
        market_state: Optional[MarketState] = None,
    ) -> Tuple[bool, str, Optional[OrderIntent]]:
        """
        Validate an order against risk constraints.

        Returns:
            Tuple of (is_valid, message, adjusted_order)
            If invalid, adjusted_order is None.
            If valid but adjusted, returns the adjusted order.
        """
        # Check kill switch first
        if self.state.kill_switch_active:
            return False, "Kill switch is active. All trading halted.", None

        # Check circuit breaker state
        risk_action = self.check_circuit_breakers(self.state.daily_pnl_pct)

        if risk_action == RiskAction.KILL_SWITCH:
            return False, "Kill switch triggered by circuit breaker.", None

        if risk_action == RiskAction.FORCE_LIQUIDATE:
            # Only allow sell orders during force liquidation
            if order.side == "BUY":
                return False, "Force liquidation active. Only sell orders allowed.", None

        if risk_action == RiskAction.HALT_NEW:
            # Only allow exits, not new positions or additions
            if order.side == "BUY":
                if order.symbol not in portfolio.positions:
                    return False, "New positions halted due to daily loss limit.", None
                # Allow adding to existing positions? No, be conservative
                return False, "Position additions halted due to daily loss limit.", None

        # Calculate potential position after order
        current_position = portfolio.positions.get(order.symbol)
        current_qty = current_position.quantity if current_position else 0

        if order.side == "BUY":
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        # Get current price for value calculations
        price = (
            market_state.close if market_state else
            (order.limit_price if order.limit_price else
             (current_position.current_price if current_position else 0))
        )

        if price <= 0:
            return False, "Cannot validate order without price data.", None

        new_position_value = abs(new_qty * price)
        new_position_pct = new_position_value / portfolio.total_value if portfolio.total_value > 0 else 0

        # Check maximum single position
        max_position = self.config.max_single_position
        if self._is_crypto(order.symbol):
            max_position = min(max_position, self.config.max_crypto_position)

        if new_position_pct > max_position:
            # Calculate reduced quantity to fit within limit
            max_value = portfolio.total_value * max_position
            if order.side == "BUY":
                allowed_qty = (max_value / price) - current_qty
            else:
                allowed_qty = order.quantity  # Sells are always allowed

            if allowed_qty <= 0:
                return False, f"Position limit exceeded. Max {max_position*100:.1f}% allowed.", None

            # Adjust order
            adjusted_order = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                quantity=allowed_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                source_signal_id=order.source_signal_id,
            )
            logger.warning(
                f"Order adjusted: {order.symbol} quantity reduced from {order.quantity} to {allowed_qty}"
            )
            return True, f"Order adjusted to fit position limit ({max_position*100:.1f}%)", adjusted_order

        # Check cash buffer for buys
        if order.side == "BUY":
            order_cost = order.quantity * price
            cash_after = portfolio.cash - order_cost
            min_cash = portfolio.total_value * self.config.min_cash_buffer

            if cash_after < min_cash:
                available_for_trade = portfolio.cash - min_cash
                if available_for_trade <= 0:
                    return False, f"Insufficient cash. Min {self.config.min_cash_buffer*100:.1f}% buffer required.", None

                allowed_qty = available_for_trade / price
                adjusted_order = OrderIntent(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=allowed_qty,
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    source_signal_id=order.source_signal_id,
                )
                return True, "Order adjusted to maintain cash buffer.", adjusted_order

        # Check sector exposure
        sector = self._get_sector(order.symbol)
        if sector and order.side == "BUY":
            sector_exposure = self._calculate_sector_exposure(portfolio, sector)
            if sector_exposure > self.config.max_sector_exposure:
                return False, f"Sector exposure limit exceeded. {sector} at {sector_exposure*100:.1f}%", None

        # Apply size reduction if circuit breaker triggered
        if risk_action == RiskAction.REDUCE_SIZE:
            reduced_qty = order.quantity * self.state.reduced_size_factor
            adjusted_order = OrderIntent(
                symbol=order.symbol,
                side=order.side,
                quantity=reduced_qty,
                order_type=order.order_type,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                source_signal_id=order.source_signal_id,
            )
            return True, f"Order size reduced by {(1-self.state.reduced_size_factor)*100:.0f}% due to daily loss.", adjusted_order

        # Check concentration warning (soft limit)
        if new_position_pct > self.config.concentration_warning:
            logger.warning(
                f"Concentration warning: {order.symbol} would be {new_position_pct*100:.1f}% of portfolio"
            )

        return True, "Order validated successfully.", order

    def check_circuit_breakers(self, daily_pnl_pct: float) -> RiskAction:
        """
        Check circuit breaker thresholds and return appropriate action.

        Args:
            daily_pnl_pct: Daily P&L as percentage (e.g., -0.05 for -5%)

        Returns:
            RiskAction indicating what action to take
        """
        if self.state.kill_switch_active:
            return RiskAction.KILL_SWITCH

        if daily_pnl_pct <= self.config.daily_loss_force_liquidate:
            self._trigger_circuit_breaker(RiskAction.FORCE_LIQUIDATE, daily_pnl_pct)
            return RiskAction.FORCE_LIQUIDATE

        if daily_pnl_pct <= self.config.daily_loss_halt_new:
            self._trigger_circuit_breaker(RiskAction.HALT_NEW, daily_pnl_pct)
            return RiskAction.HALT_NEW

        if daily_pnl_pct <= self.config.daily_loss_reduce_size:
            self._trigger_circuit_breaker(RiskAction.REDUCE_SIZE, daily_pnl_pct)
            self.state.reduced_size_factor = 0.5  # Reduce position sizes by 50%
            return RiskAction.REDUCE_SIZE

        # Reset size factor if we're back to normal
        self.state.reduced_size_factor = 1.0
        self.state.current_action = RiskAction.NORMAL
        return RiskAction.NORMAL

    def _trigger_circuit_breaker(self, action: RiskAction, daily_pnl_pct: float):
        """Record circuit breaker trigger."""
        now = datetime.now()

        # Check if already triggered
        if self.state.current_action == action:
            return

        self.state.current_action = action
        self.state.circuit_breaker_triggered_at = now

        violation = RiskViolation(
            timestamp=now,
            violation_type="circuit_breaker",
            description=f"Circuit breaker triggered: {action.value}",
            current_value=daily_pnl_pct,
            limit_value=self._get_limit_for_action(action),
            action_taken=action,
        )
        self.state.violations.append(violation)

        logger.critical(
            f"CIRCUIT BREAKER: {action.value} triggered at {daily_pnl_pct*100:.2f}% daily loss"
        )

    def _get_limit_for_action(self, action: RiskAction) -> float:
        """Get the limit value that corresponds to an action."""
        mapping = {
            RiskAction.REDUCE_SIZE: self.config.daily_loss_reduce_size,
            RiskAction.HALT_NEW: self.config.daily_loss_halt_new,
            RiskAction.FORCE_LIQUIDATE: self.config.daily_loss_force_liquidate,
        }
        return mapping.get(action, 0.0)

    def calculate_position_size(
        self,
        signal: Signal,
        market_state: MarketState,
        portfolio: Portfolio,
    ) -> float:
        """
        Calculate position size using enhanced ATR-based sizing.

        Uses the formula: Position_Size = (Account_Equity × risk_per_trade) / (ATR × atr_multiplier)

        Adjustments applied:
        - Volatility (ATR)
        - Signal confidence
        - Drawdown-based scaling
        - Portfolio constraints
        """
        # Get ATR from market state
        atr = market_state.technical_indicators.get("ATR", 0)
        if atr <= 0:
            # Fallback: use daily range as proxy
            atr = market_state.high - market_state.low

        price = market_state.close

        # Calculate risk amount (1% of equity by default)
        risk_amount = portfolio.total_value * self.config.default_risk_per_trade

        # Position size based on ATR stop
        # Formula: Position_Size = (Account_Equity × 0.01) / (ATR × 2.0)
        stop_distance = atr * self.config.atr_multiplier
        if stop_distance <= 0:
            stop_distance = price * 0.02  # Fallback: 2% stop

        # Base position size (in shares)
        position_size = risk_amount / stop_distance

        # Adjust for signal confidence
        position_size *= signal.confidence

        # Adjust for signal strength
        position_size *= signal.strength

        # Apply drawdown-based scaling
        drawdown_factor = self._calculate_drawdown_factor(portfolio)
        position_size *= drawdown_factor

        # Apply target position percentage cap
        max_position_value = portfolio.total_value * abs(signal.target_position_pct)
        max_shares = max_position_value / price if price > 0 else 0

        position_size = min(position_size, max_shares)

        # Apply circuit breaker size reduction
        position_size *= self.state.reduced_size_factor

        # Apply max single position limit
        max_position = self.config.max_single_position
        if self._is_crypto(signal.symbol):
            max_position = min(max_position, self.config.max_crypto_position)

        max_from_limit = (portfolio.total_value * max_position) / price if price > 0 else 0
        position_size = min(position_size, max_from_limit)

        # Round for lot sizes (A-shares)
        if self._is_a_share(signal.symbol):
            position_size = int(position_size / self.config.min_lot_size) * self.config.min_lot_size

        return max(0, position_size)

    def _calculate_drawdown_factor(self, portfolio: Portfolio) -> float:
        """
        Calculate position size multiplier based on current drawdown.

        Implements drawdown-based position scaling:
        - 5-10% drawdown: reduce by 25%
        - 10-15% drawdown: reduce by 50%
        - 15-20% drawdown: halt new positions
        - 20%+ drawdown: stop all trading
        """
        drawdown = portfolio.current_drawdown

        if drawdown >= 0:
            return 1.0  # No drawdown, full size

        if drawdown <= self.config.drawdown_stop_all:
            # -20% or worse: stop trading
            return 0.0

        if drawdown <= self.config.drawdown_halt_new:
            # -15% to -20%: halt new positions (but may exit)
            return 0.0

        if drawdown <= self.config.drawdown_reduce_50pct:
            # -10% to -15%: reduce by 50%
            return 0.5

        if drawdown <= self.config.drawdown_reduce_25pct:
            # -5% to -10%: reduce by 25%
            return 0.75

        return 1.0

    def calculate_kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        portfolio: Portfolio,
        price: float,
    ) -> float:
        """
        Calculate position size using Fractional Kelly Criterion.

        Kelly Formula: f* = (bp - q) / b
        where:
            b = avg_win / avg_loss (win/loss ratio)
            p = probability of winning
            q = probability of losing (1 - p)
            f* = fraction of capital to risk

        We use 1/4 Kelly (kelly_fraction = 0.25) for conservative sizing.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0

        # Win/loss ratio
        b = avg_win / avg_loss

        # Kelly fraction
        p = win_rate
        q = 1 - p

        kelly_f = (b * p - q) / b

        # Apply fractional Kelly (typically 1/4 to 1/2)
        fractional_kelly = kelly_f * self.config.kelly_fraction

        # Cap at max kelly position
        fractional_kelly = min(fractional_kelly, self.config.max_kelly_position)

        # Convert to position size
        if fractional_kelly <= 0:
            return 0

        position_value = portfolio.total_value * fractional_kelly
        position_size = position_value / price if price > 0 else 0

        # Apply drawdown scaling
        drawdown_factor = self._calculate_drawdown_factor(portfolio)
        position_size *= drawdown_factor

        return max(0, position_size)

    def kill_switch(self, reason: str = "Manual activation") -> None:
        """
        Activate kill switch to halt all trading.

        This is the emergency stop mechanism.
        """
        self.state.kill_switch_active = True
        self.state.kill_switch_activated_at = datetime.now()
        self.state.current_action = RiskAction.KILL_SWITCH

        violation = RiskViolation(
            timestamp=datetime.now(),
            violation_type="kill_switch",
            description=f"Kill switch activated: {reason}",
            current_value=0,
            limit_value=0,
            action_taken=RiskAction.KILL_SWITCH,
        )
        self.state.violations.append(violation)

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch (requires explicit action)."""
        self.state.kill_switch_active = False
        self.state.current_action = RiskAction.NORMAL
        logger.warning("Kill switch deactivated")

    def update_daily_pnl(self, portfolio: Portfolio) -> None:
        """
        Update daily P&L tracking.

        Should be called at start of day and after each trade.
        """
        today = date.today()
        state_date = self.state.timestamp.date() if self.state.timestamp else None

        # Reset at start of new day
        if state_date != today:
            self.state.day_start_value = portfolio.total_value
            self.state.current_action = RiskAction.NORMAL
            self.state.reduced_size_factor = 1.0
            self.state.circuit_breaker_triggered_at = None

        self.state.current_value = portfolio.total_value
        self.state.timestamp = datetime.now()

        if self.state.day_start_value > 0:
            self.state.daily_pnl = portfolio.total_value - self.state.day_start_value
            self.state.daily_pnl_pct = self.state.daily_pnl / self.state.day_start_value
        else:
            self.state.daily_pnl = 0
            self.state.daily_pnl_pct = 0

    def get_force_liquidation_orders(
        self,
        portfolio: Portfolio,
        target_cash_pct: float = 0.5,
    ) -> List[OrderIntent]:
        """
        Generate orders to force liquidate to target cash percentage.

        Used when FORCE_LIQUIDATE circuit breaker is triggered.
        """
        orders = []

        target_cash = portfolio.total_value * target_cash_pct
        current_cash = portfolio.cash
        need_to_sell = target_cash - current_cash

        if need_to_sell <= 0:
            return orders

        # Sort positions by size (sell largest first)
        positions_by_size = sorted(
            portfolio.positions.values(),
            key=lambda p: abs(p.market_value),
            reverse=True,
        )

        sold_value = 0
        for position in positions_by_size:
            if sold_value >= need_to_sell:
                break

            if position.quantity > 0:
                # Determine quantity to sell
                remaining_to_sell = need_to_sell - sold_value
                sell_value = min(position.market_value, remaining_to_sell)
                sell_qty = sell_value / position.current_price if position.current_price > 0 else 0

                # For A-shares, round to lot size
                if self._is_a_share(position.symbol):
                    sell_qty = int(sell_qty / self.config.min_lot_size) * self.config.min_lot_size
                    sell_qty = min(sell_qty, position.available_to_sell)

                if sell_qty > 0:
                    order = OrderIntent(
                        symbol=position.symbol,
                        side="SELL",
                        quantity=sell_qty,
                        order_type="MARKET",
                    )
                    orders.append(order)
                    sold_value += sell_qty * position.current_price

        return orders

    def set_sector_mapping(self, mapping: Dict[str, str]) -> None:
        """Set the sector mapping for sector exposure checks."""
        self._sector_mapping = mapping

    def _get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol."""
        return self._sector_mapping.get(symbol)

    def _calculate_sector_exposure(self, portfolio: Portfolio, sector: str) -> float:
        """Calculate current exposure to a sector."""
        sector_value = 0
        for symbol, position in portfolio.positions.items():
            if self._get_sector(symbol) == sector:
                sector_value += abs(position.market_value)

        return sector_value / portfolio.total_value if portfolio.total_value > 0 else 0

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        crypto_suffixes = ["USDT", "USD", "BTC", "ETH"]
        return any(symbol.endswith(suffix) for suffix in crypto_suffixes) or symbol.startswith("BTC") or symbol.startswith("ETH")

    def _is_a_share(self, symbol: str) -> bool:
        """Check if symbol is an A-share (China market)."""
        # A-shares typically have 6-digit codes
        return symbol.isdigit() and len(symbol) == 6

    def get_status(self) -> Dict:
        """Get current risk engine status."""
        return {
            "kill_switch_active": self.state.kill_switch_active,
            "current_action": self.state.current_action.value,
            "daily_pnl_pct": self.state.daily_pnl_pct,
            "reduced_size_factor": self.state.reduced_size_factor,
            "violation_count": len(self.state.violations),
            "circuit_breaker_triggered": self.state.circuit_breaker_triggered_at is not None,
        }
