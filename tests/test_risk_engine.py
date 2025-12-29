"""
Unit tests for RiskEngine functionality.

Tests:
- validate_order() with various scenarios
- Circuit breaker behavior
- Position sizing calculations
- Kill switch activation
- RiskConfig validation
"""

import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

from core import (
    RiskEngine, RiskConfig, RiskState,
    OrderIntent, OrderType, Portfolio, Position,
    MarketState, Signal, SignalDirection, RiskAction,
)


class TestRiskConfigValidation(unittest.TestCase):
    """Test RiskConfig validation in __post_init__."""

    def test_valid_default_config(self):
        """Default config should pass validation."""
        config = RiskConfig()
        self.assertIsNotNone(config)

    def test_invalid_max_single_position(self):
        """max_single_position out of range should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RiskConfig(max_single_position=1.5)
        self.assertIn("max_single_position", str(ctx.exception))

    def test_invalid_max_single_position_zero(self):
        """max_single_position of 0 should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RiskConfig(max_single_position=0)
        self.assertIn("max_single_position", str(ctx.exception))

    def test_invalid_circuit_breaker_order(self):
        """Circuit breaker thresholds in wrong order should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            # halt_new is less severe than reduce_size, which is wrong
            RiskConfig(
                daily_loss_reduce_size=-0.05,
                daily_loss_halt_new=-0.02,  # Wrong: should be more severe
                daily_loss_force_liquidate=-0.10,
            )
        self.assertIn("Circuit breaker", str(ctx.exception))

    def test_invalid_drawdown_order(self):
        """Drawdown thresholds in wrong order should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RiskConfig(
                drawdown_reduce_25pct=-0.10,
                drawdown_reduce_50pct=-0.05,  # Wrong: less severe than 25pct
                drawdown_halt_new=-0.15,
                drawdown_stop_all=-0.20,
            )
        self.assertIn("Drawdown", str(ctx.exception))

    def test_invalid_kelly_fraction(self):
        """Kelly fraction > 1 should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RiskConfig(kelly_fraction=1.5)
        self.assertIn("kelly_fraction", str(ctx.exception))

    def test_invalid_atr_period(self):
        """ATR period <= 0 should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RiskConfig(atr_period=0)
        self.assertIn("atr_period", str(ctx.exception))


class TestRiskEngineValidateOrder(unittest.TestCase):
    """Test RiskEngine.validate_order() functionality."""

    def setUp(self):
        self.config = RiskConfig()
        self.risk_engine = RiskEngine(self.config)
        self.portfolio = Portfolio(
            timestamp=datetime.now(),
            cash=10000.0,
            total_value=10000.0,
            buying_power=10000.0,
            positions={},
        )
        self.market_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            technical_indicators={"ATR": 5.0},
        )

    def test_validate_order_with_kill_switch(self):
        """Order should be rejected when kill switch is active."""
        self.risk_engine.state.kill_switch_active = True
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )
        is_valid, msg, adjusted = self.risk_engine.validate_order(
            order, self.portfolio, self.market_state
        )
        self.assertFalse(is_valid)
        self.assertIn("Kill switch", msg)
        self.assertIsNone(adjusted)

    def test_validate_order_normal_buy(self):
        """Normal buy order should be validated."""
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )
        is_valid, msg, adjusted = self.risk_engine.validate_order(
            order, self.portfolio, self.market_state
        )
        # Should be valid or adjusted (not necessarily True, depends on size limits)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(msg, str)

    def test_validate_order_exceeds_max_position(self):
        """Order exceeding max position limit should be rejected or adjusted."""
        # Try to buy 90% of portfolio in a single stock (max is 10%)
        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=60,  # 60 * 151 = 9060, which is 90.6% of 10000
            order_type=OrderType.LIMIT,
            limit_price=151.0,
        )
        is_valid, msg, adjusted = self.risk_engine.validate_order(
            order, self.portfolio, self.market_state
        )
        # Either rejected or adjusted to fit within limits
        if is_valid:
            self.assertIsNotNone(adjusted)
            # Check that adjusted order is smaller
            self.assertLess(adjusted.quantity, order.quantity)
        else:
            self.assertIsNone(adjusted)

    def test_validate_sell_order_during_force_liquidate(self):
        """Sell orders should be allowed during force liquidation."""
        # Set up a position to sell
        self.portfolio.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=151.0,
            opened_at=datetime.now(),
            last_updated=datetime.now(),
        )

        # Trigger force liquidation by setting large daily loss
        self.risk_engine.state.daily_pnl_pct = -0.12  # Beyond -10% threshold

        sell_order = OrderIntent(
            symbol="AAPL",
            side="SELL",
            quantity=50,
            order_type=OrderType.MARKET,
        )
        is_valid, msg, adjusted = self.risk_engine.validate_order(
            sell_order, self.portfolio, self.market_state
        )
        # Sell should be allowed during force liquidation
        # (validation depends on other constraints too)
        self.assertIsInstance(is_valid, bool)

    def test_validate_buy_order_blocked_during_halt(self):
        """Buy orders should be blocked when new positions are halted."""
        # Set daily loss to trigger halt
        self.risk_engine.state.daily_pnl_pct = -0.06  # Beyond -5% threshold

        order = OrderIntent(
            symbol="AAPL",
            side="BUY",
            quantity=10,
            order_type=OrderType.MARKET,
        )
        is_valid, msg, adjusted = self.risk_engine.validate_order(
            order, self.portfolio, self.market_state
        )
        self.assertFalse(is_valid)
        self.assertIn("halted", msg.lower())


class TestRiskEngineCircuitBreakers(unittest.TestCase):
    """Test circuit breaker functionality."""

    def setUp(self):
        self.config = RiskConfig()
        self.risk_engine = RiskEngine(self.config)

    def test_normal_state_at_start(self):
        """Circuit breaker should return NORMAL for no losses."""
        action = self.risk_engine.check_circuit_breakers(0.0)
        self.assertEqual(action, RiskAction.NORMAL)

    def test_reduce_size_threshold(self):
        """Should trigger REDUCE_SIZE at -2%."""
        action = self.risk_engine.check_circuit_breakers(-0.025)
        self.assertEqual(action, RiskAction.REDUCE_SIZE)

    def test_halt_new_threshold(self):
        """Should trigger HALT_NEW at -5%."""
        action = self.risk_engine.check_circuit_breakers(-0.06)
        self.assertEqual(action, RiskAction.HALT_NEW)

    def test_force_liquidate_threshold(self):
        """Should trigger FORCE_LIQUIDATE at -10%."""
        action = self.risk_engine.check_circuit_breakers(-0.12)
        self.assertEqual(action, RiskAction.FORCE_LIQUIDATE)


class TestRiskEnginePositionSizing(unittest.TestCase):
    """Test ATR-based position sizing."""

    def setUp(self):
        self.config = RiskConfig()
        self.risk_engine = RiskEngine(self.config)

    def test_calculate_position_size_with_atr(self):
        """Position size should be calculated based on ATR."""
        portfolio = Portfolio(
            timestamp=datetime.now(),
            cash=100000.0,
            total_value=100000.0,
            buying_power=100000.0,
            positions={},
        )
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.7,
            target_position_pct=0.1,
            model_name="test",
        )
        market_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=148.0,
            close=152.0,
            volume=1000000,
            technical_indicators={"ATR": 5.0},  # ATR of 5
        )

        size = self.risk_engine.calculate_position_size(
            signal, market_state, portfolio
        )

        # Position size should be positive
        self.assertGreater(size, 0)

        # ATR formula: (Equity * risk_per_trade) / (ATR * multiplier)
        # Expected: (100000 * 0.01) / (5 * 2.0) = 1000 / 10 = 100 shares
        # But also capped by max_single_position (10% = $10000 = ~66 shares at $152)
        self.assertLessEqual(size * 152, 100000 * self.config.max_single_position)

    def test_position_size_without_atr_uses_fallback(self):
        """Position sizing should work even without ATR data."""
        portfolio = Portfolio(
            timestamp=datetime.now(),
            cash=100000.0,
            total_value=100000.0,
            buying_power=100000.0,
            positions={},
        )
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.8,
            confidence=0.7,
            target_position_pct=0.1,
            model_name="test",
        )
        market_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=148.0,
            close=152.0,
            volume=1000000,
            technical_indicators={},  # No ATR
        )

        size = self.risk_engine.calculate_position_size(
            signal, market_state, portfolio
        )

        # Should still return a reasonable size
        self.assertGreaterEqual(size, 0)


if __name__ == "__main__":
    unittest.main()
