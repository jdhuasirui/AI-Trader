"""
Integration tests for AI-Trader core modules.

Tests the new infrastructure:
- Risk engine with circuit breakers
- Order manager with state machine
- LLM validator for hallucination mitigation
- Signal aggregator for multi-model fusion
- Execution algorithms with slippage
"""

import json
import unittest
from datetime import datetime
from typing import Dict, List


class TestCoreImports(unittest.TestCase):
    """Test that all core modules can be imported."""

    def test_import_data_structures(self):
        """Test data structures import."""
        from core import (
            MarketState, Signal, TargetPortfolio, OrderIntent,
            ExecutionReport, Position, Portfolio, RiskAction,
            OrderStatus, OrderType, SignalDirection, Regime,
        )
        self.assertTrue(True)

    def test_import_risk_engine(self):
        """Test risk engine import."""
        from core import RiskEngine, RiskConfig, RiskState
        self.assertTrue(True)

    def test_import_order_manager(self):
        """Test order manager import."""
        from core import OrderManager, OrderStateMachine, OrderState, RetryConfig
        self.assertTrue(True)

    def test_import_observability(self):
        """Test observability import."""
        from core import MetricsCollector, AuditLogger, AlertManager
        self.assertTrue(True)

    def test_import_execution_algos(self):
        """Test execution algorithms import."""
        from core import (
            ExecutionAlgorithm, MarketOrderExecution, TWAPExecution,
            LimitFirstExecution, SlippageSimulator, ExecutionConfig,
        )
        self.assertTrue(True)

    def test_import_regime_detector(self):
        """Test regime detector import."""
        from core import RegimeDetector, RegimeConfig, RegimeFeatures
        self.assertTrue(True)

    def test_import_signal_aggregator(self):
        """Test signal aggregator import."""
        from core import SignalAggregator, AggregatorConfig, ConfidenceCalibrator
        self.assertTrue(True)

    def test_import_llm_validator(self):
        """Test LLM validator import."""
        from core import LLMValidator, ValidationResult, create_structured_prompt
        self.assertTrue(True)


class TestRiskEngine(unittest.TestCase):
    """Test risk engine functionality."""

    def setUp(self):
        from core import RiskEngine, RiskConfig
        # Use default config (no arguments)
        self.config = RiskConfig()
        self.risk_engine = RiskEngine(self.config)

    def test_risk_engine_creation(self):
        """Test risk engine can be created."""
        self.assertIsNotNone(self.risk_engine)

    def test_calculate_position_size(self):
        """Test position sizing calculation."""
        from core import Portfolio, Signal, SignalDirection, MarketState
        portfolio = Portfolio(
            timestamp=datetime.now(),
            cash=10000.0,
            positions={},
            total_value=10000.0,
            buying_power=10000.0,
        )
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            direction=SignalDirection.LONG,
            strength=0.7,
            confidence=0.8,
            target_position_pct=0.1,
            model_name="test",
        )
        market_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            technical_indicators={"ATR": 5.0},
        )
        size = self.risk_engine.calculate_position_size(
            signal=signal,
            market_state=market_state,
            portfolio=portfolio,
        )
        self.assertIsInstance(size, float)
        self.assertGreaterEqual(size, 0)


class TestOrderManager(unittest.TestCase):
    """Test order manager functionality."""

    def setUp(self):
        from core import OrderManager, RetryConfig
        # Use default config
        self.config = RetryConfig()
        self.order_manager = OrderManager(retry_config=self.config)

    def test_order_manager_creation(self):
        """Test order manager can be created."""
        self.assertIsNotNone(self.order_manager)


class TestLLMValidator(unittest.TestCase):
    """Test LLM validator functionality."""

    def setUp(self):
        from core import LLMValidator
        self.validator = LLMValidator(
            tolerance_pct=5.0,
            require_reasoning=True,
            min_reasoning_length=10,
        )

    def test_validator_creation(self):
        """Test validator can be created."""
        self.assertIsNotNone(self.validator)

    def test_validate_valid_json(self):
        """Test validation of valid JSON output."""
        valid_output = json.dumps({
            "analysis": "Market analysis shows bullish momentum for tech stocks.",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.75,
                    "reasoning": "Strong earnings report and positive guidance."
                }
            ],
            "risk_assessment": "Portfolio risk is moderate with 5% cash buffer."
        })

        result = self.validator.validate(valid_output)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_validate_invalid_json(self):
        """Test validation of invalid JSON output."""
        invalid_output = "This is not valid JSON at all"

        result = self.validator.validate(invalid_output)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_validate_missing_fields(self):
        """Test validation catches missing required fields."""
        incomplete_output = json.dumps({
            "analysis": "Market analysis"
            # Missing signals and risk_assessment
        })

        result = self.validator.validate(incomplete_output)
        self.assertFalse(result.is_valid)

    def test_validate_invalid_action(self):
        """Test validation catches invalid action."""
        invalid_action = json.dumps({
            "analysis": "Market analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "INVALID_ACTION",  # Should be BUY, SELL, or HOLD
                    "quantity": 10,
                    "confidence": 0.75,
                    "reasoning": "Some reasoning here for the trade."
                }
            ],
            "risk_assessment": "Risk assessment"
        })

        result = self.validator.validate(invalid_action)
        self.assertFalse(result.is_valid)

    def test_validate_confidence_out_of_range(self):
        """Test validation catches confidence out of range."""
        bad_confidence = json.dumps({
            "analysis": "Market analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 1.5,  # Should be between 0 and 1
                    "reasoning": "Some reasoning here for the trade."
                }
            ],
            "risk_assessment": "Risk assessment"
        })

        result = self.validator.validate(bad_confidence)
        self.assertFalse(result.is_valid)


class TestSignalAggregator(unittest.TestCase):
    """Test signal aggregator functionality."""

    def setUp(self):
        from core import SignalAggregator, AggregatorConfig
        # Use default config
        self.config = AggregatorConfig()
        self.aggregator = SignalAggregator(self.config)

    def test_aggregator_creation(self):
        """Test aggregator can be created."""
        self.assertIsNotNone(self.aggregator)

    def test_aggregate_signals(self):
        """Test signal aggregation."""
        from core import Signal, SignalDirection

        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.8,
                confidence=0.75,
                target_position_pct=0.1,
                model_name="gpt-4o",
            ),
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                direction=SignalDirection.LONG,
                strength=0.7,
                confidence=0.65,
                target_position_pct=0.08,
                model_name="claude-3",
            ),
        ]

        aggregated = self.aggregator.aggregate_signals(signals)
        self.assertIsNotNone(aggregated)


class TestSlippageModeling(unittest.TestCase):
    """Test slippage modeling in tool_trade."""

    @classmethod
    def setUpClass(cls):
        """Check if fastmcp is available."""
        try:
            import fastmcp
            cls.fastmcp_available = True
        except ImportError:
            cls.fastmcp_available = False

    def test_calculate_slippage(self):
        """Test slippage calculation."""
        if not self.fastmcp_available:
            self.skipTest("fastmcp not installed")

        from agent_tools.tool_trade import calculate_slippage

        slippage = calculate_slippage(
            order_qty=100,
            daily_volume=1000000,
            is_buy=True,
        )

        self.assertIsInstance(slippage, float)
        self.assertGreaterEqual(slippage, 0)

    def test_apply_slippage_to_price(self):
        """Test applying slippage to price."""
        if not self.fastmcp_available:
            self.skipTest("fastmcp not installed")

        from agent_tools.tool_trade import apply_slippage_to_price

        effective_price, slippage_pct = apply_slippage_to_price(
            base_price=100.0,
            order_qty=100,
            daily_volume=1000000,
            is_buy=True,
        )

        # Buy orders should have higher effective price
        self.assertGreaterEqual(effective_price, 100.0)
        self.assertGreaterEqual(slippage_pct, 0)


class TestRegimeDetector(unittest.TestCase):
    """Test regime detector functionality."""

    def setUp(self):
        from core import RegimeDetector, RegimeConfig
        # Use default config
        self.config = RegimeConfig()
        self.detector = RegimeDetector(self.config)

    def test_detector_creation(self):
        """Test detector can be created."""
        self.assertIsNotNone(self.detector)

    def test_get_current_regime(self):
        """Test getting current regime for unknown symbol returns UNKNOWN."""
        from core import Regime

        regime = self.detector.get_current_regime("UNKNOWN_SYMBOL")
        self.assertEqual(regime, Regime.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
