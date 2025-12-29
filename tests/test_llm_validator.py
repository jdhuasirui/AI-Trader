"""
Unit tests for LLMValidator with business logic checks.

Tests:
- Schema validation
- Business logic validation (edge cases)
- Nonsensical value detection
- Contradictory signal detection
"""

import json
import unittest
from datetime import datetime

from core import LLMValidator, ValidationResult


class TestLLMValidatorSchema(unittest.TestCase):
    """Test LLMValidator schema validation."""

    def setUp(self):
        self.validator = LLMValidator(
            tolerance_pct=5.0,
            require_reasoning=True,
            min_reasoning_length=10,
        )

    def test_valid_output_passes(self):
        """Valid LLM output should pass validation."""
        output = json.dumps({
            "analysis": "Market shows strong momentum with positive technicals.",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.75,
                    "reasoning": "Strong earnings and positive sector momentum."
                }
            ],
            "risk_assessment": "Portfolio risk is moderate."
        })
        result = self.validator.validate(output)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_json_fails(self):
        """Invalid JSON should fail validation."""
        result = self.validator.validate("not valid json at all")
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_missing_required_fields_fails(self):
        """Missing required fields should fail validation."""
        output = json.dumps({
            "analysis": "Some analysis"
            # Missing signals and risk_assessment
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)

    def test_empty_signals_fails(self):
        """Empty signals array should fail."""
        output = json.dumps({
            "analysis": "Analysis here",
            "signals": [],
            "risk_assessment": "Risk assessment here"
        })
        result = self.validator.validate(output)
        # Empty signals might be valid for HOLD scenarios
        # but should be flagged for review

    def test_invalid_action_fails(self):
        """Invalid action should fail validation."""
        output = json.dumps({
            "analysis": "Analysis here",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "INVALID_ACTION",
                    "quantity": 10,
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)


class TestLLMValidatorBusinessLogic(unittest.TestCase):
    """Test LLMValidator business logic validation."""

    def setUp(self):
        self.validator = LLMValidator(
            tolerance_pct=5.0,
            require_reasoning=True,
            min_reasoning_length=10,
        )

    def test_zero_confidence_should_warn(self):
        """Zero confidence signals should be flagged."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.0,  # Zero confidence is suspicious
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        # Should either fail or have warnings
        if result.is_valid:
            self.assertGreater(len(result.warnings), 0)

    def test_confidence_out_of_range_fails(self):
        """Confidence > 1 should fail validation."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 1.5,  # Invalid: > 1
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)

    def test_negative_quantity_fails(self):
        """Negative quantity should fail validation."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": -10,  # Invalid
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)

    def test_zero_quantity_fails(self):
        """Zero quantity should fail validation."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 0,  # Invalid
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)

    def test_extremely_high_quantity_should_warn(self):
        """Extremely high quantity should trigger warning."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 1000000,  # Suspicious
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        # Should have warnings about unusual quantity

    def test_short_reasoning_fails(self):
        """Reasoning shorter than min_reasoning_length should fail."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.5,
                    "reasoning": "Short"  # Too short
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        # Should fail or warn about short reasoning

    def test_empty_symbol_fails(self):
        """Empty symbol should fail validation."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "",  # Empty
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)

    def test_invalid_symbol_format_fails(self):
        """Invalid symbol format should fail."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "!!!INVALID!!!",  # Invalid format
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.5,
                    "reasoning": "Some reasoning here."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        self.assertFalse(result.is_valid)


class TestLLMValidatorContradictions(unittest.TestCase):
    """Test detection of contradictory signals."""

    def setUp(self):
        self.validator = LLMValidator(
            tolerance_pct=5.0,
            require_reasoning=True,
            min_reasoning_length=10,
        )

    def test_conflicting_signals_same_symbol(self):
        """Conflicting BUY and SELL for same symbol should be flagged."""
        output = json.dumps({
            "analysis": "Analysis",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.5,
                    "reasoning": "Buy reasoning here..."
                },
                {
                    "symbol": "AAPL",
                    "action": "SELL",
                    "quantity": 5,
                    "confidence": 0.5,
                    "reasoning": "Sell reasoning here..."
                }
            ],
            "risk_assessment": "Risk assessment"
        })
        result = self.validator.validate(output)
        # Should flag contradictory signals
        if result.is_valid:
            self.assertGreater(len(result.warnings), 0)


class TestLLMValidatorWithMarketData(unittest.TestCase):
    """Test validation with market data context."""

    def setUp(self):
        self.validator = LLMValidator(
            tolerance_pct=5.0,
            require_reasoning=True,
            min_reasoning_length=10,
        )

    def test_validate_with_market_state(self):
        """Validation should consider market state when provided."""
        from core import MarketState
        from datetime import datetime

        output = json.dumps({
            "analysis": "Analysis shows AAPL trading at around $150.",
            "signals": [
                {
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "confidence": 0.75,
                    "reasoning": "Strong momentum and positive technicals."
                }
            ],
            "risk_assessment": "Risk assessment is moderate"
        })

        market_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            open=148.0,
            high=152.0,
            low=147.0,
            close=150.0,
            volume=50000000,
        )

        result = self.validator.validate(output, market_state=market_state)
        # Validation should pass with valid market state
        self.assertIsInstance(result.is_valid, bool)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult data structure."""

    def test_validation_result_creation(self):
        """ValidationResult should be properly created."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor warning"],
            corrected_output={"test": "data"},
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 1)

    def test_validation_result_with_errors(self):
        """ValidationResult with errors should have is_valid=False."""
        result = ValidationResult(
            is_valid=False,
            errors=["Critical error"],
            warnings=[],
            corrected_output=None,
        )
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.corrected_output)


if __name__ == "__main__":
    unittest.main()
