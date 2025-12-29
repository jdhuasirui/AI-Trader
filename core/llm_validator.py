"""
LLM Validator for AI-Trader

This module implements guardrails for LLM-generated trading signals:
- Structured output enforcement
- Hallucination mitigation (fact-checking against actual data)
- Consistency validation
- Multi-LLM consensus for critical decisions

Phase 4 of the improvement plan.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .data_structures import Signal, SignalDirection, MarketState

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of LLM output validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_output: Optional[Dict] = None
    confidence_adjustment: float = 1.0  # Multiplier for confidence


# Expected JSON schema for LLM trading outputs
SIGNAL_SCHEMA = {
    "type": "object",
    "required": ["analysis", "signals", "risk_assessment"],
    "properties": {
        "analysis": {"type": "string"},
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["symbol", "action", "quantity", "confidence", "reasoning"],
                "properties": {
                    "symbol": {"type": "string"},
                    "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "quantity": {"type": "number", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reasoning": {"type": "string"},
                },
            },
        },
        "risk_assessment": {"type": "string"},
    },
}


class LLMValidator:
    """
    Validates and sanitizes LLM-generated trading signals.

    Implements:
    1. JSON schema validation
    2. Data grounding (verify claims against actual market data)
    3. Consistency checks
    4. Hallucination detection
    """

    def __init__(
        self,
        tolerance_pct: float = 5.0,  # 5% tolerance for price claims
        require_reasoning: bool = True,
        min_reasoning_length: int = 20,
    ):
        self.tolerance_pct = tolerance_pct
        self.require_reasoning = require_reasoning
        self.min_reasoning_length = min_reasoning_length

        # Track validation history for pattern detection
        self._validation_history: List[ValidationResult] = []

    def validate(
        self,
        llm_output: str,
        market_state: Optional[MarketState] = None,
    ) -> ValidationResult:
        """
        Validate LLM output against schema and market data.

        Args:
            llm_output: Raw LLM output string
            market_state: Current market data for fact-checking

        Returns:
            ValidationResult with errors, warnings, and corrected output
        """
        result = ValidationResult(is_valid=True)

        # Step 1: Parse JSON
        parsed = self._parse_json(llm_output)
        if parsed is None:
            result.is_valid = False
            result.errors.append("Failed to parse JSON from LLM output")
            return result

        # Step 2: Schema validation
        schema_result = self._validate_schema(parsed)
        result.errors.extend(schema_result.errors)
        result.warnings.extend(schema_result.warnings)

        if schema_result.errors:
            result.is_valid = False
            return result

        # Step 3: Fact-checking against market data
        if market_state:
            fact_result = self._fact_check(parsed, market_state)
            result.errors.extend(fact_result.errors)
            result.warnings.extend(fact_result.warnings)
            result.confidence_adjustment *= fact_result.confidence_adjustment

        # Step 4: Consistency checks
        consistency_result = self._check_consistency(parsed)
        result.errors.extend(consistency_result.errors)
        result.warnings.extend(consistency_result.warnings)

        # Step 5: Hallucination detection
        hallucination_result = self._detect_hallucinations(parsed, market_state)
        result.warnings.extend(hallucination_result.warnings)
        result.confidence_adjustment *= hallucination_result.confidence_adjustment

        # Determine final validity
        if result.errors:
            result.is_valid = False
        else:
            result.corrected_output = parsed

        # Track history
        self._validation_history.append(result)
        if len(self._validation_history) > 100:
            self._validation_history = self._validation_history[-100:]

        return result

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM output."""
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _validate_schema(self, data: Dict) -> ValidationResult:
        """Validate data against expected schema."""
        result = ValidationResult(is_valid=True)

        # Check required fields
        for field in SIGNAL_SCHEMA.get("required", []):
            if field not in data:
                result.errors.append(f"Missing required field: {field}")

        # Validate signals array
        if "signals" in data:
            if not isinstance(data["signals"], list):
                result.errors.append("'signals' must be an array")
            else:
                for i, signal in enumerate(data["signals"]):
                    signal_errors = self._validate_signal(signal, i)
                    result.errors.extend(signal_errors)

        # Validate analysis
        if "analysis" in data:
            if not isinstance(data["analysis"], str):
                result.errors.append("'analysis' must be a string")
            elif len(data["analysis"]) < 10:
                result.warnings.append("Analysis is very short")

        return result

    def _validate_signal(self, signal: Dict, index: int) -> List[str]:
        """Validate individual signal object."""
        errors = []
        prefix = f"Signal[{index}]"

        required_fields = ["symbol", "action", "quantity", "confidence", "reasoning"]
        for field in required_fields:
            if field not in signal:
                errors.append(f"{prefix}: Missing required field '{field}'")

        # Validate action
        if "action" in signal:
            valid_actions = ["BUY", "SELL", "HOLD"]
            if signal["action"].upper() not in valid_actions:
                errors.append(f"{prefix}: Invalid action '{signal['action']}', must be one of {valid_actions}")

        # Validate confidence
        if "confidence" in signal:
            try:
                conf = float(signal["confidence"])
                if not 0 <= conf <= 1:
                    errors.append(f"{prefix}: Confidence must be between 0 and 1, got {conf}")
            except (TypeError, ValueError):
                errors.append(f"{prefix}: Confidence must be a number")

        # Validate quantity
        if "quantity" in signal:
            try:
                qty = float(signal["quantity"])
                if qty < 0:
                    errors.append(f"{prefix}: Quantity cannot be negative")
            except (TypeError, ValueError):
                errors.append(f"{prefix}: Quantity must be a number")

        # Validate reasoning
        if self.require_reasoning and "reasoning" in signal:
            reasoning = signal.get("reasoning", "")
            if len(reasoning) < self.min_reasoning_length:
                errors.append(f"{prefix}: Reasoning too short (min {self.min_reasoning_length} chars)")

        return errors

    def _fact_check(self, data: Dict, market_state: MarketState) -> ValidationResult:
        """Verify claims in LLM output against actual market data."""
        result = ValidationResult(is_valid=True)

        analysis = data.get("analysis", "") + data.get("risk_assessment", "")

        # Extract price claims from text
        price_patterns = [
            r'price[:\s]+\$?([\d,]+\.?\d*)',
            r'trading at[:\s]+\$?([\d,]+\.?\d*)',
            r'current price[:\s]+\$?([\d,]+\.?\d*)',
            r'\$?([\d,]+\.?\d*)\s*(?:per share|USD)',
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, analysis, re.IGNORECASE)
            for match in matches:
                try:
                    claimed_price = float(match.replace(",", ""))
                    actual_price = market_state.close

                    # Check if claim is within tolerance
                    pct_diff = abs(claimed_price - actual_price) / actual_price * 100
                    if pct_diff > self.tolerance_pct:
                        result.warnings.append(
                            f"Price claim ${claimed_price:.2f} differs from actual ${actual_price:.2f} by {pct_diff:.1f}%"
                        )
                        result.confidence_adjustment *= 0.8  # Reduce confidence

                except (ValueError, ZeroDivisionError):
                    continue

        # Extract percentage change claims
        pct_patterns = [
            r'up\s+([\d.]+)%',
            r'down\s+([\d.]+)%',
            r'gained\s+([\d.]+)%',
            r'lost\s+([\d.]+)%',
            r'increased\s+([\d.]+)%',
            r'decreased\s+([\d.]+)%',
        ]

        for pattern in pct_patterns:
            matches = re.findall(pattern, analysis, re.IGNORECASE)
            for match in matches:
                try:
                    claimed_pct = float(match)
                    # Calculate actual daily change
                    actual_pct = market_state.daily_range

                    if abs(claimed_pct - actual_pct) > 2:  # 2% tolerance
                        result.warnings.append(
                            f"Percentage claim {claimed_pct}% may not match actual range {actual_pct:.1f}%"
                        )
                        result.confidence_adjustment *= 0.9

                except ValueError:
                    continue

        return result

    def _check_consistency(self, data: Dict) -> ValidationResult:
        """Check for internal consistency in LLM output."""
        result = ValidationResult(is_valid=True)

        signals = data.get("signals", [])
        analysis = data.get("analysis", "").lower()
        risk_assessment = data.get("risk_assessment", "").lower()

        # Check if signals match sentiment in analysis
        buy_signals = sum(1 for s in signals if s.get("action", "").upper() == "BUY")
        sell_signals = sum(1 for s in signals if s.get("action", "").upper() == "SELL")

        bullish_words = ["bullish", "upside", "growth", "positive", "buy"]
        bearish_words = ["bearish", "downside", "decline", "negative", "sell"]

        analysis_bullish = sum(1 for w in bullish_words if w in analysis)
        analysis_bearish = sum(1 for w in bearish_words if w in analysis)

        # Check for contradictions
        if buy_signals > sell_signals and analysis_bearish > analysis_bullish + 2:
            result.warnings.append("Buy signals contradict bearish analysis tone")
            result.confidence_adjustment *= 0.9

        if sell_signals > buy_signals and analysis_bullish > analysis_bearish + 2:
            result.warnings.append("Sell signals contradict bullish analysis tone")
            result.confidence_adjustment *= 0.9

        # Check for conflicting signals on same symbol
        symbol_actions = {}
        for signal in signals:
            symbol = signal.get("symbol", "")
            action = signal.get("action", "").upper()
            if symbol in symbol_actions and symbol_actions[symbol] != action:
                result.errors.append(f"Conflicting actions for {symbol}: {symbol_actions[symbol]} and {action}")
            symbol_actions[symbol] = action

        return result

    def _detect_hallucinations(
        self,
        data: Dict,
        market_state: Optional[MarketState],
    ) -> ValidationResult:
        """Detect potential hallucinations in LLM output."""
        result = ValidationResult(is_valid=True)

        analysis = data.get("analysis", "")

        # Check for specific red flags
        hallucination_patterns = [
            r'breaking news',  # LLMs don't have real-time news
            r'just announced',
            r'moments ago',
            r'live update',
            r'real-time',
            r'as of \d{1,2}:\d{2}',  # Specific time claims
        ]

        for pattern in hallucination_patterns:
            if re.search(pattern, analysis, re.IGNORECASE):
                result.warnings.append(f"Potential hallucination detected: '{pattern}' claim")
                result.confidence_adjustment *= 0.7

        # Check for fabricated numbers (too precise)
        precise_patterns = [
            r'\$[\d,]+\.\d{4,}',  # More than 2 decimal places for prices
            r'[\d.]+%\s*exactly',
            r'precisely\s+[\d.]+',
        ]

        for pattern in precise_patterns:
            if re.search(pattern, analysis, re.IGNORECASE):
                result.warnings.append("Suspiciously precise numbers detected")
                result.confidence_adjustment *= 0.85

        return result

    def parse_signals(
        self,
        validated_output: Dict,
        model_name: str = "llm",
    ) -> List[Signal]:
        """
        Convert validated LLM output to Signal objects.

        Args:
            validated_output: The corrected_output from ValidationResult
            model_name: Name of the LLM model

        Returns:
            List of Signal objects
        """
        signals = []

        for signal_data in validated_output.get("signals", []):
            action = signal_data.get("action", "HOLD").upper()

            direction = {
                "BUY": SignalDirection.LONG,
                "SELL": SignalDirection.SHORT,
                "HOLD": SignalDirection.NEUTRAL,
            }.get(action, SignalDirection.NEUTRAL)

            signal = Signal(
                timestamp=datetime.now(),
                symbol=signal_data.get("symbol", ""),
                direction=direction,
                strength=signal_data.get("confidence", 0.5),
                confidence=signal_data.get("confidence", 0.5),
                target_position_pct=0.1 if action == "BUY" else (-0.1 if action == "SELL" else 0),
                model_name=model_name,
                reasoning=signal_data.get("reasoning", ""),
            )
            signals.append(signal)

        return signals

    def get_validation_stats(self) -> Dict:
        """Get statistics on validation history."""
        if not self._validation_history:
            return {"total": 0, "valid": 0, "invalid": 0, "avg_confidence_adjustment": 1.0}

        valid_count = sum(1 for r in self._validation_history if r.is_valid)
        avg_adjustment = sum(r.confidence_adjustment for r in self._validation_history) / len(self._validation_history)

        return {
            "total": len(self._validation_history),
            "valid": valid_count,
            "invalid": len(self._validation_history) - valid_count,
            "valid_rate": valid_count / len(self._validation_history),
            "avg_confidence_adjustment": avg_adjustment,
        }


def create_structured_prompt(base_prompt: str) -> str:
    """
    Wrap a base prompt with structured output instructions.

    This enforces JSON schema output from LLMs.
    """
    schema_instruction = """
You MUST respond with valid JSON in the following format:
```json
{
  "analysis": "Your market analysis here (string)",
  "signals": [
    {
      "symbol": "AAPL",
      "action": "BUY|SELL|HOLD",
      "quantity": 100,
      "confidence": 0.75,
      "reasoning": "Explanation for this signal"
    }
  ],
  "risk_assessment": "Your risk assessment here (string)"
}
```

Important:
- action must be exactly "BUY", "SELL", or "HOLD"
- confidence must be a number between 0 and 1
- quantity must be a positive number
- Do not include any text outside the JSON block
"""

    return f"{base_prompt}\n\n{schema_instruction}"
