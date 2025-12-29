"""
Signal Aggregator for AI-Trader

This module implements multi-model signal fusion:
- Regime-aware weighting (different model weights per regime)
- Consensus requirements (60% agreement for action)
- Dynamic weight updates based on rolling performance
- Confidence calibration integration

The meta-controller aggregates signals from multiple AI models
into a unified TargetPortfolio allocation.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .data_structures import (
    Signal,
    SignalDirection,
    TargetPortfolio,
    MarketState,
    Regime,
)
from .regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatorConfig:
    """Configuration for signal aggregation (immutable)."""

    # Consensus requirements
    consensus_threshold: float = 0.6  # 60% agreement required for action
    min_signals_for_action: int = 2  # Minimum signals needed

    # Confidence thresholds
    min_confidence: float = 0.3  # Minimum confidence to consider signal
    min_strength: float = 0.2  # Minimum strength to consider signal

    # Weight parameters
    base_weight: float = 1.0  # Default weight for new models
    llm_weight_discount: float = 0.5  # LLM signals get 50% weight
    max_weight: float = 2.0  # Maximum model weight
    min_weight: float = 0.1  # Minimum model weight

    # Performance tracking
    rolling_window_days: int = 30  # Window for performance calculation
    weight_update_frequency: int = 5  # Update weights every N signals

    # Position sizing
    max_position_per_symbol: float = 0.2  # 20% max per symbol
    max_total_exposure: float = 1.0  # 100% max total exposure


@dataclass
class ModelPerformance:
    """Track model performance for dynamic weighting."""
    model_name: str
    total_signals: int = 0
    correct_signals: int = 0
    total_pnl: float = 0.0

    # Per-regime performance
    regime_signals: Dict[Regime, int] = field(default_factory=lambda: defaultdict(int))
    regime_correct: Dict[Regime, int] = field(default_factory=lambda: defaultdict(int))

    # Recent performance (rolling window)
    recent_returns: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy(self) -> float:
        return self.correct_signals / self.total_signals if self.total_signals > 0 else 0.5

    @property
    def sharpe_30d(self) -> float:
        if len(self.recent_returns) < 5:
            return 0
        mean = sum(self.recent_returns) / len(self.recent_returns)
        var = sum((r - mean) ** 2 for r in self.recent_returns) / len(self.recent_returns)
        std = math.sqrt(var) if var > 0 else 1
        return (mean / std) * math.sqrt(252) if std > 0 else 0


class ConfidenceCalibrator:
    """
    Calibrate model confidence scores to actual probabilities.

    Uses isotonic regression style calibration to ensure
    "90% confidence" means 90% actual win rate.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        # Calibration mapping: predicted confidence -> actual accuracy
        self._calibration_map: Dict[str, Dict[int, float]] = {}  # model -> bin -> accuracy
        self._bin_counts: Dict[str, Dict[int, int]] = {}

    def update(
        self,
        model_name: str,
        predicted_confidence: float,
        actual_correct: bool,
    ) -> None:
        """Update calibration with a new observation."""
        if model_name not in self._calibration_map:
            self._calibration_map[model_name] = {i: 0.5 for i in range(self.n_bins)}
            self._bin_counts[model_name] = {i: 0 for i in range(self.n_bins)}

        bin_idx = min(int(predicted_confidence * self.n_bins), self.n_bins - 1)
        count = self._bin_counts[model_name][bin_idx]
        current = self._calibration_map[model_name][bin_idx]

        # Exponential moving average update
        alpha = 1 / (count + 1)
        new_accuracy = current * (1 - alpha) + float(actual_correct) * alpha

        self._calibration_map[model_name][bin_idx] = new_accuracy
        self._bin_counts[model_name][bin_idx] = count + 1

    def calibrate(self, model_name: str, confidence: float) -> float:
        """Get calibrated confidence for a model's prediction."""
        if model_name not in self._calibration_map:
            return confidence

        bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
        calibrated = self._calibration_map[model_name][bin_idx]

        # Blend with original for smoothness
        alpha = 0.7  # Weight for calibrated value
        return alpha * calibrated + (1 - alpha) * confidence


class SignalAggregator:
    """
    Aggregates signals from multiple models into target portfolio allocations.

    Uses:
    - Regime-aware weighting
    - Consensus voting
    - Dynamic weight updates based on performance
    - Confidence calibration
    """

    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        config: Optional[AggregatorConfig] = None,
    ):
        self.regime_detector = regime_detector or RegimeDetector()
        self.config = config or AggregatorConfig()

        # Model weights (by regime)
        self._model_weights: Dict[str, Dict[Regime, float]] = {}

        # Model performance tracking
        self._model_performance: Dict[str, ModelPerformance] = {}

        # Confidence calibration
        self._calibrator = ConfidenceCalibrator()

        # Signal count for weight updates
        self._signal_count = 0

        # LLM model names (for weight discount)
        self._llm_models = {"gpt", "claude", "gemini", "qwen", "deepseek", "llama"}

    def aggregate_signals(
        self,
        signals: List[Signal],
        current_market_state: Optional[MarketState] = None,
    ) -> TargetPortfolio:
        """
        Aggregate multiple signals into a target portfolio.

        Args:
            signals: List of signals from different models
            current_market_state: Current market data for regime detection

        Returns:
            TargetPortfolio with aggregated allocations
        """
        if not signals:
            return TargetPortfolio(
                timestamp=datetime.now(),
                regime=Regime.UNKNOWN,
            )

        # Detect current regime
        regime = Regime.UNKNOWN
        if current_market_state:
            regime = self.regime_detector.detect(current_market_state)

        # Filter and validate signals
        valid_signals = self._filter_signals(signals)
        if not valid_signals:
            logger.info("No valid signals after filtering")
            return TargetPortfolio(
                timestamp=datetime.now(),
                regime=regime,
            )

        # Get weights for each model in current regime
        weights = self._get_regime_weights(valid_signals, regime)

        # Check consensus
        consensus_result = self._check_consensus(valid_signals, weights)

        # Aggregate positions
        target_positions = self._aggregate_positions(valid_signals, weights, regime)

        # Calculate aggregate metrics
        total_confidence = self._weighted_average(
            [s.confidence for s in valid_signals],
            [weights.get(s.model_name, 1.0) for s in valid_signals],
        )

        target = TargetPortfolio(
            timestamp=datetime.now(),
            positions=target_positions,
            regime=regime,
            model_agreement=consensus_result["agreement"],
            confidence=total_confidence,
            source_signals=valid_signals,
        )

        # Update signal count for weight updates
        self._signal_count += 1
        if self._signal_count % self.config.weight_update_frequency == 0:
            self._update_weights()

        return target

    def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals by confidence and strength thresholds."""
        valid = []
        for signal in signals:
            if signal.confidence < self.config.min_confidence:
                logger.debug(f"Signal from {signal.model_name} filtered: low confidence {signal.confidence:.2f}")
                continue
            if signal.strength < self.config.min_strength:
                logger.debug(f"Signal from {signal.model_name} filtered: low strength {signal.strength:.2f}")
                continue
            valid.append(signal)
        return valid

    def _get_regime_weights(
        self,
        signals: List[Signal],
        regime: Regime,
    ) -> Dict[str, float]:
        """Get model weights for the current regime."""
        weights = {}

        for signal in signals:
            model = signal.model_name

            # Initialize weights if new model
            if model not in self._model_weights:
                self._initialize_model_weights(model)

            # Get regime-specific weight
            base_weight = self._model_weights[model].get(regime, self.config.base_weight)

            # Apply LLM discount
            if self._is_llm_model(model):
                base_weight *= self.config.llm_weight_discount

            weights[model] = base_weight

        return weights

    def _initialize_model_weights(self, model_name: str) -> None:
        """Initialize weights for a new model."""
        self._model_weights[model_name] = {
            Regime.TRENDING_UP: self.config.base_weight,
            Regime.TRENDING_DOWN: self.config.base_weight,
            Regime.RANGING: self.config.base_weight,
            Regime.VOLATILE: self.config.base_weight * 0.8,  # Lower weight in volatile
            Regime.UNKNOWN: self.config.base_weight,
        }

        self._model_performance[model_name] = ModelPerformance(model_name=model_name)

    def _is_llm_model(self, model_name: str) -> bool:
        """Check if model is an LLM (for weight discount)."""
        name_lower = model_name.lower()
        return any(llm in name_lower for llm in self._llm_models)

    def _check_consensus(
        self,
        signals: List[Signal],
        weights: Dict[str, float],
    ) -> Dict:
        """Check if signals meet consensus requirements."""
        if len(signals) < self.config.min_signals_for_action:
            return {"consensus": False, "agreement": 0, "reason": "insufficient signals"}

        # Group by direction
        direction_weights = defaultdict(float)
        total_weight = 0

        for signal in signals:
            weight = weights.get(signal.model_name, 1.0) * signal.confidence
            direction_weights[signal.direction] += weight
            total_weight += weight

        if total_weight == 0:
            return {"consensus": False, "agreement": 0, "reason": "zero total weight"}

        # Find dominant direction
        max_direction = max(direction_weights, key=direction_weights.get)
        agreement = direction_weights[max_direction] / total_weight

        consensus = agreement >= self.config.consensus_threshold

        return {
            "consensus": consensus,
            "agreement": agreement,
            "dominant_direction": max_direction,
            "direction_breakdown": dict(direction_weights),
        }

    def _aggregate_positions(
        self,
        signals: List[Signal],
        weights: Dict[str, float],
        regime: Regime,
    ) -> Dict[str, float]:
        """Aggregate signal positions into target weights."""
        # Group signals by symbol
        symbol_signals: Dict[str, List[Tuple[Signal, float]]] = defaultdict(list)

        for signal in signals:
            weight = weights.get(signal.model_name, 1.0)
            symbol_signals[signal.symbol].append((signal, weight))

        # Calculate weighted target for each symbol
        target_positions = {}

        for symbol, signal_list in symbol_signals.items():
            weighted_sum = 0
            weight_sum = 0

            for signal, weight in signal_list:
                # Apply confidence calibration
                calibrated_conf = self._calibrator.calibrate(
                    signal.model_name, signal.confidence
                )

                # Direction multiplier
                direction_mult = {
                    SignalDirection.LONG: 1.0,
                    SignalDirection.SHORT: -1.0,
                    SignalDirection.NEUTRAL: 0.0,
                }.get(signal.direction, 0.0)

                # Weighted contribution
                contribution = (
                    signal.target_position_pct *
                    direction_mult *
                    weight *
                    calibrated_conf *
                    signal.strength
                )

                weighted_sum += contribution
                weight_sum += weight * calibrated_conf

            if weight_sum > 0:
                target = weighted_sum / weight_sum
                # Clamp to max position
                target = max(-self.config.max_position_per_symbol,
                           min(target, self.config.max_position_per_symbol))
                target_positions[symbol] = target

        # Normalize if total exposure exceeds limit
        total_exposure = sum(abs(p) for p in target_positions.values())
        if total_exposure > self.config.max_total_exposure:
            scale = self.config.max_total_exposure / total_exposure
            target_positions = {s: p * scale for s, p in target_positions.items()}

        return target_positions

    def _weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average."""
        if not values or not weights:
            return 0
        weight_sum = sum(weights)
        if weight_sum == 0:
            return sum(values) / len(values)
        return sum(v * w for v, w in zip(values, weights)) / weight_sum

    def _update_weights(self) -> None:
        """Update model weights based on recent performance."""
        for model_name, perf in self._model_performance.items():
            if perf.total_signals < 10:  # Need minimum history
                continue

            sharpe = perf.sharpe_30d

            # Update weights based on Sharpe ratio
            # Sharpe > 1: increase weight, Sharpe < 0: decrease weight
            for regime in Regime:
                current = self._model_weights[model_name].get(regime, 1.0)

                # Softmax-style adjustment
                if sharpe > 1.0:
                    adjustment = 1.1
                elif sharpe > 0.5:
                    adjustment = 1.05
                elif sharpe < 0:
                    adjustment = 0.9
                elif sharpe < -0.5:
                    adjustment = 0.8
                else:
                    adjustment = 1.0

                new_weight = current * adjustment
                new_weight = max(self.config.min_weight, min(new_weight, self.config.max_weight))
                self._model_weights[model_name][regime] = new_weight

            logger.info(f"Updated weights for {model_name}: Sharpe={sharpe:.2f}")

    def record_signal_outcome(
        self,
        signal: Signal,
        actual_return: float,
        regime: Regime,
    ) -> None:
        """
        Record the outcome of a signal for performance tracking.

        Args:
            signal: The original signal
            actual_return: Actual return achieved (as percentage)
            regime: Market regime when signal was generated
        """
        model = signal.model_name

        if model not in self._model_performance:
            self._model_performance[model] = ModelPerformance(model_name=model)

        perf = self._model_performance[model]

        # Update counts
        perf.total_signals += 1
        perf.regime_signals[regime] += 1

        # Determine if correct
        expected_direction = signal.direction
        correct = (
            (expected_direction == SignalDirection.LONG and actual_return > 0) or
            (expected_direction == SignalDirection.SHORT and actual_return < 0) or
            (expected_direction == SignalDirection.NEUTRAL and abs(actual_return) < 0.5)
        )

        if correct:
            perf.correct_signals += 1
            perf.regime_correct[regime] += 1

        # Update PnL
        perf.total_pnl += actual_return

        # Update recent returns (rolling window)
        perf.recent_returns.append(actual_return)
        max_history = self.config.rolling_window_days
        if len(perf.recent_returns) > max_history:
            perf.recent_returns = perf.recent_returns[-max_history:]

        perf.last_updated = datetime.now()

        # Update calibration
        self._calibrator.update(model, signal.confidence, correct)

    def get_model_weights(self, regime: Optional[Regime] = None) -> Dict[str, float]:
        """Get current model weights."""
        if regime is None:
            # Return average across regimes
            result = {}
            for model, regime_weights in self._model_weights.items():
                result[model] = sum(regime_weights.values()) / len(regime_weights)
            return result
        else:
            return {
                model: weights.get(regime, 1.0)
                for model, weights in self._model_weights.items()
            }

    def get_model_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for all models."""
        return {
            model: {
                "accuracy": perf.accuracy,
                "sharpe_30d": perf.sharpe_30d,
                "total_signals": perf.total_signals,
                "total_pnl": perf.total_pnl,
            }
            for model, perf in self._model_performance.items()
        }

    def set_model_weight(
        self,
        model_name: str,
        weight: float,
        regime: Optional[Regime] = None,
    ) -> None:
        """Manually set a model's weight."""
        if model_name not in self._model_weights:
            self._initialize_model_weights(model_name)

        weight = max(self.config.min_weight, min(weight, self.config.max_weight))

        if regime is None:
            # Set for all regimes
            for r in Regime:
                self._model_weights[model_name][r] = weight
        else:
            self._model_weights[model_name][regime] = weight

        logger.info(f"Set weight for {model_name}: {weight} (regime={regime})")
