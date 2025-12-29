"""
TradingEngine - Integrated Trading Infrastructure

This module provides a unified trading engine that integrates all core components:
- DriftDetector: Monitor model performance and detect concept drift
- ConfidenceCalibrator: Calibrate LLM confidence outputs
- RiskEngine: ATR-based position sizing and risk management
- SlippageSimulator: Execution cost estimation
- SignalAggregator: Multi-model signal fusion

Usage:
    engine = TradingEngine(config)
    engine.initialize()

    # Process a trading signal
    result = engine.process_signal(model_id, raw_signal, market_data)

    # Check if model needs attention
    if engine.check_model_health(model_id):
        # Model is healthy
        pass
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

# Import core modules
from .drift_detector import DriftDetector, DriftConfig, DriftState, DriftAction
from .confidence_calibration import (
    ConfidenceCalibrator,
    PlattScaling,
    IsotonicCalibration,
    CalibrationConfig,
)
from .risk_engine import RiskEngine, RiskConfig, RiskState
from .execution_algos import SlippageSimulator, ExecutionConfig
from .signal_aggregator import SignalAggregator, AggregatorConfig
from .data_structures import Signal, SignalDirection, Regime, MarketState


logger = logging.getLogger(__name__)


class EngineMode(Enum):
    """Trading engine operation mode"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class TradingEngineConfig:
    """Configuration for the trading engine (immutable)"""
    # General settings
    mode: EngineMode = EngineMode.PAPER
    initial_capital: float = 100000.0

    # Drift detection settings
    drift_enabled: bool = True
    drift_config: Optional[DriftConfig] = None

    # Confidence calibration settings
    calibration_enabled: bool = True
    calibration_config: Optional[CalibrationConfig] = None

    # Risk management settings
    risk_enabled: bool = True
    risk_config: Optional[RiskConfig] = None

    # Execution settings
    slippage_enabled: bool = True
    execution_config: Optional[ExecutionConfig] = None

    # Signal aggregation settings
    aggregation_enabled: bool = True
    aggregator_config: Optional[AggregatorConfig] = None

    # Logging settings
    log_dir: str = "./data/engine_logs"
    log_level: str = "INFO"


@dataclass
class ModelState:
    """State for a single model/agent"""
    model_id: str
    drift_detector: DriftDetector
    calibrator: ConfidenceCalibrator
    predictions: List[Tuple[float, float]] = field(default_factory=list)  # (prediction, actual)
    signals_generated: int = 0
    last_signal_time: Optional[datetime] = None
    cumulative_pnl: float = 0.0
    is_healthy: bool = True
    weight_adjustment: float = 1.0


@dataclass
class TradeDecision:
    """Result of trade decision processing"""
    symbol: str
    direction: SignalDirection
    raw_confidence: float
    calibrated_confidence: float
    position_size: float
    estimated_slippage: float
    risk_approved: bool
    drift_warning: bool
    model_weight: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


class TradingEngine:
    """
    Integrated trading engine that orchestrates all core components.

    This engine provides:
    1. Per-model drift detection and health monitoring
    2. Confidence calibration for LLM outputs
    3. ATR-based position sizing with drawdown protection
    4. Slippage estimation for execution planning
    5. Multi-model signal aggregation
    """

    def __init__(self, config: Optional[TradingEngineConfig] = None):
        """
        Initialize the trading engine.

        Args:
            config: Engine configuration, uses defaults if None
        """
        self.config = config or TradingEngineConfig()

        # Initialize component configs with defaults if not provided
        if self.config.drift_config is None:
            self.config.drift_config = DriftConfig()
        if self.config.calibration_config is None:
            self.config.calibration_config = CalibrationConfig()
        if self.config.risk_config is None:
            self.config.risk_config = RiskConfig()
        if self.config.execution_config is None:
            self.config.execution_config = ExecutionConfig()
        if self.config.aggregator_config is None:
            self.config.aggregator_config = AggregatorConfig()

        # Core components
        self.risk_engine: Optional[RiskEngine] = None
        self.slippage_simulator: Optional[SlippageSimulator] = None
        self.signal_aggregator: Optional[SignalAggregator] = None

        # Per-model state
        self.model_states: Dict[str, ModelState] = {}

        # Current market state
        self.current_regime: Regime = Regime.RANGING
        self.current_volatility: float = 0.02  # 2% default

        # Portfolio state
        self.portfolio_value: float = self.config.initial_capital
        self.cash: float = self.config.initial_capital
        self.positions: Dict[str, float] = {}

        # Metrics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0

        # Logging
        self._setup_logging()

        self._initialized = False

    def _setup_logging(self) -> None:
        """Set up logging for the engine"""
        os.makedirs(self.config.log_dir, exist_ok=True)

        log_file = os.path.join(
            self.config.log_dir,
            f"engine_{datetime.now().strftime('%Y%m%d')}.log"
        )

        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def initialize(self) -> None:
        """Initialize all engine components"""
        logger.info("Initializing TradingEngine...")

        # Initialize risk engine
        if self.config.risk_enabled:
            self.risk_engine = RiskEngine(self.config.risk_config)
            logger.info("RiskEngine initialized")

        # Initialize slippage simulator
        if self.config.slippage_enabled:
            self.slippage_simulator = SlippageSimulator(self.config.execution_config)
            logger.info("SlippageSimulator initialized")

        # Initialize signal aggregator
        if self.config.aggregation_enabled:
            self.signal_aggregator = SignalAggregator(self.config.aggregator_config)
            logger.info("SignalAggregator initialized")

        self._initialized = True
        logger.info("TradingEngine initialization complete")

    def register_model(self, model_id: str) -> None:
        """
        Register a new model/agent with the engine.

        Args:
            model_id: Unique identifier for the model
        """
        if model_id in self.model_states:
            logger.warning(f"Model {model_id} already registered")
            return

        # Create drift detector for this model
        drift_detector = DriftDetector(self.config.drift_config)

        # Create calibrator for this model
        calibrator = ConfidenceCalibrator(self.config.calibration_config)

        # Create model state
        self.model_states[model_id] = ModelState(
            model_id=model_id,
            drift_detector=drift_detector,
            calibrator=calibrator,
        )

        logger.info(f"Registered model: {model_id}")

    def update_market_state(
        self,
        regime: Regime,
        volatility: float,
        atr: Optional[float] = None
    ) -> None:
        """
        Update current market state.

        Args:
            regime: Current market regime
            volatility: Current volatility estimate
            atr: Average True Range (optional)
        """
        self.current_regime = regime
        self.current_volatility = volatility

        if self.signal_aggregator:
            self.signal_aggregator.set_current_regime(regime)

        if self.risk_engine and atr:
            # ATR can be used for position sizing
            pass

        logger.debug(f"Market state updated: regime={regime}, vol={volatility:.4f}")

    def update_portfolio(
        self,
        portfolio_value: float,
        cash: float,
        positions: Dict[str, float]
    ) -> None:
        """
        Update portfolio state.

        Args:
            portfolio_value: Total portfolio value
            cash: Available cash
            positions: Current positions {symbol: quantity}
        """
        self.portfolio_value = portfolio_value
        self.cash = cash
        self.positions = positions.copy()

    def process_signal(
        self,
        model_id: str,
        symbol: str,
        direction: SignalDirection,
        raw_confidence: float,
        price: float,
        atr: float,
        volume: float,
        reasoning: str = ""
    ) -> TradeDecision:
        """
        Process a trading signal from a model.

        This method:
        1. Calibrates the confidence score
        2. Checks for drift
        3. Calculates position size using ATR
        4. Estimates slippage
        5. Validates with risk engine

        Args:
            model_id: Model that generated the signal
            symbol: Trading symbol
            direction: BUY/SELL/HOLD
            raw_confidence: Raw confidence from model (0-1)
            price: Current price
            atr: Average True Range
            volume: Current volume
            reasoning: Model's reasoning (optional)

        Returns:
            TradeDecision with all computed values
        """
        if not self._initialized:
            raise RuntimeError("TradingEngine not initialized. Call initialize() first.")

        # Ensure model is registered
        if model_id not in self.model_states:
            self.register_model(model_id)

        model_state = self.model_states[model_id]

        # 1. Calibrate confidence
        calibrated_confidence = raw_confidence
        if self.config.calibration_enabled and model_state.calibrator.is_fitted:
            calibrated_scores = model_state.calibrator.calibrate([raw_confidence])
            calibrated_confidence = calibrated_scores[0] if calibrated_scores else raw_confidence

        # 2. Check for drift
        drift_warning = False
        model_weight = 1.0
        if self.config.drift_enabled:
            drift_state = model_state.drift_detector.update(
                value=calibrated_confidence,
                prediction=calibrated_confidence,
                actual=None  # Will be updated later with actual outcome
            )
            drift_warning = drift_state.drift_detected
            model_weight = model_state.drift_detector.get_model_weight_adjustment()
            model_state.weight_adjustment = model_weight
            model_state.is_healthy = not drift_state.should_halt

        # 3. Calculate position size
        position_size = 0.0
        if self.config.risk_enabled and self.risk_engine:
            # Use fractional Kelly with ATR
            position_size = self.risk_engine.calculate_kelly_position_size(
                win_rate=calibrated_confidence,
                win_loss_ratio=2.0,  # Assume 2:1 reward-risk
                portfolio_value=self.portfolio_value
            )

            # Apply ATR-based sizing adjustment
            # Position = (Account * 1%) / (ATR * 2)
            risk_per_trade = self.portfolio_value * 0.01
            atr_position = risk_per_trade / (atr * 2.0) if atr > 0 else 0
            position_size = min(position_size, atr_position * price)

            # Apply model weight adjustment
            position_size *= model_weight

        # 4. Estimate slippage
        estimated_slippage = 0.0
        if self.config.slippage_enabled and self.slippage_simulator:
            order_value = position_size
            estimated_slippage = self.slippage_simulator._calculate_enhanced_slippage(
                base_slippage=self.config.execution_config.base_slippage_bps / 10000,
                order_value=order_value,
                average_volume=volume * price,
                volatility=self.current_volatility,
                is_crypto=(symbol.endswith("-USDT") or symbol.endswith("USDT"))
            )

        # 5. Risk validation
        risk_approved = True
        if self.config.risk_enabled and self.risk_engine:
            # Check against risk limits
            current_position = self.positions.get(symbol, 0)
            if direction == SignalDirection.BUY:
                new_position = current_position + position_size
            else:
                new_position = current_position - position_size

            # Basic position limit check
            max_position = self.portfolio_value * 0.2  # 20% max per position
            if abs(new_position * price) > max_position:
                risk_approved = False
                position_size = 0.0

        # Update model state
        model_state.signals_generated += 1
        model_state.last_signal_time = datetime.now()

        # Create decision
        decision = TradeDecision(
            symbol=symbol,
            direction=direction,
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            position_size=position_size,
            estimated_slippage=estimated_slippage,
            risk_approved=risk_approved,
            drift_warning=drift_warning,
            model_weight=model_weight,
            reasoning=reasoning,
        )

        logger.info(
            f"Signal processed: {model_id} -> {symbol} {direction.name} "
            f"conf={calibrated_confidence:.2f} size={position_size:.2f} "
            f"slippage={estimated_slippage:.4f} approved={risk_approved}"
        )

        return decision

    def aggregate_signals(
        self,
        signals: List[Signal]
    ) -> Tuple[Signal, Dict[str, Any]]:
        """
        Aggregate signals from multiple models.

        Args:
            signals: List of signals from different models

        Returns:
            Tuple of (aggregated_signal, metadata)
        """
        if not self.config.aggregation_enabled or not self.signal_aggregator:
            # Return first signal if aggregation disabled
            if signals:
                return signals[0], {"method": "first", "models": 1}
            return None, {"error": "no_signals"}

        # Apply model weight adjustments
        weighted_signals = []
        for signal in signals:
            if signal.model_id in self.model_states:
                state = self.model_states[signal.model_id]
                # Adjust confidence by model weight
                adjusted_signal = Signal(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    confidence=signal.confidence * state.weight_adjustment,
                    model_id=signal.model_id,
                    timestamp=signal.timestamp,
                    metadata={**signal.metadata, "weight_adjusted": state.weight_adjustment}
                )
                weighted_signals.append(adjusted_signal)
            else:
                weighted_signals.append(signal)

        # Aggregate
        for signal in weighted_signals:
            self.signal_aggregator.add_signal(signal)

        aggregated = self.signal_aggregator.get_aggregated_signal(
            weighted_signals[0].symbol if weighted_signals else ""
        )

        metadata = {
            "method": "weighted_consensus",
            "models": len(signals),
            "regime": self.current_regime.value if self.current_regime else "unknown",
        }

        return aggregated, metadata

    def record_outcome(
        self,
        model_id: str,
        prediction: float,
        actual: float,
        pnl: float
    ) -> None:
        """
        Record the outcome of a prediction for drift detection and calibration.

        Args:
            model_id: Model that made the prediction
            prediction: Predicted value/confidence
            actual: Actual outcome
            pnl: Profit/loss from the trade
        """
        if model_id not in self.model_states:
            logger.warning(f"Unknown model: {model_id}")
            return

        state = self.model_states[model_id]

        # Update drift detector
        if self.config.drift_enabled:
            state.drift_detector.update(
                value=pnl,
                prediction=prediction,
                actual=actual
            )

        # Store for calibration updates
        state.predictions.append((prediction, actual))
        state.cumulative_pnl += pnl

        # Update metrics
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        self.total_trades += 1

        logger.debug(f"Outcome recorded: {model_id} pred={prediction:.3f} actual={actual:.3f} pnl={pnl:.2f}")

    def check_model_health(self, model_id: str) -> bool:
        """
        Check if a model is healthy and should continue trading.

        Args:
            model_id: Model to check

        Returns:
            True if model is healthy, False otherwise
        """
        if model_id not in self.model_states:
            return False

        state = self.model_states[model_id]

        # Check drift detector
        if self.config.drift_enabled:
            if state.drift_detector.should_retrain():
                logger.warning(f"Model {model_id} needs retraining")
                return False

        return state.is_healthy

    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """
        Get statistics for a model.

        Args:
            model_id: Model to get stats for

        Returns:
            Dictionary of model statistics
        """
        if model_id not in self.model_states:
            return {"error": f"Unknown model: {model_id}"}

        state = self.model_states[model_id]

        return {
            "model_id": model_id,
            "signals_generated": state.signals_generated,
            "last_signal_time": state.last_signal_time.isoformat() if state.last_signal_time else None,
            "cumulative_pnl": state.cumulative_pnl,
            "is_healthy": state.is_healthy,
            "weight_adjustment": state.weight_adjustment,
            "drift_state": state.drift_detector.get_state().__dict__ if self.config.drift_enabled else None,
        }

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get overall engine statistics.

        Returns:
            Dictionary of engine statistics
        """
        win_rate = self.winning_trades / max(self.total_trades, 1)

        return {
            "mode": self.config.mode.value,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "current_regime": self.current_regime.value if self.current_regime else "unknown",
            "current_volatility": self.current_volatility,
            "registered_models": list(self.model_states.keys()),
            "model_count": len(self.model_states),
        }

    def save_state(self, filepath: str) -> None:
        """
        Save engine state to file.

        Args:
            filepath: Path to save state
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "engine_stats": self.get_engine_stats(),
            "model_stats": {
                model_id: self.get_model_stats(model_id)
                for model_id in self.model_states
            },
            "positions": self.positions,
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Engine state saved to {filepath}")

    def __str__(self) -> str:
        return (
            f"TradingEngine(mode={self.config.mode.value}, "
            f"models={len(self.model_states)}, "
            f"portfolio={self.portfolio_value:.2f})"
        )


# Convenience function to create a pre-configured engine
def create_trading_engine(
    mode: str = "paper",
    initial_capital: float = 100000.0,
    enable_all: bool = True
) -> TradingEngine:
    """
    Create a trading engine with sensible defaults.

    Args:
        mode: "backtest", "paper", or "live"
        initial_capital: Starting capital
        enable_all: Enable all components

    Returns:
        Configured TradingEngine instance
    """
    mode_enum = EngineMode[mode.upper()]

    config = TradingEngineConfig(
        mode=mode_enum,
        initial_capital=initial_capital,
        drift_enabled=enable_all,
        calibration_enabled=enable_all,
        risk_enabled=enable_all,
        slippage_enabled=enable_all,
        aggregation_enabled=enable_all,
    )

    engine = TradingEngine(config)
    engine.initialize()

    return engine
