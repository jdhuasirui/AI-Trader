"""
AI-Trader Core Module

This module contains the core infrastructure for the AI trading system:
- Data structures for market state, signals, and orders
- Risk management engine with circuit breakers and ATR-based position sizing
- Order management with state machine
- Execution algorithms (TWAP, limit-first) with enhanced slippage model
- Regime detection using HMM
- Signal aggregation for multi-model fusion
- Confidence calibration for LLM outputs (Platt scaling, Isotonic)
- Concept drift detection (ADWIN, Page-Hinkley)
- Walk-forward validation with Deflated Sharpe Ratio
- Observability layer for metrics and monitoring
"""

from .data_structures import (
    MarketState,
    Signal,
    TargetPortfolio,
    OrderIntent,
    ExecutionReport,
    Position,
    Portfolio,
    RiskAction,
    OrderStatus,
    OrderType,
    SignalDirection,
    Regime,
    TradeRecord,
    RiskViolation,
)

from .risk_engine import RiskEngine, RiskConfig, RiskState
from .order_manager import OrderManager, OrderStateMachine, OrderState, RetryConfig
from .observability import MetricsCollector, AuditLogger, AlertManager, PerformanceMetrics
from .execution_algos import (
    ExecutionAlgorithm,
    MarketOrderExecution,
    TWAPExecution,
    LimitFirstExecution,
    SlippageSimulator,
    ExecutionConfig,
    get_execution_algorithm,
)
from .regime_detector import RegimeDetector, RegimeConfig, RegimeFeatures
from .signal_aggregator import SignalAggregator, AggregatorConfig, ConfidenceCalibrator
from .llm_validator import LLMValidator, ValidationResult, create_structured_prompt

# New modules for enhanced trading
from .drift_detector import (
    DriftDetector,
    DriftConfig,
    DriftState,
    DriftType,
    DriftAction,
    ADWIN,
    PageHinkley,
)
from .validation import (
    ValidationFramework,
    WalkForwardOptimizer,
    CPCV,
    DeflatedSharpeRatio,
    ValidationConfig,
    ValidationResult as BacktestValidationResult,
    WalkForwardResult,
)
from .confidence_calibration import (
    ConfidenceCalibrator as AdvancedCalibrator,
    PlattScaling,
    IsotonicCalibration,
    TemperatureScaling,
    ReliabilityCurve,
    CalibrationConfig,
    CalibrationResult,
)

# Integrated Trading Engine
from .trading_engine import (
    TradingEngine,
    TradingEngineConfig,
    EngineMode,
    TradeDecision,
    ModelState,
    create_trading_engine,
)

__all__ = [
    # Data Structures
    "MarketState",
    "Signal",
    "TargetPortfolio",
    "OrderIntent",
    "ExecutionReport",
    "Position",
    "Portfolio",
    "RiskAction",
    "OrderStatus",
    "OrderType",
    "SignalDirection",
    "Regime",
    "TradeRecord",
    "RiskViolation",
    # Risk Engine
    "RiskEngine",
    "RiskConfig",
    "RiskState",
    # Order Manager
    "OrderManager",
    "OrderStateMachine",
    "OrderState",
    "RetryConfig",
    # Observability
    "MetricsCollector",
    "AuditLogger",
    "AlertManager",
    "PerformanceMetrics",
    # Execution Algorithms
    "ExecutionAlgorithm",
    "MarketOrderExecution",
    "TWAPExecution",
    "LimitFirstExecution",
    "SlippageSimulator",
    "ExecutionConfig",
    "get_execution_algorithm",
    # Regime Detection
    "RegimeDetector",
    "RegimeConfig",
    "RegimeFeatures",
    # Signal Aggregation
    "SignalAggregator",
    "AggregatorConfig",
    "ConfidenceCalibrator",
    # LLM Validation
    "LLMValidator",
    "ValidationResult",
    "create_structured_prompt",
    # Drift Detection (NEW)
    "DriftDetector",
    "DriftConfig",
    "DriftState",
    "DriftType",
    "DriftAction",
    "ADWIN",
    "PageHinkley",
    # Validation Framework (NEW)
    "ValidationFramework",
    "WalkForwardOptimizer",
    "CPCV",
    "DeflatedSharpeRatio",
    "ValidationConfig",
    "BacktestValidationResult",
    "WalkForwardResult",
    # Confidence Calibration (NEW)
    "AdvancedCalibrator",
    "PlattScaling",
    "IsotonicCalibration",
    "TemperatureScaling",
    "ReliabilityCurve",
    "CalibrationConfig",
    "CalibrationResult",
    # Trading Engine (NEW)
    "TradingEngine",
    "TradingEngineConfig",
    "EngineMode",
    "TradeDecision",
    "ModelState",
    "create_trading_engine",
]
