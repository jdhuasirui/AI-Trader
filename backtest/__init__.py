"""
Backtesting Framework for AI-Trader

This module provides rigorous backtesting validation:
- Walk-forward optimization (WFO)
- Deflated Sharpe Ratio (DSR) for multiple testing correction
- Combinatorial Purged Cross-Validation (CPCV) for PBO calculation
"""

from .walk_forward import WalkForwardOptimizer, WFOConfig, WFOResult
from .metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from .cpcv import CPCVValidator, CPCVConfig, CPCVResult

__all__ = [
    # Walk Forward
    "WalkForwardOptimizer",
    "WFOConfig",
    "WFOResult",
    # Metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "calculate_max_drawdown",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    # CPCV
    "CPCVValidator",
    "CPCVConfig",
    "CPCVResult",
]
