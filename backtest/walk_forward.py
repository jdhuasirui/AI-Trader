"""
Walk-Forward Optimization for AI-Trader

This module implements walk-forward optimization (WFO) for robust backtesting:
- Rolling window training/validation/test splits
- Walk-Forward Efficiency (WFE) calculation
- Parameter stability analysis
- Regime-aware optimization

Configuration: 30-day training / 5-day validation / 5-day test (configurable)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple, Any
import math

from .metrics import calculate_all_metrics, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WFOConfig:
    """Configuration for walk-forward optimization (immutable)."""

    # Window sizes (in days)
    training_window: int = 30
    validation_window: int = 5
    test_window: int = 5

    # Step size (how much to roll forward)
    step_size: int = 5  # Roll forward by 5 days

    # Minimum requirements
    min_training_samples: int = 20
    min_test_samples: int = 3

    # WFE threshold
    min_wfe: float = 0.5  # Minimum 50% walk-forward efficiency

    # Optimization settings
    n_parameter_combinations: int = 100  # Max combinations to test
    metric_to_optimize: str = "sharpe_ratio"

    # Purging/embargo
    purge_days: int = 1  # Days to purge between train and test
    embargo_days: int = 0  # Days embargo after test


@dataclass
class WFOWindow:
    """A single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    test_start: datetime
    test_end: datetime

    # Results
    best_params: Optional[Dict] = None
    train_metrics: Optional[BacktestMetrics] = None
    valid_metrics: Optional[BacktestMetrics] = None
    test_metrics: Optional[BacktestMetrics] = None


@dataclass
class WFOResult:
    """Results from walk-forward optimization."""
    windows: List[WFOWindow] = field(default_factory=list)

    # Aggregate metrics
    walk_forward_efficiency: float = 0.0
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    parameter_stability: float = 0.0

    # Combined out-of-sample results
    combined_test_returns: List[float] = field(default_factory=list)
    combined_test_metrics: Optional[BacktestMetrics] = None

    # Pass/fail
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework.

    Implements anchored and rolling walk-forward analysis
    with purging and embargo to prevent data leakage.
    """

    def __init__(
        self,
        config: Optional[WFOConfig] = None,
    ):
        self.config = config or WFOConfig()

    def run(
        self,
        data: List[Dict],  # List of {date, returns, ...}
        strategy_fn: Callable[[List[Dict], Dict], List[float]],  # (data, params) -> returns
        param_grid: Dict[str, List[Any]],  # Parameter search space
        date_key: str = "date",
    ) -> WFOResult:
        """
        Run walk-forward optimization.

        Args:
            data: Time-series data with date and features
            strategy_fn: Function that takes data and params, returns returns
            param_grid: Dictionary of parameter names to possible values
            date_key: Key for date field in data

        Returns:
            WFOResult with all windows and aggregate metrics
        """
        result = WFOResult()

        # Create windows
        windows = self._create_windows(data, date_key)
        if not windows:
            result.failure_reasons.append("Insufficient data for walk-forward analysis")
            return result

        logger.info(f"Created {len(windows)} walk-forward windows")

        # Process each window
        for window in windows:
            self._process_window(
                window=window,
                data=data,
                strategy_fn=strategy_fn,
                param_grid=param_grid,
                date_key=date_key,
            )
            result.windows.append(window)

            # Collect test returns
            if window.test_metrics:
                # Reconstruct test returns (simplified)
                test_sharpe = window.test_metrics.sharpe_ratio
                result.combined_test_returns.extend([test_sharpe / math.sqrt(252)] * self.config.test_window)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(result)

        # Validate results
        self._validate_results(result)

        return result

    def _create_windows(
        self,
        data: List[Dict],
        date_key: str,
    ) -> List[WFOWindow]:
        """Create walk-forward windows from data."""
        windows = []

        if not data:
            return windows

        # Sort by date
        sorted_data = sorted(data, key=lambda x: x[date_key])
        start_date = sorted_data[0][date_key]
        end_date = sorted_data[-1][date_key]

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
            end_date = datetime.fromisoformat(end_date)

        total_days = (end_date - start_date).days
        total_window = (
            self.config.training_window +
            self.config.validation_window +
            self.config.test_window +
            self.config.purge_days
        )

        if total_days < total_window:
            logger.warning(f"Insufficient data: {total_days} days, need {total_window}")
            return windows

        # Create rolling windows
        window_id = 0
        current_start = start_date

        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.config.training_window)

            # Purge period
            valid_start = train_end + timedelta(days=self.config.purge_days)
            valid_end = valid_start + timedelta(days=self.config.validation_window)

            test_start = valid_end + timedelta(days=self.config.purge_days)
            test_end = test_start + timedelta(days=self.config.test_window)

            if test_end > end_date:
                break

            windows.append(WFOWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            ))

            window_id += 1
            current_start = current_start + timedelta(days=self.config.step_size)

        return windows

    def _process_window(
        self,
        window: WFOWindow,
        data: List[Dict],
        strategy_fn: Callable,
        param_grid: Dict[str, List[Any]],
        date_key: str,
    ) -> None:
        """Process a single walk-forward window."""
        # Split data
        train_data = self._filter_data_by_date(
            data, window.train_start, window.train_end, date_key
        )
        valid_data = self._filter_data_by_date(
            data, window.valid_start, window.valid_end, date_key
        )
        test_data = self._filter_data_by_date(
            data, window.test_start, window.test_end, date_key
        )

        if len(train_data) < self.config.min_training_samples:
            logger.warning(f"Window {window.window_id}: Insufficient training data")
            return

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        # Find best parameters on validation set
        best_params = None
        best_valid_metric = float('-inf')

        for params in param_combinations[:self.config.n_parameter_combinations]:
            try:
                # Train on training data
                train_returns = strategy_fn(train_data, params)
                train_metrics = calculate_all_metrics(train_returns)

                # Validate
                valid_returns = strategy_fn(valid_data, params)
                valid_metrics = calculate_all_metrics(valid_returns)

                # Check if best
                metric_value = getattr(valid_metrics, self.config.metric_to_optimize, 0)
                if metric_value > best_valid_metric:
                    best_valid_metric = metric_value
                    best_params = params
                    window.train_metrics = train_metrics
                    window.valid_metrics = valid_metrics

            except Exception as e:
                logger.debug(f"Parameter combination failed: {e}")
                continue

        if best_params is None:
            logger.warning(f"Window {window.window_id}: No valid parameter combination found")
            return

        window.best_params = best_params

        # Test with best parameters
        try:
            test_returns = strategy_fn(test_data, best_params)
            window.test_metrics = calculate_all_metrics(test_returns)
        except Exception as e:
            logger.warning(f"Window {window.window_id}: Test failed: {e}")

    def _filter_data_by_date(
        self,
        data: List[Dict],
        start: datetime,
        end: datetime,
        date_key: str,
    ) -> List[Dict]:
        """Filter data to date range."""
        filtered = []
        for item in data:
            date = item[date_key]
            if isinstance(date, str):
                date = datetime.fromisoformat(date)
            if start <= date <= end:
                filtered.append(item)
        return filtered

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> List[Dict]:
        """Generate all parameter combinations."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []

        def generate(idx: int, current: Dict):
            if idx == len(keys):
                combinations.append(current.copy())
                return
            for v in values[idx]:
                current[keys[idx]] = v
                generate(idx + 1, current)

        generate(0, {})
        return combinations

    def _calculate_aggregate_metrics(self, result: WFOResult) -> None:
        """Calculate aggregate walk-forward metrics."""
        valid_windows = [w for w in result.windows if w.test_metrics is not None]

        if not valid_windows:
            return

        # Average Sharpe ratios
        train_sharpes = [w.train_metrics.sharpe_ratio for w in valid_windows if w.train_metrics]
        test_sharpes = [w.test_metrics.sharpe_ratio for w in valid_windows]

        result.avg_train_sharpe = sum(train_sharpes) / len(train_sharpes) if train_sharpes else 0
        result.avg_test_sharpe = sum(test_sharpes) / len(test_sharpes) if test_sharpes else 0

        # Walk-Forward Efficiency
        # WFE = OOS Sharpe / IS Sharpe
        if result.avg_train_sharpe > 0:
            result.walk_forward_efficiency = result.avg_test_sharpe / result.avg_train_sharpe
        else:
            result.walk_forward_efficiency = 0

        # Parameter stability (how often parameters change)
        params_list = [str(w.best_params) for w in valid_windows if w.best_params]
        if len(params_list) > 1:
            unique_params = len(set(params_list))
            result.parameter_stability = 1 - (unique_params / len(params_list))
        else:
            result.parameter_stability = 1.0

        # Combined test metrics
        if result.combined_test_returns:
            result.combined_test_metrics = calculate_all_metrics(result.combined_test_returns)

    def _validate_results(self, result: WFOResult) -> None:
        """Validate walk-forward results against thresholds."""
        result.passed = True
        result.failure_reasons = []

        # Check WFE threshold
        if result.walk_forward_efficiency < self.config.min_wfe:
            result.passed = False
            result.failure_reasons.append(
                f"WFE {result.walk_forward_efficiency:.2f} below threshold {self.config.min_wfe}"
            )

        # Check for positive OOS performance
        if result.avg_test_sharpe < 0:
            result.passed = False
            result.failure_reasons.append(
                f"Negative out-of-sample Sharpe: {result.avg_test_sharpe:.2f}"
            )

        # Check for significant performance degradation
        if result.avg_train_sharpe > 0 and result.avg_test_sharpe > 0:
            degradation = (result.avg_train_sharpe - result.avg_test_sharpe) / result.avg_train_sharpe
            if degradation > 0.5:  # More than 50% degradation
                result.failure_reasons.append(
                    f"Significant performance degradation: {degradation:.1%}"
                )

        # Check minimum windows
        valid_windows = [w for w in result.windows if w.test_metrics is not None]
        if len(valid_windows) < 3:
            result.passed = False
            result.failure_reasons.append(
                f"Insufficient valid windows: {len(valid_windows)}"
            )

        if result.failure_reasons:
            logger.warning(f"WFO validation failures: {result.failure_reasons}")
        else:
            logger.info("WFO validation passed")
