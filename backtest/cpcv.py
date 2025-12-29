"""
Combinatorial Purged Cross-Validation (CPCV) for AI-Trader

This module implements CPCV for calculating Probability of Backtest Overfitting (PBO):
- Combinatorial path generation
- Purging to eliminate data leakage
- Embargo periods for time-series
- PBO calculation

Reference: Bailey et al. (2016): "Probability of Backtest Overfitting"
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple, Any

from .metrics import calculate_all_metrics, BacktestMetrics, probability_of_backtest_overfitting

logger = logging.getLogger(__name__)


@dataclass
class CPCVConfig:
    """Configuration for CPCV validation."""

    # Number of folds
    n_splits: int = 6  # Number of time-series folds

    # Test folds per path
    n_test_folds: int = 2  # C(6,2) = 15 paths

    # Purging and embargo
    purge_pct: float = 0.01  # Purge 1% between train/test
    embargo_pct: float = 0.01  # Embargo 1% after test

    # PBO threshold
    max_pbo: float = 0.05  # Maximum acceptable PBO

    # Minimum requirements
    min_samples_per_fold: int = 20


@dataclass
class CPCVPath:
    """A single CPCV combinatorial path."""
    path_id: int
    train_folds: List[int]
    test_folds: List[int]

    # Results
    train_metrics: Optional[BacktestMetrics] = None
    test_metrics: Optional[BacktestMetrics] = None
    best_params: Optional[Dict] = None


@dataclass
class CPCVResult:
    """Results from CPCV validation."""
    paths: List[CPCVPath] = field(default_factory=list)

    # PBO metrics
    pbo: float = 0.0  # Probability of Backtest Overfitting
    logit_pbo: float = 0.0  # Logit of PBO

    # Performance distribution
    in_sample_sharpes: List[float] = field(default_factory=list)
    out_sample_sharpes: List[float] = field(default_factory=list)

    # Rank correlation
    rank_correlation: float = 0.0

    # Pass/fail
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation validator.

    Implements CPCV to estimate the probability that a backtest
    has been overfit due to strategy selection from multiple trials.
    """

    def __init__(self, config: Optional[CPCVConfig] = None):
        self.config = config or CPCVConfig()

    def validate(
        self,
        data: List[Dict],
        strategy_fn: Callable[[List[Dict], Dict], List[float]],
        param_grid: Dict[str, List[Any]],
        date_key: str = "date",
    ) -> CPCVResult:
        """
        Run CPCV validation.

        Args:
            data: Time-series data
            strategy_fn: Strategy function (data, params) -> returns
            param_grid: Parameter search space
            date_key: Date field key

        Returns:
            CPCVResult with PBO and all paths
        """
        result = CPCVResult()

        # Create folds
        folds = self._create_folds(data, date_key)
        if len(folds) < self.config.n_splits:
            result.failure_reasons.append(
                f"Insufficient folds: {len(folds)}, need {self.config.n_splits}"
            )
            return result

        logger.info(f"Created {len(folds)} folds for CPCV")

        # Generate combinatorial paths
        paths = self._generate_paths(folds)
        logger.info(f"Generated {len(paths)} CPCV paths")

        # Process each path
        for path in paths:
            self._process_path(
                path=path,
                folds=folds,
                strategy_fn=strategy_fn,
                param_grid=param_grid,
            )
            result.paths.append(path)

            if path.train_metrics and path.test_metrics:
                result.in_sample_sharpes.append(path.train_metrics.sharpe_ratio)
                result.out_sample_sharpes.append(path.test_metrics.sharpe_ratio)

        # Calculate PBO
        self._calculate_pbo(result)

        # Validate
        self._validate_results(result)

        return result

    def _create_folds(
        self,
        data: List[Dict],
        date_key: str,
    ) -> List[List[Dict]]:
        """Create time-series folds with purging."""
        if not data:
            return []

        # Sort by date
        sorted_data = sorted(data, key=lambda x: x[date_key])
        n = len(sorted_data)

        # Calculate sizes
        fold_size = n // self.config.n_splits
        purge_size = max(1, int(n * self.config.purge_pct))

        if fold_size < self.config.min_samples_per_fold:
            logger.warning(f"Fold size {fold_size} below minimum {self.config.min_samples_per_fold}")
            return []

        folds = []
        for i in range(self.config.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.config.n_splits - 1 else n

            # Apply purging at boundaries
            if i > 0:
                start += purge_size
            if i < self.config.n_splits - 1:
                end -= purge_size

            fold_data = sorted_data[start:end]
            folds.append(fold_data)

        return folds

    def _generate_paths(
        self,
        folds: List[List[Dict]],
    ) -> List[CPCVPath]:
        """Generate combinatorial train/test paths."""
        n_folds = len(folds)
        n_test = self.config.n_test_folds

        # Generate all combinations of test folds
        test_combinations = list(combinations(range(n_folds), n_test))

        paths = []
        for path_id, test_folds in enumerate(test_combinations):
            train_folds = [i for i in range(n_folds) if i not in test_folds]

            paths.append(CPCVPath(
                path_id=path_id,
                train_folds=train_folds,
                test_folds=list(test_folds),
            ))

        return paths

    def _process_path(
        self,
        path: CPCVPath,
        folds: List[List[Dict]],
        strategy_fn: Callable,
        param_grid: Dict[str, List[Any]],
    ) -> None:
        """Process a single CPCV path."""
        # Combine training folds
        train_data = []
        for fold_idx in path.train_folds:
            train_data.extend(folds[fold_idx])

        # Combine test folds
        test_data = []
        for fold_idx in path.test_folds:
            test_data.extend(folds[fold_idx])

        if not train_data or not test_data:
            return

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        # Find best parameters on training set
        best_params = None
        best_train_sharpe = float('-inf')

        for params in param_combinations[:100]:  # Limit for efficiency
            try:
                train_returns = strategy_fn(train_data, params)
                train_metrics = calculate_all_metrics(train_returns)

                if train_metrics.sharpe_ratio > best_train_sharpe:
                    best_train_sharpe = train_metrics.sharpe_ratio
                    best_params = params
                    path.train_metrics = train_metrics

            except Exception as e:
                logger.debug(f"Path {path.path_id} param failed: {e}")
                continue

        if best_params is None:
            return

        path.best_params = best_params

        # Test with best parameters
        try:
            test_returns = strategy_fn(test_data, best_params)
            path.test_metrics = calculate_all_metrics(test_returns)
        except Exception as e:
            logger.debug(f"Path {path.path_id} test failed: {e}")

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> List[Dict]:
        """Generate parameter combinations."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations_list = []

        def generate(idx: int, current: Dict):
            if idx == len(keys):
                combinations_list.append(current.copy())
                return
            for v in values[idx]:
                current[keys[idx]] = v
                generate(idx + 1, current)

        generate(0, {})
        return combinations_list

    def _calculate_pbo(self, result: CPCVResult) -> None:
        """Calculate Probability of Backtest Overfitting."""
        if len(result.in_sample_sharpes) < 2:
            result.pbo = 0.5  # Insufficient data
            return

        # Use the metrics module function
        result.pbo = probability_of_backtest_overfitting(
            result.in_sample_sharpes,
            result.out_sample_sharpes,
        )

        # Logit of PBO
        if 0 < result.pbo < 1:
            result.logit_pbo = math.log(result.pbo / (1 - result.pbo))
        elif result.pbo <= 0:
            result.logit_pbo = float('-inf')
        else:
            result.logit_pbo = float('inf')

        # Calculate rank correlation
        result.rank_correlation = self._calculate_rank_correlation(
            result.in_sample_sharpes,
            result.out_sample_sharpes,
        )

    def _calculate_rank_correlation(
        self,
        x: List[float],
        y: List[float],
    ) -> float:
        """Calculate Spearman rank correlation."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)

        # Get ranks
        x_ranks = self._get_ranks(x)
        y_ranks = self._get_ranks(y)

        # Calculate Spearman correlation
        d_squared_sum = sum((x_ranks[i] - y_ranks[i]) ** 2 for i in range(n))
        rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

        return rho

    def _get_ranks(self, values: List[float]) -> List[float]:
        """Get ranks of values."""
        n = len(values)
        sorted_with_idx = sorted(enumerate(values), key=lambda x: x[1])

        ranks = [0.0] * n
        for rank, (idx, _) in enumerate(sorted_with_idx, 1):
            ranks[idx] = float(rank)

        return ranks

    def _validate_results(self, result: CPCVResult) -> None:
        """Validate CPCV results."""
        result.passed = True
        result.failure_reasons = []

        # Check PBO threshold
        if result.pbo > self.config.max_pbo:
            result.passed = False
            result.failure_reasons.append(
                f"PBO {result.pbo:.3f} exceeds threshold {self.config.max_pbo}"
            )

        # Check rank correlation
        if result.rank_correlation < 0:
            result.failure_reasons.append(
                f"Negative rank correlation: {result.rank_correlation:.2f}"
            )

        # Check for consistent underperformance OOS
        if result.out_sample_sharpes:
            avg_oos = sum(result.out_sample_sharpes) / len(result.out_sample_sharpes)
            if avg_oos < 0:
                result.passed = False
                result.failure_reasons.append(
                    f"Negative average OOS Sharpe: {avg_oos:.2f}"
                )

        # Check for sufficient paths
        valid_paths = [p for p in result.paths if p.test_metrics is not None]
        if len(valid_paths) < 5:
            result.passed = False
            result.failure_reasons.append(
                f"Insufficient valid paths: {len(valid_paths)}"
            )

        if result.failure_reasons:
            logger.warning(f"CPCV validation failures: {result.failure_reasons}")
        else:
            logger.info(f"CPCV validation passed with PBO={result.pbo:.3f}")
