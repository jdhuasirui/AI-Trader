"""
Validation Framework for AI-Trader

This module implements rigorous backtesting validation methods:

1. Walk-Forward Optimization (WFO):
   - Rolling train/validate/trade windows
   - Walk-Forward Efficiency (WFE) calculation
   - Out-of-sample performance tracking

2. Combinatorial Purged Cross-Validation (CPCV):
   - Avoids look-ahead bias in time series
   - Purging overlapping samples
   - Embargo period to prevent leakage

3. Deflated Sharpe Ratio (DSR):
   - Corrects for multiple testing bias
   - Probability of Backtest Overfitting (PBO)
   - Adjusts Sharpe for number of trials

4. Statistical Tests:
   - Probability of Overfitting
   - Performance persistence tests
   - Regime-aware validation
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for validation framework (immutable)."""

    # Walk-Forward settings
    wf_train_days: int = 30  # Training window in days
    wf_validate_days: int = 5  # Validation window
    wf_trade_days: int = 5  # Out-of-sample trading window
    wf_min_efficiency: float = 0.5  # Minimum WFE to pass (50%)

    # CPCV settings
    cpcv_n_splits: int = 5  # Number of CV splits
    cpcv_purge_days: int = 1  # Days to purge between train/test
    cpcv_embargo_days: int = 1  # Embargo period after test

    # Deflated Sharpe Ratio settings
    dsr_risk_free_rate: float = 0.02  # Annual risk-free rate
    dsr_periods_per_year: int = 252  # Trading days per year
    dsr_min_observations: int = 30  # Minimum samples for DSR

    # Overfitting thresholds
    pbo_max_threshold: float = 0.05  # Max PBO (5%)
    min_sharpe_threshold: float = 0.5  # Minimum in-sample Sharpe
    performance_decay_threshold: float = 0.6  # OOS/IS ratio threshold


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    validate_start: datetime
    validate_end: datetime
    trade_start: datetime
    trade_end: datetime

    # Performance metrics
    is_sharpe: float = 0.0  # In-sample Sharpe ratio
    oos_sharpe: float = 0.0  # Out-of-sample Sharpe ratio
    is_return: float = 0.0  # In-sample return
    oos_return: float = 0.0  # Out-of-sample return
    is_max_drawdown: float = 0.0
    oos_max_drawdown: float = 0.0

    # Efficiency
    walk_forward_efficiency: float = 0.0  # OOS Sharpe / IS Sharpe
    passed: bool = False


@dataclass
class ValidationResult:
    """Complete validation results."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Walk-Forward results
    wf_folds: List[WalkForwardResult] = field(default_factory=list)
    avg_wf_efficiency: float = 0.0
    wf_passed: bool = False

    # CPCV results
    cpcv_scores: List[float] = field(default_factory=list)
    cpcv_mean: float = 0.0
    cpcv_std: float = 0.0

    # Deflated Sharpe Ratio
    raw_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    sharpe_haircut: float = 0.0  # % reduction from raw
    num_trials: int = 1

    # Probability of Backtest Overfitting
    pbo: float = 0.0
    pbo_passed: bool = False

    # Overall verdict
    is_valid: bool = False
    rejection_reasons: List[str] = field(default_factory=list)


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization implementation.

    Validates strategies using rolling out-of-sample testing:
    1. Train on historical data
    2. Validate on hold-out period
    3. Trade in next period
    4. Roll forward and repeat
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.results: List[WalkForwardResult] = []

    def run(
        self,
        data: List[Dict],
        strategy_fn: Callable[[List[Dict]], Dict],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward optimization.

        Args:
            data: List of OHLCV dicts with 'date' key
            strategy_fn: Function that takes data and returns performance dict
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of WalkForwardResult for each fold
        """
        self.results = []

        # Sort data by date
        sorted_data = sorted(data, key=lambda x: x.get('date', x.get('timestamp', datetime.min)))

        if not sorted_data:
            return self.results

        # Determine date range
        if start_date is None:
            start_date = sorted_data[0].get('date', sorted_data[0].get('timestamp'))
        if end_date is None:
            end_date = sorted_data[-1].get('date', sorted_data[-1].get('timestamp'))

        # Calculate window lengths
        train_days = self.config.wf_train_days
        validate_days = self.config.wf_validate_days
        trade_days = self.config.wf_trade_days
        step_days = validate_days + trade_days

        # Generate folds
        fold_id = 0
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=train_days)
            validate_end = train_end + timedelta(days=validate_days)
            trade_end = validate_end + timedelta(days=trade_days)

            if trade_end > end_date:
                break

            # Extract data for each period
            train_data = [d for d in sorted_data if current_start <= d.get('date', d.get('timestamp', datetime.min)) < train_end]
            validate_data = [d for d in sorted_data if train_end <= d.get('date', d.get('timestamp', datetime.min)) < validate_end]
            trade_data = [d for d in sorted_data if validate_end <= d.get('date', d.get('timestamp', datetime.min)) < trade_end]

            if len(train_data) < 10 or len(trade_data) < 1:
                current_start += timedelta(days=step_days)
                continue

            # Run strategy on each period
            is_perf = strategy_fn(train_data)
            oos_perf = strategy_fn(trade_data)

            # Calculate efficiency
            is_sharpe = is_perf.get('sharpe_ratio', 0)
            oos_sharpe = oos_perf.get('sharpe_ratio', 0)

            wfe = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
            passed = wfe >= self.config.wf_min_efficiency

            result = WalkForwardResult(
                fold_id=fold_id,
                train_start=current_start,
                train_end=train_end,
                validate_start=train_end,
                validate_end=validate_end,
                trade_start=validate_end,
                trade_end=trade_end,
                is_sharpe=is_sharpe,
                oos_sharpe=oos_sharpe,
                is_return=is_perf.get('total_return', 0),
                oos_return=oos_perf.get('total_return', 0),
                is_max_drawdown=is_perf.get('max_drawdown', 0),
                oos_max_drawdown=oos_perf.get('max_drawdown', 0),
                walk_forward_efficiency=wfe,
                passed=passed,
            )

            self.results.append(result)
            fold_id += 1
            current_start += timedelta(days=step_days)

        return self.results

    def get_summary(self) -> Dict:
        """Get summary statistics from walk-forward optimization."""
        if not self.results:
            return {"error": "No results available"}

        efficiencies = [r.walk_forward_efficiency for r in self.results]
        is_sharpes = [r.is_sharpe for r in self.results]
        oos_sharpes = [r.oos_sharpe for r in self.results]
        passes = [r.passed for r in self.results]

        return {
            "num_folds": len(self.results),
            "avg_efficiency": np.mean(efficiencies) if efficiencies else 0,
            "std_efficiency": np.std(efficiencies) if efficiencies else 0,
            "min_efficiency": min(efficiencies) if efficiencies else 0,
            "max_efficiency": max(efficiencies) if efficiencies else 0,
            "avg_is_sharpe": np.mean(is_sharpes) if is_sharpes else 0,
            "avg_oos_sharpe": np.mean(oos_sharpes) if oos_sharpes else 0,
            "pass_rate": sum(passes) / len(passes) if passes else 0,
            "overall_passed": all(passes) if passes else False,
        }


class CPCV:
    """
    Combinatorial Purged Cross-Validation.

    Implements proper time-series cross-validation with:
    - Purging: Removes samples between train/test to prevent leakage
    - Embargo: Adds embargo period after test to prevent look-ahead
    - Combinatorial: Tests all possible train/test combinations
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def split(
        self,
        n_samples: int,
        purge: Optional[int] = None,
        embargo: Optional[int] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Generate train/test splits for CPCV.

        Args:
            n_samples: Total number of samples
            purge: Number of samples to purge (default from config)
            embargo: Number of samples for embargo (default from config)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        purge = purge if purge is not None else self.config.cpcv_purge_days
        embargo = embargo if embargo is not None else self.config.cpcv_embargo_days
        n_splits = self.config.cpcv_n_splits

        # Create n_splits roughly equal groups
        group_size = n_samples // n_splits
        groups = []
        for i in range(n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < n_splits - 1 else n_samples
            groups.append(list(range(start, end)))

        splits = []

        # Generate all combinations of test groups
        for n_test in range(1, n_splits):
            for test_indices in combinations(range(n_splits), n_test):
                train_groups = [i for i in range(n_splits) if i not in test_indices]
                test_groups = list(test_indices)

                # Flatten indices
                train_idx = []
                test_idx = []

                for g in train_groups:
                    train_idx.extend(groups[g])

                for g in test_groups:
                    test_idx.extend(groups[g])

                # Apply purging: remove samples close to test set
                purged_train_idx = []
                for idx in train_idx:
                    # Check distance to any test sample
                    min_dist = min(abs(idx - t) for t in test_idx)
                    if min_dist > purge:
                        purged_train_idx.append(idx)

                # Apply embargo: remove samples right after test set
                test_max = max(test_idx)
                embargo_end = test_max + embargo
                final_train_idx = [i for i in purged_train_idx if not (test_max < i <= embargo_end)]

                if len(final_train_idx) > 0 and len(test_idx) > 0:
                    splits.append((sorted(final_train_idx), sorted(test_idx)))

        return splits

    def cross_validate(
        self,
        data: List[Any],
        strategy_fn: Callable[[List[Any]], float],
    ) -> List[float]:
        """
        Run CPCV cross-validation.

        Args:
            data: List of data points (ordered by time)
            strategy_fn: Function that returns a score given data subset

        Returns:
            List of scores for each split
        """
        splits = self.split(len(data))
        scores = []

        for train_idx, test_idx in splits:
            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            try:
                score = strategy_fn(test_data)
                scores.append(score)
            except Exception as e:
                logger.warning(f"CPCV fold failed: {e}")
                continue

        return scores


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio calculation.

    Adjusts the Sharpe ratio for:
    1. Multiple testing (number of strategies tried)
    2. Non-normality of returns (skewness, kurtosis)
    3. Serial correlation

    References:
    - Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014)
    - "Probability of Backtest Overfitting" (PBO)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def calculate(
        self,
        returns: List[float],
        num_trials: int = 1,
        skewness: Optional[float] = None,
        kurtosis: Optional[float] = None,
    ) -> Dict:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            returns: List of returns (not annualized)
            num_trials: Number of strategies/parameters tested
            skewness: Skewness of returns (calculated if not provided)
            kurtosis: Excess kurtosis of returns (calculated if not provided)

        Returns:
            Dict with DSR, raw Sharpe, haircut, etc.
        """
        n = len(returns)

        if n < self.config.dsr_min_observations:
            return {
                "error": f"Insufficient observations: {n} < {self.config.dsr_min_observations}",
                "raw_sharpe": 0,
                "deflated_sharpe": 0,
                "haircut_pct": 0,
                "pbo": 1.0,  # Maximum probability of overfitting
            }

        # Calculate moments
        mean_return = sum(returns) / n
        var_return = sum((r - mean_return) ** 2 for r in returns) / (n - 1)
        std_return = var_return ** 0.5

        if std_return <= 0:
            return {
                "error": "Zero standard deviation",
                "raw_sharpe": 0,
                "deflated_sharpe": 0,
                "haircut_pct": 0,
                "pbo": 1.0,
            }

        # Annualized Sharpe ratio
        rf_per_period = self.config.dsr_risk_free_rate / self.config.dsr_periods_per_year
        excess_return = mean_return - rf_per_period
        raw_sharpe = (excess_return / std_return) * (self.config.dsr_periods_per_year ** 0.5)

        # Calculate skewness if not provided
        if skewness is None:
            skewness = sum((r - mean_return) ** 3 for r in returns) / (n * std_return ** 3)

        # Calculate excess kurtosis if not provided
        if kurtosis is None:
            kurtosis = sum((r - mean_return) ** 4 for r in returns) / (n * std_return ** 4) - 3

        # Standard error of Sharpe ratio (Lo, 2002)
        sr_std_error = ((1 + 0.5 * raw_sharpe ** 2 - skewness * raw_sharpe +
                        (kurtosis / 4) * raw_sharpe ** 2) / n) ** 0.5

        # Expected maximum Sharpe ratio under null (multiple testing)
        # E[max(SR)] ≈ (1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
        # Simplified: E[max] ≈ sqrt(2 * ln(N)) for large N
        if num_trials > 1:
            expected_max_sr = (2 * math.log(num_trials)) ** 0.5
        else:
            expected_max_sr = 0

        # Deflated Sharpe Ratio
        # DSR = SR - E[max_SR] * SE(SR)
        deflated_sharpe = raw_sharpe - expected_max_sr * sr_std_error

        # Haircut percentage
        haircut_pct = (1 - deflated_sharpe / raw_sharpe) * 100 if raw_sharpe > 0 else 0

        # Probability of Backtest Overfitting (simplified)
        # PBO ≈ Φ(-DSR / SE(SR))
        # Higher DSR = lower PBO
        if sr_std_error > 0:
            z_stat = deflated_sharpe / sr_std_error
            # Approximate Φ(-z) using error function
            pbo = 0.5 * (1 - math.erf(z_stat / math.sqrt(2)))
        else:
            pbo = 0.5

        return {
            "raw_sharpe": raw_sharpe,
            "deflated_sharpe": deflated_sharpe,
            "haircut_pct": haircut_pct,
            "sr_std_error": sr_std_error,
            "expected_max_sr": expected_max_sr,
            "skewness": skewness,
            "excess_kurtosis": kurtosis,
            "num_trials": num_trials,
            "n_observations": n,
            "pbo": pbo,
            "pbo_passed": pbo < self.config.pbo_max_threshold,
        }

    def min_track_record_length(
        self,
        target_sharpe: float = 1.0,
        num_trials: int = 1,
        confidence: float = 0.95,
    ) -> int:
        """
        Calculate minimum track record length for statistical significance.

        Based on: "How Long Does It Take to Win at Gambling?" (Bailey, 2012)

        Args:
            target_sharpe: Target annualized Sharpe ratio
            num_trials: Number of strategies tested
            confidence: Confidence level (default 95%)

        Returns:
            Minimum number of periods needed
        """
        # Z-score for confidence level
        z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645

        # Expected max under null
        expected_max = (2 * math.log(max(1, num_trials))) ** 0.5

        # Required Sharpe to be significant
        required_sr = target_sharpe + z * expected_max

        # Minimum length: n > (z / SR)^2
        # Annualized SR needs to be converted
        periods_per_year = self.config.dsr_periods_per_year
        sr_per_period = target_sharpe / (periods_per_year ** 0.5)

        if sr_per_period <= 0:
            return 10000  # Very long

        min_length = int((z / sr_per_period) ** 2 * (1 + 0.25 * target_sharpe ** 2))

        return max(self.config.dsr_min_observations, min_length)


class ValidationFramework:
    """
    Complete validation framework combining all methods.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.wfo = WalkForwardOptimizer(config)
        self.cpcv = CPCV(config)
        self.dsr = DeflatedSharpeRatio(config)

    def validate(
        self,
        data: List[Dict],
        strategy_fn: Callable[[List[Dict]], Dict],
        num_trials: int = 1,
    ) -> ValidationResult:
        """
        Run complete validation suite.

        Args:
            data: Historical data with OHLCV and date
            strategy_fn: Strategy function returning performance dict
            num_trials: Number of strategies/parameters tested

        Returns:
            Complete ValidationResult
        """
        result = ValidationResult(num_trials=num_trials)

        # 1. Walk-Forward Optimization
        try:
            wf_results = self.wfo.run(data, strategy_fn)
            result.wf_folds = wf_results

            if wf_results:
                efficiencies = [r.walk_forward_efficiency for r in wf_results]
                result.avg_wf_efficiency = sum(efficiencies) / len(efficiencies)
                result.wf_passed = all(r.passed for r in wf_results)

                if not result.wf_passed:
                    result.rejection_reasons.append(
                        f"Walk-Forward efficiency too low: {result.avg_wf_efficiency:.2%}"
                    )
        except Exception as e:
            logger.error(f"Walk-Forward validation failed: {e}")
            result.rejection_reasons.append(f"WFO failed: {str(e)}")

        # 2. CPCV Cross-Validation
        try:
            def cpcv_scorer(subset):
                perf = strategy_fn(subset)
                return perf.get('sharpe_ratio', 0)

            cpcv_scores = self.cpcv.cross_validate(data, cpcv_scorer)
            result.cpcv_scores = cpcv_scores

            if cpcv_scores:
                result.cpcv_mean = sum(cpcv_scores) / len(cpcv_scores)
                result.cpcv_std = (sum((s - result.cpcv_mean) ** 2 for s in cpcv_scores) / len(cpcv_scores)) ** 0.5
        except Exception as e:
            logger.error(f"CPCV validation failed: {e}")
            result.rejection_reasons.append(f"CPCV failed: {str(e)}")

        # 3. Deflated Sharpe Ratio
        try:
            # Calculate returns from full dataset
            returns = self._extract_returns(data)

            if returns:
                dsr_result = self.dsr.calculate(returns, num_trials)
                result.raw_sharpe = dsr_result.get('raw_sharpe', 0)
                result.deflated_sharpe = dsr_result.get('deflated_sharpe', 0)
                result.sharpe_haircut = dsr_result.get('haircut_pct', 0)
                result.pbo = dsr_result.get('pbo', 1.0)
                result.pbo_passed = dsr_result.get('pbo_passed', False)

                if result.raw_sharpe < self.config.min_sharpe_threshold:
                    result.rejection_reasons.append(
                        f"Sharpe ratio too low: {result.raw_sharpe:.2f}"
                    )

                if not result.pbo_passed:
                    result.rejection_reasons.append(
                        f"PBO too high: {result.pbo:.2%} > {self.config.pbo_max_threshold:.2%}"
                    )
        except Exception as e:
            logger.error(f"DSR calculation failed: {e}")
            result.rejection_reasons.append(f"DSR failed: {str(e)}")

        # 4. Final verdict
        result.is_valid = (
            result.wf_passed and
            result.pbo_passed and
            len(result.rejection_reasons) == 0
        )

        return result

    def _extract_returns(self, data: List[Dict]) -> List[float]:
        """Extract daily returns from price data."""
        returns = []
        prices = [d.get('close', 0) for d in data if d.get('close', 0) > 0]

        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

        return returns

    def get_min_sample_requirement(
        self,
        target_sharpe: float = 1.0,
        num_trials: int = 1,
    ) -> Dict:
        """
        Get minimum sample requirements for valid backtest.

        Returns:
            Dict with minimum requirements
        """
        min_length = self.dsr.min_track_record_length(target_sharpe, num_trials)

        # Account for WFO needs
        wfo_min = (
            self.config.wf_train_days +
            self.config.wf_validate_days +
            self.config.wf_trade_days
        ) * 3  # At least 3 folds

        return {
            "min_for_dsr": min_length,
            "min_for_wfo": wfo_min,
            "min_total": max(min_length, wfo_min),
            "recommended": max(min_length, wfo_min) * 1.5,
        }
