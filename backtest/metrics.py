"""
Backtesting Metrics for AI-Trader

This module provides statistical metrics for backtesting validation:
- Standard metrics (Sharpe, Sortino, Calmar, Max Drawdown)
- Deflated Sharpe Ratio (DSR) for multiple testing correction
- Statistical significance tests

References:
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
- Bailey et al. (2016): "Probability of Backtest Overfitting"
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.04,
    annualization_factor: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: List of periodic returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free rate to periodic
    periodic_rf = risk_free_rate / annualization_factor

    # Calculate excess returns
    excess_returns = [r - periodic_rf for r in returns]

    # Mean and standard deviation
    mean = sum(excess_returns) / len(excess_returns)
    variance = sum((r - mean) ** 2 for r in excess_returns) / len(excess_returns)
    std = math.sqrt(variance) if variance > 0 else 1e-10

    # Annualize
    sharpe = (mean / std) * math.sqrt(annualization_factor)
    return sharpe


def calculate_sortino_ratio(
    returns: List[float],
    risk_free_rate: float = 0.04,
    annualization_factor: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Args:
        returns: List of periodic returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    periodic_rf = risk_free_rate / annualization_factor

    # Calculate excess returns
    excess_returns = [r - periodic_rf for r in returns]
    mean = sum(excess_returns) / len(excess_returns)

    # Downside deviation (only negative excess returns)
    downside = [min(0, r) ** 2 for r in excess_returns]
    downside_variance = sum(downside) / len(downside)
    downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 1e-10

    # Annualize
    sortino = (mean / downside_std) * math.sqrt(annualization_factor)
    return sortino


def calculate_calmar_ratio(
    returns: List[float],
    max_drawdown: Optional[float] = None,
    annualization_factor: int = 252,
) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: List of periodic returns
        max_drawdown: Pre-calculated max drawdown (if available)
        annualization_factor: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    # Calculate annualized return
    mean_return = sum(returns) / len(returns)
    annual_return = mean_return * annualization_factor

    # Calculate max drawdown if not provided
    if max_drawdown is None:
        max_drawdown = calculate_max_drawdown(returns)

    if max_drawdown <= 0:
        return 0.0 if annual_return <= 0 else float('inf')

    return annual_return / max_drawdown


def calculate_max_drawdown(returns: List[float]) -> float:
    """
    Calculate maximum drawdown from returns.

    Args:
        returns: List of periodic returns

    Returns:
        Maximum drawdown as positive decimal (e.g., 0.20 for 20%)
    """
    if len(returns) < 2:
        return 0.0

    # Calculate cumulative returns
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0

    for r in returns:
        cumulative *= (1 + r)
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    variance_sharpe: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_observations: int = 252,
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR) to correct for multiple testing.

    The DSR adjusts the Sharpe ratio for:
    - Number of trials/strategies tested
    - Non-normality of returns (skewness, kurtosis)
    - Variance of Sharpe ratio estimator

    Args:
        sharpe: Observed Sharpe ratio
        n_trials: Number of strategy variations tested
        variance_sharpe: Variance of Sharpe ratio estimates
        skewness: Skewness of returns
        kurtosis: Excess kurtosis of returns
        n_observations: Number of observations

    Returns:
        Deflated Sharpe ratio (probability that true Sharpe > 0)

    Reference:
        Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
    """
    if n_trials <= 0 or variance_sharpe <= 0:
        return 0.0

    # Expected maximum Sharpe from random trials
    # E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/n) + γ * Φ^{-1}(1 - 1/(n*e))
    # Simplified approximation:
    from math import log, sqrt, erf

    def norm_ppf(p: float) -> float:
        """Approximate inverse normal CDF."""
        # Rational approximation
        if p <= 0:
            return -6.0
        if p >= 1:
            return 6.0

        t = sqrt(-2 * log(min(p, 1 - p)))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        result = t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)
        return result if p > 0.5 else -result

    def norm_cdf(x: float) -> float:
        """Normal CDF."""
        return 0.5 * (1 + erf(x / sqrt(2)))

    # Expected maximum Sharpe under null
    euler_gamma = 0.5772156649
    expected_max_sharpe = (
        (1 - euler_gamma) * norm_ppf(1 - 1/n_trials) +
        euler_gamma * norm_ppf(1 - 1/(n_trials * math.e))
    ) * sqrt(variance_sharpe)

    # Adjust for non-normality
    sr_std = sqrt(
        (1 + 0.5 * sharpe**2 - skewness * sharpe + ((kurtosis - 3) / 4) * sharpe**2)
        / n_observations
    )

    if sr_std <= 0:
        return 0.0

    # DSR is probability that true Sharpe > expected_max under null
    z_score = (sharpe - expected_max_sharpe) / sr_std
    dsr = norm_cdf(z_score)

    return dsr


def probability_of_backtest_overfitting(
    in_sample_sharpes: List[float],
    out_sample_sharpes: List[float],
) -> float:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    PBO measures the probability that the best in-sample strategy
    performs below median out-of-sample.

    Args:
        in_sample_sharpes: Sharpe ratios on training data
        out_sample_sharpes: Sharpe ratios on test data

    Returns:
        PBO probability (0 to 1, lower is better, < 0.05 is good)

    Reference:
        Bailey et al. (2016): "Probability of Backtest Overfitting"
    """
    if len(in_sample_sharpes) != len(out_sample_sharpes):
        raise ValueError("Must have same number of in-sample and out-sample results")

    if len(in_sample_sharpes) < 2:
        return 0.5  # Insufficient data

    n = len(in_sample_sharpes)

    # Rank correlation between in-sample and out-sample
    # Count inversions (pairs where in-sample rank != out-sample rank direction)

    # Get ranks
    in_sample_ranks = _get_ranks(in_sample_sharpes)
    out_sample_ranks = _get_ranks(out_sample_sharpes)

    # Find best in-sample strategy
    best_in_sample_idx = in_sample_sharpes.index(max(in_sample_sharpes))

    # Check if best in-sample is below median out-sample
    out_sample_median = sorted(out_sample_sharpes)[n // 2]
    best_oos_performance = out_sample_sharpes[best_in_sample_idx]

    # Calculate logit using rank correlation
    # Simplified: fraction of combinations where best IS underperforms OOS
    underperform_count = 0
    for i in range(n):
        if out_sample_sharpes[i] > best_oos_performance:
            underperform_count += 1

    pbo = underperform_count / n

    return pbo


def _get_ranks(values: List[float]) -> List[int]:
    """Get ranks of values (1 = highest)."""
    sorted_with_idx = sorted(enumerate(values), key=lambda x: -x[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(sorted_with_idx, 1):
        ranks[idx] = rank
    return ranks


@dataclass
class BacktestMetrics:
    """Collection of backtest performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    total_return: float
    annual_return: float
    volatility: float
    win_rate: float
    profit_factor: float
    expectancy: float
    n_trades: int

    # Statistical significance
    dsr: Optional[float] = None  # Deflated Sharpe Ratio
    pbo: Optional[float] = None  # Probability of Backtest Overfitting

    def is_statistically_significant(self) -> bool:
        """Check if strategy passes statistical tests."""
        if self.dsr is not None and self.dsr < 0.95:
            return False
        if self.pbo is not None and self.pbo > 0.05:
            return False
        return True


def calculate_all_metrics(
    returns: List[float],
    trades: Optional[List[float]] = None,
    risk_free_rate: float = 0.04,
) -> BacktestMetrics:
    """
    Calculate all backtest metrics.

    Args:
        returns: List of periodic returns
        trades: List of individual trade P&L (optional)
        risk_free_rate: Annual risk-free rate

    Returns:
        BacktestMetrics with all calculated values
    """
    if not returns:
        return BacktestMetrics(
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, total_return=0, annual_return=0,
            volatility=0, win_rate=0, profit_factor=0,
            expectancy=0, n_trades=0,
        )

    # Basic metrics
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns, max_dd)

    # Total and annual return
    cumulative = 1.0
    for r in returns:
        cumulative *= (1 + r)
    total_return = cumulative - 1
    annual_return = (cumulative ** (252 / len(returns))) - 1 if len(returns) > 0 else 0

    # Volatility
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    volatility = math.sqrt(variance * 252)

    # Trade statistics
    if trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        n_trades = len(trades)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    else:
        n_trades = 0
        win_rate = 0
        profit_factor = 0
        expectancy = 0

    return BacktestMetrics(
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        total_return=total_return,
        annual_return=annual_return,
        volatility=volatility,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        n_trades=n_trades,
    )
