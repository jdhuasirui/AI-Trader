"""
Confidence Calibration for AI-Trader

This module implements confidence calibration methods to ensure that
model-reported confidence scores accurately reflect true probabilities:

1. Platt Scaling:
   - Fits sigmoid function to map raw scores to calibrated probabilities
   - P(y=1|x) = 1 / (1 + exp(A*f(x) + B))

2. Isotonic Regression:
   - Non-parametric monotonic regression
   - More flexible than Platt scaling
   - Better for non-sigmoid shaped calibration

3. Reliability Curves (Calibration Curves):
   - Visual diagnostic for calibration quality
   - Expected Calibration Error (ECE)
   - Maximum Calibration Error (MCE)

4. Temperature Scaling:
   - Simple single-parameter scaling
   - P(y|x) = softmax(logits / T)

Use cases:
- Comparing confidence across different models
- Proper uncertainty quantification for position sizing
- Ensemble weighting based on calibrated confidence
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for confidence calibration (immutable)."""

    # Platt scaling parameters
    platt_max_iter: int = 100  # Maximum iterations for fitting
    platt_learning_rate: float = 0.01  # Learning rate for gradient descent
    platt_tolerance: float = 1e-6  # Convergence tolerance

    # Isotonic regression parameters
    iso_min_samples: int = 10  # Minimum samples per bin
    iso_increasing: bool = True  # Enforce increasing calibration

    # Reliability curve parameters
    n_bins: int = 10  # Number of bins for reliability curve
    strategy: str = "uniform"  # "uniform" or "quantile" binning

    # Temperature scaling
    temp_init: float = 1.0  # Initial temperature
    temp_max_iter: int = 50  # Max iterations for temperature optimization

    # Validation
    min_samples: int = 50  # Minimum samples for calibration


@dataclass
class CalibrationResult:
    """Results from calibration fitting."""
    method: str = ""
    is_fitted: bool = False

    # Platt parameters
    platt_A: float = 0.0
    platt_B: float = 0.0

    # Isotonic mapping (list of (threshold, probability) pairs)
    iso_thresholds: List[float] = field(default_factory=list)
    iso_probabilities: List[float] = field(default_factory=list)

    # Temperature
    temperature: float = 1.0

    # Calibration metrics
    ece_before: float = 0.0  # Expected Calibration Error before
    ece_after: float = 0.0  # Expected Calibration Error after
    mce_before: float = 0.0  # Maximum Calibration Error before
    mce_after: float = 0.0  # Maximum Calibration Error after
    brier_before: float = 0.0  # Brier score before
    brier_after: float = 0.0  # Brier score after


class PlattScaling:
    """
    Platt Scaling for probability calibration.

    Maps raw model scores to calibrated probabilities using a sigmoid:
    P(y=1|x) = 1 / (1 + exp(A*f(x) + B))

    Parameters A and B are fitted to minimize negative log-likelihood.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.A = 0.0  # Slope parameter
        self.B = 0.0  # Intercept parameter
        self.is_fitted = False

    def fit(
        self,
        scores: List[float],
        labels: List[int],
    ) -> CalibrationResult:
        """
        Fit Platt scaling parameters.

        Args:
            scores: Raw model scores/confidences (0-1 or unbounded)
            labels: Binary labels (0 or 1)

        Returns:
            CalibrationResult with fitted parameters
        """
        n = len(scores)
        if n < self.config.min_samples:
            logger.warning(f"Insufficient samples for Platt scaling: {n}")
            return CalibrationResult(method="platt", is_fitted=False)

        scores = np.array(scores)
        labels = np.array(labels)

        # Initialize parameters
        n_pos = sum(labels)
        n_neg = n - n_pos

        # Prior correction (Platt's heuristic)
        hi_target = (n_pos + 1) / (n_pos + 2)
        lo_target = 1 / (n_neg + 2)

        # Initialize A and B
        A = 0.0
        B = math.log((n_neg + 1) / (n_pos + 1))

        # Target values with prior correction
        t = np.where(labels == 1, hi_target, lo_target)

        # Gradient descent
        for iteration in range(self.config.platt_max_iter):
            # Forward pass
            fApB = A * scores + B
            # Numerical stability
            fApB = np.clip(fApB, -200, 200)
            p = 1.0 / (1.0 + np.exp(fApB))

            # Gradient
            d1 = t - p
            d2 = p * (1 - p)

            # Update A (gradient w.r.t. A)
            grad_A = np.sum(d1 * scores)
            # Update B (gradient w.r.t. B)
            grad_B = np.sum(d1)

            # Hessian diagonal (for Newton step)
            H_A = np.sum(d2 * scores * scores) + 1e-10
            H_B = np.sum(d2) + 1e-10

            # Newton update
            A_new = A + grad_A / H_A
            B_new = B + grad_B / H_B

            # Check convergence
            if abs(A_new - A) < self.config.platt_tolerance and \
               abs(B_new - B) < self.config.platt_tolerance:
                A, B = A_new, B_new
                break

            A, B = A_new, B_new

        self.A = float(A)
        self.B = float(B)
        self.is_fitted = True

        # Calculate metrics
        result = CalibrationResult(method="platt", is_fitted=True)
        result.platt_A = self.A
        result.platt_B = self.B

        # ECE before and after
        result.ece_before = self._calculate_ece(scores.tolist(), labels.tolist())
        calibrated = self.transform(scores.tolist())
        result.ece_after = self._calculate_ece(calibrated, labels.tolist())

        # Brier score
        result.brier_before = np.mean((scores - labels) ** 2)
        result.brier_after = np.mean((np.array(calibrated) - labels) ** 2)

        return result

    def transform(self, scores: List[float]) -> List[float]:
        """
        Transform raw scores to calibrated probabilities.

        Args:
            scores: Raw model scores

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Platt scaling not fitted, returning raw scores")
            return scores

        calibrated = []
        for s in scores:
            fApB = self.A * s + self.B
            # Numerical stability
            fApB = max(-200, min(200, fApB))
            p = 1.0 / (1.0 + math.exp(fApB))
            calibrated.append(p)

        return calibrated

    def _calculate_ece(
        self,
        probs: List[float],
        labels: List[int],
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error."""
        bins = defaultdict(list)
        labels_by_bin = defaultdict(list)

        for p, y in zip(probs, labels):
            bin_idx = min(int(p * n_bins), n_bins - 1)
            bins[bin_idx].append(p)
            labels_by_bin[bin_idx].append(y)

        ece = 0.0
        n = len(probs)

        for bin_idx in bins:
            bin_probs = bins[bin_idx]
            bin_labels = labels_by_bin[bin_idx]
            bin_size = len(bin_probs)

            avg_confidence = sum(bin_probs) / bin_size
            avg_accuracy = sum(bin_labels) / bin_size

            ece += (bin_size / n) * abs(avg_confidence - avg_accuracy)

        return ece


class IsotonicCalibration:
    """
    Isotonic Regression for probability calibration.

    Non-parametric method that fits a monotonically increasing
    step function to map scores to probabilities.

    More flexible than Platt scaling, especially when the
    calibration curve is not sigmoid-shaped.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.thresholds: List[float] = []
        self.probabilities: List[float] = []
        self.is_fitted = False

    def fit(
        self,
        scores: List[float],
        labels: List[int],
    ) -> CalibrationResult:
        """
        Fit isotonic regression.

        Uses Pool Adjacent Violators Algorithm (PAVA).

        Args:
            scores: Raw model scores
            labels: Binary labels

        Returns:
            CalibrationResult
        """
        n = len(scores)
        if n < self.config.min_samples:
            logger.warning(f"Insufficient samples for isotonic calibration: {n}")
            return CalibrationResult(method="isotonic", is_fitted=False)

        # Sort by scores
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0])
        sorted_scores = [p[0] for p in sorted_pairs]
        sorted_labels = [p[1] for p in sorted_pairs]

        # Pool Adjacent Violators Algorithm
        n_samples = len(sorted_labels)
        y = np.array(sorted_labels, dtype=float)
        weights = np.ones(n_samples)

        # Forward pass
        blocks = [[i] for i in range(n_samples)]
        block_weights = weights.copy()
        block_values = y.copy()

        i = 0
        while i < len(blocks) - 1:
            if block_values[i] > block_values[i + 1]:
                # Merge blocks
                new_weight = block_weights[i] + block_weights[i + 1]
                new_value = (block_weights[i] * block_values[i] +
                            block_weights[i + 1] * block_values[i + 1]) / new_weight

                blocks[i].extend(blocks[i + 1])
                block_weights[i] = new_weight
                block_values[i] = new_value

                del blocks[i + 1]
                del block_weights[i + 1]
                del block_values[i + 1]

                # Move back to check previous
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Build threshold-probability mapping
        self.thresholds = []
        self.probabilities = []

        for block, value in zip(blocks, block_values):
            # Use the maximum score in the block as threshold
            block_scores = [sorted_scores[i] for i in block]
            threshold = max(block_scores)
            self.thresholds.append(float(threshold))
            self.probabilities.append(float(value))

        self.is_fitted = True

        # Calculate metrics
        result = CalibrationResult(method="isotonic", is_fitted=True)
        result.iso_thresholds = self.thresholds
        result.iso_probabilities = self.probabilities

        # ECE before and after
        platt = PlattScaling(self.config)
        result.ece_before = platt._calculate_ece(scores, labels)
        calibrated = self.transform(scores)
        result.ece_after = platt._calculate_ece(calibrated, labels)

        # Brier score
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        result.brier_before = np.mean((scores_arr - labels_arr) ** 2)
        result.brier_after = np.mean((np.array(calibrated) - labels_arr) ** 2)

        return result

    def transform(self, scores: List[float]) -> List[float]:
        """
        Transform raw scores to calibrated probabilities.

        Uses linear interpolation between isotonic thresholds.
        """
        if not self.is_fitted:
            return scores

        calibrated = []
        for s in scores:
            # Find surrounding thresholds
            if s <= self.thresholds[0]:
                p = self.probabilities[0]
            elif s >= self.thresholds[-1]:
                p = self.probabilities[-1]
            else:
                # Linear interpolation
                for i in range(len(self.thresholds) - 1):
                    if self.thresholds[i] <= s <= self.thresholds[i + 1]:
                        t1, t2 = self.thresholds[i], self.thresholds[i + 1]
                        p1, p2 = self.probabilities[i], self.probabilities[i + 1]
                        if t2 > t1:
                            p = p1 + (p2 - p1) * (s - t1) / (t2 - t1)
                        else:
                            p = (p1 + p2) / 2
                        break
                else:
                    p = self.probabilities[-1]

            calibrated.append(p)

        return calibrated


class TemperatureScaling:
    """
    Temperature Scaling for neural network calibration.

    Simple single-parameter scaling:
    P(y|x) = softmax(logits / T)

    For binary case: P(y=1|x) = sigmoid(logit / T)
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.temperature = 1.0
        self.is_fitted = False

    def fit(
        self,
        logits: List[float],
        labels: List[int],
    ) -> CalibrationResult:
        """
        Fit temperature parameter.

        Optimizes temperature to minimize negative log-likelihood.
        """
        n = len(logits)
        if n < self.config.min_samples:
            return CalibrationResult(method="temperature", is_fitted=False)

        logits = np.array(logits)
        labels = np.array(labels)

        # Grid search for temperature
        best_temp = 1.0
        best_nll = float('inf')

        for temp in np.linspace(0.1, 5.0, 50):
            scaled_logits = logits / temp
            # Numerical stability
            scaled_logits = np.clip(scaled_logits, -200, 200)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))

            # Negative log-likelihood
            nll = -np.mean(labels * np.log(probs + 1e-10) +
                          (1 - labels) * np.log(1 - probs + 1e-10))

            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = float(best_temp)
        self.is_fitted = True

        result = CalibrationResult(method="temperature", is_fitted=True)
        result.temperature = self.temperature

        return result

    def transform(self, logits: List[float]) -> List[float]:
        """Transform logits to calibrated probabilities."""
        if not self.is_fitted:
            return [1.0 / (1.0 + math.exp(-l)) for l in logits]

        calibrated = []
        for l in logits:
            scaled = l / self.temperature
            scaled = max(-200, min(200, scaled))
            p = 1.0 / (1.0 + math.exp(-scaled))
            calibrated.append(p)

        return calibrated


class ReliabilityCurve:
    """
    Calculate and analyze reliability (calibration) curves.

    A well-calibrated model should have predictions that match
    the true frequency of positive outcomes.
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()

    def calculate(
        self,
        probs: List[float],
        labels: List[int],
    ) -> Dict:
        """
        Calculate reliability curve data.

        Returns:
            Dict with bin edges, mean predicted, mean observed, and metrics
        """
        n_bins = self.config.n_bins

        if self.config.strategy == "uniform":
            # Uniform bins
            bins = np.linspace(0, 1, n_bins + 1)
        else:
            # Quantile bins
            bins = np.percentile(probs, np.linspace(0, 100, n_bins + 1))

        bin_data = defaultdict(lambda: {"probs": [], "labels": []})

        for p, y in zip(probs, labels):
            for i in range(len(bins) - 1):
                if bins[i] <= p < bins[i + 1] or (i == len(bins) - 2 and p == bins[-1]):
                    bin_data[i]["probs"].append(p)
                    bin_data[i]["labels"].append(y)
                    break

        # Calculate bin statistics
        mean_predicted = []
        mean_observed = []
        bin_counts = []
        bin_edges = []

        for i in range(n_bins):
            if bin_data[i]["probs"]:
                mean_predicted.append(np.mean(bin_data[i]["probs"]))
                mean_observed.append(np.mean(bin_data[i]["labels"]))
                bin_counts.append(len(bin_data[i]["probs"]))
            else:
                mean_predicted.append((bins[i] + bins[i + 1]) / 2)
                mean_observed.append(0)
                bin_counts.append(0)
            bin_edges.append((float(bins[i]), float(bins[i + 1])))

        # Calculate ECE and MCE
        n = sum(bin_counts)
        ece = sum(
            (count / n) * abs(pred - obs)
            for count, pred, obs in zip(bin_counts, mean_predicted, mean_observed)
        ) if n > 0 else 0

        mce = max(
            abs(pred - obs)
            for pred, obs in zip(mean_predicted, mean_observed)
        ) if mean_predicted else 0

        return {
            "bin_edges": bin_edges,
            "mean_predicted": mean_predicted,
            "mean_observed": mean_observed,
            "bin_counts": bin_counts,
            "ece": ece,
            "mce": mce,
            "n_samples": len(probs),
        }


class ConfidenceCalibrator:
    """
    Main interface for confidence calibration.

    Supports multiple calibration methods and provides
    easy-to-use calibration and evaluation.
    """

    def __init__(self, method: str = "platt", config: Optional[CalibrationConfig] = None):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ("platt", "isotonic", "temperature")
            config: Configuration
        """
        self.config = config or CalibrationConfig()
        self.method = method
        self.reliability = ReliabilityCurve(self.config)

        if method == "platt":
            self.calibrator = PlattScaling(self.config)
        elif method == "isotonic":
            self.calibrator = IsotonicCalibration(self.config)
        elif method == "temperature":
            self.calibrator = TemperatureScaling(self.config)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        self.result: Optional[CalibrationResult] = None

    def fit(
        self,
        scores: List[float],
        labels: List[int],
    ) -> CalibrationResult:
        """
        Fit the calibrator.

        Args:
            scores: Raw model scores/confidences
            labels: Binary labels (0 or 1)

        Returns:
            CalibrationResult with metrics
        """
        self.result = self.calibrator.fit(scores, labels)

        # Add reliability curve data
        if self.result.is_fitted:
            rel_data = self.reliability.calculate(scores, labels)
            self.result.ece_before = rel_data["ece"]
            self.result.mce_before = rel_data["mce"]

            calibrated = self.calibrator.transform(scores)
            rel_after = self.reliability.calculate(calibrated, labels)
            self.result.ece_after = rel_after["ece"]
            self.result.mce_after = rel_after["mce"]

        return self.result

    def calibrate(self, scores: List[float]) -> List[float]:
        """
        Calibrate new scores.

        Args:
            scores: Raw model scores to calibrate

        Returns:
            Calibrated probabilities
        """
        return self.calibrator.transform(scores)

    def calibrate_single(self, score: float) -> float:
        """Calibrate a single score."""
        return self.calibrator.transform([score])[0]

    def get_reliability_curve(
        self,
        probs: List[float],
        labels: List[int],
    ) -> Dict:
        """Get reliability curve data for visualization."""
        return self.reliability.calculate(probs, labels)

    def is_well_calibrated(self, threshold: float = 0.1) -> bool:
        """
        Check if model is well-calibrated.

        Args:
            threshold: Maximum acceptable ECE

        Returns:
            True if ECE after calibration is below threshold
        """
        if self.result is None:
            return False
        return self.result.ece_after < threshold
