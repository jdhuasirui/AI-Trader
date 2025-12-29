"""
Concept Drift Detection for AI-Trader

This module implements adaptive drift detection methods (ADDM) to detect
when model predictions or market conditions have significantly changed:

- ADWIN (Adaptive Windowing): Detects distribution changes in streaming data
- Page-Hinkley Test: Detects changes in mean of a sequence
- SETAR (Self-Exciting Threshold AutoRegressive): Regime-based threshold alerts
- Prediction Error Monitoring: Tracks model accuracy degradation

When drift is detected, the system can:
1. Stop model trading
2. Reduce model weight in signal aggregation
3. Trigger model retraining
4. Alert for human review
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of detected drift."""
    NONE = "NONE"
    WARNING = "WARNING"  # Early warning, may revert
    DRIFT = "DRIFT"  # Confirmed drift
    SEVERE = "SEVERE"  # Severe drift requiring immediate action


class DriftAction(str, Enum):
    """Actions to take when drift is detected."""
    CONTINUE = "CONTINUE"  # No action needed
    REDUCE_WEIGHT = "REDUCE_WEIGHT"  # Reduce model weight in aggregation
    HALT_MODEL = "HALT_MODEL"  # Stop using this model
    RETRAIN = "RETRAIN"  # Trigger retraining
    ALERT = "ALERT"  # Send alert for human review


@dataclass
class DriftConfig:
    """Configuration for drift detection."""

    # ADWIN parameters
    adwin_delta: float = 0.002  # Confidence parameter (smaller = more sensitive)
    adwin_max_buckets: int = 5  # Maximum number of bucket levels

    # Page-Hinkley parameters
    ph_delta: float = 0.005  # Minimum magnitude of change to detect
    ph_lambda: float = 50.0  # Threshold for detection
    ph_alpha: float = 0.9999  # Forgetting factor

    # SETAR parameters (for regime-based alerts)
    setar_threshold_low: float = -2.0  # Lower threshold (z-score)
    setar_threshold_high: float = 2.0  # Upper threshold (z-score)
    setar_window: int = 20  # Lookback window for statistics

    # Prediction error monitoring
    error_window: int = 50  # Window for error calculation
    error_threshold: float = 1.5  # Alert if error increases by 50%
    baseline_error_window: int = 200  # Baseline error window

    # Action thresholds
    warning_threshold: float = 0.7  # Drift score for warning
    drift_threshold: float = 0.85  # Drift score for confirmed drift
    severe_threshold: float = 0.95  # Drift score for severe drift

    # Cooldown
    cooldown_periods: int = 10  # Periods to wait after drift before rechecking


@dataclass
class DriftState:
    """Current state of drift detection."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Current drift status
    drift_type: DriftType = DriftType.NONE
    drift_score: float = 0.0  # 0-1, higher = more drift

    # Individual detector states
    adwin_detected: bool = False
    page_hinkley_detected: bool = False
    setar_regime_change: bool = False
    error_spike_detected: bool = False

    # Statistics
    current_mean: float = 0.0
    current_std: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 0.0

    # Error tracking
    current_error: float = 0.0
    baseline_error: float = 0.0
    error_ratio: float = 1.0

    # Recommended action
    action: DriftAction = DriftAction.CONTINUE
    action_reason: str = ""

    # History
    drift_history: List[Tuple[datetime, DriftType, float]] = field(default_factory=list)

    # Cooldown tracking
    cooldown_remaining: int = 0


class ADWIN:
    """
    Adaptive Windowing algorithm for drift detection.

    ADWIN maintains a variable-length window of recent data and
    detects distributional changes by comparing subwindows.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5):
        self.delta = delta
        self.max_buckets = max_buckets

        # Bucket structure: list of (sum, variance, count) at each level
        self.buckets: List[List[Tuple[float, float, int]]] = [[] for _ in range(max_buckets)]
        self.total_count = 0
        self.total_sum = 0.0
        self.variance = 0.0
        self.width = 0

    def update(self, value: float) -> bool:
        """
        Add a new value and check for drift.

        Returns True if drift is detected.
        """
        # Add to first bucket level
        self._insert_element(value)
        self.total_count += 1
        self.total_sum += value
        self.width += 1

        # Compress buckets if needed
        self._compress_buckets()

        # Check for drift
        return self._detect_change()

    def _insert_element(self, value: float):
        """Insert element at first bucket level."""
        self.buckets[0].append((value, 0.0, 1))

    def _compress_buckets(self):
        """Compress buckets following exponential histogram."""
        for level in range(self.max_buckets - 1):
            if len(self.buckets[level]) >= 2:
                # Merge two oldest buckets into one at next level
                b1 = self.buckets[level].pop(0)
                b2 = self.buckets[level].pop(0)

                new_sum = b1[0] + b2[0]
                new_count = b1[2] + b2[2]
                # Combined variance approximation
                new_var = b1[1] + b2[1] + (b1[0]/b1[2] - b2[0]/b2[2])**2 * b1[2] * b2[2] / new_count if b1[2] > 0 and b2[2] > 0 else 0

                self.buckets[level + 1].append((new_sum, new_var, new_count))

    def _detect_change(self) -> bool:
        """Check for statistical change between window halves."""
        if self.width < 10:
            return False

        # Calculate statistics for each half
        n1, sum1 = 0, 0.0
        n2, sum2 = 0, 0.0

        # Divide data into two halves using bucket structure
        half_count = self.width // 2
        current_count = 0

        for level in range(self.max_buckets):
            for bucket in self.buckets[level]:
                if current_count < half_count:
                    n1 += bucket[2]
                    sum1 += bucket[0]
                else:
                    n2 += bucket[2]
                    sum2 += bucket[0]
                current_count += bucket[2]

        if n1 == 0 or n2 == 0:
            return False

        mean1 = sum1 / n1
        mean2 = sum2 / n2

        # Hoeffding bound for difference detection
        m = 1.0 / (1.0 / n1 + 1.0 / n2)
        epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))

        if abs(mean1 - mean2) > epsilon:
            # Drift detected - reduce window
            self._reduce_window()
            return True

        return False

    def _reduce_window(self):
        """Remove oldest buckets after drift detection."""
        # Remove approximately half the data
        removed = 0
        target = self.width // 2

        for level in range(self.max_buckets - 1, -1, -1):
            while self.buckets[level] and removed < target:
                bucket = self.buckets[level].pop(0)
                removed += bucket[2]
                self.total_sum -= bucket[0]

        self.width -= removed
        self.total_count -= removed

    def get_mean(self) -> float:
        """Get current mean of the window."""
        return self.total_sum / self.width if self.width > 0 else 0.0

    def reset(self):
        """Reset the detector."""
        self.buckets = [[] for _ in range(self.max_buckets)]
        self.total_count = 0
        self.total_sum = 0.0
        self.variance = 0.0
        self.width = 0


class PageHinkley:
    """
    Page-Hinkley test for mean change detection.

    Detects if the mean of a sequence has changed significantly.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999):
        self.delta = delta  # Minimum magnitude of change to detect
        self.threshold = threshold  # Detection threshold (lambda)
        self.alpha = alpha  # Forgetting factor

        self.sum = 0.0
        self.min_sum = float('inf')
        self.max_sum = float('-inf')
        self.mean = 0.0
        self.count = 0

    def update(self, value: float) -> Tuple[bool, str]:
        """
        Update with new value and check for drift.

        Returns (drift_detected, direction).
        """
        self.count += 1

        # Update mean estimate with forgetting
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value

        # Update cumulative sum
        self.sum += value - self.mean - self.delta

        self.min_sum = min(self.min_sum, self.sum)
        self.max_sum = max(self.max_sum, self.sum)

        # Check for upward drift
        ph_up = self.sum - self.min_sum
        if ph_up > self.threshold:
            self.reset()
            return True, "UP"

        # Check for downward drift
        ph_down = self.max_sum - self.sum
        if ph_down > self.threshold:
            self.reset()
            return True, "DOWN"

        return False, ""

    def reset(self):
        """Reset the detector."""
        self.sum = 0.0
        self.min_sum = float('inf')
        self.max_sum = float('-inf')
        self.count = 0


class DriftDetector:
    """
    Main drift detection system combining multiple detection methods.

    Uses ensemble of:
    1. ADWIN for distribution drift
    2. Page-Hinkley for mean shift
    3. SETAR-style thresholds for regime changes
    4. Prediction error monitoring
    """

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self.state = DriftState()

        # Individual detectors
        self.adwin = ADWIN(
            delta=self.config.adwin_delta,
            max_buckets=self.config.adwin_max_buckets,
        )
        self.page_hinkley = PageHinkley(
            delta=self.config.ph_delta,
            threshold=self.config.ph_lambda,
            alpha=self.config.ph_alpha,
        )

        # Data windows for SETAR and error monitoring
        self._value_window: deque = deque(maxlen=self.config.setar_window)
        self._error_window: deque = deque(maxlen=self.config.error_window)
        self._baseline_errors: deque = deque(maxlen=self.config.baseline_error_window)

        # Model performance tracking
        self._predictions: deque = deque(maxlen=100)
        self._actuals: deque = deque(maxlen=100)

    def update(
        self,
        value: float,
        prediction: Optional[float] = None,
        actual: Optional[float] = None,
    ) -> DriftState:
        """
        Update drift detection with new data.

        Args:
            value: The value to monitor (e.g., return, spread, indicator)
            prediction: Model prediction (optional, for error monitoring)
            actual: Actual value (optional, for error monitoring)

        Returns:
            Updated DriftState with detection results
        """
        self.state.timestamp = datetime.now()

        # Check cooldown
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self.state

        # Update ADWIN
        self.state.adwin_detected = self.adwin.update(value)

        # Update Page-Hinkley
        ph_detected, ph_direction = self.page_hinkley.update(value)
        self.state.page_hinkley_detected = ph_detected

        # Update SETAR-style detection
        self._value_window.append(value)
        self.state.setar_regime_change = self._check_setar_threshold(value)

        # Update error monitoring
        if prediction is not None and actual is not None:
            error = abs(prediction - actual)
            self._error_window.append(error)
            self._baseline_errors.append(error)
            self.state.error_spike_detected = self._check_error_spike()

        # Update statistics
        self._update_statistics()

        # Calculate combined drift score
        self.state.drift_score = self._calculate_drift_score()

        # Determine drift type
        self.state.drift_type = self._classify_drift()

        # Determine action
        self.state.action, self.state.action_reason = self._determine_action()

        # Record in history
        if self.state.drift_type != DriftType.NONE:
            self.state.drift_history.append(
                (self.state.timestamp, self.state.drift_type, self.state.drift_score)
            )

        return self.state

    def _check_setar_threshold(self, value: float) -> bool:
        """Check if value crosses SETAR thresholds (regime change)."""
        if len(self._value_window) < self.config.setar_window // 2:
            return False

        values = list(self._value_window)
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

        if std <= 0:
            return False

        z_score = (value - mean) / std

        return z_score < self.config.setar_threshold_low or z_score > self.config.setar_threshold_high

    def _check_error_spike(self) -> bool:
        """Check if prediction error has spiked above baseline."""
        if len(self._error_window) < 10 or len(self._baseline_errors) < 50:
            return False

        current_errors = list(self._error_window)
        baseline_errors = list(self._baseline_errors)

        current_mean = sum(current_errors) / len(current_errors)
        baseline_mean = sum(baseline_errors) / len(baseline_errors)

        self.state.current_error = current_mean
        self.state.baseline_error = baseline_mean
        self.state.error_ratio = current_mean / baseline_mean if baseline_mean > 0 else 1.0

        return self.state.error_ratio > self.config.error_threshold

    def _update_statistics(self):
        """Update current and baseline statistics."""
        if len(self._value_window) > 0:
            values = list(self._value_window)
            self.state.current_mean = sum(values) / len(values)
            self.state.current_std = (sum((v - self.state.current_mean) ** 2 for v in values) / len(values)) ** 0.5

    def _calculate_drift_score(self) -> float:
        """
        Calculate combined drift score from all detectors.

        Returns 0-1 score where higher = more likely drift.
        """
        scores = []
        weights = []

        # ADWIN contribution (weight: 0.3)
        if self.state.adwin_detected:
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(0.3)

        # Page-Hinkley contribution (weight: 0.3)
        if self.state.page_hinkley_detected:
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(0.3)

        # SETAR contribution (weight: 0.2)
        if self.state.setar_regime_change:
            scores.append(1.0)
        else:
            scores.append(0.0)
        weights.append(0.2)

        # Error spike contribution (weight: 0.2)
        if self.state.error_spike_detected:
            # Scale by how much error increased
            error_score = min(1.0, (self.state.error_ratio - 1) / (self.config.error_threshold - 1))
            scores.append(error_score)
        else:
            scores.append(0.0)
        weights.append(0.2)

        # Weighted average
        total_weight = sum(weights)
        drift_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return drift_score

    def _classify_drift(self) -> DriftType:
        """Classify drift level based on score and detector agreement."""
        score = self.state.drift_score

        # Count detectors that triggered
        detectors_triggered = sum([
            self.state.adwin_detected,
            self.state.page_hinkley_detected,
            self.state.setar_regime_change,
            self.state.error_spike_detected,
        ])

        if score >= self.config.severe_threshold or detectors_triggered >= 3:
            return DriftType.SEVERE
        elif score >= self.config.drift_threshold or detectors_triggered >= 2:
            return DriftType.DRIFT
        elif score >= self.config.warning_threshold or detectors_triggered >= 1:
            return DriftType.WARNING
        else:
            return DriftType.NONE

    def _determine_action(self) -> Tuple[DriftAction, str]:
        """Determine recommended action based on drift status."""
        if self.state.drift_type == DriftType.SEVERE:
            self.state.cooldown_remaining = self.config.cooldown_periods
            return DriftAction.HALT_MODEL, "Severe drift detected - multiple detectors triggered"

        elif self.state.drift_type == DriftType.DRIFT:
            self.state.cooldown_remaining = self.config.cooldown_periods // 2
            if self.state.error_spike_detected:
                return DriftAction.RETRAIN, "Confirmed drift with prediction error spike"
            return DriftAction.REDUCE_WEIGHT, "Confirmed drift - reduce model weight"

        elif self.state.drift_type == DriftType.WARNING:
            return DriftAction.ALERT, "Early drift warning - monitor closely"

        return DriftAction.CONTINUE, "No significant drift detected"

    def get_model_weight_adjustment(self) -> float:
        """
        Get suggested weight adjustment factor for the model.

        Returns multiplier (0-1) to apply to model weight in signal aggregation.
        """
        if self.state.drift_type == DriftType.SEVERE:
            return 0.0  # Don't use this model
        elif self.state.drift_type == DriftType.DRIFT:
            return 0.3  # Heavily reduce weight
        elif self.state.drift_type == DriftType.WARNING:
            return 0.7  # Slightly reduce weight
        else:
            return 1.0  # Full weight

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        return self.state.action == DriftAction.RETRAIN or self.state.action == DriftAction.HALT_MODEL

    def reset(self):
        """Reset all detectors."""
        self.adwin.reset()
        self.page_hinkley = PageHinkley(
            delta=self.config.ph_delta,
            threshold=self.config.ph_lambda,
            alpha=self.config.ph_alpha,
        )
        self._value_window.clear()
        self._error_window.clear()
        self._baseline_errors.clear()
        self.state = DriftState()

    def get_status(self) -> Dict:
        """Get current drift detection status."""
        return {
            "drift_type": self.state.drift_type.value,
            "drift_score": self.state.drift_score,
            "action": self.state.action.value,
            "action_reason": self.state.action_reason,
            "detectors": {
                "adwin": self.state.adwin_detected,
                "page_hinkley": self.state.page_hinkley_detected,
                "setar": self.state.setar_regime_change,
                "error_spike": self.state.error_spike_detected,
            },
            "error_ratio": self.state.error_ratio,
            "model_weight_adjustment": self.get_model_weight_adjustment(),
            "should_retrain": self.should_retrain(),
        }
