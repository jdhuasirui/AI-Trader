"""
Regime Detector for AI-Trader

This module implements market regime detection using:
- Hidden Markov Model (HMM) with 3-4 states
- Features: ADX, ATR percentile, volume ratio, price momentum
- Regime classification: Trending (up/down), Ranging, Volatile

The regime affects signal fusion weights and risk parameters.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

from .data_structures import MarketState, Regime

logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    # Feature thresholds
    adx_trending_threshold: float = 25.0  # ADX > 25 = trending
    adx_ranging_threshold: float = 20.0  # ADX < 20 = ranging
    atr_volatile_percentile: float = 80.0  # ATR > 80th percentile = volatile
    volume_spike_threshold: float = 2.0  # Volume > 2x average = spike

    # HMM parameters
    n_regimes: int = 4  # Number of hidden states
    lookback_window: int = 60  # Days of history for feature calculation
    update_frequency: int = 1  # Update regime every N bars

    # Smoothing
    regime_persistence: float = 0.7  # Probability of staying in same regime
    min_regime_duration: int = 5  # Minimum bars before regime change


@dataclass
class RegimeFeatures:
    """Extracted features for regime detection."""
    timestamp: datetime
    symbol: str

    # Trend indicators
    adx: float = 0.0  # Average Directional Index
    plus_di: float = 0.0  # +DI
    minus_di: float = 0.0  # -DI
    trend_strength: float = 0.0  # Derived from DI difference

    # Volatility indicators
    atr: float = 0.0  # Average True Range
    atr_percentile: float = 0.0  # ATR relative to history
    realized_vol: float = 0.0  # Historical volatility

    # Volume indicators
    volume_ratio: float = 0.0  # Current volume / average volume
    volume_trend: float = 0.0  # Volume momentum

    # Momentum indicators
    price_momentum_5d: float = 0.0  # 5-day return
    price_momentum_20d: float = 0.0  # 20-day return
    rsi: float = 50.0  # RSI for momentum context


class SimpleHMM:
    """
    Simple Hidden Markov Model implementation.

    Uses Gaussian emissions and fixed transition probabilities.
    This is a lightweight alternative to hmmlearn for environments
    where that package isn't available.
    """

    def __init__(self, n_states: int = 4, persistence: float = 0.7):
        self.n_states = n_states
        self.persistence = persistence

        # Initialize transition matrix (favor staying in same state)
        self.transition_matrix = self._init_transition_matrix()

        # State means and variances (learned from data)
        self.state_means: Dict[int, List[float]] = {}
        self.state_vars: Dict[int, List[float]] = {}

        # Current state probabilities
        self.state_probs = [1.0 / n_states] * n_states

    def _init_transition_matrix(self) -> List[List[float]]:
        """Initialize transition matrix with persistence bias."""
        off_diag = (1 - self.persistence) / (self.n_states - 1)
        matrix = []
        for i in range(self.n_states):
            row = []
            for j in range(self.n_states):
                row.append(self.persistence if i == j else off_diag)
            matrix.append(row)
        return matrix

    def fit(self, features: List[List[float]]) -> None:
        """
        Fit HMM parameters using simple clustering.

        Args:
            features: List of feature vectors
        """
        if len(features) < self.n_states * 2:
            logger.warning("Insufficient data for HMM fitting")
            return

        # Simple k-means style clustering
        # Sort by first feature (typically ADX) and divide into states
        indexed_features = [(i, f[0]) for i, f in enumerate(features)]
        sorted_by_first = sorted(indexed_features, key=lambda x: x[1])
        sorted_indices = [x[0] for x in sorted_by_first]

        chunk_size = len(features) // self.n_states
        n_features = len(features[0]) if features else 0

        for i in range(self.n_states):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.n_states - 1 else len(features)
            state_indices = sorted_indices[start:end]
            state_features = [features[idx] for idx in state_indices]

            # Calculate mean
            means = []
            vars = []
            for j in range(n_features):
                vals = [f[j] for f in state_features]
                mean = sum(vals) / len(vals) if vals else 0
                var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 1e-6
                means.append(mean)
                vars.append(var + 1e-6)  # Avoid zero variance

            self.state_means[i] = means
            self.state_vars[i] = vars

    def predict(self, features: List[float]) -> int:
        """
        Predict the most likely state for given features.

        Uses forward algorithm with current state probabilities.
        """
        if not self.state_means:
            # Not fitted yet, use simple rule-based detection
            return 0

        # Calculate emission probabilities for each state
        emissions = []
        for state in range(self.n_states):
            mean = self.state_means[state]
            var = self.state_vars[state]
            # Gaussian log-likelihood
            log_prob = 0
            for j in range(len(features)):
                log_prob += -0.5 * (((features[j] - mean[j]) ** 2) / var[j] + math.log(2 * math.pi * var[j]))
            emissions.append(math.exp(log_prob))

        # Normalize
        emissions_sum = sum(emissions) + 1e-10
        emissions = [e / emissions_sum for e in emissions]

        # Update state probabilities using transition matrix
        new_probs = []
        for j in range(self.n_states):
            prob = sum(self.transition_matrix[i][j] * self.state_probs[i] for i in range(self.n_states))
            new_probs.append(prob * emissions[j])

        # Normalize
        probs_sum = sum(new_probs) + 1e-10
        self.state_probs = [p / probs_sum for p in new_probs]

        # Return argmax
        max_idx = 0
        max_val = self.state_probs[0]
        for i, v in enumerate(self.state_probs):
            if v > max_val:
                max_val = v
                max_idx = i
        return max_idx

    def get_state_probabilities(self) -> List[float]:
        """Get current state probabilities."""
        return self.state_probs.copy()


class RegimeDetector:
    """
    Market regime detector using HMM and technical indicators.

    Classifies market into regimes:
    - TRENDING_UP: Strong upward trend (ADX > 25, +DI > -DI)
    - TRENDING_DOWN: Strong downward trend (ADX > 25, -DI > +DI)
    - RANGING: No clear trend (ADX < 20)
    - VOLATILE: High volatility regime (ATR > historical norm)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.hmm = SimpleHMM(
            n_states=self.config.n_regimes,
            persistence=self.config.regime_persistence,
        )

        # Historical data for feature calculation
        self._price_history: Dict[str, deque] = {}  # symbol -> prices
        self._volume_history: Dict[str, deque] = {}
        self._atr_history: Dict[str, deque] = {}

        # Current regime state
        self._current_regime: Dict[str, Regime] = {}
        self._regime_duration: Dict[str, int] = {}
        self._last_update: Dict[str, datetime] = {}

        # Feature history for HMM fitting
        self._feature_history: List[List[float]] = []

    def update(self, market_state: MarketState) -> Regime:
        """
        Update regime detection with new market data.

        Args:
            market_state: Current market state

        Returns:
            Detected regime
        """
        symbol = market_state.symbol

        # Initialize history if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.config.lookback_window)
            self._volume_history[symbol] = deque(maxlen=self.config.lookback_window)
            self._atr_history[symbol] = deque(maxlen=self.config.lookback_window)
            self._current_regime[symbol] = Regime.UNKNOWN
            self._regime_duration[symbol] = 0

        # Add current data to history
        self._price_history[symbol].append({
            "open": market_state.open,
            "high": market_state.high,
            "low": market_state.low,
            "close": market_state.close,
        })
        self._volume_history[symbol].append(market_state.volume)

        # Calculate ATR if we have enough data
        if len(self._price_history[symbol]) >= 2:
            atr = self._calculate_atr(symbol)
            self._atr_history[symbol].append(atr)

        # Need enough history for feature extraction
        if len(self._price_history[symbol]) < 20:
            return Regime.UNKNOWN

        # Extract features
        features = self._extract_features(symbol, market_state)

        # Detect regime
        regime = self._detect_regime(symbol, features, market_state)

        # Update state
        self._current_regime[symbol] = regime
        self._last_update[symbol] = market_state.timestamp

        return regime

    def detect(self, market_state: MarketState) -> Regime:
        """
        Detect current regime (alias for update).

        This is the main entry point for regime detection.
        """
        return self.update(market_state)

    def get_current_regime(self, symbol: str) -> Regime:
        """Get the current regime for a symbol."""
        return self._current_regime.get(symbol, Regime.UNKNOWN)

    def get_regime_probabilities(self, symbol: str) -> Dict[Regime, float]:
        """Get probability distribution over regimes."""
        probs = self.hmm.get_state_probabilities()

        # Map HMM states to regimes
        regime_map = {
            0: Regime.RANGING,
            1: Regime.TRENDING_UP,
            2: Regime.TRENDING_DOWN,
            3: Regime.VOLATILE,
        }

        return {regime_map.get(i, Regime.UNKNOWN): float(probs[i]) for i in range(len(probs))}

    def _extract_features(self, symbol: str, market_state: MarketState) -> RegimeFeatures:
        """Extract features for regime detection."""
        features = RegimeFeatures(
            timestamp=market_state.timestamp,
            symbol=symbol,
        )

        prices = list(self._price_history[symbol])
        volumes = list(self._volume_history[symbol])
        atrs = list(self._atr_history[symbol])

        # Get technical indicators from market state if available
        indicators = market_state.technical_indicators

        # ADX and directional indicators
        features.adx = indicators.get("ADX", self._calculate_adx(prices))
        features.plus_di = indicators.get("+DI", 0)
        features.minus_di = indicators.get("-DI", 0)
        features.trend_strength = abs(features.plus_di - features.minus_di)

        # ATR and volatility
        if atrs:
            features.atr = atrs[-1]
            features.atr_percentile = self._calculate_percentile(atrs[-1], atrs)
        features.realized_vol = self._calculate_volatility(prices)

        # Volume
        if volumes:
            avg_volume = sum(volumes) / len(volumes)
            features.volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
            if len(volumes) >= 5:
                recent_avg = sum(list(volumes)[-5:]) / 5
                features.volume_trend = recent_avg / avg_volume if avg_volume > 0 else 1.0

        # Momentum
        closes = [p["close"] for p in prices]
        if len(closes) >= 5:
            features.price_momentum_5d = (closes[-1] / closes[-5] - 1) * 100
        if len(closes) >= 20:
            features.price_momentum_20d = (closes[-1] / closes[-20] - 1) * 100

        # RSI
        features.rsi = indicators.get("RSI", self._calculate_rsi(closes))

        return features

    def _detect_regime(
        self,
        symbol: str,
        features: RegimeFeatures,
        market_state: MarketState,
    ) -> Regime:
        """Determine regime from features."""

        # Rule-based detection first
        rule_regime = self._rule_based_detection(features)

        # HMM-based detection
        feature_vector = [
            features.adx,
            features.atr_percentile,
            features.volume_ratio,
            features.price_momentum_20d,
            features.rsi - 50,  # Center around 0
        ]

        # Add to history for fitting
        self._feature_history.append(feature_vector)

        # Fit HMM periodically
        if len(self._feature_history) >= 100 and len(self._feature_history) % 20 == 0:
            self.hmm.fit(self._feature_history[-200:])

        # Get HMM prediction if fitted
        hmm_regime = Regime.UNKNOWN
        if self.hmm.state_means:
            hmm_state = self.hmm.predict(feature_vector)
            hmm_regime = self._map_hmm_state_to_regime(hmm_state, features)

        # Combine rule-based and HMM
        # Prefer rule-based for clear signals, HMM for ambiguous cases
        if rule_regime != Regime.UNKNOWN:
            final_regime = rule_regime
        else:
            final_regime = hmm_regime

        # Apply regime persistence (minimum duration)
        current = self._current_regime.get(symbol, Regime.UNKNOWN)
        duration = self._regime_duration.get(symbol, 0)

        if final_regime != current:
            if duration < self.config.min_regime_duration:
                # Stay in current regime
                self._regime_duration[symbol] = duration + 1
                return current
            else:
                # Switch to new regime
                self._regime_duration[symbol] = 0
                return final_regime
        else:
            self._regime_duration[symbol] = duration + 1
            return final_regime

    def _rule_based_detection(self, features: RegimeFeatures) -> Regime:
        """Simple rule-based regime detection."""

        # Volatile regime takes priority
        if features.atr_percentile > self.config.atr_volatile_percentile:
            return Regime.VOLATILE

        # Trending regimes
        if features.adx > self.config.adx_trending_threshold:
            if features.plus_di > features.minus_di:
                return Regime.TRENDING_UP
            else:
                return Regime.TRENDING_DOWN

        # Ranging regime
        if features.adx < self.config.adx_ranging_threshold:
            return Regime.RANGING

        # Ambiguous - let HMM decide
        return Regime.UNKNOWN

    def _map_hmm_state_to_regime(self, state: int, features: RegimeFeatures) -> Regime:
        """Map HMM state index to Regime enum."""
        # Use feature context to determine regime meaning
        if state == 0:
            return Regime.RANGING
        elif state == 1:
            return Regime.TRENDING_UP if features.price_momentum_20d > 0 else Regime.TRENDING_DOWN
        elif state == 2:
            return Regime.TRENDING_DOWN if features.price_momentum_20d < 0 else Regime.TRENDING_UP
        else:
            return Regime.VOLATILE

    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range."""
        prices = list(self._price_history[symbol])
        if len(prices) < 2:
            return 0

        true_ranges = []
        for i in range(1, min(len(prices), period + 1)):
            high = prices[-i]["high"]
            low = prices[-i]["low"]
            prev_close = prices[-i - 1]["close"]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0

    def _calculate_adx(self, prices: List[dict], period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        if len(prices) < period + 1:
            return 0

        # Simplified ADX calculation
        plus_dm = []
        minus_dm = []

        for i in range(1, len(prices)):
            high_diff = prices[i]["high"] - prices[i - 1]["high"]
            low_diff = prices[i - 1]["low"] - prices[i]["low"]

            plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
            minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)

        if len(plus_dm) < period:
            return 0

        # Smoothed averages
        plus_di = sum(plus_dm[-period:]) / period
        minus_di = sum(minus_dm[-period:]) / period

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0

        dx = abs(plus_di - minus_di) / di_sum * 100
        return dx  # Simplified - should be smoothed ADX

    def _calculate_volatility(self, prices: List[dict], period: int = 20) -> float:
        """Calculate realized volatility (annualized)."""
        if len(prices) < period + 1:
            return 0

        closes = [p["close"] for p in prices[-(period + 1):]]
        returns = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))]

        if not returns:
            return 0

        std = (sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / len(returns)) ** 0.5
        return std * (252 ** 0.5) * 100  # Annualized percentage

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50

        gains = []
        losses = []

        for i in range(1, min(len(closes), period + 1)):
            change = closes[-i] - closes[-i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_percentile(self, value: float, history: List[float]) -> float:
        """Calculate percentile of value in history."""
        if not history:
            return 50

        below = sum(1 for h in history if h < value)
        return (below / len(history)) * 100
