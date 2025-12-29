"""
Macro Economic Data Provider for AI-Trader

This module provides macro economic factors for risk appetite adjustment:
- DXY (Dollar Index) - inverse correlation with risk assets
- VIX - volatility/fear index
- Fed Funds Rate expectations
- Treasury yields
- Economic indicators (GDP, CPI, etc.)

Used to set overall risk appetite in regime detection.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class MacroFactors:
    """Macro economic factors."""
    timestamp: datetime

    # Currency
    dxy: Optional[float] = None  # Dollar Index
    dxy_change_1d: Optional[float] = None  # 1-day change %
    dxy_change_1w: Optional[float] = None  # 1-week change %

    # Volatility
    vix: Optional[float] = None  # CBOE Volatility Index
    vix_term_structure: Optional[float] = None  # VIX contango/backwardation

    # Interest rates
    fed_funds_rate: Optional[float] = None
    fed_funds_expectation_3m: Optional[float] = None  # Expected in 3 months
    us_10y_yield: Optional[float] = None
    us_2y_yield: Optional[float] = None
    yield_curve: Optional[float] = None  # 10Y - 2Y spread

    # Economic indicators
    cpi_yoy: Optional[float] = None  # CPI year-over-year
    pce_yoy: Optional[float] = None  # PCE year-over-year
    gdp_growth: Optional[float] = None  # GDP growth rate
    unemployment_rate: Optional[float] = None
    nfp_change: Optional[float] = None  # Non-farm payrolls

    # Commodity prices
    gold_price: Optional[float] = None
    oil_price: Optional[float] = None

    # Crypto-specific
    stablecoin_supply: Optional[float] = None  # Total stablecoin market cap

    def get_risk_appetite(self) -> float:
        """
        Calculate overall risk appetite based on macro factors.

        Returns:
            Risk appetite score from 0 (risk-off) to 1 (risk-on)
        """
        scores = []
        weights = []

        # VIX (lower is risk-on)
        if self.vix is not None:
            if self.vix < 15:
                vix_score = 0.9
            elif self.vix < 20:
                vix_score = 0.7
            elif self.vix < 25:
                vix_score = 0.5
            elif self.vix < 30:
                vix_score = 0.3
            else:
                vix_score = 0.1
            scores.append(vix_score)
            weights.append(2.0)  # Higher weight for VIX

        # DXY (lower is risk-on for crypto/stocks)
        if self.dxy is not None and self.dxy_change_1d is not None:
            if self.dxy_change_1d < -0.5:
                dxy_score = 0.8
            elif self.dxy_change_1d < 0:
                dxy_score = 0.6
            elif self.dxy_change_1d < 0.5:
                dxy_score = 0.4
            else:
                dxy_score = 0.2
            scores.append(dxy_score)
            weights.append(1.5)

        # Yield curve (inverted is risk-off)
        if self.yield_curve is not None:
            if self.yield_curve > 0.5:
                curve_score = 0.8
            elif self.yield_curve > 0:
                curve_score = 0.6
            elif self.yield_curve > -0.5:
                curve_score = 0.4
            else:
                curve_score = 0.2
            scores.append(curve_score)
            weights.append(1.0)

        if not scores:
            return 0.5  # Default neutral

        # Weighted average
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight


class MacroDataProvider:
    """
    Provider for macro economic data.

    Data sources:
    - Federal Reserve FRED API
    - Alpha Vantage (for forex, commodities)
    - Yahoo Finance (for VIX, indices)
    """

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        cache_ttl_seconds: int = 3600,  # 1 hour cache
    ):
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        self.alpha_vantage_key = alpha_vantage_key or os.environ.get("ALPHAADVANTAGE_API_KEY")
        self.cache_ttl = cache_ttl_seconds

        # Cache
        self._cache: Dict[str, tuple] = {}

        # FRED series IDs
        self._fred_series = {
            "fed_funds_rate": "FEDFUNDS",
            "us_10y_yield": "DGS10",
            "us_2y_yield": "DGS2",
            "cpi_yoy": "CPIAUCSL",
            "pce_yoy": "PCEPI",
            "gdp_growth": "A191RL1Q225SBEA",
            "unemployment_rate": "UNRATE",
        }

    def get_factors(self) -> MacroFactors:
        """
        Get current macro economic factors.

        Returns:
            MacroFactors with available data
        """
        # Check cache
        cache_key = "macro_factors"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl):
                return data

        factors = MacroFactors(timestamp=datetime.now())

        # Fetch from FRED
        if self.fred_api_key:
            self._fetch_fred_data(factors)

        # Fetch from Alpha Vantage
        if self.alpha_vantage_key:
            self._fetch_alpha_vantage_data(factors)

        # If no API keys, use simulated data
        if not self.fred_api_key and not self.alpha_vantage_key:
            factors = self._get_simulated_factors()

        # Calculate derived metrics
        if factors.us_10y_yield and factors.us_2y_yield:
            factors.yield_curve = factors.us_10y_yield - factors.us_2y_yield

        # Cache result
        self._cache[cache_key] = (factors, datetime.now())

        return factors

    def _fetch_fred_data(self, factors: MacroFactors) -> None:
        """Fetch data from FRED API."""
        try:
            import urllib.request

            base_url = "https://api.stlouisfed.org/fred/series/observations"

            for attr, series_id in self._fred_series.items():
                try:
                    url = (
                        f"{base_url}?series_id={series_id}"
                        f"&api_key={self.fred_api_key}"
                        "&file_type=json&sort_order=desc&limit=1"
                    )

                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode())

                        if "observations" in data and data["observations"]:
                            value = data["observations"][0]["value"]
                            if value != ".":
                                setattr(factors, attr, float(value))

                except Exception as e:
                    logger.debug(f"FRED {series_id} fetch failed: {e}")

        except Exception as e:
            logger.error(f"FRED API error: {e}")

    def _fetch_alpha_vantage_data(self, factors: MacroFactors) -> None:
        """Fetch forex and commodity data from Alpha Vantage."""
        try:
            import urllib.request

            # Fetch DXY (via EUR/USD inverse proxy)
            try:
                url = (
                    f"https://www.alphavantage.co/query?"
                    f"function=CURRENCY_EXCHANGE_RATE&from_currency=EUR&to_currency=USD"
                    f"&apikey={self.alpha_vantage_key}"
                )
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    if "Realtime Currency Exchange Rate" in data:
                        rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                        # Approximate DXY from EUR/USD (EUR is ~57% of DXY)
                        factors.dxy = 100 / rate * 0.57 + 50 * 0.43

            except Exception as e:
                logger.debug(f"DXY fetch failed: {e}")

            # Fetch VIX
            try:
                url = (
                    f"https://www.alphavantage.co/query?"
                    f"function=GLOBAL_QUOTE&symbol=VIX"
                    f"&apikey={self.alpha_vantage_key}"
                )
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    if "Global Quote" in data and "05. price" in data["Global Quote"]:
                        factors.vix = float(data["Global Quote"]["05. price"])

            except Exception as e:
                logger.debug(f"VIX fetch failed: {e}")

            # Fetch Gold
            try:
                url = (
                    f"https://www.alphavantage.co/query?"
                    f"function=GLOBAL_QUOTE&symbol=GLD"
                    f"&apikey={self.alpha_vantage_key}"
                )
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())

                    if "Global Quote" in data and "05. price" in data["Global Quote"]:
                        # GLD is roughly 1/10 of gold price
                        factors.gold_price = float(data["Global Quote"]["05. price"]) * 10

            except Exception as e:
                logger.debug(f"Gold fetch failed: {e}")

        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")

    def _get_simulated_factors(self) -> MacroFactors:
        """Generate simulated macro factors for testing."""
        import random

        # Seed for consistency within the hour
        random.seed(int(datetime.now().timestamp() // 3600))

        return MacroFactors(
            timestamp=datetime.now(),
            dxy=random.uniform(100, 110),
            dxy_change_1d=random.uniform(-1, 1),
            dxy_change_1w=random.uniform(-2, 2),
            vix=random.uniform(12, 35),
            fed_funds_rate=random.uniform(4.5, 5.5),
            us_10y_yield=random.uniform(4.0, 5.0),
            us_2y_yield=random.uniform(4.0, 5.0),
            cpi_yoy=random.uniform(2.5, 4.0),
            unemployment_rate=random.uniform(3.5, 4.5),
            gold_price=random.uniform(1800, 2200),
            oil_price=random.uniform(70, 90),
        )

    def get_historical_factors(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[MacroFactors]:
        """Get historical macro factors for backtesting."""
        # In production, this would fetch from FRED historical data
        # For now, return simulated data
        factors = []
        current = start_date

        while current <= end_date:
            f = self._get_simulated_factors()
            f.timestamp = current
            factors.append(f)
            current += timedelta(days=1)

        return factors

    def get_risk_environment(self) -> str:
        """
        Get current risk environment assessment.

        Returns:
            One of: "RISK_ON", "RISK_OFF", "NEUTRAL"
        """
        factors = self.get_factors()
        appetite = factors.get_risk_appetite()

        if appetite > 0.65:
            return "RISK_ON"
        elif appetite < 0.35:
            return "RISK_OFF"
        else:
            return "NEUTRAL"

    def is_available(self) -> bool:
        """Check if macro data is available."""
        return bool(self.fred_api_key or self.alpha_vantage_key)
