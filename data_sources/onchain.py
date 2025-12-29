"""
On-Chain Analytics for Cryptocurrency Trading

This module provides on-chain metrics for crypto trading signals:
- MVRV Z-Score: Market Value to Realized Value ratio
- SOPR: Spent Output Profit Ratio
- Exchange Net Flow: Coins entering/leaving exchanges
- Whale Ratio: Large transaction dominance

Data sources: Glassnode API, CryptoQuant API (requires API keys)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain metrics for a cryptocurrency."""
    timestamp: datetime
    symbol: str  # e.g., "BTC", "ETH"

    # Valuation metrics
    mvrv_ratio: Optional[float] = None  # Market Value / Realized Value
    mvrv_zscore: Optional[float] = None  # Z-score of MVRV
    nvt_ratio: Optional[float] = None  # Network Value to Transactions

    # Profit/Loss metrics
    sopr: Optional[float] = None  # Spent Output Profit Ratio
    nupl: Optional[float] = None  # Net Unrealized Profit/Loss

    # Exchange flow metrics
    exchange_net_flow: Optional[float] = None  # Positive = inflow (bearish)
    exchange_reserve: Optional[float] = None  # Total on exchanges

    # Whale activity
    whale_ratio: Optional[float] = None  # Large txn % of total
    whale_transaction_count: Optional[int] = None

    # Mining metrics
    hash_rate: Optional[float] = None
    mining_difficulty: Optional[float] = None
    puell_multiple: Optional[float] = None  # Miner revenue ratio

    # Network activity
    active_addresses: Optional[int] = None
    new_addresses: Optional[int] = None
    transaction_count: Optional[int] = None

    def get_signal(self) -> Dict[str, str]:
        """Get trading signals based on on-chain metrics."""
        signals = {}

        # MVRV Z-Score signals
        if self.mvrv_zscore is not None:
            if self.mvrv_zscore < 0.8:
                signals["mvrv"] = "BUY"  # Undervalued
            elif self.mvrv_zscore > 3.0:
                signals["mvrv"] = "SELL"  # Overvalued
            else:
                signals["mvrv"] = "HOLD"

        # SOPR signals
        if self.sopr is not None:
            if self.sopr < 1.0:
                signals["sopr"] = "BUY"  # Selling at loss, capitulation
            elif self.sopr > 1.1:
                signals["sopr"] = "SELL"  # Taking profits
            else:
                signals["sopr"] = "HOLD"

        # Exchange flow signals
        if self.exchange_net_flow is not None:
            if self.exchange_net_flow < -1000:  # Outflow
                signals["exchange_flow"] = "BUY"  # Coins leaving exchanges
            elif self.exchange_net_flow > 1000:  # Inflow
                signals["exchange_flow"] = "SELL"  # Coins entering exchanges
            else:
                signals["exchange_flow"] = "HOLD"

        # Whale activity signals
        if self.whale_ratio is not None:
            if self.whale_ratio > 0.478:
                signals["whale"] = "ALERT"  # High whale activity

        return signals


class OnChainAnalytics:
    """
    Provider for on-chain cryptocurrency analytics.

    Supports multiple data sources:
    - Glassnode (comprehensive on-chain data)
    - CryptoQuant (exchange flows, miner data)
    """

    def __init__(
        self,
        glassnode_api_key: Optional[str] = None,
        cryptoquant_api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,  # 5 minute cache
    ):
        self.glassnode_api_key = glassnode_api_key or os.environ.get("GLASSNODE_API_KEY")
        self.cryptoquant_api_key = cryptoquant_api_key or os.environ.get("CRYPTOQUANT_API_KEY")
        self.cache_ttl = cache_ttl_seconds

        # Simple cache
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)

        # Supported assets
        self.supported_assets = ["BTC", "ETH", "SOL", "AVAX"]

    def get_metrics(self, symbol: str) -> OnChainMetrics:
        """
        Get current on-chain metrics for a cryptocurrency.

        Args:
            symbol: Cryptocurrency symbol (e.g., "BTC", "ETH")

        Returns:
            OnChainMetrics with available data
        """
        symbol = symbol.upper()

        # Check cache
        cache_key = f"metrics_{symbol}"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl):
                return data

        metrics = OnChainMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
        )

        # Try Glassnode first
        if self.glassnode_api_key:
            glassnode_data = self._fetch_glassnode(symbol)
            if glassnode_data:
                self._populate_from_glassnode(metrics, glassnode_data)

        # Try CryptoQuant for additional/fallback data
        if self.cryptoquant_api_key:
            cryptoquant_data = self._fetch_cryptoquant(symbol)
            if cryptoquant_data:
                self._populate_from_cryptoquant(metrics, cryptoquant_data)

        # If no API keys, use simulated data for testing
        if not self.glassnode_api_key and not self.cryptoquant_api_key:
            metrics = self._get_simulated_metrics(symbol)

        # Cache result
        self._cache[cache_key] = (metrics, datetime.now())

        return metrics

    def _fetch_glassnode(self, symbol: str) -> Optional[Dict]:
        """Fetch data from Glassnode API."""
        if not self.glassnode_api_key:
            return None

        try:
            import urllib.request

            base_url = "https://api.glassnode.com/v1/metrics"

            # Fetch multiple metrics
            endpoints = {
                "mvrv": f"{base_url}/market/mvrv",
                "sopr": f"{base_url}/indicators/sopr",
                "exchange_balance": f"{base_url}/distribution/balance_exchanges",
                "active_addresses": f"{base_url}/addresses/active_count",
            }

            results = {}
            for metric_name, url in endpoints.items():
                full_url = f"{url}?a={symbol}&api_key={self.glassnode_api_key}"
                try:
                    req = urllib.request.Request(full_url)
                    with urllib.request.urlopen(req, timeout=10) as response:
                        data = json.loads(response.read().decode())
                        if data:
                            results[metric_name] = data[-1]["v"] if isinstance(data, list) else data
                except Exception as e:
                    logger.debug(f"Glassnode {metric_name} fetch failed: {e}")

            return results if results else None

        except Exception as e:
            logger.error(f"Glassnode API error: {e}")
            return None

    def _fetch_cryptoquant(self, symbol: str) -> Optional[Dict]:
        """Fetch data from CryptoQuant API."""
        if not self.cryptoquant_api_key:
            return None

        try:
            import urllib.request

            base_url = "https://api.cryptoquant.com/v1"

            # Fetch exchange flow data
            url = f"{base_url}/{symbol.lower()}/exchange-flows/netflow"
            headers = {"Authorization": f"Bearer {self.cryptoquant_api_key}"}

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data

        except Exception as e:
            logger.error(f"CryptoQuant API error: {e}")
            return None

    def _populate_from_glassnode(self, metrics: OnChainMetrics, data: Dict) -> None:
        """Populate metrics from Glassnode data."""
        if "mvrv" in data:
            metrics.mvrv_ratio = data["mvrv"]
        if "sopr" in data:
            metrics.sopr = data["sopr"]
        if "exchange_balance" in data:
            metrics.exchange_reserve = data["exchange_balance"]
        if "active_addresses" in data:
            metrics.active_addresses = int(data["active_addresses"])

    def _populate_from_cryptoquant(self, metrics: OnChainMetrics, data: Dict) -> None:
        """Populate metrics from CryptoQuant data."""
        if "netflow" in data:
            metrics.exchange_net_flow = data["netflow"]
        if "whale_ratio" in data:
            metrics.whale_ratio = data["whale_ratio"]

    def _get_simulated_metrics(self, symbol: str) -> OnChainMetrics:
        """
        Generate simulated metrics for testing.

        In production, this would be replaced with actual API calls.
        """
        import random

        # Seed based on current hour for consistency within the hour
        hour_seed = int(datetime.now().timestamp() // 3600)
        random.seed(hour_seed + hash(symbol))

        return OnChainMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            mvrv_ratio=random.uniform(0.5, 4.0),
            mvrv_zscore=random.uniform(-1.0, 4.0),
            nvt_ratio=random.uniform(10, 100),
            sopr=random.uniform(0.9, 1.1),
            nupl=random.uniform(-0.5, 0.8),
            exchange_net_flow=random.uniform(-5000, 5000),
            exchange_reserve=random.uniform(2000000, 3000000),
            whale_ratio=random.uniform(0.3, 0.6),
            whale_transaction_count=random.randint(500, 2000),
            active_addresses=random.randint(500000, 1500000),
            new_addresses=random.randint(10000, 50000),
            transaction_count=random.randint(200000, 500000),
        )

    def get_historical_metrics(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[OnChainMetrics]:
        """Get historical on-chain metrics for backtesting."""
        # In production, this would fetch historical data from APIs
        # For now, return simulated data
        metrics = []
        current = start_date

        while current <= end_date:
            m = self._get_simulated_metrics(symbol)
            m.timestamp = current
            metrics.append(m)
            current += timedelta(days=1)

        return metrics

    def is_available(self) -> bool:
        """Check if on-chain data is available."""
        return bool(self.glassnode_api_key or self.cryptoquant_api_key)
