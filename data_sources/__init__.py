"""
Data Sources for AI-Trader

This module provides alternative data integrations:
- On-chain analytics (crypto)
- Sentiment analysis
- Macro economic factors
"""

from .onchain import OnChainAnalytics, OnChainMetrics
from .sentiment import SentimentAnalyzer, SentimentResult
from .macro import MacroDataProvider, MacroFactors

__all__ = [
    "OnChainAnalytics",
    "OnChainMetrics",
    "SentimentAnalyzer",
    "SentimentResult",
    "MacroDataProvider",
    "MacroFactors",
]
