"""
Cache Module for AI Trader

Provides a unified caching layer with TTL support for API data.
"""

from .cache_manager import CacheManager, get_cache_manager, cache_with_ttl

__all__ = ["CacheManager", "get_cache_manager", "cache_with_ttl"]
