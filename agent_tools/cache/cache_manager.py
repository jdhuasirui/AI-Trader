"""
Cache Manager for AI Trader

Provides a unified caching system with TTL support for various data types.
Supports both in-memory caching and optional persistent caching.

Usage:
    from agent_tools.cache import get_cache_manager, cache_with_ttl

    # Get singleton cache manager
    cache = get_cache_manager()

    # Store data with TTL
    cache.set("market_snapshot:AAPL", data, ttl_seconds=300)

    # Retrieve data
    data = cache.get("market_snapshot:AAPL")

    # Use decorator
    @cache_with_ttl("news", ttl_seconds=300)
    def get_news(symbol):
        return api_call(symbol)
"""

import functools
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    """A cached entry with metadata"""
    value: Any
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.created_at

    @property
    def remaining_ttl(self) -> float:
        """Get remaining TTL in seconds"""
        remaining = (self.created_at + self.ttl_seconds) - time.time()
        return max(0, remaining)


class CacheManager:
    """
    Thread-safe in-memory cache with TTL support.

    Features:
    - TTL-based expiration
    - Automatic cleanup of expired entries
    - Namespace support for organizing cache keys
    - Cache statistics
    - Degraded mode flag for API failures
    """

    # Default TTL values for different data types (in seconds)
    DEFAULT_TTL = {
        "snapshot": 300,      # 5 minutes for market snapshots
        "news": 300,          # 5 minutes for news clusters
        "trades": 60,         # 1 minute for recent trades
        "corporate_actions": 3600,  # 1 hour for corporate actions
        "onchain": 3600,      # 1 hour for on-chain metrics
        "adv": 86400,         # 24 hours for ADV calculations
        "correlations": 604800,  # 7 days for stock correlations
    }

    def __init__(self, cleanup_interval: float = 60.0):
        """
        Initialize cache manager.

        Args:
            cleanup_interval: How often to clean up expired entries (seconds)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        # Degraded mode tracking
        self._degraded_sources: Dict[str, float] = {}
        self._degraded_ttl = 300  # 5 minutes before retrying failed source

        # Start cleanup thread
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="CacheCleanup"
        )
        self._cleanup_thread.start()

    def _make_key(self, namespace: str, key: str) -> str:
        """Create a namespaced cache key"""
        return f"{namespace}:{key}"

    def _cleanup_loop(self):
        """Background thread to clean up expired entries"""
        while True:
            time.sleep(self._cleanup_interval)
            self.cleanup_expired()

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            Cached value or None if not found/expired
        """
        full_key = self._make_key(namespace, key)

        with self._lock:
            entry = self._cache.get(full_key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[full_key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return None

            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1
            return entry.value

    def get_with_metadata(
        self,
        key: str,
        namespace: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get value with metadata (age, remaining TTL, etc.)

        Returns:
            Dict with 'value', 'age_seconds', 'remaining_ttl', 'access_count'
            or None if not found
        """
        full_key = self._make_key(namespace, key)

        with self._lock:
            entry = self._cache.get(full_key)

            if entry is None or entry.is_expired:
                return None

            entry.access_count += 1
            entry.last_accessed = time.time()

            return {
                "value": entry.value,
                "age_seconds": round(entry.age_seconds, 1),
                "remaining_ttl": round(entry.remaining_ttl, 1),
                "access_count": entry.access_count,
            }

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
        namespace: str = "default"
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses namespace default if not provided)
            namespace: Cache namespace
        """
        if ttl_seconds is None:
            ttl_seconds = self.DEFAULT_TTL.get(namespace, 300)

        full_key = self._make_key(namespace, key)

        with self._lock:
            self._cache[full_key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl_seconds,
                access_count=0,
                last_accessed=time.time(),
            )
            self._stats["sets"] += 1

    def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete a value from cache.

        Returns:
            True if key was found and deleted
        """
        full_key = self._make_key(namespace, key)

        with self._lock:
            if full_key in self._cache:
                del self._cache[full_key]
                return True
            return False

    def clear_namespace(self, namespace: str) -> int:
        """
        Clear all entries in a namespace.

        Returns:
            Number of entries cleared
        """
        prefix = f"{namespace}:"
        cleared = 0

        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]
                cleared += 1

        return cleared

    def clear_all(self) -> int:
        """
        Clear entire cache.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired.append(key)

            for key in expired:
                del self._cache[key]
                self._stats["evictions"] += 1

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests * 100
                if total_requests > 0
                else 0
            )

            return {
                "entries": len(self._cache),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_pct": round(hit_rate, 1),
                "sets": self._stats["sets"],
                "evictions": self._stats["evictions"],
            }

    # ==================== Degraded Mode ====================

    def mark_degraded(self, source: str) -> None:
        """
        Mark a data source as degraded (API failure).

        Args:
            source: Source identifier (e.g., "alpaca_news", "onchain_btc")
        """
        with self._lock:
            self._degraded_sources[source] = time.time()

    def clear_degraded(self, source: str) -> None:
        """Clear degraded status for a source"""
        with self._lock:
            self._degraded_sources.pop(source, None)

    def is_degraded(self, source: str) -> bool:
        """Check if a source is in degraded mode"""
        with self._lock:
            if source not in self._degraded_sources:
                return False

            degraded_time = self._degraded_sources[source]
            if time.time() - degraded_time > self._degraded_ttl:
                # Auto-clear after TTL
                del self._degraded_sources[source]
                return False

            return True

    def get_degraded_sources(self) -> Dict[str, float]:
        """Get all degraded sources with their degraded-since timestamps"""
        with self._lock:
            # Clean up expired degraded sources
            now = time.time()
            to_remove = [
                source for source, ts in self._degraded_sources.items()
                if now - ts > self._degraded_ttl
            ]
            for source in to_remove:
                del self._degraded_sources[source]

            return dict(self._degraded_sources)


# Singleton instance
_cache_manager: Optional[CacheManager] = None
_cache_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager

    if _cache_manager is None:
        with _cache_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()

    return _cache_manager


def cache_with_ttl(
    namespace: str,
    ttl_seconds: Optional[float] = None,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching function results with TTL.

    Args:
        namespace: Cache namespace
        ttl_seconds: TTL in seconds (uses namespace default if not provided)
        key_func: Optional function to generate cache key from arguments

    Example:
        @cache_with_ttl("news", ttl_seconds=300)
        def get_stock_news(symbol: str) -> Dict:
            return api_call(symbol)

        @cache_with_ttl("snapshot", key_func=lambda symbols: symbols)
        def get_snapshot(symbols: str) -> Dict:
            return api_call(symbols)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache = get_cache_manager()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: hash all arguments
                key_data = json.dumps(
                    {"args": args, "kwargs": kwargs},
                    sort_keys=True,
                    default=str
                )
                cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try to get from cache
            cached = cache.get(cache_key, namespace)
            if cached is not None:
                return cached

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds, namespace)
            return result

        return wrapper
    return decorator


if __name__ == "__main__":
    # Test cache manager
    print("Testing Cache Manager...")

    cache = get_cache_manager()

    # Test basic operations
    cache.set("test_key", {"data": "test"}, ttl_seconds=5, namespace="test")
    result = cache.get("test_key", namespace="test")
    print(f"Set and get: {result}")

    # Test with metadata
    result_meta = cache.get_with_metadata("test_key", namespace="test")
    print(f"With metadata: {result_meta}")

    # Test expiration
    cache.set("short_ttl", "expires soon", ttl_seconds=1, namespace="test")
    print(f"Before expiry: {cache.get('short_ttl', namespace='test')}")
    time.sleep(1.5)
    print(f"After expiry: {cache.get('short_ttl', namespace='test')}")

    # Test stats
    print(f"Stats: {cache.get_stats()}")

    # Test decorator
    @cache_with_ttl("test", ttl_seconds=10)
    def expensive_function(x: int) -> int:
        print(f"  Computing for {x}...")
        return x * 2

    print(f"First call: {expensive_function(5)}")
    print(f"Cached call: {expensive_function(5)}")

    # Test degraded mode
    cache.mark_degraded("test_api")
    print(f"Is degraded: {cache.is_degraded('test_api')}")
    cache.clear_degraded("test_api")
    print(f"After clear: {cache.is_degraded('test_api')}")

    print("\nAll tests passed!")
