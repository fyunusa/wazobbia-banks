import hashlib
import logging
from typing import Optional
import redis.asyncio as aioredis
from config.settings import settings

logger = logging.getLogger("store.redis_client")


class RedisStore:
    """Async wrapper for Redis cache operations using redis.asyncio."""

    def __init__(self) -> None:
        self.client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,  # Automatically decode responses to strings
        )

    async def get_cached_query(self, cache_key: str) -> Optional[str]:
        """Retrieves a cached value by its key. Returns None if not found/expired."""
        try:
            val = await self.client.get(cache_key)
            if val:
                logger.info(f"Cache hit for key: {cache_key}")
            return val
        except Exception as e:
            logger.error(f"Error reading from Redis cache: {e}")
            return None

    async def cache_query_result(self, cache_key: str, result: str, ttl_seconds: int = 3600) -> None:
        """Stores query results in cache with a TTL."""
        try:
            await self.client.set(cache_key, result, ex=ttl_seconds)
            logger.debug(f"Cached key: {cache_key} with TTL: {ttl_seconds}s")
        except Exception as e:
            logger.error(f"Error writing to Redis cache: {e}")

    def build_cache_key(self, query: str, institution_slug: str, language: str) -> str:
        """Generates a unique SHA256 cache key based on query parameters."""
        raw_key = f"{query.strip().lower()}:{institution_slug}:{language}"
        sha_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"wazobia:cache:query:{institution_slug}:{sha_hash}"

    async def invalidate_institution_cache(self, institution_slug: str) -> None:
        """Scan and delete all query cache keys, stats, and metadata for this institution."""
        pattern = f"wazobia:cache:query:{institution_slug}:*"
        try:
            cursor = 0
            # Scan and delete keys matching pattern
            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.client.delete(*keys)
                if cursor == 0:
                    break

            # Invalidate stats and metadata caches
            await self.client.delete(f"wazobia:cache:institutions:detail:{institution_slug}")
            await self.client.delete("wazobia:cache:institutions:all")
            await self.client.delete("wazobia:cache:qdrant_stats")
            logger.info(f"Successfully invalidated Redis caches for {institution_slug}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache for {institution_slug}: {e}")

    async def close(self) -> None:
        """Closes the Redis connections."""
        await self.client.aclose()
