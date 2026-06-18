from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore
from store.postgres import get_db


from openai import AsyncOpenAI
from config.settings import settings
from ingestion.processors.embedder import Embedder


def get_qdrant_store(request: Request) -> QdrantStore:
    """Dependency to retrieve the QdrantStore instance from app state."""
    return request.app.state.qdrant_store


def get_redis_store(request: Request) -> RedisStore:
    """Dependency to retrieve the RedisStore instance from app state."""
    return request.app.state.redis_store


def get_openai_client() -> AsyncOpenAI:
    """Dependency providing a configured AsyncOpenAI client instance."""
    return AsyncOpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY or "mock-key",
    )


def get_embedder(request: Request) -> Embedder:
    """Dependency to retrieve the Embedder instance from app state."""
    return request.app.state.embedder


import time
import logging

logger = logging.getLogger("api.dependencies")

class SlidingWindowRateLimiter:
    """FastAPI dependency wrapper for Redis-backed sliding-window rate limiting."""

    def __init__(self, key_prefix: str, limit: int, window_seconds: int):
        self.key_prefix = key_prefix
        self.limit = limit
        self.window = window_seconds

    async def __call__(self, request: Request, redis: RedisStore = Depends(get_redis_store)) -> None:
        ip = request.client.host if request.client else "127.0.0.1"
        key = f"wazobia:rate_limit:{self.key_prefix}:{ip}"
        now = time.time()
        
        try:
            pipe = redis.client.pipeline()
            pipe.zremrangebyscore(key, 0, now - self.window)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window)
            _, _, count, _ = await pipe.execute()
            
            if count > self.limit:
                # Find earliest timestamp in window to compute Retry-After
                earliest_entries = await redis.client.zrange(key, 0, 0, withscores=True)
                if earliest_entries:
                    earliest_time = earliest_entries[0][1]
                    retry_after = max(1, int(self.window - (now - earliest_time)))
                else:
                    retry_after = 1
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Too Many Requests: Rate limit exceeded (maximum {self.limit} requests per {self.window} seconds).",
                    headers={"Retry-After": str(retry_after)}
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiter Redis error: {e}. Allowing request by default.")
            return
