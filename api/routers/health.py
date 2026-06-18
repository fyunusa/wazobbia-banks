import logging
import json
from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI

from config.settings import settings
from config.metrics import wazobia_qdrant_points_total
from store.postgres import get_db
from store.redis_client import RedisStore
from store.qdrant_client import QdrantStore
from api.dependencies import get_db, get_redis_store, get_qdrant_store, get_openai_client
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger("api.routers.health")

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: RedisStore = Depends(get_redis_store),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """Enhances health checks by pinging DB, Redis, Qdrant, and OpenAI."""
    postgres_ok = False
    redis_ok = False
    qdrant_ok = False
    llm_ok = False
    
    # 1. Check PostgreSQL
    try:
        await db.execute(select(1))
        postgres_ok = True
    except Exception as e:
        logger.error(f"Health check Postgres failure: {e}")
        postgres_ok = False
        
    # 2. Check Redis
    try:
        await redis.client.ping()
        redis_ok = True
    except Exception as e:
        logger.error(f"Health check Redis failure: {e}")
        redis_ok = False
        
    # 3. Check Qdrant
    try:
        collections = await qdrant.client.get_collections()
        exists = any(c.name == qdrant.collection_name for c in collections.collections)
        qdrant_ok = exists
    except Exception as e:
        logger.error(f"Health check Qdrant failure: {e}")
        qdrant_ok = False
        
    # 4. Check LLM Reachability
    try:
        await openai_client.models.list()
        llm_ok = True
    except Exception as e:
        logger.error(f"Health check LLM failure: {e}")
        llm_ok = False
        
    checks = {
        "postgres": "healthy" if postgres_ok else "unhealthy",
        "redis": "healthy" if redis_ok else "unhealthy",
        "qdrant": "healthy" if qdrant_ok else "unhealthy",
        "llm": "healthy" if llm_ok else "unhealthy",
    }
    
    # Critical components: Postgres, Redis, Qdrant
    if postgres_ok and redis_ok and qdrant_ok:
        if llm_ok:
            status = "healthy"
        else:
            status = "degraded"
    else:
        status = "unhealthy"
        
    return {
        "status": status,
        "environment": settings.ENVIRONMENT,
        "checks": checks,
    }


@router.get("/metrics")
async def metrics(
    redis: RedisStore = Depends(get_redis_store),
    qdrant: QdrantStore = Depends(get_qdrant_store)
):
    """Exposes Prometheus format metrics, dynamically updating Qdrant stats (cached 5min)."""
    cache_key = "wazobia:cache:qdrant_stats"
    stats_data = None
    try:
        cached = await redis.client.get(cache_key)
        if cached:
            stats_data = json.loads(cached)
    except Exception as e:
        logger.error(f"Failed to read Qdrant stats from cache for metrics: {e}")

    if not stats_data:
        try:
            stats = await qdrant.get_collection_stats()
            stats_data = {
                "total_points": stats.total_points,
                "points_per_institution": stats.points_per_institution,
                "indexed_at": stats.indexed_at.isoformat(),
            }
            await redis.client.set(cache_key, json.dumps(stats_data), ex=300)
        except Exception as e:
            logger.error(f"Failed to fetch collection stats for metrics: {e}")

    if stats_data:
        points_per_inst = stats_data.get("points_per_institution", {})
        for inst_slug, count in points_per_inst.items():
            wazobia_qdrant_points_total.labels(institution=inst_slug).set(count)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
