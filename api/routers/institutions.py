import logging
import json
import bcrypt
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Header, HTTPException, Depends, Request
from celery.result import AsyncResult

from config.settings import settings
from registry.institutions import list_institutions, get_institution, Institution
from ingestion.tasks import scrape_institution, celery_app
from api.dependencies import get_qdrant_store, get_redis_store, SlidingWindowRateLimiter
from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore

logger = logging.getLogger("api.routers.institutions")

router = APIRouter(tags=["Institutions"])

from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Instantiate sliding window rate limiter for ingest (60 requests per hour)
ingest_limiter = SlidingWindowRateLimiter("ingest", 60, 3600)


async def verify_admin_key(
    request: Request,
    x_api_key: str = Depends(api_key_header)
) -> str:
    """Dependency verifying the presence of valid administrative API key header via bcrypt."""
    hashed_key = settings.ADMIN_API_KEY
    ip = request.client.host if request.client else "127.0.0.1"
    
    is_valid = False
    try:
        is_valid = bcrypt.checkpw(x_api_key.encode("utf-8"), hashed_key.encode("utf-8"))
    except Exception as e:
        logger.error(f"Bcrypt key comparison error: {e}. Rejecting access.")
        is_valid = False

    if not is_valid:
        logger.warning(f"Attempted administrative access with invalid API key from IP: {ip}.")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid administrative X-API-Key header credential.",
        )
    
    logger.info(
        f"Admin access authorized for IP: {ip}. Path: {request.url.path} Method: {request.method}",
        extra={
            "user_ip": ip,
            "action": f"admin_{request.method.lower()}_{request.url.path.replace('/', '_')}",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    return x_api_key


@router.get("/institutions", response_model=List[Institution], dependencies=[Depends(verify_admin_key)])
async def list_all_institutions(redis: RedisStore = Depends(get_redis_store)):
    """Lists all registered financial institutions in the system registry (cached 24h)."""
    cache_key = "wazobia:cache:institutions:all"
    try:
        cached = await redis.client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Failed to read from institutions cache: {e}")

    insts = list_institutions(active_only=False)
    
    try:
        serialized = json.dumps([inst.model_dump() for inst in insts])
        await redis.client.set(cache_key, serialized, ex=86400)  # 24h TTL
    except Exception as e:
        logger.error(f"Failed to write to institutions cache: {e}")

    return insts


@router.get("/institutions/{slug}", response_model=Institution, dependencies=[Depends(verify_admin_key)])
async def get_institution_detail(slug: str, redis: RedisStore = Depends(get_redis_store)):
    """Retrieves detailed profile metadata for a specific institution registry slug (cached 24h)."""
    # Validate slug
    try:
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    cache_key = f"wazobia:cache:institutions:detail:{slug}"
    try:
        cached = await redis.client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Failed to read from institution detail cache: {e}")

    inst = get_institution(slug)

    try:
        await redis.client.set(cache_key, inst.model_dump_json(), ex=86400)  # 24h TTL
    except Exception as e:
        logger.error(f"Failed to write to institution detail cache: {e}")

    return inst


@router.post("/institutions/{slug}/ingest", status_code=202, dependencies=[Depends(verify_admin_key), Depends(ingest_limiter)])
async def trigger_institution_ingest(slug: str):
    """Triggers an ingestion task in Celery worker thread queue for target bank/slug."""
    try:
        # Verify slug exists first
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Queue the Celery task
    task = scrape_institution.delay(slug)
    logger.info(f"Queued scrape_institution task for {slug}. Task ID: {task.id}")
    return {
        "slug": slug,
        "task_id": task.id,
        "status": "queued",
        "detail": f"Scrape task for {slug} successfully sent to worker queues.",
    }


@router.get("/institutions/{slug}/stats", dependencies=[Depends(verify_admin_key)])
async def get_institution_qdrant_stats(
    slug: str,
    qdrant: QdrantStore = Depends(get_qdrant_store),
    redis: RedisStore = Depends(get_redis_store),
):
    """Retrieves vector counts and indices filtered for a target slug from Qdrant store (cached 5min)."""
    try:
        # Validate slug
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    cache_key = "wazobia:cache:qdrant_stats"
    stats_data = None
    try:
        cached = await redis.client.get(cache_key)
        if cached:
            stats_data = json.loads(cached)
    except Exception as e:
        logger.error(f"Failed to read from Qdrant stats cache: {e}")

    if not stats_data:
        try:
            stats = await qdrant.get_collection_stats()
            stats_data = {
                "total_points": stats.total_points,
                "points_per_institution": stats.points_per_institution,
                "indexed_at": stats.indexed_at.isoformat(),
            }
            await redis.client.set(cache_key, json.dumps(stats_data), ex=300)  # 5min TTL
        except Exception as e:
            logger.error(f"Error querying collection statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to query Qdrant collection stats: {e}")

    points_count = stats_data.get("points_per_institution", {}).get(slug, 0)
    return {
        "institution_slug": slug,
        "collection_name": qdrant.collection_name,
        "total_collection_points": stats_data.get("total_points", 0),
        "institution_points_count": points_count,
        "indexed_at": stats_data.get("indexed_at"),
    }


@router.get("/ingest/tasks/{task_id}", dependencies=[Depends(verify_admin_key)])
async def get_ingestion_task_status(task_id: str):
    """Queries and returns Celery task execution status and outcomes."""
    res = AsyncResult(task_id, app=celery_app)

    response_data = {
        "task_id": task_id,
        "status": res.status,
        "ready": res.ready(),
        "result": None,
    }

    if res.ready():
        if res.failed():
            # If the task failed, return string error trace/representation
            response_data["result"] = {"error": str(res.result)}
        else:
            response_data["result"] = res.result

    return response_data
