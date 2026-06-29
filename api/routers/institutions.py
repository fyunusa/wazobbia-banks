import logging
import json
import bcrypt
import uuid
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Header, HTTPException, Depends, Request, UploadFile, File
from celery.result import AsyncResult
from qdrant_client.http import models

from config.settings import settings
from registry.institutions import list_institutions, get_institution, Institution
from ingestion.tasks import scrape_institution, ingest_uploaded_documents, celery_app
from ingestion.processors.parsers import UniversalFileParser
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
    # Validate task_id format (should be a valid UUID)
    if not task_id or len(task_id) < 8:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id format: {task_id}"
        )
    
    res = AsyncResult(task_id, app=celery_app)

    # Celery returns "PENDING" for non-existent tasks, so we need to distinguish
    # A task is truly pending if it was recently submitted, otherwise it's likely invalid
    # Check if the task has any state recorded in the backend
    task_state = celery_app.backend.get_state(task_id)
    
    if task_state is None and res.status == "PENDING":
        # Task doesn't exist in backend - it was never registered
        logger.warning(f"Task {task_id} not found in Celery backend")
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found. Check that you're using the correct task_id (not batch_id)."
        )

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


# Instantiate sliding window rate limiter for uploads (20 requests per hour)
upload_limiter = SlidingWindowRateLimiter("upload", 20, 3600)


@router.post("/institutions/{slug}/upload", status_code=202, dependencies=[Depends(verify_admin_key), Depends(upload_limiter)])
async def upload_institution_documents(
    slug: str,
    file: UploadFile = File(..., description="Select a file to upload (DOCX, JSON, or PDF)"),
):
    """Upload a document (DOCX, JSON, or PDF) for an institution and trigger ingestion.
    
    Accepts a single file of supported format and queues it for processing through
    the standard ingestion pipeline (cleaning, chunking, embedding, Qdrant storage).
    
    Args:
        slug: Institution slug identifier
        file: File to upload (DOCX, JSON, or PDF format)
        
    Returns:
        Batch upload response with task ID and file processing details
    """
    # Validate institution exists
    try:
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided or file has no filename")
    
    # Validate and parse file
    parser = UniversalFileParser()
    raw_documents = []
    file_summaries = []
    
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail=f"File {file.filename} is empty")
        
        # Limit file size (50MB per file)
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds 50MB limit"
            )
        
        # Parse the file
        raw_doc = await parser.parse(content, file.filename, slug)
        raw_documents.append(raw_doc)
        
        file_summaries.append({
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(content),
            "status": "parsed",
        })
        
    except ValueError as e:
        logger.error(f"Validation error parsing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process file {file.filename}: {str(e)}"
        )
    
    # Generate unique batch ID
    upload_batch_id = str(uuid.uuid4())
    
    # Serialize raw documents for Celery task
    raw_docs_dicts = [doc.model_dump() for doc in raw_documents]
    
    # Queue the Celery task
    task = ingest_uploaded_documents.delay(slug, raw_docs_dicts, upload_batch_id)
    
    logger.info(
        f"Queued upload ingestion task for {slug}. "
        f"Batch ID: {upload_batch_id}, Task ID: {task.id}, "
        f"File: {file.filename}",
        extra={
            "institution_slug": slug,
            "upload_batch_id": upload_batch_id,
            "task_id": task.id,
            "file_name": file.filename,
        }
    )
    
    return {
        "upload_batch_id": upload_batch_id,
        "institution_slug": slug,
        "task_id": task.id,
        "status": "queued",
        "file": file_summaries[0],
        "detail": f"Upload batch {upload_batch_id} successfully sent to processing queue.",
    }


@router.get("/institutions/{slug}/uploads/{batch_id}/chunks", dependencies=[Depends(verify_admin_key)])
async def get_upload_chunks(
    slug: str,
    batch_id: str,
    qdrant: QdrantStore = Depends(get_qdrant_store),
):
    """Retrieve all extracted chunks from an uploaded document batch.
    
    Args:
        slug: Institution slug
        batch_id: Upload batch ID to retrieve chunks for
        
    Returns:
        List of chunks with metadata (content, category, source, etc.)
    """
    # Validate institution exists
    try:
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        chunks = await qdrant.search_by_batch_id(batch_id)
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found for batch {batch_id} in institution {slug}"
            )
        
        logger.info(
            f"Retrieved {len(chunks)} chunks for upload batch {batch_id} in {slug}",
            extra={
                "institution_slug": slug,
                "upload_batch_id": batch_id,
                "chunk_count": len(chunks),
            }
        )
        
        return {
            "upload_batch_id": batch_id,
            "institution_slug": slug,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrieving chunks for batch {batch_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving chunks: {str(e)}"
        )


@router.get("/institutions/{slug}/debug/points", dependencies=[Depends(verify_admin_key)])
async def debug_institution_points(
    slug: str,
    limit: int = 10,
    qdrant: QdrantStore = Depends(get_qdrant_store),
):
    """Debug endpoint: Show raw points for an institution to verify upload_batch_id storage.
    
    Args:
        slug: Institution slug
        limit: Max points to return
        
    Returns:
        Raw point data with all payloads
    """
    # Validate institution exists
    try:
        get_institution(slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    try:
        filter_condition = models.FieldCondition(
            key="institution_slug",
            match=models.MatchValue(value=slug),
        )
        
        response = await qdrant.client.scroll(
            collection_name=qdrant.collection_name,
            scroll_filter=models.Filter(must=[filter_condition]),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        points = response[0] if isinstance(response, tuple) else response
        
        debug_data = []
        for hit in points:
            payload = hit.payload or {}
            debug_data.append({
                "id": str(hit.id),
                "upload_batch_id": payload.get("upload_batch_id"),
                "category": payload.get("category"),
                "content_preview": payload.get("content", "")[:100],
            })
        
        logger.info(f"Debug: Retrieved {len(debug_data)} points for {slug}")
        
        return {
            "institution_slug": slug,
            "point_count": len(debug_data),
            "points": debug_data,
        }
    except Exception as e:
        logger.error(f"Debug query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
