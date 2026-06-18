import time
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore
from store.postgres import get_db, QueryLog
from openai import AsyncOpenAI
from api.dependencies import get_qdrant_store, get_redis_store, get_embedder, get_openai_client, SlidingWindowRateLimiter
from api.routers.institutions import verify_admin_key
from registry.institutions import get_institution
from agents.rag_query import RAGQueryEngine, QueryRequest, QueryResponse
from agents.orchestrator import MultiAgentOrchestrator, OrchestratorResponse
from ingestion.processors.embedder import Embedder

logger = logging.getLogger("api.routers.query")

router = APIRouter(tags=["Query"])


import re

class MultiQueryRequest(BaseModel):
    query: str = Field(..., max_length=500, description="The user query (max 500 chars)")
    language: str = "en"
    institution_slug: Optional[str] = None


# Instantiate sliding window rate limiter (30 queries per minute)
query_limiter = SlidingWindowRateLimiter("query", 30, 60)


def sanitize_query(text: str) -> str:
    """Sanitizes query text by rejecting null bytes, stripping HTML tags, and truncating to 500 chars."""
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Bad Request: Query contains null bytes.")
    # Strip HTML tags
    clean_text = re.sub(r"<[^>]*>", "", text)
    clean_text = clean_text.strip()
    if len(clean_text) > 500:
        clean_text = clean_text[:500]
    return clean_text


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(query_limiter)])
async def query_rag_engine(
    query_req: QueryRequest,
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: Embedder = Depends(get_embedder),
    redis: RedisStore = Depends(get_redis_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """Submits query to RAG Engine returning retrieved context and LLM answer."""
    # 1. Validate institution slug against registry strictly
    try:
        get_institution(query_req.institution_slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 2. Sanitize and validate query
    query_req.query = sanitize_query(query_req.query)

    # 3. Invoke RAG Query Engine
    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis,
        openai_client=openai_client,
    )
    return await engine.query(query_req)


@router.get("/query/institutions/{slug}/suggested-questions", response_model=List[str])
async def get_suggested_questions(slug: str):
    """Returns 5 suggested starter questions for a specific institution."""
    try:
        inst = get_institution(slug)
        name = inst.name
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [
        f"What are {name} transfer fees?",
        f"How do I open a {name} account?",
        f"What is the {name} USSD code?",
        f"What is the {name} customer care line?",
        f"Does {name} charge for card maintenance?",
    ]


@router.post("/query/multi", response_model=OrchestratorResponse, dependencies=[Depends(query_limiter)])
async def query_multi_agent(
    req: MultiQueryRequest,
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: Embedder = Depends(get_embedder),
    redis: RedisStore = Depends(get_redis_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """Submits query to Multi-Agent Orchestrator returning mapped responses."""
    # 1. Sanitize and validate query
    req.query = sanitize_query(req.query)

    # 2. Validate institution slug against registry strictly if present
    if req.institution_slug:
        try:
            get_institution(req.institution_slug)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    orchestrator = MultiAgentOrchestrator(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis,
        openai_client=openai_client,
    )
    return await orchestrator.route(
        query=req.query,
        language=req.language,
        institution_slug=req.institution_slug,
    )


@router.get("/query/logs", dependencies=[Depends(verify_admin_key)])
async def get_query_logs(
    institution_slug: Optional[str] = None,
    language: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    cache_hit: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Retrieves paginated and filtered QueryLog history from PostgreSQL (admin only)."""
    stmt = select(QueryLog)
    conditions = []

    if institution_slug:
        conditions.append(QueryLog.institution_slug == institution_slug)
    if language:
        conditions.append(QueryLog.language == language)
    if start_date:
        conditions.append(QueryLog.created_at >= start_date)
    if end_date:
        conditions.append(QueryLog.created_at <= end_date)
    if cache_hit is not None:
        conditions.append(QueryLog.cache_hit == cache_hit)

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(QueryLog.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    logs = result.scalars().all()
    return logs
