import json
import logging
import time
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import tiktoken
from openai import AsyncOpenAI
from fastapi import HTTPException

from config.settings import settings
from config.metrics import wazobia_queries_total, wazobia_query_latency_seconds
from store.qdrant_client import QdrantStore, SearchResult
from store.redis_client import RedisStore
from store.postgres import AsyncSessionLocal, QueryLog
from registry.institutions import get_institution
from agents.system_prompts import build_system_prompt
from ingestion.processors.embedder import Embedder

logger = logging.getLogger("agents.rag_query")


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=500, description="The user query (max 500 chars)")
    institution_slug: str
    language: str = "en"
    top_k: int = Field(5, ge=1, le=20)
    use_hybrid: bool = True
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    institution_slug: str
    language: str
    confidence: float
    cache_hit: bool
    latency_ms: int


class RAGQueryEngine:
    """Orchestrates the query RAG loop: caching, translating, retrieving, and generating responses."""

    def __init__(
        self,
        qdrant: QdrantStore,
        embedder: Embedder,
        redis_store: RedisStore,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder
        self.redis_store = redis_store
        self.openai_client = openai_client
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def _translate_text(self, text: str, source_or_target: str, to_english: bool = True) -> str:
        """Translates text between English and target low-resource Nigerian languages using LLM."""
        if to_english:
            prompt = f"Translate the following text to standard English. Reply ONLY with the translated text. Do not add any conversational preamble or explanations.\n\nText: {text}"
        else:
            prompt = f"Translate the following English text to {source_or_target}. Reply ONLY with the translated text. Maintain formatting and disclaimers. Do not add any conversational preamble or explanations.\n\nText: {text}"

        try:
            # Call Llama 3.1 model via vLLM
            response = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional multilingual translator specialized in Nigerian languages (Yoruba, Hausa, Igbo, Pidgin English) and English."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.0,
                extra_body={"repetition_penalty": 1.1},
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )
            translated = response.choices[0].message.content or text
            return translated.strip()
        except Exception as e:
            logger.error(f"Translation failed: {e}. Returning original text.", exc_info=True)
            return text

    async def _log_query_to_db(self, slug: str, query: str, lang: str, answer: str, latency: int, hit: bool) -> None:
        """Saves query log event to PostgreSQL DB in background."""
        try:
            async with AsyncSessionLocal() as session:
                log = QueryLog(
                    institution_slug=slug,
                    query_text=query,
                    language=lang,
                    response_text=answer,
                    latency_ms=latency,
                    cache_hit=hit,
                )
                session.add(log)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed logging query execution to DB: {e}")

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Executes the complete RAG query lifecycle."""
        start_time = time.time()

        # Step 1 — Cache check
        cache_key = self.redis_store.build_cache_key(
            query=request.query,
            institution_slug=request.institution_slug,
            language=request.language,
        )
        cached_val = await self.redis_store.get_cached_query(cache_key)
        if cached_val:
            try:
                data = json.loads(cached_val)
                data["cache_hit"] = True
                latency_ms = int((time.time() - start_time) * 1000)
                data["latency_ms"] = latency_ms
                
                # Record metrics
                wazobia_queries_total.labels(
                    institution=request.institution_slug,
                    language=request.language,
                    cache_hit="true"
                ).inc()
                wazobia_query_latency_seconds.observe(latency_ms / 1000.0)

                # Log structured query (Cache Hit)
                logger.info(
                    f"RAG query completed (Cache Hit) for {request.institution_slug}",
                    extra={
                        "institution_slug": request.institution_slug,
                        "language": request.language,
                        "cache_hit": True,
                        "top_score": data.get("confidence", 0.0),
                        "llm_tokens_used": 0,
                    }
                )

                # Background log database record
                asyncio.create_task(
                    self._log_query_to_db(
                        slug=request.institution_slug,
                        query=request.query,
                        lang=request.language,
                        answer=data["answer"],
                        latency=data["latency_ms"],
                        hit=True,
                    )
                )
                return QueryResponse(**data)
            except Exception as e:
                logger.warning(f"Failed parsing cached query payload: {e}")

        # Resolve institution display name for fallbacks
        try:
            inst_details = get_institution(request.institution_slug)
            institution_name = inst_details.name
        except Exception:
            institution_name = request.institution_slug.upper()

        # Step 2 — Query translation (if not english)
        english_query = request.query
        if request.language != "en":
            english_query = await self._translate_text(
                text=request.query,
                source_or_target=request.language,
                to_english=True,
            )
            logger.info(f"Translated query '{request.query}' to English: '{english_query}'")

        # Step 3 — Embed query
        try:
            query_vector = await self.embedder.embed_query(english_query)
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error: failed generating query vector.")

        # Step 4 — Retrieve from Qdrant
        if request.use_hybrid:
            hits = await self.qdrant.hybrid_search(
                query_vector=query_vector,
                query_text=english_query,
                institution_slug=request.institution_slug,
                top_k=request.top_k,
            )
        else:
            hits = await self.qdrant.search(
                query_vector=query_vector,
                institution_slug=request.institution_slug,
                top_k=request.top_k,
            )

        # Retrieve top confidence score
        confidence = hits[0].score if hits else 0.0

        # Graceful "no information" fallback if empty or low scores (all scores < 0.35)
        if not hits or all(hit.score < 0.35 for hit in hits):
            fallback_ans = f"I don't have information about that for {institution_name}."
            if request.language != "en":
                fallback_ans = await self._translate_text(fallback_ans, request.language, to_english=False)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Record metrics
            wazobia_queries_total.labels(
                institution=request.institution_slug,
                language=request.language,
                cache_hit="false"
            ).inc()
            wazobia_query_latency_seconds.observe(latency_ms / 1000.0)

            # Log structured query (Fallback)
            logger.info(
                f"RAG query completed (Fallback) for {request.institution_slug}",
                extra={
                    "institution_slug": request.institution_slug,
                    "language": request.language,
                    "cache_hit": False,
                    "top_score": confidence,
                    "llm_tokens_used": 0,
                }
            )

            asyncio.create_task(
                self._log_query_to_db(
                    slug=request.institution_slug,
                    query=request.query,
                    lang=request.language,
                    answer=fallback_ans,
                    latency=latency_ms,
                    hit=False,
                )
            )
            return QueryResponse(
                answer=fallback_ans,
                sources=[],
                institution_slug=request.institution_slug,
                language=request.language,
                confidence=confidence,
                cache_hit=False,
                latency_ms=latency_ms,
            )

        # Step 5 — Build context
        context_parts = []
        sources = []
        current_tokens = 0
        token_limit = 3000

        for hit in hits:
            hit_tokens = len(self.encoder.encode(hit.content))
            if current_tokens + hit_tokens <= token_limit:
                context_parts.append(hit.content)
                current_tokens += hit_tokens
                if hit.source_url not in sources:
                    sources.append(hit.source_url)
            else:
                break

        context = "\n---\n".join(context_parts)

        # Step 6 — LLM generation
        system_prompt = build_system_prompt(request.institution_slug, request.language)
        user_message = f"Context:\n{context}\n\nQuestion: {english_query}"

        try:
            completion = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=600,
                temperature=0.1,
                extra_body={"repetition_penalty": 1.1},
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )
            answer = completion.choices[0].message.content or ""
            answer = answer.strip()
        except Exception as e:
            logger.error(f"LLM Chat completion failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM generation error: {e}")

        # Step 7 — Translate response (if requested language not English)
        if request.language != "en":
            answer = await self._translate_text(
                text=answer,
                source_or_target=request.language,
                to_english=False,
            )

        latency_ms = int((time.time() - start_time) * 1000)
        tokens = completion.usage.total_tokens if (completion and completion.usage) else 0

        # Record metrics
        wazobia_queries_total.labels(
            institution=request.institution_slug,
            language=request.language,
            cache_hit="false"
        ).inc()
        wazobia_query_latency_seconds.observe(latency_ms / 1000.0)

        # Log structured query (Success)
        logger.info(
            f"RAG query completed for {request.institution_slug}",
            extra={
                "institution_slug": request.institution_slug,
                "language": request.language,
                "cache_hit": False,
                "top_score": confidence,
                "llm_tokens_used": tokens,
            }
        )

        # Step 8 — Cache result and background log DB event
        response_obj = QueryResponse(
            answer=answer,
            sources=sources,
            institution_slug=request.institution_slug,
            language=request.language,
            confidence=confidence,
            cache_hit=False,
            latency_ms=latency_ms,
        )

        # Cache for 1 hour
        await self.redis_store.cache_query_result(
            cache_key=cache_key,
            result=response_obj.model_dump_json(),
            ttl_seconds=3600,
        )

        asyncio.create_task(
            self._log_query_to_db(
                slug=request.institution_slug,
                query=request.query,
                lang=request.language,
                answer=answer,
                latency=latency_ms,
                hit=False,
            )
        )

        return response_obj
