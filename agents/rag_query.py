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
import re
from urllib.parse import urlparse

def get_source_reference(url: str, scraped_at_str: Optional[str] = None) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Identify source name
    if "cbn.gov.ng" in domain:
        source = "CBN"
    elif "punchng.com" in domain:
        source = "Punch News"
    elif "vanguardngr.com" in domain:
        source = "Vanguard News"
    elif "guardian.ng" in domain:
        source = "The Guardian"
    elif "premiumtimesng.com" in domain:
        source = "Premium Times"
    elif "nairametrics.com" in domain:
        source = "Nairametrics"
    elif "techpoint.africa" in domain:
        source = "Techpoint"
    elif "gtbank.com" in domain:
        source = "GTBank Website"
    elif "zenithbank.com" in domain:
        source = "Zenith Bank Website"
    elif "accessbankplc.com" in domain:
        source = "Access Bank Website"
    elif "kudabank.com" in domain or "kuda.com" in domain:
        source = "Kuda Bank Website"
    elif "opayweb.com" in domain or "opay.com" in domain:
        source = "OPay Website"
    elif "moniepoint.com" in domain:
        source = "Moniepoint Website"
    elif "palmpay.com" in domain:
        source = "PalmPay Website"
    elif "firstbank" in domain:
        source = "FirstBank Website"
    elif "ubagroup.com" in domain:
        source = "UBA Website"
    elif "unionbank" in domain:
        source = "Union Bank Website"
    elif "sterling" in domain:
        source = "Sterling Bank Website"
    elif "wema" in domain or "alat" in domain:
        source = "Wema Bank Website"
    elif "fidelity" in domain:
        source = "Fidelity Bank Website"
    elif "fcmb" in domain:
        source = "FCMB Website"
    elif "stanbic" in domain:
        source = "Stanbic IBTC Website"
    else:
        parts = domain.split('.')
        if len(parts) > 1:
            source = parts[-2].capitalize() + " Website"
        else:
            source = domain.capitalize()

    # Determine year from URL
    year_match = re.search(r'/20(2[0-9])/', url)
    if year_match:
        year = f"20{year_match.group(1)}"
    else:
        year_match_alt = re.search(r'\b20(2[0-9])\b', url)
        if year_match_alt:
            year = f"20{year_match_alt.group(1)}"
        elif scraped_at_str:
            try:
                # Handle ISO datetime parsing
                year_part = scraped_at_str.split("T")[0]
                year = year_part.split("-")[0]
                if not year.isdigit() or len(year) != 4:
                    year = "2026"
            except Exception:
                year = "2026"
        else:
            year = "2026"

    return f"{source} {year}"

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
        # Map language codes to full names
        lang_map = {
            "yo": "Yoruba",
            "ha": "Hausa",
            "ig": "Igbo",
            "pcm": "Nigerian Pidgin",
            "en": "English",
        }
        target_lang = lang_map.get(source_or_target, source_or_target)
        
        if to_english:
            prompt = f"Translate the following text to standard English. Reply ONLY with the translated text. Do not add any conversational preamble or explanations.\n\nText: {text}"
        else:
            prompt = f"Translate the following English text to {target_lang}. Reply ONLY with the translated text. Maintain formatting and disclaimers. Do not add any conversational preamble or explanations.\n\nText: {text}"

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

        # Step 2 — Query processing
        # NOTE: Queries are expected to be in English.
        # The 'language' parameter controls the RESPONSE language, not the input.
        # If multilingual query input is needed, add separate input_language field.
        english_query = request.query.strip()
        logger.info(f"Processing query: '{english_query}' for language: {request.language}")

        # Step 3 — Embed query (with rate limit protection)
        try:
            await asyncio.sleep(0.5)  # Rate limit queries to Cohere API
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
            ref = get_source_reference(hit.source_url, hit.scraped_at)
            formatted_content = f"[Reference: {ref} | URL: {hit.source_url}]\n{hit.content}"
            hit_tokens = len(self.encoder.encode(formatted_content))
            if current_tokens + hit_tokens <= token_limit:
                context_parts.append(formatted_content)
                current_tokens += hit_tokens
                if hit.source_url not in sources:
                    sources.append(hit.source_url)
            else:
                break

        context = "\n---\n".join(context_parts)

        # Step 6 — LLM generation (ALWAYS in English, llama3.1 doesn't speak low-resource languages)
        system_prompt = build_system_prompt(request.institution_slug, "en")  # Always generate in English
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

        # Step 7 — Translate response to target language (if not English)
        # llama3.1 can only generate in English, so we translate the English answer
        if request.language != "en":
            logger.info(f"Translating answer from English to {request.language}")
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
