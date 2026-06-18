import asyncio
import logging
from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from config.settings import settings
from registry.institutions import list_institutions, get_institution
from agents.rag_query import RAGQueryEngine, QueryRequest, QueryResponse
from agents.intent_classifier import IntentClassifier

logger = logging.getLogger("agents.orchestrator")


class OrchestratorResponse(BaseModel):
    response_type: Literal["single_agent", "comparative", "unknown_institution", "out_of_scope"]
    answers: List[QueryResponse] = Field(default_factory=list)
    synthesis: Optional[str] = None
    institutions_queried: List[str] = Field(default_factory=list)


class MultiAgentOrchestrator:
    """Orchestrates query classification and handles parallel institutional RAG retrieval or synthesis comparison."""

    def __init__(
        self,
        qdrant,
        embedder,
        redis_store,
        openai_client: AsyncOpenAI,
    ) -> None:
        self.qdrant = qdrant
        self.embedder = embedder
        self.redis_store = redis_store
        self.openai_client = openai_client
        self.intent_classifier = IntentClassifier(openai_client=openai_client)
        self.query_engine = RAGQueryEngine(
            qdrant=qdrant,
            embedder=embedder,
            redis_store=redis_store,
            openai_client=openai_client,
        )

    async def synthesize_comparison(self, query: str, responses: List[QueryResponse], language: str) -> str:
        """Synthesizes retrieval answers from multiple banks into a comparative markdown table response."""
        context_parts = []
        for resp in responses:
            try:
                inst = get_institution(resp.institution_slug)
                inst_name = inst.name
            except Exception:
                inst_name = resp.institution_slug.upper()

            context_parts.append(
                f"Bank: {inst_name}\n"
                f"Information: {resp.answer}\n"
                f"Sources: {', '.join(resp.sources)}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        # Build comparison prompt
        prompt = (
            f"Based on these answers from different Nigerian banks, provide a concise comparison table in markdown. "
            f"Compare their fees, USSD codes, features, or relevant details mentioned. "
            f"Only use the provided answers context below — do not make up or extrapolate facts.\n\n"
            f"Answers Context:\n{context_str}\n\n"
            f"User Comparison Request: {query}\n\n"
            f"Requirements:\n"
            f"1. Generate a structured markdown table.\n"
            f"2. Add a short summary comparing the details.\n"
            f"3. Do not add conversational preamble. Reply only with the comparison table and summary.\n"
        )

        lang_names = {
            "en": "English",
            "ha": "Hausa",
            "yo": "Yoruba",
            "ig": "Igbo",
            "pcm": "Nigerian Pidgin",
        }
        target_lang = lang_names.get(language, "English")

        try:
            completion = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a precise financial comparisons assistant. You formulate responses in {target_lang}."
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.1,
                extra_body={"repetition_penalty": 1.1},
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )
            synthesis = completion.choices[0].message.content or ""
            return synthesis.strip()
        except Exception as e:
            logger.error(f"Failed comparative synthesis: {e}", exc_info=True)
            return "Failed to synthesize comparison due to an internal error."

    async def route(self, query: str, language: str, institution_slug: Optional[str] = None) -> OrchestratorResponse:
        """Routes user queries dynamically to single target agents or multi-agent comparative flows."""
        # a) If institution_slug is provided:
        if institution_slug:
            try:
                get_institution(institution_slug)
            except ValueError:
                return OrchestratorResponse(
                    response_type="unknown_institution",
                    answers=[],
                    synthesis=None,
                    institutions_queried=[institution_slug],
                )

            req = QueryRequest(
                query=query,
                institution_slug=institution_slug,
                language=language,
            )
            resp = await self.query_engine.query(req)
            return OrchestratorResponse(
                response_type="single_agent",
                answers=[resp],
                synthesis=None,
                institutions_queried=[institution_slug],
            )

        # b) If institution_slug is None:
        # Check if query targets a single institution via aliases
        extracted_slug = await self.intent_classifier.extract_institution_slug(query)
        if extracted_slug:
            req = QueryRequest(
                query=query,
                institution_slug=extracted_slug,
                language=language,
            )
            resp = await self.query_engine.query(req)
            return OrchestratorResponse(
                response_type="single_agent",
                answers=[resp],
                synthesis=None,
                institutions_queried=[extracted_slug],
            )

        # Check if it is a comparative query
        is_comparative = await self.intent_classifier.is_comparative_query(query)
        if is_comparative:
            active_insts = list_institutions(active_only=True)
            inst_slugs = [inst.slug for inst in active_insts]

            # Helper to run a RAG query with error isolation and 10s timeout
            async def run_single_query(slug: str):
                req_obj = QueryRequest(
                    query=query,
                    institution_slug=slug,
                    language=language,
                )
                try:
                    return await asyncio.wait_for(self.query_engine.query(req_obj), timeout=10.0)
                except asyncio.TimeoutError as te:
                    logger.warning(f"RAG query for bank '{slug}' timed out (10s limit).")
                    return te
                except Exception as ex:
                    logger.error(f"RAG query for bank '{slug}' failed with exception: {ex}", exc_info=True)
                    return ex

            # True parallel fan-out
            tasks = [run_single_query(slug) for slug in inst_slugs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_responses = []
            queried_slugs = []

            for slug, res in zip(inst_slugs, results):
                if isinstance(res, asyncio.TimeoutError):
                    # Timeout occurred, exclude from synthesis but log already done inside wrapper
                    continue
                elif isinstance(res, Exception):
                    # Exception occurred, exclude
                    continue
                else:
                    queried_slugs.append(slug)
                    # Filter out responses with low confidence (< 0.4)
                    if res.confidence >= 0.4:
                        valid_responses.append(res)

            synthesis = None
            if valid_responses:
                synthesis = await self.synthesize_comparison(
                    query=query,
                    responses=valid_responses,
                    language=language,
                )
            else:
                synthesis = "No high-confidence information was retrieved from any active institution to compile a comparison."

            return OrchestratorResponse(
                response_type="comparative",
                answers=valid_responses,
                synthesis=synthesis,
                institutions_queried=queried_slugs,
            )

        # Not comparative, no targeted bank
        return OrchestratorResponse(
            response_type="out_of_scope",
            answers=[],
            synthesis=None,
            institutions_queried=[],
        )
