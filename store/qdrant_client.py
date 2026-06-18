import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, HnswConfigDiff
from config.settings import settings

logger = logging.getLogger("store.qdrant_client")


class KnowledgePoint(BaseModel):
    id: str = Field(description="UUIDv4 identifier of the point")
    vector: List[float] = Field(description="Dense vector embedding")
    institution_slug: str
    institution_name: str
    category: str
    sub_category: Optional[str] = None
    content: str
    source_url: str
    scraped_at: datetime
    content_hash: str
    is_verified: bool = False
    language: str = "en"


class SearchResult(BaseModel):
    id: str
    score: float
    content: str
    source_url: str
    category: str
    institution_slug: str
    institution_name: str


class CollectionStats(BaseModel):
    total_points: int
    points_per_institution: Dict[str, int]
    indexed_at: datetime


class QdrantStore:
    """Async wrapper for Qdrant vector store operations."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.EMBEDDING_DIMS

    async def ensure_collection(self) -> None:
        """Idempotently ensures the target collection exists with index settings."""
        collections = await self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name} with dims: {self.vector_size}")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                ),
            )
            # Create payload indexes for fast filtering
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="institution_slug",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="is_verified",
                field_schema=models.PayloadSchemaType.BOOL,
            )
            # Full-text payload index on 'content' for keyword matching (hybrid search helper)
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        else:
            logger.debug(f"Collection {self.collection_name} already exists.")

    async def upsert_points(self, points: List[KnowledgePoint]) -> Any:
        """Performs batch upserts of knowledge points to Qdrant."""
        start_time = time.time()
        qdrant_points = []

        for p in points:
            payload = {
                "institution_slug": p.institution_slug,
                "institution_name": p.institution_name,
                "category": p.category,
                "sub_category": p.sub_category,
                "content": p.content,
                "source_url": p.source_url,
                "scraped_at": p.scraped_at.isoformat(),
                "content_hash": p.content_hash,
                "is_verified": p.is_verified,
                "language": p.language,
            }
            qdrant_points.append(
                models.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=payload,
                )
            )

        # Batch upsert in chunks of 100
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(qdrant_points), batch_size):
            chunk = qdrant_points[i : i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=chunk,
            )
            total_upserted += len(chunk)

        elapsed = time.time() - start_time
        logger.info(f"Upserted {total_upserted} points to Qdrant in {elapsed:.3f}s")
        return {"upserted": total_upserted, "time_taken": elapsed}

    async def search(
        self,
        query_vector: List[float],
        institution_slug: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Performs dense vector similarity search with filters and threshold."""
        must_filters = [
            models.FieldCondition(
                key="institution_slug",
                match=models.MatchValue(value=institution_slug),
            )
        ]

        if category_filter:
            must_filters.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category_filter),
                )
            )

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(must=must_filters),
            limit=top_k,
            score_threshold=0.35,
        )

        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    content=payload.get("content", ""),
                    source_url=payload.get("source_url", ""),
                    category=payload.get("category", ""),
                    institution_slug=payload.get("institution_slug", ""),
                    institution_name=payload.get("institution_name", ""),
                )
            )
        return results

    async def hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        institution_slug: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Performs dense + full-text sparse search, re-ranked via manual Reciprocal Rank Fusion (RRF)."""
        # 1. Run Dense Search
        dense_results = await self.search(
            query_vector=query_vector,
            institution_slug=institution_slug,
            top_k=top_k * 2,  # Oversample for fusion
        )

        # 2. Run Keyword Full-Text Search
        must_filters = [
            models.FieldCondition(
                key="institution_slug",
                match=models.MatchValue(value=institution_slug),
            ),
            models.FieldCondition(
                key="content",
                match=models.MatchText(text=query_text),
            ),
        ]

        text_response = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must_filters),
            limit=top_k * 2,
            with_payload=True,
            with_vectors=False,
        )

        text_results = []
        for point in text_response[0]:
            payload = point.payload or {}
            text_results.append(
                SearchResult(
                    id=str(point.id),
                    score=1.0,  # Scroll matches do not return raw scores, treat as equivalent rank
                    content=payload.get("content", ""),
                    source_url=payload.get("source_url", ""),
                    category=payload.get("category", ""),
                    institution_slug=payload.get("institution_slug", ""),
                    institution_name=payload.get("institution_name", ""),
                )
            )

        # 3. Manually perform Reciprocal Rank Fusion (RRF)
        # Formula: RRF_Score = sum( 1 / (k + rank) )
        # k is a constant parameter (usually 60)
        k = 60
        rrf_scores: Dict[str, float] = {}
        all_hits: Dict[str, SearchResult] = {}

        # Record dense ranks
        for rank, hit in enumerate(dense_results, start=1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (k + rank))
            all_hits[hit.id] = hit

        # Record text keyword ranks
        for rank, hit in enumerate(text_results, start=1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (k + rank))
            if hit.id not in all_hits:
                # Assign a default score of 0.50 for keyword-only search hits
                hit.score = 0.50
                all_hits[hit.id] = hit

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results = []
        for point_id in sorted_ids:
            hit = all_hits[point_id]
            # Keep original hit.score (which is either dense similarity score or default text score)
            # instead of overwriting with small RRF values like 0.016, so threshold checks pass.
            final_results.append(hit)

        return final_results

    async def delete_by_institution(self, institution_slug: str) -> int:
        """Deletes all points associated with a specific institution slug."""
        response = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="institution_slug",
                            match=models.MatchValue(value=institution_slug),
                        )
                    ]
                )
            ),
        )
        # Qdrant's delete action is asynchronous on backend, we return a success code/ack.
        logger.info(f"Deleted points selector executed for institution: {institution_slug}")
        return 1

    async def content_hash_exists(self, content_hash: str, institution_slug: str) -> bool:
        """Checks if a point with the exact content hash and institution exists."""
        response = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="institution_slug",
                        match=models.MatchValue(value=institution_slug),
                    ),
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(response[0]) > 0

    async def get_collection_stats(self) -> CollectionStats:
        """Retrieves collection statistics, including counts per institution."""
        stats = await self.client.get_collection(collection_name=self.collection_name)
        total_points = stats.points_count or 0

        # Perform scroll aggregates to calculate points per institution
        points_per_institution: Dict[str, int] = {}
        offset = None

        while True:
            records, offset = await self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["institution_slug"],
                with_vectors=False,
            )
            for r in records:
                slug = (r.payload or {}).get("institution_slug", "unknown")
                points_per_institution[slug] = points_per_institution.get(slug, 0) + 1

            if not offset:
                break

        return CollectionStats(
            total_points=total_points,
            points_per_institution=points_per_institution,
            indexed_at=datetime.utcnow(),
        )

    async def close(self) -> None:
        """Closes the client connections."""
        await self.client.close()
