import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
from celery import Celery
from sqlalchemy import select, update

from config.settings import settings
from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore
from store.postgres import AsyncSessionLocal, ScrapeJob
from config.metrics import wazobia_scrape_runs_total
from registry.institutions import list_institutions
from ingestion.processors.cleaner import DocumentCleaner
from ingestion.processors.chunker import SemanticChunker
from ingestion.processors.embedder import Embedder

# Scrapers mappings
from ingestion.scrapers.gtbank import GTBankScraper
from ingestion.scrapers.zenith import ZenithScraper
from ingestion.scrapers.access import AccessScraper
from ingestion.scrapers.kuda import KudaScraper
from ingestion.scrapers.opay import OPayScraper
from ingestion.scrapers.cbn_regulatory import CBNRegulatoryScraper

SCRAPER_MAP = {
    "gtbank": GTBankScraper,
    "zenith": ZenithScraper,
    "access": AccessScraper,
    "kuda": KudaScraper,
    "opay": OPayScraper,
    "cbn": CBNRegulatoryScraper,
}

logger = logging.getLogger("ingestion.tasks")

# Initialize Celery Application
celery_app = Celery(
    "wazobia_ingestion",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Africa/Lagos",  # Use WAT timezone
    enable_utc=True,
    task_soft_time_limit=600,
    task_time_limit=900,
)


async def _scrape_institution_async(institution_slug: str) -> Dict[str, Any]:
    """Asynchronous implementation running the complete ingestion pipeline."""
    start_time = time.time()
    logger.info(f"Starting async ingestion pipeline for institution: {institution_slug}")

    pages_scraped = 0
    chunks_created = 0
    points_upserted = 0
    skipped_dedup = 0

    # Create ScrapeJob record in Postgres
    async with AsyncSessionLocal() as session:
        job = ScrapeJob(
            institution_slug=institution_slug,
            status="running",
            started_at=datetime.utcnow(),
            pages_scraped=0,
            points_upserted=0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id

    qdrant = QdrantStore()
    cleaner = DocumentCleaner()
    chunker = SemanticChunker()
    embedder = Embedder()

    try:
        # 1. Dynamically load the scraper
        if institution_slug not in SCRAPER_MAP:
            raise ValueError(f"Scraper implementation not found for slug: {institution_slug}")

        scraper_cls = SCRAPER_MAP[institution_slug]
        scraper = scraper_cls()

        # Ensure vector store collection exists
        await qdrant.ensure_collection()

        # 2. Scrape raw documents
        raw_docs = await scraper.scrape()
        pages_scraped = len(raw_docs)

        # 3. Clean raw documents
        cleaned_docs = []
        for raw in raw_docs:
            try:
                cleaned = await cleaner.clean(raw)
                cleaned_docs.append(cleaned)
            except Exception as e:
                logger.error(f"Failed cleaning raw document {raw.url}: {e}", exc_info=True)

        # 4. Chunk cleaned documents
        all_chunks = []
        for cleaned in cleaned_docs:
            try:
                doc_chunks = chunker.chunk(cleaned)
                all_chunks.extend(doc_chunks)
            except Exception as e:
                logger.error(f"Failed chunking cleaned document {cleaned.url}: {e}", exc_info=True)

        chunks_created = len(all_chunks)

        # 5. Filter duplicates (skip if content_hash already in Qdrant)
        filtered_chunks = []
        skipped_dedup = 0
        for chunk in all_chunks:
            try:
                exists = await qdrant.content_hash_exists(chunk.content_hash, chunk.institution_slug)
                if not exists:
                    filtered_chunks.append(chunk)
                else:
                    skipped_dedup += 1
            except Exception as e:
                logger.error(f"Failed checking dedup index in Qdrant for hash {chunk.content_hash}: {e}")
                # Safe fallback: keep chunk if check fails
                filtered_chunks.append(chunk)

        # 6. Generate embeddings
        if filtered_chunks:
            points = await embedder.embed_chunks(filtered_chunks)
            # 7. Upsert to Qdrant
            upsert_result = await qdrant.upsert_points(points)
            points_upserted = len(points)

        duration = time.time() - start_time

        # Update ScrapeJob in Postgres to Completed
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(ScrapeJob)
                .where(ScrapeJob.id == job_id)
                .values(
                    status="completed",
                    completed_at=datetime.utcnow(),
                    pages_scraped=pages_scraped,
                    points_upserted=points_upserted,
                )
            )
            await session.commit()

        await qdrant.close()

        # Invalidate Redis caches for institution_slug on successful completion
        try:
            redis_store = RedisStore()
            await redis_store.invalidate_institution_cache(institution_slug)
            await redis_store.close()
        except Exception as redis_err:
            logger.error(f"Failed to invalidate cache for {institution_slug}: {redis_err}", exc_info=True)

        wazobia_scrape_runs_total.labels(institution=institution_slug, status="completed").inc()

        summary = {
            "institution_slug": institution_slug,
            "pages_scraped": pages_scraped,
            "chunks_created": chunks_created,
            "points_upserted": points_upserted,
            "skipped_dedup": skipped_dedup,
            "duration_seconds": round(duration, 2),
        }
        logger.info(
            f"Ingestion pipeline completed successfully for {institution_slug}",
            extra=summary
        )
        return summary

    except Exception as exc:
        duration = time.time() - start_time
        wazobia_scrape_runs_total.labels(institution=institution_slug, status="failed").inc()
        logger.error(
            f"Ingestion pipeline failed for {institution_slug}: {exc}",
            exc_info=True,
            extra={
                "institution_slug": institution_slug,
                "pages_scraped": pages_scraped,
                "chunks_created": chunks_created,
                "duration_seconds": round(duration, 2),
                "error": str(exc),
            }
        )

        # Update ScrapeJob in Postgres to Failed
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(ScrapeJob)
                .where(ScrapeJob.id == job_id)
                .values(
                    status="failed",
                    completed_at=datetime.utcnow(),
                    error_message=str(exc),
                )
            )
            await session.commit()

        await qdrant.close()
        raise exc


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def scrape_institution(self, institution_slug: str) -> dict:
    """Orchestrates document scraping, cleaning, chunking, and indexing for an institution."""
    try:
        # Run async orchestrator synchronously within celery worker thread context
        return asyncio.run(_scrape_institution_async(institution_slug))
    except Exception as exc:
        # Implement Celery retries with exponential backoff on exceptions
        logger.warning(f"Retrying scrape task for {institution_slug} due to exception: {exc}")
        raise self.retry(exc=exc, countdown=60 * (2**self.request.retries))


@celery_app.task
def scrape_all_institutions() -> List[str]:
    """Fans out scraping jobs for all active institutions registered in the system."""
    active_institutions = list_institutions(active_only=True)
    task_ids = []
    for inst in active_institutions:
        # Trigger scrape task asynchronously via delay
        res = scrape_institution.delay(inst.slug)
        task_ids.append(res.id)
    logger.info(f"Fanned out scrape tasks for {len(active_institutions)} institutions. Task IDs: {task_ids}")
    return task_ids


@celery_app.task
def scrape_cbn_regulatory() -> dict:
    """Specialized task to scrape CBN regulatory guidelines and circulars."""
    try:
        return asyncio.run(_scrape_institution_async("cbn"))
    except Exception as exc:
        logger.error(f"CBN regulatory scrape failed: {exc}", exc_info=True)
        raise exc


async def _reindex_institution_async(institution_slug: str) -> dict:
    """Deletes existing points for an institution in Qdrant and then triggers scraping."""
    logger.warning(f"PERFORMING FULL RE-INDEX: Deleting all Qdrant vectors for {institution_slug}!")
    qdrant = QdrantStore()
    await qdrant.ensure_collection()
    await qdrant.delete_by_institution(institution_slug)
    await qdrant.close()

    # Trigger scrape institution async helper
    return await _scrape_institution_async(institution_slug)


@celery_app.task
def reindex_institution(institution_slug: str) -> dict:
    """Wipes an institution's vectors from Qdrant and triggers a fresh scrape run."""
    return asyncio.run(_reindex_institution_async(institution_slug))
