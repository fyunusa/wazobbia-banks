# Wazobia Agent — AI Agent Build Prompts
## RAG Pipeline: Step-by-Step Agent Instructions

> Feed each prompt to your AI coding agent **in sequence**.
> Each step builds on the previous. Do not skip steps.
> Stack: Python 3.11+ · FastAPI · Qdrant · PostgreSQL · Redis · Celery

---

## STEP 0 — Project Scaffold & Docker Environment

```
You are a senior Python engineer. Scaffold a production-grade FastAPI project called "wazobia-agent".

Requirements:
- Python 3.11+, use `uv` for dependency management (not pip directly)
- Create the following directory structure EXACTLY:

wazobia-agent/
├── api/
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   ├── voice.py
│   │   ├── institutions.py
│   │   └── health.py
│   ├── __init__.py
│   ├── dependencies.py
│   └── main.py
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── rag_query.py
│   ├── system_prompts.py
│   └── intent_classifier.py
├── voice/
│   ├── stt/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── whisper_engine.py
│   │   └── mms_engine.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── mms_tts.py
│   │   └── coqui_engine.py
│   ├── __init__.py
│   ├── vad.py
│   ├── language_detector.py
│   └── normalizer.py
├── ingestion/
│   ├── scrapers/
│   │   ├── __init__.py
│   │   ├── base_scraper.py
│   │   ├── gtbank.py
│   │   ├── zenith.py
│   │   ├── access.py
│   │   ├── kuda.py
│   │   ├── opay.py
│   │   └── cbn_regulatory.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── cleaner.py
│   ├── __init__.py
│   ├── tasks.py
│   └── scheduler.py
├── store/
│   ├── __init__.py
│   ├── qdrant_client.py
│   ├── postgres.py
│   └── redis_client.py
├── registry/
│   ├── __init__.py
│   └── institutions.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_voice.py
│   └── test_ingestion.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md

Infrastructure (docker-compose.yml):
- Qdrant: port 6333 (HTTP) + 6334 (gRPC), persist volume to ./data/qdrant
- PostgreSQL 16: port 5432, db=wazobia, user=wazobia, persist volume
- Redis 7: port 6379, persist volume
- All services on a shared Docker network: wazobia-net
- NO app service in docker-compose — we run FastAPI locally during dev

pyproject.toml dependencies (include all):
fastapi, uvicorn[standard], pydantic-settings, pydantic[email],
sqlalchemy[asyncio], asyncpg, alembic,
qdrant-client, openai,
celery[redis], redis,
httpx, playwright, trafilatura, beautifulsoup4, scrapy,
pdfplumber, langdetect, lingua-language-detector,
transformers, torch, torchaudio, soundfile,
python-multipart, aiofiles,
pytest, pytest-asyncio, pytest-cov

.env.example must include:
# ── LLM (Self-hosted vLLM on RunPod) ──────────────────────
LLM_BACKEND=vllm
LLM_BASE_URL=http://<your-runpod-ip>:8001/v1
LLM_MODEL=casperhansen/llama-3.1-8b-instruct-awq
LLM_API_KEY=<your-vllm-secret-key>
LLM_MAX_TOKENS=600
LLM_TEMPERATURE=0.1

# ── Embeddings (Cohere — free tier) ───────────────────────
COHERE_API_KEY=<from-dashboard.cohere.com>
EMBEDDING_MODEL=embed-multilingual-v3.0
EMBEDDING_DIMS=1024

# ── Infrastructure ─────────────────────────────────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=wazobia_kb
POSTGRES_DSN=postgresql+asyncpg://wazobia:wazobia@localhost:5432/wazobia
REDIS_URL=redis://localhost:6379/0

# ── App ────────────────────────────────────────────────────
ENVIRONMENT=production
LOG_LEVEL=INFO
ADMIN_API_KEY=<strong-random-secret>
VOICE_INFERENCE_ENABLED=true

# ── NO OpenAI. NO Groq. NO Anthropic. ─────────────────────


config/settings.py: Pydantic BaseSettings class loading from .env, with typed fields and sensible defaults.

config/logging.py: Structured JSON logging using Python's logging module, level from settings. Include request_id in all log records via contextvars.

api/main.py: 
- FastAPI app with lifespan context manager
- Mount all routers with /v1 prefix
- Global exception handler returning RFC 7807 Problem Details format
- CORS middleware (configurable origins from settings)
- Request ID middleware (UUID injected into each request, propagated to logs)
- Health router mounted at root (no /v1 prefix)

Do not implement router logic yet — just stubs returning {"status": "not_implemented"}.
Write clean, typed, production-grade Python throughout.
```

---

## STEP 1 — Institution Registry

```
You are a senior Python engineer working on the Wazobia Agent project.

Implement `registry/institutions.py` — the single source of truth for all Nigerian financial institutions the platform supports.

Requirements:

1. Define a Pydantic model `Institution` with fields:
   - slug: str (URL-safe, e.g. "gtbank")
   - name: str (short display name)
   - full_name: str (official CBN-registered name)
   - cbn_license_type: Literal["Commercial Bank", "Merchant Bank", "Microfinance Bank", "Fintech/MFB", "Mobile Money Operator"]
   - ussd_code: Optional[str]
   - customer_care: Optional[str]
   - hq: str
   - website: str
   - scrape_targets: list[ScrapeTarget]
   - logo_slug: str (filename without extension)
   - active: bool = True

2. Define `ScrapeTarget` Pydantic model:
   - url: str
   - category: Literal["fees", "products", "ussd", "rates", "complaints", "faq", "regulatory"]
   - requires_js: bool = False  # True = use Playwright, False = httpx+BS4
   - notes: Optional[str]

3. Populate the registry with these institutions (research their real public URLs):
   - gtbank, zenith, access, firstbank, uba, union, sterling, wema,
     fidelity, fcmb, stanbic, opay, kuda, moniepoint, palmpay

4. Expose:
   - INSTITUTIONS: dict[str, Institution]  — keyed by slug
   - get_institution(slug: str) -> Institution  — raises ValueError if not found
   - list_institutions(active_only: bool = True) -> list[Institution]

5. Also define LANGUAGE_CONFIG dict:
   {
     "en": {"name": "English", "mms_stt_code": None, "mms_tts_model": None, "whisper_lang": "en"},
     "ha": {"name": "Hausa", "mms_stt_code": "hau", "mms_tts_model": "facebook/mms-tts-hau", "whisper_lang": None},
     "yo": {"name": "Yoruba", "mms_stt_code": "yor", "mms_tts_model": "facebook/mms-tts-yor", "whisper_lang": None},
     "ig": {"name": "Igbo", "mms_stt_code": "ibo", "mms_tts_model": "facebook/mms-tts-ibo", "whisper_lang": None},
     "pcm": {"name": "Nigerian Pidgin", "mms_stt_code": None, "mms_tts_model": None, "whisper_lang": "en"},
   }

Write full type annotations. No placeholder strings — use real URLs from public bank websites.
```

---

## STEP 2 — Vector Store Client (Qdrant)

```
You are a senior Python engineer working on Wazobia Agent.

Implement `store/qdrant_client.py` — the async Qdrant wrapper for all vector store operations.

Requirements:

1. Use `qdrant-client` async client: `AsyncQdrantClient`
2. Collection name from settings: `settings.QDRANT_COLLECTION` (default: "wazobia_kb")
3. Vector size: 1536 (OpenAI text-embedding-3-small) — configurable via settings

4. Implement `QdrantStore` class with these async methods:

   a) `async def ensure_collection(self) -> None`
      - Idempotent: create collection only if it doesn't exist
      - Use Cosine distance
      - Enable payload indexing on: institution_slug, category, is_verified
      - Use HNSW index config: m=16, ef_construct=200 (production-quality)

   b) `async def upsert_points(self, points: list[KnowledgePoint]) -> UpsertResult`
      - Batch upsert, batch_size=100
      - KnowledgePoint is a Pydantic model:
        {
          id: str (UUIDv4),
          vector: list[float],
          institution_slug: str,
          institution_name: str,
          category: str,
          sub_category: Optional[str],
          content: str,
          source_url: str,
          scraped_at: datetime,
          content_hash: str,
          is_verified: bool = False,
          language: str = "en"
        }
      - Log upsert count and timing

   c) `async def search(self, query_vector: list[float], institution_slug: str, top_k: int = 5, category_filter: Optional[str] = None) -> list[SearchResult]`
      - Filter MUST include institution_slug match
      - Optionally also filter by category
      - Return SearchResult: {id, score, content, source_url, category, institution_slug, institution_name}
      - score_threshold: 0.35 (don't return garbage matches)

   d) `async def hybrid_search(self, query_vector: list[float], query_text: str, institution_slug: str, top_k: int = 5) -> list[SearchResult]`
      - Dense search (vector similarity) + sparse keyword match using Qdrant's built-in sparse vectors
      - Re-rank with RRF (Reciprocal Rank Fusion) — implement RRF manually
      - Return merged, deduplicated top_k results

   e) `async def delete_by_institution(self, institution_slug: str) -> int`
      - Delete all points for an institution (for full re-index)
      - Return count deleted

   f) `async def content_hash_exists(self, content_hash: str, institution_slug: str) -> bool`
      - Check if a content hash already exists (skip re-embedding unchanged content)

   g) `async def get_collection_stats(self) -> CollectionStats`
      - Return {total_points, points_per_institution: dict[str, int], indexed_at: datetime}

5. `store/redis_client.py`: implement async Redis wrapper with:
   - `get_cached_query(cache_key: str) -> Optional[str]`
   - `cache_query_result(cache_key: str, result: str, ttl_seconds: int = 3600) -> None`
   - `build_cache_key(query: str, institution_slug: str, language: str) -> str` (SHA256 hash)

6. `store/postgres.py`: implement async SQLAlchemy setup:
   - Async engine + session factory using `asyncpg`
   - Base declarative model
   - Models:
     - `ScrapeJob`: id, institution_slug, status, started_at, completed_at, pages_scraped, points_upserted, error_message
     - `QueryLog`: id, institution_slug, query_text, language, response_text, latency_ms, cache_hit, created_at
   - `get_db()` async generator for FastAPI dependency injection
   - Alembic configured for async migrations

Use dependency injection pattern — `QdrantStore` instantiated once at startup via FastAPI lifespan and injected via `api/dependencies.py`.
```

---

## STEP 3 — Document Ingestion: Scraper Layer

```
You are a senior Python engineer working on Wazobia Agent.

Implement the scraper layer in `ingestion/scrapers/`.

1. `ingestion/scrapers/base_scraper.py` — Abstract base class `BaseScraper`:

   Abstract interface:
   - `slug: str` (class-level)
   - `async def scrape(self) -> list[RawDocument]`

   `RawDocument` Pydantic model:
   - url: str
   - raw_html: Optional[str]
   - raw_text: Optional[str]
   - pdf_bytes: Optional[bytes]
   - category: str
   - sub_category: Optional[str]
   - scraped_at: datetime
   - institution_slug: str
   - http_status: int
   - content_type: str  # "html" | "pdf" | "text"

   Base class must provide:
   - `async def fetch_html(self, url: str, use_playwright: bool = False) -> tuple[str, int]`
     - httpx with 30s timeout, retry 3x with exponential backoff for 5xx
     - Playwright fallback for JS-rendered pages (use async playwright)
     - Rotate User-Agent strings (list of 5 real browser UAs)
     - Respect robots.txt — check before scraping
     - Rate limit: 2 req/sec per domain (asyncio.Semaphore)
   - `async def fetch_pdf(self, url: str) -> tuple[bytes, int]`
   - Structured logging for every fetch (url, status, timing, institution)

2. Implement scrapers for these institutions using their real public URLs.
   Each scraper must iterate through its institution's `scrape_targets` from the registry.
   Implement: `gtbank.py`, `zenith.py`, `access.py`, `kuda.py`, `opay.py`

   Each scraper:
   - Inherits BaseScraper
   - Sets `slug` matching registry key
   - `scrape()` iterates through `INSTITUTIONS[slug].scrape_targets`
   - Routes to `fetch_html()` or `fetch_pdf()` based on `content_type` and `requires_js`
   - Returns list of RawDocument

3. `ingestion/scrapers/cbn_regulatory.py`:
   - Scrapes CBN's public circulars page (https://www.cbn.gov.ng/supervision/Cir.asp)
   - Fetches and downloads any PDF circulars related to banking tariffs/fees
   - Marks category as "regulatory"
   - institution_slug = "cbn" (applies to all institutions)

Write production-grade async code. Robust error handling — a failed scrape for one URL must not abort the entire scraper run. Log all failures with full context.
```

---

## STEP 4 — Document Processing Pipeline

```
You are a senior Python engineer working on Wazobia Agent.

Implement the document processors in `ingestion/processors/`.

1. `ingestion/processors/cleaner.py` — `DocumentCleaner` class:

   `async def clean(self, doc: RawDocument) -> CleanedDocument`

   CleanedDocument model:
   - All RawDocument fields (minus raw_html/pdf_bytes)
   - clean_text: str
   - title: Optional[str]
   - extracted_tables: list[str]  # Tables serialized as markdown
   - word_count: int
   - language_detected: str  # ISO 639-1

   Cleaning logic:
   - HTML: use Trafilatura for main content extraction (`trafilatura.extract()`), fallback to BeautifulSoup
   - Strip navigation, footers, cookie banners, ads (CSS selector blocklist)
   - PDF: use pdfplumber — extract text page by page, also extract tables as markdown
   - Normalize Nigerian Naira: "N5,000", "5000 naira", "NGN5000" → "₦5,000"
   - Normalize USSD codes: ensure format `*XXX#`
   - Remove excessive whitespace, normalize unicode
   - Detect language using `langdetect` (expect mostly "en")
   - Extract page `<title>` or first `<h1>` as document title

2. `ingestion/processors/chunker.py` — `SemanticChunker` class:

   Strategy: sentence-boundary-aware chunking (NOT naive character split)

   `def chunk(self, doc: CleanedDocument) -> list[Chunk]`

   Chunk model:
   - id: str (UUIDv4)
   - content: str
   - chunk_index: int
   - total_chunks: int
   - institution_slug: str
   - category: str
   - sub_category: Optional[str]
   - source_url: str
   - title: Optional[str]
   - scraped_at: datetime
   - content_hash: str  # SHA256 of content

   Chunking rules:
   - Target chunk size: 512 tokens (use tiktoken cl100k_base for counting)
   - Overlap: 64 tokens between consecutive chunks
   - NEVER split mid-sentence — use sentence tokenization (split on ". ", "? ", "! ")
   - NEVER split a table row — keep extracted_tables as atomic chunks (one table = one chunk regardless of size)
   - Always prepend document title + category to every chunk:
     `"[GTBank | Fees & Charges]\n\n{content}"`
     This gives the LLM context even without the surrounding document.
   - Minimum chunk size: 50 tokens — discard smaller fragments
   - content_hash: SHA256(content) — used for dedup

3. `ingestion/processors/embedder.py` — `Embedder` class:

   `async def embed_chunks(self, chunks: list[Chunk]) -> list[KnowledgePoint]`

   - Use OpenAI `text-embedding-3-small` (1536 dims) via async `openai.AsyncOpenAI`
   - Batch chunks into groups of 100 (API limit)
   - Retry with exponential backoff on rate limit (429) errors
   - Map each chunk → KnowledgePoint (for Qdrant upsert)
   - Log total tokens consumed per batch (for cost tracking)
   - Support BGE-M3 as offline alternative: detect from `settings.EMBEDDING_BACKEND`
     ("openai" | "bge") and dispatch accordingly
   - BGE-M3 path: use `transformers` SentenceTransformer — run on CPU by default

All processors must be stateless and independently testable. Write unit tests in `tests/test_ingestion.py` covering: cleaning a real HTML snippet, chunking a 2000-word document, verifying chunk overlap, and dedup via content_hash.
```

---

## STEP 5 — Full Ingestion Pipeline Orchestration

```
You are a senior Python engineer working on Wazobia Agent.

Implement the orchestrated ingestion pipeline that wires scrapers → cleaner → chunker → embedder → Qdrant.

1. `ingestion/tasks.py` — Celery tasks:

   Configure Celery:
   - Broker: Redis (settings.REDIS_URL)
   - Result backend: Redis
   - Serializer: json
   - Task soft_time_limit: 600s, hard_time_limit: 900s
   - Enable task retries with exponential backoff

   Tasks:

   a) `@celery_app.task(bind=True, max_retries=3)`
      `def scrape_institution(self, institution_slug: str) -> dict`
      - Run the full pipeline for one institution:
        1. Get scraper for slug (dynamic import from scrapers/)
        2. scraper.scrape() → list[RawDocument]
        3. For each doc: cleaner.clean() → CleanedDocument
        4. chunker.chunk() → list[Chunk]
        5. Filter chunks: skip if content_hash already in Qdrant (dedup)
        6. embedder.embed_chunks() → list[KnowledgePoint]
        7. qdrant.upsert_points()
        8. Log ScrapeJob record to PostgreSQL
      - Return summary dict: {institution_slug, pages_scraped, chunks_created, points_upserted, skipped_dedup, duration_seconds}
      - On failure: update ScrapeJob.status = "failed", store error_message, re-raise for retry

   b) `@celery_app.task`
      `def scrape_all_institutions() -> list[dict]`
      - Fan out: `scrape_institution.delay(slug)` for all active institutions
      - Return list of task IDs (not blocking)

   c) `@celery_app.task`
      `def scrape_cbn_regulatory() -> dict`
      - Run CBN scraper only
      - Upsert results into Qdrant without institution_slug filter (institution_slug = "cbn")

   d) `@celery_app.task`
      `def reindex_institution(institution_slug: str) -> dict`
      - Delete ALL existing Qdrant points for institution (full re-index)
      - Then run scrape_institution
      - Use with caution — log a warning before deletion

2. `ingestion/scheduler.py` — Celery Beat schedule:

   - Weekly full scrape: Monday 02:00 WAT (UTC+1)
   - Daily CBN regulatory: 06:00 WAT
   - Dedup cleanup task: runs after each upsert batch (inline, not scheduled)

3. `api/routers/institutions.py` — expose ingestion control endpoints:

   `GET /v1/institutions` — list all institutions with status
   `GET /v1/institutions/{slug}` — single institution detail
   `POST /v1/institutions/{slug}/ingest` — trigger scrape_institution.delay(), return task_id
   `GET /v1/institutions/{slug}/stats` — Qdrant collection stats filtered by slug
   `GET /v1/ingest/tasks/{task_id}` — Celery task status (PENDING/STARTED/SUCCESS/FAILURE + result)

   All endpoints: require API key auth via `X-API-Key` header (check against settings.ADMIN_API_KEY).

Write the Celery worker startup command in README. The pipeline must be runnable with:
   `celery -A ingestion.tasks worker --loglevel=info -c 4`
   `celery -A ingestion.tasks beat --loglevel=info`
```

---

## STEP 6 — RAG Query Engine

```
You are a senior Python engineer working on Wazobia Agent.

Implement the RAG query engine in `agents/rag_query.py` and `agents/system_prompts.py`.

IMPORTANT ARCHITECTURE DECISION:
LLM is self-hosted Llama 3.1 8B (AWQ quantized) via vLLM on RunPod.
- Use openai.AsyncOpenAI client pointing at settings.LLM_BASE_URL
- Add stop tokens: ["<|eot_id|>", "<|end_of_text|>"]
- Add repetition_penalty: 1.1 via extra_body
- No OpenAI API key — use settings.LLM_API_KEY (vLLM's own key)
- Dev fallback: Ollama on localhost:11434 (same interface)
- Do NOT import anthropic or openai packages for LLM — only openai SDK
  pointed at vLLM base URL
  

1. `agents/system_prompts.py`:

   `def build_system_prompt(institution_slug: str, language: str) -> str`

   The prompt must:
   - Address the AI as a dedicated agent for that specific institution
   - Instruct it to ONLY use provided context — no hallucination
   - Tell it to explicitly say "I don't have that information" if context is insufficient
   - Include institution facts: full name, USSD code, customer care, CBN license type
   - Specify response language (en | ha | yo | ig | pcm)
   - Include Nigerian financial context rules:
     * Always quote amounts in ₦
     * Mention USSD codes where relevant
     * Add disclaimer: "Please verify directly with the bank or CBN for current rates"
     * Never give financial advice — information only
   - Keep prompt under 800 tokens (costs matter at scale)

2. `agents/rag_query.py` — `RAGQueryEngine` class:

   Dependencies injected via constructor: QdrantStore, Embedder, Redis cache, AsyncOpenAI

   `async def query(self, request: QueryRequest) -> QueryResponse`

   QueryRequest model:
   - query: str (max 500 chars)
   - institution_slug: str
   - language: str = "en"
   - top_k: int = 5
   - use_hybrid: bool = True  # dense + sparse search
   - user_id: Optional[str]

   QueryResponse model:
   - answer: str
   - sources: list[str] (URLs)
   - institution_slug: str
   - language: str
   - confidence: float  # top retrieval score
   - cache_hit: bool
   - latency_ms: int

   Implementation:

   Step 1 — Cache check:
   - Build cache key: SHA256(f"{query}:{institution_slug}:{language}")
   - Return cached response if exists (mark cache_hit=True)

   Step 2 — Query translation:
   - If language != "en": translate query to English using GPT with a minimal prompt
   - Keep original query for response translation later

   Step 3 — Embed query:
   - Embed the English query text

   Step 4 — Retrieve:
   - If use_hybrid=True: call qdrant.hybrid_search()
   - Else: call qdrant.search()
   - If results empty or all scores < 0.35: return "I don't have information about that for {institution_name}"

   Step 5 — Build context:
   - Join top_k chunks with separator
   - Truncate total context to 3000 tokens (tiktoken)
   - Include source URLs

   Step 6 — LLM generation:
   - OpenAI chat completion (settings.LLM_MODEL)
   - system: build_system_prompt(slug, language)
   - user: f"Context:\n{context}\n\nQuestion: {original_query}"
   - max_tokens: 600, temperature: 0.1 (factual, low creativity)
   - Stream: False for REST, True for WebSocket endpoint

   Step 7 — Translate response:
   - If language != "en": translate LLM response to target language

   Step 8 — Cache + log:
   - Cache response with 1-hour TTL
   - Log QueryLog record to PostgreSQL (async, non-blocking)
   - Return QueryResponse with latency_ms

3. `agents/intent_classifier.py`:

   `async def is_comparative_query(self, query: str) -> bool`
   - Detect if query compares multiple institutions
   - Use simple keyword matching first (fast path): ["which bank", "compare", "best bank", "cheapest", "vs", "versus"]
   - If ambiguous: one LLM call with classification prompt
   - Return bool

   `async def extract_institution_slug(self, query: str) -> Optional[str]`
   - Try to identify which institution the query is about
   - Match institution names, common aliases ("GT", "Access", "First Bank", "Konga"→None)
   - Return slug or None

4. `api/routers/query.py` — expose query endpoints:

   `POST /v1/query`
   Body: QueryRequest
   Response: QueryResponse
   - Validate institution_slug against registry
   - No auth required (public endpoint)
   - Rate limit: 30 req/min per IP (Redis sliding window)

   `GET /v1/query/institutions/{slug}/suggested-questions`
   - Return 5 hardcoded starter questions per institution
   - Example for GTBank: ["What are GTBank transfer fees?", "How do I open a GTBank account?", ...]

Write unit tests in `tests/test_rag.py`:
- Mock Qdrant returning 3 results → verify answer contains institution name
- Mock Qdrant returning empty → verify graceful "no information" response
- Verify cache hit path skips embedding + LLM calls
- Verify query with language="yo" triggers translation steps
```

---

## STEP 7 — Multi-Agent Orchestrator

```
You are a senior Python engineer working on Wazobia Agent.

Implement `agents/orchestrator.py` — the multi-agent orchestration layer for cross-institution queries.

1. `MultiAgentOrchestrator` class:

   `async def route(self, query: str, language: str, institution_slug: Optional[str] = None) -> OrchestratorResponse`

   OrchestratorResponse model:
   - response_type: Literal["single_agent", "comparative", "unknown_institution", "out_of_scope"]
   - answers: list[QueryResponse]  # 1 for single, N for comparative
   - synthesis: Optional[str]  # Only for comparative — the merged comparison answer
   - institutions_queried: list[str]

   Routing logic:
   a) If institution_slug is provided:
      - Validate slug exists in registry
      - Route directly to RAGQueryEngine.query()
      - Return response_type="single_agent"

   b) If institution_slug is None:
      - Run intent_classifier.extract_institution_slug(query)
      - If slug found: route to single agent
      - If not found: run intent_classifier.is_comparative_query(query)
        - If comparative: fan out to ALL active institutions (asyncio.gather)
        - Filter out responses with confidence < 0.4 (institution doesn't have this info)
        - Synthesize comparison (see below)
        - Return response_type="comparative"
      - If not comparative and no institution: return response_type="out_of_scope"

2. Comparative synthesis:
   `async def synthesize_comparison(self, query: str, responses: list[QueryResponse], language: str) -> str`
   
   - Build a structured comparison prompt:
     "Based on these answers from different Nigerian banks, provide a concise comparison table in markdown..."
   - Include each institution's answer as context
   - Ask LLM to format as a markdown table where applicable
   - Translate to target language if needed

3. Fan-out implementation must be:
   - True async parallel: `asyncio.gather(*tasks, return_exceptions=True)`
   - Catch exceptions per-institution (one bank failing must not kill the whole response)
   - Timeout per agent: 10 seconds (asyncio.wait_for)
   - Log which institutions timed out

4. `api/routers/query.py` — add new endpoint:

   `POST /v1/query/multi`
   Body: {query: str, language: str = "en", institution_slug: Optional[str]}
   Response: OrchestratorResponse
   - Routes through MultiAgentOrchestrator
   - Same rate limiting as /v1/query

5. Add to `api/routers/query.py`:
   `GET /v1/query/logs` (admin only, API key required)
   - Return paginated QueryLog records from PostgreSQL
   - Filter by: institution_slug, language, date range, cache_hit
   - Use for analytics — what are Nigerians actually asking?
```

---

## STEP 8 — STT Engine (Speech-to-Text)

```
You are a senior Python engineer working on Wazobia Agent.

Implement the STT (Speech-to-Text) layer in `voice/stt/`.

1. `voice/stt/base.py` — `BaseSTTEngine` abstract class:
   
   `async def transcribe(self, audio_bytes: bytes, language_hint: Optional[str] = None) -> TranscriptionResult`
   
   TranscriptionResult model:
   - transcript: str
   - detected_language: str  # ISO 639-1
   - confidence: Optional[float]
   - duration_seconds: float
   - engine_used: str

2. `voice/stt/whisper_engine.py` — `WhisperSTTEngine`:

   - Load model: `openai/whisper-large-v3` via HuggingFace `transformers`
   - Use `AutomaticSpeechRecognitionPipeline`
   - Device: CUDA if available, else CPU
   - Supported languages: "en", "pcm" (route Pidgin as "en" with Nigerian context hint)
   - Audio preprocessing:
     * Accept: WAV, WebM, MP3, OGG (convert to 16kHz mono WAV using torchaudio)
     * Max duration: 60 seconds — reject longer audio
     * Normalize audio amplitude before inference
   - Implement as singleton (model loaded once at startup, not per-request)
   - Return transcript + detected_language from Whisper's output

3. `voice/stt/mms_engine.py` — `MMSSTTEngine`:

   - Use Meta MMS model: `facebook/mms-300m` via HuggingFace `Wav2Vec2ForCTC`
   - Supported language codes: "ha" (hau), "yo" (yor), "ig" (ibo)
   - Language routing:
     * Map our ISO codes to MMS adapter codes: {"ha": "hau", "yo": "yor", "ig": "ibo"}
     * Load adapter: `model.load_adapter(mms_code)`
     * `processor.tokenizer.set_target_lang(mms_code)`
   - Load model once at startup, swap adapters per request (NOT reload model)
   - Audio preprocessing: same as Whisper (16kHz mono WAV)
   - Handle `asyncio.run_in_executor` for CPU-bound inference (keep FastAPI non-blocking)

4. `voice/language_detector.py` — `LanguageDetector`:

   `def detect(self, text: Optional[str] = None, audio_bytes: Optional[bytes] = None) -> str`

   - If text provided: use `lingua-language-detector` (more accurate than langdetect for African langs)
     * LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.YORUBA, Language.HAUSA, Language.IGBO)
     * Fallback: langdetect
   - If audio only: return "auto" (will be determined post-transcription)
   - If confidence < 0.7: default to "en"

5. `voice/normalizer.py` — `TranscriptNormalizer`:

   `def normalize(self, transcript: str, language: str) -> str`

   Rules:
   - Nigerian Pidgin corrections: "abeg" stays, "wetin" stays (preserve Pidgin)
   - Code-switching handling: if Yoruba transcript contains English bank terms, don't translate those terms
   - Named entity preservation: bank names, USSD codes, ₦ amounts must survive normalization unchanged
   - Remove filler words: "erm", "uhh", "like" (English fillers only)
   - Normalize "naira" → "₦", "kobo" stays as is
   - Strip leading/trailing whitespace, normalize internal whitespace

All inference (Whisper, MMS) must run in `asyncio.run_in_executor(None, ...)` — never block the event loop.
Both engines must be lazy-loaded (model downloaded on first use, cached to ~/.cache/huggingface).
Write a CLI test script `scripts/test_stt.py` that takes an audio file path + language code and prints the transcript.
```

---

## STEP 9 — TTS Engine (Text-to-Speech)

```
You are a senior Python engineer working on Wazobia Agent.

Implement the TTS (Text-to-Speech) layer in `voice/tts/`.

1. `voice/tts/base.py` — `BaseTTSEngine` abstract class:

   `async def synthesize(self, text: str, language: str) -> TTSResult`

   TTSResult model:
   - audio_bytes: bytes  # WAV format, 22050Hz
   - language: str
   - duration_seconds: float
   - engine_used: str
   - sample_rate: int

2. `voice/tts/mms_tts.py` — `MMSTTSEngine`:

   Supported languages: "ha", "yo", "ig"
   Models:
   - Hausa: `facebook/mms-tts-hau`
   - Yoruba: `facebook/mms-tts-yor`
   - Igbo: `facebook/mms-tts-ibo`

   Implementation:
   - Use HuggingFace `VitsModel` + `AutoTokenizer`
   - Load ALL three models at startup (memory trade-off for latency)
   - Each language has its own model instance (no adapter swapping for TTS)
   - `model(**inputs).waveform` → convert tensor to WAV bytes via `soundfile`
   - Target sample rate: 16000Hz (MMS TTS native)
   - Text preprocessing per language:
     * Strip markdown formatting before TTS
     * Split long text (>500 chars) into sentences, synthesize each, concatenate audio
     * Nigerian financial terms that should be spelled out: "₦" → "naira", "*737#" → "star seven three seven hash"
   - Run inference in executor (CPU-bound)

3. `voice/tts/coqui_engine.py` — `CoquiTTSEngine`:

   Supported languages: "en", "pcm" (Nigerian English/Pidgin fallback)
   - Use Coqui TTS: `TTS` class from `TTS` package
   - Model: `tts_models/en/ljspeech/tacotron2-DDC` (fast, good quality)
   - For Nigerian accent: use `tts_models/en/vctk/vits` with a speaker close to West African accent
   - Synthesize to WAV bytes (write to BytesIO, not disk)
   - Run in executor

4. `voice/tts/router.py` — `TTSRouter`:

   `async def synthesize(self, text: str, language: str) -> TTSResult`
   
   Routes to correct engine:
   - "ha" | "yo" | "ig" → MMSTTSEngine
   - "en" | "pcm" → CoquiTTSEngine
   - Unknown language → CoquiTTSEngine (safe fallback)

5. `voice/vad.py` — `VoiceActivityDetector`:

   For WebSocket streaming:
   `def is_end_of_speech(self, audio_buffer: bytes, sample_rate: int = 16000) -> bool`
   
   - Use energy-based VAD: compute RMS energy of last 500ms
   - If energy drops below threshold for 1.5s after speech detected → end of speech
   - Simple but effective for Nigerian network conditions (don't use heavy WebRTC VAD)
   - Returns True when user has finished speaking

Write a CLI test script `scripts/test_tts.py` that takes text + language code and saves output as `output.wav`.
Write unit tests mocking the model inference to test routing logic and text preprocessing.
```

---

## STEP 10 — Voice API Endpoints

```
You are a senior Python engineer working on Wazobia Agent.

Implement `api/routers/voice.py` — the full voice API surface.

1. `POST /v1/voice/query` — REST voice endpoint:

   Request: multipart/form-data
   - audio: UploadFile (WAV | WebM | MP3 | OGG, max 10MB)
   - institution_slug: str (Form field)
   - preferred_language: str = "auto" (Form field)
   - user_id: Optional[str] (Form field)

   Response: StreamingResponse (audio/wav)
   Headers on response:
   - X-Transcript: URL-encoded transcript text
   - X-Language: detected language code
   - X-Institution: institution slug
   - X-Sources: JSON array of source URLs
   - X-Confidence: float string
   - X-Latency-Ms: total end-to-end latency

   Pipeline:
   1. Validate file size and mime type
   2. Read audio bytes
   3. Language detection from preferred_language ("auto" → detect post-STT)
   4. STT: WhisperSTTEngine if lang in ["en","pcm","auto"], MMSSTTEngine if lang in ["ha","yo","ig"]
   5. Normalize transcript
   6. RAG query (QueryRequest with detected language)
   7. TTS synthesis (TTSRouter)
   8. Return audio bytes as StreamingResponse
   9. Log full pipeline metrics (STT latency, RAG latency, TTS latency, total)

2. `WebSocket /v1/voice/stream/{institution_slug}`:

   Protocol:
   Client sends: binary frames (raw audio chunks, 16kHz PCM)
   Client sends: JSON frame {"type": "language", "lang": "yo"} to set language
   Client sends: JSON frame {"type": "end"} to signal manual end of speech

   Server sends:
   - {"type": "transcript", "text": "...", "lang": "yo"} — as soon as STT completes
   - {"type": "answer_text", "text": "..."} — RAG answer text
   - binary frame: TTS audio bytes
   - {"type": "done", "sources": [...], "confidence": 0.87}
   - {"type": "error", "message": "..."}

   Implementation:
   - Buffer audio chunks until VAD detects end of speech OR client sends {"type": "end"}
   - Run full pipeline (STT → RAG → TTS)
   - Send responses interleaved (transcript first for fast feedback UX)
   - Handle WebSocket disconnect gracefully (cleanup buffers)
   - Timeout: if no audio for 30s, close connection

3. Rate limiting on voice endpoints:
   - 10 voice requests/min per IP (REST endpoint)
   - Max 5 concurrent WebSocket connections per IP
   - Implement using Redis (sliding window counter)

4. Add audio format validation utility:
   `voice/utils.py` → `validate_and_convert_audio(audio_bytes: bytes) -> bytes`
   - Detect format using file magic bytes
   - Convert to 16kHz mono WAV using torchaudio if not already
   - Raise HTTPException(415) for unsupported formats

Include full OpenAPI documentation on all endpoints (description, response schema, example values).
Write integration test in `tests/test_voice.py` using a real 3-second WAV test fixture.
```

---

## STEP 11 — Hardening, Observability & Production Readiness

```
You are a senior Python engineer working on Wazobia Agent.

Harden the application for production deployment.

1. Security:

   a) Rate limiting (already partial — make comprehensive):
      - Sliding window in Redis for all public endpoints
      - Endpoint-specific limits: query=30/min, voice=10/min, ingest=5/hour
      - Return 429 with Retry-After header

   b) Input validation (add to all endpoints):
      - Sanitize query text: strip HTML, limit to 500 chars, reject null bytes
      - Validate institution_slug strictly against registry (no arbitrary slugs)
      - Audio upload: validate magic bytes (not just mimetype header)

   c) API key middleware:
      - Admin endpoints (ingest trigger, query logs) require X-API-Key
      - Validate against settings.ADMIN_API_KEY (bcrypt comparison, not plain string)
      - Log all admin actions with IP and timestamp

2. Observability:

   a) Structured logging (enhance existing):
      - Every request: method, path, status, latency_ms, request_id, user_ip
      - Every RAG query: institution_slug, language, cache_hit, top_score, llm_tokens_used
      - Every voice request: stt_engine, tts_engine, audio_duration, stt_latency, tts_latency
      - Every scrape: institution, pages_scraped, chunks_created, duration, errors

   b) `/health` endpoint (enhance existing):
      - Check Qdrant connection + collection exists
      - Check PostgreSQL connection
      - Check Redis connection
      - Check OpenAI API reachability (lightweight ping)
      - Return {"status": "healthy"|"degraded"|"unhealthy", "checks": {...}}

   c) `/metrics` endpoint (Prometheus format):
      - wazobia_queries_total (labels: institution, language, cache_hit)
      - wazobia_query_latency_seconds (histogram)
      - wazobia_voice_requests_total (labels: stt_engine, tts_engine, language)
      - wazobia_scrape_runs_total (labels: institution, status)
      - wazobia_qdrant_points_total (gauge, per institution)
      - Use `prometheus-client` library

3. Caching strategy (complete):
   - Query results: 1h TTL (already implemented — verify it works)
   - Institution metadata: 24h TTL (rarely changes)
   - Qdrant collection stats: 5min TTL
   - Cache invalidation: on successful ingest, clear all cache keys for that institution_slug

4. Async correctness audit:
   - Review all DB calls — must use async SQLAlchemy (no sync calls in async context)
   - Review all Qdrant calls — must use AsyncQdrantClient
   - Review all model inference — must use run_in_executor (no blocking in event loop)
   - Review Celery tasks — they are sync workers by design (correct)

5. Graceful shutdown:
   - FastAPI lifespan: on shutdown, complete in-flight requests, close DB pool, close Qdrant client
   - Celery: SIGTERM handling for graceful task completion

6. Docker production build:
   - Multi-stage Dockerfile: builder stage (install deps) + runtime stage (copy only needed files)
   - Non-root user in container
   - Health check instruction in Dockerfile
   - Environment variables via .env (never bake secrets into image)

7. Alembic migrations:
   - Generate initial migration for ScrapeJob + QueryLog tables
   - Add `alembic upgrade head` to application startup (with lock to prevent race condition on multi-instance deploy)

8. README.md — write complete setup guide:
   - Prerequisites
   - `cp .env.example .env` and fill values
   - `docker compose up -d`
   - `uv run alembic upgrade head`
   - `uv run uvicorn api.main:app --reload`
   - Celery worker + beat commands
   - How to trigger first ingest
   - How to run tests: `uv run pytest tests/ -v --cov`
```

---

## STEP 12 — End-to-End Integration Test & Demo Script

```
You are a senior Python engineer working on Wazobia Agent.

Write a complete end-to-end integration test and a CLI demo script.

1. `tests/test_e2e.py` — full pipeline integration test (requires live services):

   Mark all tests with `@pytest.mark.integration` (skip in CI without services).

   Test cases:

   a) `test_gtbank_text_query_english`:
      - Trigger GTBank ingest (mock scraper returning 3 real-looking documents)
      - Query: "What are the charges for transferring money with GTBank?"
      - Assert: response contains "₦" and a numeric amount
      - Assert: sources list non-empty
      - Assert: confidence > 0.4

   b) `test_zenith_query_yoruba`:
      - Query: "Elo ni owo transfer fun Zenith Bank?" (How much is transfer fee for Zenith Bank?)
      - language = "yo"
      - Assert: response is in Yoruba (detect with lingua)
      - Assert: response contains Zenith Bank

   c) `test_comparative_query`:
      - Query: "Which bank has the cheapest transfer fees?"
      - institution_slug = None
      - Assert: OrchestratorResponse.response_type == "comparative"
      - Assert: multiple institutions in institutions_queried

   d) `test_voice_pipeline_english`:
      - Use test fixture: `tests/fixtures/gtbank_query_en.wav` (generate via TTS in fixture setup)
      - POST to /v1/voice/query with institution_slug="gtbank"
      - Assert: HTTP 200
      - Assert: X-Transcript header non-empty
      - Assert: response body is valid WAV (check magic bytes: RIFF header)

   e) `test_cache_hit`:
      - Run same query twice
      - Second response: assert cache_hit=True, latency_ms significantly lower

   f) `test_unknown_institution_graceful`:
      - Query with institution_slug="not_a_real_bank"
      - Assert: HTTP 422 (validation error, not 500)

2. `scripts/demo.py` — CLI demo script:

   A runnable demo that showcases the full pipeline:

   ```
   python scripts/demo.py --institution gtbank --query "What are GTBank transfer fees?" --language en
   python scripts/demo.py --institution kuda --query "Kuda bank fixed deposit rates" --language en
   python scripts/demo.py --voice --institution gtbank --audio tests/fixtures/sample_query.wav --language yo
   ```

   Output format:
   ```
   ═══════════════════════════════════════════
   WAZOBIA AGENT — GTBank Agent
   ═══════════════════════════════════════════
   Query     : What are GTBank transfer fees?
   Language  : English
   Cache Hit : No
   Latency   : 1,243ms
   Confidence: 0.87
   ───────────────────────────────────────────
   ANSWER:
   GTBank charges ₦52.50 per NIP transfer...
   ───────────────────────────────────────────
   SOURCES:
   · https://www.gtbank.com/personal-banking/fees-charges
   ═══════════════════════════════════════════
   ```

3. `scripts/ingest_all.py` — one-shot ingest script:
   
   - Calls `POST /v1/institutions/{slug}/ingest` for all active institutions sequentially
   - Polls task status every 5 seconds
   - Prints progress table:
     ```
     Institution   | Status    | Pages | Chunks | Points | Duration
     GTBank        | SUCCESS   | 12    | 847    | 847    | 43s
     Zenith        | SUCCESS   | 9     | 623    | 598    | 38s
     FirstBank     | FAILED    | -     | -      | -      | error: timeout
     ```
   - Exit code 0 if all succeed, 1 if any fail

4. Generate `tests/fixtures/sample_query_en.wav`:
   - Use CoquiTTSEngine to synthesize "What are the transfer charges for GTBank?" 
   - Save as WAV to tests/fixtures/ directory
   - This fixture is used by voice integration tests

Ensure all tests pass with: `uv run pytest tests/ -v -m "not integration"`
The integration tests should pass with: `uv run pytest tests/ -v -m integration` (when services are running)
```

---

## BUILD ORDER SUMMARY

| Step | What Gets Built | Estimated Time |
|------|----------------|----------------|
| 0 | Project scaffold + Docker Compose | 2–3 hours |
| 1 | Institution registry | 1–2 hours |
| 2 | Qdrant + Redis + PostgreSQL clients | 2–3 hours |
| 3 | Scraper layer (5 banks) | 3–4 hours |
| 4 | Cleaner + Chunker + Embedder | 3–4 hours |
| 5 | Celery ingestion pipeline + admin endpoints | 2–3 hours |
| 6 | RAG query engine + query endpoint | 3–4 hours |
| 7 | Multi-agent orchestrator + comparative queries | 2–3 hours |
| 8 | STT engines (Whisper + MMS) | 3–4 hours |
| 9 | TTS engines (MMS + Coqui) | 2–3 hours |
| 10 | Voice REST + WebSocket endpoints | 3–4 hours |
| 11 | Hardening + observability + production readiness | 3–4 hours |
| 12 | E2E tests + demo scripts | 2–3 hours |
| **TOTAL** | **Full production-ready RAG + Voice backend** | **~30–40 hours** |

---

## IMPORTANT NOTES FOR YOUR AGENT

- Run steps **strictly in order** — each step imports from previous steps
- After each step, run `uv run pytest tests/ -v` before proceeding
- All secrets via `.env` — never hardcode
- Qdrant, PostgreSQL, Redis must be running (`docker compose up -d`) before any step that touches the store
- MMS and Whisper models are large (~1–3GB each) — first run will download to `~/.cache/huggingface/`
- For development, `EMBEDDING_BACKEND=bge` avoids OpenAI costs during testing
- The demo is your Twitter/X moment — make Step 12 beautiful

---

*Wazobia Agent — Build Prompts v1.0*
*Architecture: Umar Yunusa*
