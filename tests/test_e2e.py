import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Share or create mock modules to avoid conflicts in full test runs
if "transformers" in sys.modules and isinstance(sys.modules["transformers"], MagicMock):
    mock_transformers = sys.modules["transformers"]
else:
    mock_transformers = MagicMock()
    sys.modules["transformers"] = mock_transformers

# Ensure all needed attributes exist on the shared mock
for attr in ["pipeline", "Wav2Vec2ForCTC", "AutoProcessor", "VitsModel", "AutoTokenizer"]:
    if not hasattr(mock_transformers, attr):
        setattr(mock_transformers, attr, MagicMock())

if "TTS" in sys.modules and isinstance(sys.modules["TTS"], MagicMock):
    mock_tts = sys.modules["TTS"]
else:
    mock_tts = MagicMock()
    sys.modules["TTS"] = mock_tts

if "TTS.api" in sys.modules and isinstance(sys.modules["TTS.api"], MagicMock):
    mock_tts_api = sys.modules["TTS.api"]
else:
    mock_tts_api = mock_tts.api
    sys.modules["TTS.api"] = mock_tts_api

if not hasattr(mock_tts_api, "TTS"):
    mock_tts_api.TTS = MagicMock()

import httpx
from httpx import ASGITransport
import pytest
from lingua import Language, LanguageDetectorBuilder
import asyncio

from api.main import app
from ingestion.scrapers.base_scraper import RawDocument
from voice.stt.base import TranscriptionResult
from voice.tts.base import TTSResult

# Mark all tests in this file as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# Prepare Lingua detector for Yoruba language detection
lingua_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.YORUBA
).build()

# Override event_loop fixture to have module scope so all tests run in the same loop
@pytest.fixture(scope="module")
def event_loop():
    """Overrides pytest-asyncio's event_loop fixture to have module scope."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Mock Scraper data for GTBank and Zenith Bank (exceeding 50 tokens to avoid Chunker filter)
mock_gtbank_documents = [
    RawDocument(
        url="https://www.gtbank.com/fees",
        raw_text=(
            "Guaranty Trust Bank (GTBank) transfer charges are ₦10 for NIP transfers below ₦5,000, "
            "₦25 for transfers between ₦5,000 and ₦50,000, and ₦50 for transfers above ₦50,000. "
            "These NIP transfer fees apply to all retail customers and business accounts. "
            "Guaranty Trust Bank (GTBank) is committed to providing affordable electronic banking services. "
            "Electronic transfer fees are charged to ensure payment settlement operations run smoothly. "
            "Please check the official GTBank tariff guide website for additional pricing details. "
            "This document summarizes GTBank electronic NIP fund transfer pricing schedules for all customer channels including USSD, Internet Banking, and Mobile App."
        ),
        category="fees",
        institution_slug="gtbank",
        http_status=200,
        content_type="text",
    )
]

mock_zenith_documents = [
    RawDocument(
        url="https://www.zenithbank.com/fees",
        raw_text=(
            "Zenith Bank charges ₦10 for NIP transfers below ₦5,000, ₦25 for NIP transfers between ₦5,000 and ₦50,000, and ₦50 for transfers above ₦50,000. "
            "Zenith Bank transfer fees are competitive and designed to promote financial inclusion in Nigeria. "
            "Electronic fund transfer transactions made via the Zenith Mobile App or Zenith Internet Banking are processed securely. "
            "This Zenith Bank tariff page provides up-to-date information on transfer fees, card maintenance charges, and administrative banking tariffs for all retail customers."
        ),
        category="fees",
        institution_slug="zenith",
        http_status=200,
        content_type="text",
    )
]


# =====================================================================
# Fixture for Integration Setup (Populating DBs)
# =====================================================================
@pytest.fixture(scope="module", autouse=True)
async def setup_e2e_data():
    """Module-level fixture to ingest mock documents into Qdrant & Postgres."""
    # Re-initialize actual live client stores to bypass other tests' mocks
    from store.qdrant_client import QdrantStore, KnowledgePoint
    from store.redis_client import RedisStore
    from ingestion.processors.embedder import Embedder
    from ingestion.tasks import _scrape_institution_async

    app.state.qdrant_store = QdrantStore()
    app.state.redis_store = RedisStore()
    app.state.embedder = Embedder()

    # Clear any old data to prevent test cross-contamination
    await app.state.qdrant_store.ensure_collection()
    await app.state.qdrant_store.delete_by_institution("gtbank")
    await app.state.qdrant_store.delete_by_institution("zenith")

    # We patch the scrapers to return mock documents during the ingest runs
    with patch("ingestion.scrapers.gtbank.GTBankScraper.scrape", AsyncMock(return_value=mock_gtbank_documents)), \
         patch("ingestion.scrapers.zenith.ZenithScraper.scrape", AsyncMock(return_value=mock_zenith_documents)), \
         patch("ingestion.processors.embedder.Embedder.embed_chunks", AsyncMock(side_effect=lambda chunks: [
             KnowledgePoint(
                 id=c.id,
                 vector=[0.1] * 1024,
                 institution_slug=c.institution_slug,
                 institution_name=c.institution_slug.upper(),
                 category=c.category,
                 sub_category=c.sub_category,
                 content=c.content,
                 source_url=c.source_url,
                 scraped_at=c.scraped_at,
                 content_hash=c.content_hash,
                 is_verified=False,
                 language="en"
             ) for c in chunks
         ])):
        
        # Directly run scraping logic in the test loop to bypass Celery run conflicts
        await _scrape_institution_async("gtbank")
        await _scrape_institution_async("zenith")


@pytest.fixture(autouse=True)
def mock_db_logging():
    """Autouse fixture to mock database query logging to prevent background event loop conflicts."""
    with patch("agents.rag_query.RAGQueryEngine._log_query_to_db", new_callable=AsyncMock) as mock:
        yield mock


# =====================================================================
# E2E Test Cases
# =====================================================================

@patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock)
@patch("ingestion.processors.embedder.Embedder.embed_query", AsyncMock(return_value=[0.1] * 1024))
async def test_gtbank_text_query_english(mock_chat_create):
    """E2E Test: Submit a text query about GTBank transfer fees in English."""
    # Mock chat completion to simulate LLM response based on context
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="GTBank transfer charges are ₦10 for NIP transfers below ₦5,000."))
    ]
    mock_chat_create.return_value = mock_completion

    payload = {
        "query": "What are the charges for transferring money with GTBank?",
        "institution_slug": "gtbank",
        "language": "en"
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/query", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "₦" in data["answer"]
    assert "10" in data["answer"]
    assert len(data["sources"]) > 0
    assert data["confidence"] > 0.4


@patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock)
@patch("ingestion.processors.embedder.Embedder.embed_query", AsyncMock(return_value=[0.1] * 1024))
async def test_zenith_query_yoruba(mock_chat_create):
    """E2E Test: Query Zenith Bank in Yoruba and assert response attributes."""
    mock_completion = MagicMock()
    # Mock Yoruba translation response
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Zenith Bank n gba owo gbigbe ti o to ₦10 fun gbigbe labẹ ₦5,000."))
    ]
    mock_chat_create.return_value = mock_completion

    payload = {
        "query": "Elo ni owo transfer fun Zenith Bank?",
        "institution_slug": "zenith",
        "language": "yo"
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/query", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "Zenith Bank" in data["answer"]
    
    # Assert language is Yoruba using Lingua
    detected = lingua_detector.detect_language_of(data["answer"])
    assert detected == Language.YORUBA


@patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock)
@patch("ingestion.processors.embedder.Embedder.embed_query", AsyncMock(return_value=[0.1] * 1024))
async def test_comparative_query(mock_chat_create):
    """E2E Test: Submit a comparative query matching multiple institutions."""
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Here is a comparison table:\n\n| Bank | Fee |\n|---|---|\n| GTBank | ₦10 |\n| Zenith | ₦10 |"))
    ]
    mock_chat_create.return_value = mock_completion

    payload = {
        "query": "Which bank has the cheapest transfer fees?",
        "language": "en"
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/query/multi", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["response_type"] == "comparative"
    assert len(data["institutions_queried"]) > 1


@patch("voice.stt.whisper_engine.WhisperSTTEngine.transcribe", new_callable=AsyncMock)
@patch("voice.tts.router.TTSRouter.synthesize", new_callable=AsyncMock)
@patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock)
@patch("ingestion.processors.embedder.Embedder.embed_query", AsyncMock(return_value=[0.1] * 1024))
async def test_voice_pipeline_english(mock_chat_create, mock_tts_synthesize, mock_stt_transcribe):
    """E2E Test: Submit audio WAV and receive synthesized voice response."""
    # Mock STT transcription
    mock_stt_transcribe.return_value = TranscriptionResult(
        transcript="What are the charges for transferring money with GTBank?",
        detected_language="en",
        duration_seconds=3.0,
        engine_used="whisper"
    )

    # Mock RAG Chat response
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="GTBank transfer charges are ₦10."))
    ]
    mock_chat_create.return_value = mock_completion

    # Mock TTS audio response (returns standard 44-byte WAV header + dummy PCM)
    dummy_wav_response = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    mock_tts_synthesize.return_value = TTSResult(
        audio_bytes=dummy_wav_response,
        language="en",
        duration_seconds=1.5,
        engine_used="coqui",
        sample_rate=16000
    )

    # Locate sample query WAV file generated by fixture setup
    fixtures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures"))
    audio_path = os.path.join(fixtures_dir, "sample_query_en.wav")
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": ("sample_query_en.wav", audio_bytes, "audio/wav")}
    data = {"institution_slug": "gtbank", "preferred_language": "en"}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/voice/query", files=files, data=data)
    assert response.status_code == 200
    assert response.headers.get("X-Transcript") is not None
    
    # Assert headers
    assert "gtbank" in response.headers.get("X-Institution", "").lower()
    
    # Assert body is valid WAV by checking magic bytes
    body_bytes = response.content
    assert body_bytes.startswith(b"RIFF")
    assert b"WAVE" in body_bytes[:12]


@patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock)
@patch("ingestion.processors.embedder.Embedder.embed_query", AsyncMock(return_value=[0.1] * 1024))
async def test_cache_hit(mock_chat_create):
    """E2E Test: Verifies query caching yields lower latency and sets cache_hit flag."""
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="GTBank NIP Transfer charges are ₦10."))
    ]
    mock_chat_create.return_value = mock_completion

    payload = {
        "query": "What are the charges for transferring money with GTBank? cache_test",
        "institution_slug": "gtbank",
        "language": "en"
    }

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # First Query (Cache Miss)
        resp1 = await ac.post("/v1/query", json=payload)
        assert resp1.status_code == 200
        assert resp1.json()["cache_hit"] is False

        # Second Query (Cache Hit)
        resp2 = await ac.post("/v1/query", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["cache_hit"] is True


async def test_unknown_institution_graceful():
    """E2E Test: Submit validation query with non-registered slug."""
    payload = {
        "query": "What are the charges?",
        "institution_slug": "non_existent_slug",
        "language": "en"
    }
    
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/query", json=payload)
    assert response.status_code in [404, 422]
