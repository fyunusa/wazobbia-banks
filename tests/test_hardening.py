import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
import bcrypt
from datetime import datetime

from api.main import app
from api.routers.query import sanitize_query
from api.routers.voice import validate_audio_magic_bytes
from api.routers.institutions import verify_admin_key
from config.settings import settings

# Test client
client = TestClient(app)

# ==========================================
# 1. Query Sanitization Tests
# ==========================================
def test_sanitize_query_html_stripping():
    query_with_html = "What is the <script>alert('hack')</script> USSD code for <b>GTBank</b>?"
    sanitized = sanitize_query(query_with_html)
    assert "<script>" not in sanitized
    assert "<b>" not in sanitized
    assert "USSD code for GTBank?" in sanitized

def test_sanitize_query_null_byte():
    query_with_null = "What is the USSD code\x00 for GTBank?"
    with pytest.raises(HTTPException) as exc_info:
        sanitize_query(query_with_null)
    assert exc_info.value.status_code == 400
    assert "null bytes" in exc_info.value.detail

def test_sanitize_query_truncation():
    long_query = "a" * 600
    sanitized = sanitize_query(long_query)
    assert len(sanitized) == 500

# ==========================================
# 2. Audio Magic Bytes Tests
# ==========================================
def test_validate_audio_magic_bytes_valid():
    # WAV
    validate_audio_magic_bytes(b"RIFF\x00\x00\x00\x00WAVEotherbytes")
    # MP3 ID3
    validate_audio_magic_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x00")
    # MP3 raw frames
    validate_audio_magic_bytes(b"\xff\xfb\x90\x44")
    # Ogg
    validate_audio_magic_bytes(b"OggS\x00\x02\x00\x00")
    # WebM
    validate_audio_magic_bytes(b"\x1a\x45\xdf\xa3\x01\x00\x00")

def test_validate_audio_magic_bytes_invalid():
    # Too short
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_magic_bytes(b"abc")
    assert exc_info.value.status_code == 415
    assert "too short" in exc_info.value.detail.lower()

    # Invalid header
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_magic_bytes(b"invalid_header_bytes_12345")
    assert exc_info.value.status_code == 415
    assert "magic bytes do not match" in exc_info.value.detail.lower()

# ==========================================
# 3. Bcrypt API Key Middleware Tests
# ==========================================
@pytest.mark.asyncio
async def test_verify_admin_key_success():
    # The default settings.ADMIN_API_KEY hash is set to the bcrypt hash of "test-admin-secret-key-123"
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.url.path = "/v1/institutions"
    request.method = "GET"

    # Valid key check
    result = await verify_admin_key(request, "test-admin-secret-key-123")
    assert result == "test-admin-secret-key-123"

@pytest.mark.asyncio
async def test_verify_admin_key_forbidden():
    request = MagicMock()
    request.client.host = "127.0.0.1"
    request.url.path = "/v1/institutions"
    request.method = "GET"

    # Invalid key check
    with pytest.raises(HTTPException) as exc_info:
        await verify_admin_key(request, "wrong-admin-secret-key")
    assert exc_info.value.status_code == 403
    assert "Invalid administrative" in exc_info.value.detail

# ==========================================
# 4. Sliding Window Rate Limiter Tests
# ==========================================
def test_rate_limiter_exceeded():
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    # Mocking pipeline returns: [zremrangebyscore_res, zadd_res, count (12 > limit 10), expire_res]
    mock_pipe.execute = AsyncMock(return_value=[None, None, 12, None])
    mock_redis.client.pipeline.return_value = mock_pipe
    
    # Mock zrange to calculate Retry-After
    now = 1234567890.0
    mock_redis.client.zrange = AsyncMock(return_value=[("some_val", now - 20)])

    original_redis = getattr(app.state, "redis_store", None)
    app.state.redis_store = mock_redis

    try:
        # Request /v1/voice/query and verify rate limit is hit
        dummy_wav = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 20
        files = {"audio": ("test.wav", dummy_wav, "audio/wav")}
        data = {"institution_slug": "gtbank", "preferred_language": "en"}
        
        # We need to temporarily mock other dependencies to avoid startup errors
        with patch("api.dependencies.get_qdrant_store", return_value=MagicMock()), \
             patch("api.dependencies.get_embedder", return_value=MagicMock()):
            response = client.post("/v1/voice/query", files=files, data=data)
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert int(response.headers["Retry-After"]) > 0
    finally:
        if original_redis is not None:
            app.state.redis_store = original_redis
        else:
            try:
                del app.state.redis_store
            except AttributeError:
                pass

# ==========================================
# 5. Diagnostic Connection Checks (/health)
# ==========================================
@pytest.mark.asyncio
async def test_health_endpoint():
    # Save original app state dependencies
    orig_qdrant = getattr(app.state, "qdrant_store", None)
    orig_redis = getattr(app.state, "redis_store", None)
    orig_embedder = getattr(app.state, "embedder", None)

    # Mock DB session execute
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()

    # Mock Redis client ping
    mock_redis = MagicMock()
    mock_redis.client.ping = AsyncMock()

    # Mock Qdrant get_collections
    mock_qdrant = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.name = "wazobia_knowledge"
    mock_qdrant.collection_name = "wazobia_knowledge"
    
    mock_collections_list = MagicMock()
    mock_collections_list.collections = [mock_collection_info]
    mock_qdrant.client.get_collections = AsyncMock(return_value=mock_collections_list)

    # Mock OpenAI models list
    mock_openai = MagicMock()
    mock_openai.models.list = AsyncMock()

    app.state.qdrant_store = mock_qdrant
    app.state.redis_store = mock_redis

    # Override dependencies
    from api.dependencies import get_db, get_redis_store, get_qdrant_store, get_openai_client
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_redis_store] = lambda: mock_redis
    app.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant
    app.dependency_overrides[get_openai_client] = lambda: mock_openai

    try:
        # 1. Healthy state
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["checks"]["postgres"] == "healthy"
        assert data["checks"]["redis"] == "healthy"
        assert data["checks"]["qdrant"] == "healthy"
        assert data["checks"]["llm"] == "healthy"

        # 2. Degraded state (LLM fails)
        mock_openai.models.list.side_effect = Exception("LLM down")
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["checks"]["llm"] == "unhealthy"

        # 3. Unhealthy state (Postgres fails)
        mock_openai.models.list.side_effect = None
        mock_db.execute.side_effect = Exception("DB down")
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["postgres"] == "unhealthy"

    finally:
        # Restore original states and clear overrides
        if orig_qdrant:
            app.state.qdrant_store = orig_qdrant
        if orig_redis:
            app.state.redis_store = orig_redis
        app.dependency_overrides.clear()

# ==========================================
# 6. Prometheus Metrics Endpoint (/metrics)
# ==========================================
def test_metrics_endpoint():
    # Set up mock dependencies to avoid live external client hits
    mock_redis = MagicMock()
    mock_redis.client.get = AsyncMock(return_value=None)
    mock_redis.client.set = AsyncMock()

    mock_qdrant = MagicMock()
    mock_stats = MagicMock()
    mock_stats.total_points = 1500
    mock_stats.points_per_institution = {"gtbank": 500, "zenith": 1000}
    mock_stats.indexed_at = datetime.utcnow()
    mock_qdrant.get_collection_stats = AsyncMock(return_value=mock_stats)

    orig_redis = getattr(app.state, "redis_store", None)
    orig_qdrant = getattr(app.state, "qdrant_store", None)

    app.state.redis_store = mock_redis
    app.state.qdrant_store = mock_qdrant

    from api.dependencies import get_redis_store, get_qdrant_store
    app.dependency_overrides[get_redis_store] = lambda: mock_redis
    app.dependency_overrides[get_qdrant_store] = lambda: mock_qdrant

    try:
        response = client.get("/metrics")
        assert response.status_code == 200
        text = response.text
        # Check custom metrics exist in standard prometheus format
        assert "wazobia_qdrant_points_total" in text
        assert 'wazobia_qdrant_points_total{institution="gtbank"}' in text
        assert "500" in text
    finally:
        if orig_redis:
            app.state.redis_store = orig_redis
        if orig_qdrant:
            app.state.qdrant_store = orig_qdrant
        app.dependency_overrides.clear()
