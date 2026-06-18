import io
import json
import sys
from urllib.parse import unquote
import pytest
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
mock_TTS_class = mock_tts_api.TTS

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

from api.main import app
from voice.stt.base import TranscriptionResult
from voice.tts.base import TTSResult

# Populate app state for testing (lifespan is not run in Starlette testclient by default)
app.state.qdrant_store = MagicMock()
app.state.embedder = MagicMock()

# Setup mock Redis store with AsyncMock methods to prevent 'can't be used in await expression' errors
mock_redis_store = MagicMock()
mock_redis_store.client.scard = AsyncMock(return_value=0)
mock_redis_store.client.sadd = AsyncMock(return_value=1)
mock_redis_store.client.expire = AsyncMock(return_value=True)
mock_redis_store.client.srem = AsyncMock(return_value=1)
app.state.redis_store = mock_redis_store

client = TestClient(app)


# Helper to generate a 3-second dummy WAV audio file
def generate_dummy_wav(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Sine wave scaled to 16-bit range
    data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    wav_io = io.BytesIO()
    sf.write(wav_io, data, sample_rate, format="WAV", subtype="PCM_16")
    return wav_io.getvalue()


# ==========================================
# 1. REST Voice Query Endpoint Tests
# ==========================================
@patch("api.routers.voice.validate_and_convert_audio")
@patch("voice.stt.whisper_engine.WhisperSTTEngine.transcribe")
@patch("agents.rag_query.RAGQueryEngine.query")
@patch("voice.tts.router.TTSRouter.synthesize")
def test_voice_query_success(mock_synthesize, mock_query, mock_transcribe, mock_validate):
    dummy_wav = generate_dummy_wav()
    
    # 1. Configure Mocks
    mock_validate.return_value = dummy_wav
    mock_transcribe.return_value = TranscriptionResult(
        transcript="what is the USSD code for GTBank",
        detected_language="en",
        duration_seconds=3.0,
        engine_used="whisper",
    )
    
    # Mock RAG response
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "The USSD code is *737#."
    mock_rag_response.sources = ["https://gtbank.com/ussd"]
    mock_rag_response.confidence = 0.95
    mock_query.return_value = mock_rag_response

    # Mock TTS response
    mock_synthesize.return_value = TTSResult(
        audio_bytes=b"synthesized_audio_data",
        language="en",
        duration_seconds=2.0,
        engine_used="coqui",
        sample_rate=22050,
    )

    # 2. Make Request
    files = {"audio": ("test.wav", dummy_wav, "audio/wav")}
    data = {"institution_slug": "gtbank", "preferred_language": "en"}
    
    response = client.post("/v1/voice/query", files=files, data=data)

    # 3. Assertions
    assert response.status_code == 200
    assert response.content == b"synthesized_audio_data"
    
    # Check headers
    assert "X-Transcript" in response.headers
    assert unquote(response.headers["X-Transcript"]) == "what is the USSD code for GTBank"
    assert response.headers["X-Language"] == "en"
    assert response.headers["X-Institution"] == "gtbank"
    assert response.headers["X-Sources"] == json.dumps(["https://gtbank.com/ussd"])
    assert response.headers["X-Confidence"] == "0.9500"
    assert "X-Latency-Ms" in response.headers


@patch("api.routers.voice.validate_and_convert_audio")
@patch("voice.stt.whisper_engine.WhisperSTTEngine.transcribe")
@patch("agents.rag_query.RAGQueryEngine.query")
@patch("voice.tts.router.TTSRouter.synthesize")
def test_voice_query_return_url_success(mock_synthesize, mock_query, mock_transcribe, mock_validate):
    dummy_wav = generate_dummy_wav()
    
    # 1. Configure Mocks
    mock_validate.return_value = dummy_wav
    mock_transcribe.return_value = TranscriptionResult(
        transcript="what is the USSD code for GTBank",
        detected_language="en",
        duration_seconds=3.0,
        engine_used="whisper",
    )
    
    # Mock RAG response
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "The USSD code is *737#."
    mock_rag_response.sources = ["https://gtbank.com/ussd"]
    mock_rag_response.confidence = 0.95
    mock_query.return_value = mock_rag_response

    # Mock TTS response
    mock_synthesize.return_value = TTSResult(
        audio_bytes=b"synthesized_audio_data",
        language="en",
        duration_seconds=2.0,
        engine_used="coqui",
        sample_rate=22050,
    )

    # 2. Make Request
    files = {"audio": ("test.wav", dummy_wav, "audio/wav")}
    data = {"institution_slug": "gtbank", "preferred_language": "en", "return_url": "true"}
    
    response = client.post("/v1/voice/query", files=files, data=data)

    # 3. Assertions
    assert response.status_code == 200
    data_json = response.json()
    assert data_json["transcript"] == "what is the USSD code for GTBank"
    assert data_json["language"] == "en"
    assert data_json["institution_slug"] == "gtbank"
    assert data_json["confidence"] == 0.95
    assert data_json["answer"] == "The USSD code is *737#."
    assert data_json["sources"] == ["https://gtbank.com/ussd"]
    assert "audio_url" in data_json
    assert "/static/audio/" in data_json["audio_url"]


def test_voice_query_invalid_slug():
    dummy_wav = generate_dummy_wav()
    files = {"audio": ("test.wav", dummy_wav, "audio/wav")}
    data = {"institution_slug": "invalid_slug", "preferred_language": "en"}
    
    response = client.post("/v1/voice/query", files=files, data=data)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_voice_query_too_large():
    # 11MB file (exceeds 10MB limit)
    large_audio = b"\x00" * (11 * 1024 * 1024)
    files = {"audio": ("test.wav", large_audio, "audio/wav")}
    data = {"institution_slug": "gtbank", "preferred_language": "en"}
    
    response = client.post("/v1/voice/query", files=files, data=data)
    assert response.status_code == 413
    assert "Payload Too Large" in response.json()["detail"]


def test_voice_query_rate_limiting():
    # Mock Redis pipeline return value to trigger limit (>10 hits)
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[None, None, 12, None])  # 12 is > limit (10)
    mock_redis.client.pipeline.return_value = mock_pipe
    mock_redis.client.zrange = AsyncMock(return_value=[("some_val", 1234567890.0)])

    original_redis = app.state.redis_store
    app.state.redis_store = mock_redis

    try:
        dummy_wav = generate_dummy_wav()
        files = {"audio": ("test.wav", dummy_wav, "audio/wav")}
        data = {"institution_slug": "gtbank", "preferred_language": "en"}
        
        response = client.post("/v1/voice/query", files=files, data=data)
        assert response.status_code == 429
        assert "rate limit exceeded" in response.json()["detail"].lower()
    finally:
        app.state.redis_store = original_redis


# ==========================================
# 2. WebSocket Streaming Voice Endpoint Tests
# ==========================================
@pytest.mark.asyncio
@patch("voice.stt.whisper_engine.WhisperSTTEngine.transcribe")
@patch("voice.stt.mms_engine.MMSSTTEngine.transcribe")
@patch("agents.rag_query.RAGQueryEngine.query")
@patch("voice.tts.router.TTSRouter.synthesize")
@patch("voice.vad.VoiceActivityDetector.is_end_of_speech")
async def test_websocket_stream_success(
    mock_is_end, mock_synthesize, mock_query, mock_mms_transcribe, mock_whisper_transcribe
):
    # Configure mock returns
    mock_is_end.return_value = False
    
    mock_mms_transcribe.return_value = TranscriptionResult(
        transcript="kini USSD code fun GTBank",
        detected_language="yo",
        duration_seconds=3.0,
        engine_used="mms",
    )
    
    mock_rag_response = MagicMock()
    mock_rag_response.answer = "USSD code yi ni *737#."
    mock_rag_response.sources = ["https://gtbank.com"]
    mock_rag_response.confidence = 0.9
    mock_query.return_value = mock_rag_response
    
    mock_synthesize.return_value = TTSResult(
        audio_bytes=b"mock_tts_stream_bytes",
        language="yo",
        duration_seconds=2.0,
        engine_used="mms",
        sample_rate=16000,
    )

    # Establish test client connection
    with client.websocket_connect("/v1/voice/stream/gtbank") as ws:
        # 1. Send text frame configuration to set language
        ws.send_json({"type": "language", "lang": "yo"})
        
        # 2. Send small binary audio chunks
        ws.send_bytes(b"\x00\x00" * 1600)  # 100ms of silence
        
        # 3. Send manual end-of-speech frame
        ws.send_json({"type": "end"})
        
        # 4. Read interleaved messages sent from the server
        # - Transcript message
        msg_transcript = ws.receive_json()
        assert msg_transcript["type"] == "transcript"
        assert msg_transcript["text"] == "kini USSD code fun GTBank"
        assert msg_transcript["lang"] == "yo"
        
        # - RAG Answer text
        msg_answer = ws.receive_json()
        assert msg_answer["type"] == "answer_text"
        assert msg_answer["text"] == "USSD code yi ni *737#."
        
        # - TTS audio bytes (binary frame)
        audio_frame = ws.receive_bytes()
        assert audio_frame == b"mock_tts_stream_bytes"
        
        # - Done message
        msg_done = ws.receive_json()
        assert msg_done["type"] == "done"
        assert msg_done["confidence"] == 0.9
        assert msg_done["sources"] == ["https://gtbank.com"]


@pytest.mark.asyncio
async def test_websocket_stream_invalid_slug():
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/voice/stream/invalid_slug") as ws:
            ws.send_json({"type": "end"})
            
    # Starlette close code 3001 represents invalid slug
    assert exc_info.value.code == 3001


@pytest.mark.asyncio
async def test_websocket_stream_connection_limits():
    # Mock Redis scard call to show 5 active connections already
    mock_redis = MagicMock()
    mock_redis.client.scard = AsyncMock(return_value=5)
    
    original_redis = app.state.redis_store
    app.state.redis_store = mock_redis
    
    try:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/v1/voice/stream/gtbank") as ws:
                pass
        # Starlette close code 3000 represents concurrent WebSocket connection limit exceeded
        assert exc_info.value.code == 3000
    finally:
        app.state.redis_store = original_redis
