import asyncio
import io
import json
import logging
import time
import uuid
from typing import Optional
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from api.dependencies import (
    get_embedder,
    get_openai_client,
    get_qdrant_store,
    get_redis_store,
    SlidingWindowRateLimiter,
)
from config.metrics import wazobia_voice_requests_total
from config.settings import settings
from registry.institutions import get_institution
from agents.rag_query import RAGQueryEngine, QueryRequest
from voice.stt.whisper_engine import WhisperSTTEngine
from voice.stt.mms_engine import MMSSTTEngine
from voice.normalizer import TranscriptNormalizer
from voice.tts.router import TTSRouter
from voice.vad import VoiceActivityDetector
from voice.utils import validate_and_convert_audio, pcm_to_wav
from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore
from ingestion.processors.embedder import Embedder

logger = logging.getLogger("api.routers.voice")

router = APIRouter(tags=["Voice"])


# Instantiate sliding window rate limiter for voice queries (10 requests per minute)
voice_limiter = SlidingWindowRateLimiter("voice", 10, 60)


def validate_audio_magic_bytes(audio_bytes: bytes) -> None:
    """Validates that the file prefix matches supported formats: WAV, WebM, MP3, OGG."""
    if len(audio_bytes) < 4:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type: The uploaded file is too short to be a valid audio file.",
        )
    
    header = audio_bytes[:12]
    # Check WebM (starts with EBML header \x1a\x45\xdf\xa3)
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return
    # Check OGG (starts with OggS)
    if header.startswith(b"OggS"):
        return
    # Check WAV (RIFF header + WAVE format)
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return
    # Check MP3 (ID3v2 or raw frame)
    if header.startswith(b"ID3") or header.startswith(b"\xff\xfb") or header.startswith(b"\xff\xf3") or header.startswith(b"\xff\xf2"):
        return
        
    raise HTTPException(
        status_code=415,
        detail="Unsupported Media Type: The uploaded file's magic bytes do not match a supported audio format (WAV, MP3, OGG, WebM).",
    )


@router.post(
    "/voice/query",
    summary="Query RAG Engine via Voice (REST)",
    description="""
    Transcribes uploaded audio (WAV, MP3, OGG, WebM), executes a RAG financial database search
    for the target institution, synthesizes the response text to WAV audio, and returns it.
    
    Includes response metadata headers for the transcribed query (`X-Transcript`), language (`X-Language`),
    institution slug (`X-Institution`), source URLs (`X-Sources`), confidence (`X-Confidence`), and latency (`X-Latency-Ms`).
    """,
    response_description="Synthesized WAV audio bytes matching the RAG search result.",
    dependencies=[Depends(voice_limiter)],
)
async def voice_query(
    request: Request,
    audio: UploadFile = File(
        ...,
        description="Audio file to transcribe and query (WAV, WebM, MP3, OGG, max 10MB)",
    ),
    institution_slug: str = Form(
        ...,
        description="Registry slug identifier of the target Nigerian institution",
        examples=["gtbank", "zenith", "access"],
    ),
    preferred_language: str = Form(
        "auto",
        description="Language tag hint. If 'auto', automatically detects Hausa, Yoruba, Igbo, English, or Pidgin.",
        examples=["auto", "en", "pcm", "ha", "yo", "ig"],
    ),
    return_url: bool = Form(
        False,
        description="If true, returns a JSON response containing metadata and a link to the audio file on the server instead of streaming raw audio bytes.",
    ),
    gender: str = Form(
        "female",
        description="Voice gender choice ('male' or 'female')",
        examples=["female", "male"],
    ),
    user_id: Optional[str] = Form(None, description="Optional user identifier for auditing/metrics"),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: Embedder = Depends(get_embedder),
    redis: RedisStore = Depends(get_redis_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    start_time = time.time()

    # 1. Validate institution slug exists
    try:
        get_institution(institution_slug)
    except ValueError as e:
        logger.warning(f"Voice query requested for invalid institution slug: {institution_slug}")
        raise HTTPException(status_code=404, detail=str(e))

    # Validate voice gender choice
    gender_choice = gender.lower().strip()
    if gender_choice not in ("male", "female"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported voice gender: {gender}. Supported: ['male', 'female']",
        )

    # 2. Validate audio size limit (10MB)
    audio.file.seek(0, 2)
    file_size = audio.file.tell()
    audio.file.seek(0)
    if file_size > 10 * 1024 * 1024:
        logger.warning(f"Voice query upload rejected: {file_size} bytes exceeds 10MB limit")
        raise HTTPException(
            status_code=413,
            detail="Payload Too Large: Audio file size exceeds the maximum limit of 10MB.",
        )

    # 3. Read, validate magic bytes, and convert audio to 16kHz mono WAV bytes
    audio_bytes = await audio.read()
    validate_audio_magic_bytes(audio_bytes)
    try:
        pcm_wav_bytes = validate_and_convert_audio(audio_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio format or processing failure")

    # 4. Transcribe audio (STT)
    stt_start = time.time()
    whisper_engine = WhisperSTTEngine()
    mms_engine = MMSSTTEngine()
    
    lang = preferred_language.lower().strip()
    
    try:
        if lang in ("en", "pcm") or lang == "auto":
            # Auto-detect language or English/Pidgin transcribes with Whisper
            lang_hint = None if lang == "auto" else lang
            stt_result = await whisper_engine.transcribe(pcm_wav_bytes, language_hint=lang_hint)
            detected_lang = stt_result.detected_language
        elif lang in ("ha", "yo", "ig"):
            # African native languages transcribes with MMS
            stt_result = await mms_engine.transcribe(pcm_wav_bytes, language_hint=lang)
            detected_lang = lang
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language code: {preferred_language}. Supported: ['auto', 'en', 'pcm', 'ha', 'yo', 'ig']",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to transcribe speech audio")
        
    stt_latency = (time.time() - stt_start) * 1000

    # 5. Normalize STT transcription
    normalizer = TranscriptNormalizer()
    normalized_transcript = normalizer.normalize(stt_result.transcript, detected_lang)
    logger.info(f"Speech transcribed ({detected_lang}): '{normalized_transcript}'")

    if not normalized_transcript:
        raise HTTPException(
            status_code=400,
            detail="No speech content could be transcribed from the uploaded audio file.",
        )

    # 6. Execute RAG query search
    rag_start = time.time()
    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis,
        openai_client=openai_client,
    )
    query_req = QueryRequest(
        query=normalized_transcript,
        institution_slug=institution_slug,
        language=detected_lang,
    )
    try:
        query_response = await engine.query(query_req)
    except Exception as e:
        logger.error(f"RAG query execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to execute search against RAG engine")
        
    rag_latency = (time.time() - rag_start) * 1000

    # 7. Synthesize response to WAV speech (TTS)
    tts_start = time.time()
    tts_router = TTSRouter()
    try:
        tts_result = await tts_router.synthesize(query_response.answer, detected_lang, gender_choice)
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to synthesize response text to speech audio")
        
    tts_latency = (time.time() - tts_start) * 1000
    total_latency = (time.time() - start_time) * 1000

    # Record voice request metrics
    wazobia_voice_requests_total.labels(
        stt_engine=stt_result.engine_used,
        tts_engine=tts_result.engine_used,
        language=detected_lang
    ).inc()

    # Log structured latency details
    logger.info(
        f"Voice query request completed for {institution_slug}",
        extra={
            "stt_engine": stt_result.engine_used,
            "tts_engine": tts_result.engine_used,
            "audio_duration": stt_result.duration_seconds,
            "stt_latency": stt_latency,
            "tts_latency": tts_latency,
        }
    )

    # 8. Return WAV response stream or JSON containing URL to audio file
    if return_url:
        import uuid
        import os
        unique_id = str(uuid.uuid4())
        audio_filename = f"{unique_id}.wav"
        static_dir = os.path.join("static", "audio")
        os.makedirs(static_dir, exist_ok=True)
        audio_path = os.path.join(static_dir, audio_filename)
        
        with open(audio_path, "wb") as f:
            f.write(tts_result.audio_bytes)
            
        audio_url = f"{request.base_url}static/audio/{audio_filename}"
        return {
            "transcript": normalized_transcript,
            "language": detected_lang,
            "institution_slug": institution_slug,
            "confidence": query_response.confidence,
            "answer": query_response.answer,
            "normalized_answer": tts_result.normalized_text,
            "sources": query_response.sources,
            "audio_url": audio_url,
            "latency_ms": int(total_latency),
        }

    headers = {
        "X-Transcript": quote(normalized_transcript),
        "X-Language": detected_lang,
        "X-Institution": institution_slug,
        "X-Sources": json.dumps(query_response.sources),
        "X-Confidence": f"{query_response.confidence:.4f}",
        "X-Latency-Ms": f"{total_latency:.0f}",
    }
    
    from fastapi import Response
    return Response(
        content=tts_result.audio_bytes,
        media_type="audio/wav",
        headers=headers,
    )


@router.post(
    "/voice/stream-sse",
    summary="Stream Voice Query via Server-Sent Events",
    description="""
    Streams voice query processing via Server-Sent Events (SSE).
    
    Returns transcript, text response, and audio chunks progressively as they're generated.
    Better latency perception for frontend users since chunks stream incrementally.
    
    Events:
    - transcript: Recognized speech text + language + confidence
    - response: RAG text answer for the query
    - audio_chunk: Base64-encoded WAV audio chunk (playable progressively)
    - completed: Final summary with latency and sources
    """,
    dependencies=[Depends(voice_limiter)],
)
async def voice_stream_sse(
    request: Request,
    audio: UploadFile = File(..., description="Audio file (WAV, WebM, MP3, OGG, max 10MB)"),
    institution_slug: str = Form(..., description="Target institution slug"),
    preferred_language: str = Form("auto", description="Language: auto, en, ha, yo, ig, pcm"),
    gender: str = Form("female", description="Voice gender: male or female"),
    user_id: Optional[str] = Form(None, description="Optional user identifier"),
    qdrant: QdrantStore = Depends(get_qdrant_store),
    embedder: Embedder = Depends(get_embedder),
    redis: RedisStore = Depends(get_redis_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
):
    """
    Stream voice query response via Server-Sent Events (SSE).
    Frontend receives: transcript → response text → audio chunks → completion.
    """
    start_time = time.time()
    
    # Validate institution
    try:
        institution = get_institution(institution_slug)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Validate gender
    gender_choice = gender.lower().strip()
    if gender_choice not in ("male", "female"):
        raise HTTPException(status_code=400, detail=f"Invalid gender: {gender}")
    
    # Validate audio size
    audio.file.seek(0, 2)
    file_size = audio.file.tell()
    audio.file.seek(0)
    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio exceeds 10MB limit")
    
    # Read and validate audio
    audio_bytes = await audio.read()
    validate_audio_magic_bytes(audio_bytes)
    try:
        pcm_wav_bytes = validate_and_convert_audio(audio_bytes)
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio format")
    
    async def event_stream():
        """Generator for SSE events."""
        try:
            # 1. Transcribe audio (STT)
            stt_start = time.time()
            whisper_engine = WhisperSTTEngine()
            mms_engine = MMSSTTEngine()
            
            lang = preferred_language.lower().strip()
            try:
                if lang in ("en", "pcm") or lang == "auto":
                    lang_hint = None if lang == "auto" else lang
                    stt_result = await whisper_engine.transcribe(pcm_wav_bytes, language_hint=lang_hint)
                    detected_lang = stt_result.detected_language
                elif lang in ("ha", "yo", "ig"):
                    stt_result = await mms_engine.transcribe(pcm_wav_bytes, language_hint=lang)
                    detected_lang = lang
                else:
                    raise ValueError(f"Unsupported language: {preferred_language}")
            except Exception as e:
                logger.error(f"STT failed: {e}")
                yield f"event: error\ndata: {json.dumps({'error': 'Speech transcription failed'})}\n\n"
                return
            
            stt_latency = (time.time() - stt_start) * 1000
            
            # 2. Normalize transcript
            normalizer = TranscriptNormalizer()
            normalized_transcript = normalizer.normalize(stt_result.transcript, detected_lang)
            
            if not normalized_transcript:
                yield f"event: error\ndata: {json.dumps({'error': 'No speech detected'})}\n\n"
                return
            
            # Stream transcript event
            yield f"event: transcript\n"
            yield f"data: {json.dumps({\
                'text': normalized_transcript,\
                'language': detected_lang,\
                'confidence': stt_result.confidence if hasattr(stt_result, 'confidence') else 0.95\
            })}\n\n"
            
            # 3. Execute RAG query
            rag_start = time.time()
            engine = RAGQueryEngine(
                qdrant=qdrant,
                embedder=embedder,
                redis_store=redis,
                openai_client=openai_client,
            )
            query_req = QueryRequest(
                query=normalized_transcript,
                institution_slug=institution_slug,
                language=detected_lang,
            )
            try:
                query_response = await engine.query(query_req)
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                yield f"event: error\ndata: {json.dumps({'error': 'Search failed'})}\n\n"
                return
            
            rag_latency = (time.time() - rag_start) * 1000
            
            # Stream response event
            yield f"event: response\n"
            yield f"data: {json.dumps({\
                'text': query_response.answer,\
                'language': detected_lang,\
                'confidence': query_response.confidence\
            })}\n\n"
            
            # 4. Synthesize TTS
            tts_start = time.time()
            tts_router = TTSRouter()
            try:
                tts_result = await tts_router.synthesize(query_response.answer, detected_lang, gender_choice)
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                yield f"event: error\ndata: {json.dumps({'error': 'TTS synthesis failed'})}\n\n"
                return
            
            tts_latency = (time.time() - tts_start) * 1000
            
            # Stream audio chunks (split into 4KB chunks for streaming)
            import base64
            chunk_size = 4096
            for i in range(0, len(tts_result.audio_bytes), chunk_size):
                chunk = tts_result.audio_bytes[i:i + chunk_size]
                chunk_index = i // chunk_size
                yield f"event: audio_chunk\n"
                yield f"data: {json.dumps({\
                    'audio_base64': base64.b64encode(chunk).decode('utf-8'),\
                    'chunk_index': chunk_index,\
                    'is_last': (i + chunk_size >= len(tts_result.audio_bytes))\
                })}\n\n"
                
                # Small delay to prevent overwhelming the frontend
                await asyncio.sleep(0.01)
            
            # 5. Send completion event
            total_latency = (time.time() - start_time) * 1000
            yield f"event: completed\n"
            yield f"data: {json.dumps({\
                'transcript': normalized_transcript,\
                'answer': query_response.answer,\
                'language': detected_lang,\
                'institution': institution_slug,\
                'sources': query_response.sources,\
                'stt_engine': stt_result.engine_used,\
                'tts_engine': tts_result.engine_used,\
                'latency_ms': int(total_latency),\
                'stt_latency_ms': int(stt_latency),\
                'rag_latency_ms': int(rag_latency),\
                'tts_latency_ms': int(tts_latency)\
            })}\n\n"
            
            # Log metrics
            wazobia_voice_requests_total.labels(
                stt_engine=stt_result.engine_used,
                tts_engine=tts_result.engine_used,
                language=detected_lang
            ).inc()
            
            logger.info(
                f"Voice SSE stream completed for {institution_slug}",
                extra={
                    "stt_engine": stt_result.engine_used,
                    "tts_engine": tts_result.engine_used,
                    "language": detected_lang,
                    "total_latency_ms": int(total_latency),
                }
            )
            
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'Internal processing error'})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.websocket("/voice/stream/{institution_slug}")
async def websocket_voice_stream(
    websocket: WebSocket,
    institution_slug: str,
):
    """
    WebSocket endpoint for real-time streaming voice queries.
    
    Handles raw binary PCM (16kHz, mono, 16-bit) chunks. Tracks stateful end-of-speech (VAD),
    interleaves transcript, text answer, and audio bytes back to the client.
    """
    # Verify institution slug first
    try:
        get_institution(institution_slug)
    except ValueError as e:
        logger.warning(f"WebSocket rejected for invalid institution slug: {institution_slug}")
        await websocket.close(code=3001, reason=str(e))
        return

    # Extract state dependencies from the application state
    redis: RedisStore = websocket.app.state.redis_store
    qdrant: QdrantStore = websocket.app.state.qdrant_store
    embedder: Embedder = websocket.app.state.embedder
    openai_client = get_openai_client()

    ip = websocket.client.host if websocket.client else "127.0.0.1"
    conn_id = str(uuid.uuid4())
    ws_key = f"wazobia:ws_conn:{ip}"

    # 1. Enforce concurrent WebSocket connections per IP limit (Max 5)
    try:
        active_count = await redis.client.scard(ws_key)
        if active_count >= 5:
            logger.warning(f"WebSocket concurrent limit (5) reached for IP: {ip}. Rejecting.")
            await websocket.close(code=3000, reason="Max concurrent connections exceeded")
            return
            
        await redis.client.sadd(ws_key, conn_id)
        await redis.client.expire(ws_key, 3600)  # TTL of 1 hour for self-healing
    except Exception as e:
        logger.error(f"Failed to check/record WebSocket rate limit in Redis: {e}")
        pass

    logger.info(f"WebSocket client connected from {ip} (connection ID: {conn_id})")
    await websocket.accept()

    audio_buffer = bytearray()
    vad = VoiceActivityDetector()
    current_language = None  # None/auto represents auto-detect
    current_gender = "female"

    async def execute_voice_pipeline(pcm_bytes: bytes, target_lang: str, target_gender: str):
        """Helper to run the voice RAG pipeline and send frames back to client."""
        if len(pcm_bytes) < 3200:  # Less than 100ms of audio (16000 * 2 * 0.1)
            await websocket.send_json({"type": "error", "message": "Audio input is too short"})
            return

        try:
            # A. Convert raw PCM bytes to WAV format so STT models can decode them
            wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000, num_channels=1)

            # B. Speech-to-Text
            stt_start_time = time.time()
            whisper_engine = WhisperSTTEngine()
            mms_engine = MMSSTTEngine()
            
            if not target_lang or target_lang == "auto" or target_lang in ("en", "pcm"):
                lang_hint = None if (not target_lang or target_lang == "auto") else target_lang
                stt_result = await whisper_engine.transcribe(wav_bytes, language_hint=lang_hint)
                detected_lang = stt_result.detected_language
            elif target_lang in ("ha", "yo", "ig"):
                stt_result = await mms_engine.transcribe(wav_bytes, language_hint=target_lang)
                detected_lang = target_lang
            else:
                await websocket.send_json({"type": "error", "message": f"Unsupported language: {target_lang}"})
                return
            stt_latency = (time.time() - stt_start_time) * 1000

            normalizer = TranscriptNormalizer()
            normalized_transcript = normalizer.normalize(stt_result.transcript, detected_lang)

            if not normalized_transcript:
                await websocket.send_json({"type": "error", "message": "No speech detected"})
                return

            # C. Send transcript immediately to client for fast feedback UX
            await websocket.send_json(
                {"type": "transcript", "text": normalized_transcript, "lang": detected_lang}
            )

            # E. RAG Database search query
            engine = RAGQueryEngine(
                qdrant=qdrant,
                embedder=embedder,
                redis_store=redis,
                openai_client=openai_client,
            )
            query_req = QueryRequest(
                query=normalized_transcript,
                institution_slug=institution_slug,
                language=detected_lang,
            )
            query_response = await engine.query(query_req)

            # F. Send RAG Text response
            await websocket.send_json({"type": "answer_text", "text": query_response.answer})

            # G. TTS response synthesis
            tts_start_time = time.time()
            tts_router = TTSRouter()
            tts_result = await tts_router.synthesize(query_response.answer, detected_lang, target_gender)
            tts_latency = (time.time() - tts_start_time) * 1000

            # Record voice request metrics
            wazobia_voice_requests_total.labels(
                stt_engine=stt_result.engine_used,
                tts_engine=tts_result.engine_used,
                language=detected_lang
            ).inc()

            # Log structured voice details
            logger.info(
                f"WebSocket voice pipeline completed for {institution_slug} (Conn: {conn_id})",
                extra={
                    "stt_engine": stt_result.engine_used,
                    "tts_engine": tts_result.engine_used,
                    "audio_duration": stt_result.duration_seconds,
                    "stt_latency": stt_latency,
                    "tts_latency": tts_latency,
                }
            )

            # H. Send binary TTS audio bytes
            await websocket.send_bytes(tts_result.audio_bytes)

            # I. Send Done frame
            await websocket.send_json(
                {
                    "type": "done",
                    "sources": query_response.sources,
                    "confidence": query_response.confidence,
                    "normalized_answer": tts_result.normalized_text,
                }
            )

        except Exception as ex:
            logger.error(f"WebSocket voice pipeline execution error: {ex}", exc_info=True)
            await websocket.send_json({"type": "error", "message": f"Pipeline processing failed: {str(ex)}"})

    try:
        while True:
            # 2. Enforce 30 seconds inactivity timeout
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket ID {conn_id} timed out due to 30s inactivity.")
                await websocket.send_json({"type": "error", "message": "Connection timed out due to 30s inactivity."})
                break

            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
                
                # Check stateful VAD end of speech
                if vad.is_end_of_speech(bytes(audio_buffer)):
                    logger.info(f"VAD triggered end-of-speech for WebSocket ID {conn_id}.")
                    lang_hint = current_language or "auto"
                    
                    await execute_voice_pipeline(bytes(audio_buffer), lang_hint, current_gender)
                    audio_buffer = bytearray()
                    vad.speech_detected = False
                    vad.silence_duration_samples = 0
                    vad.last_buffer_len = 0
                    
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    frame_type = data.get("type")
                    if frame_type == "language":
                        current_language = data.get("lang")
                        logger.info(f"WebSocket ID {conn_id} set language context: {current_language}")
                    elif frame_type == "gender":
                        gender_val = data.get("gender", "female")
                        if gender_val in ("male", "female"):
                            current_gender = gender_val
                            logger.info(f"WebSocket ID {conn_id} set gender context: {current_gender}")
                        else:
                            await websocket.send_json({"type": "error", "message": f"Unsupported gender value: {gender_val}"})
                    elif frame_type == "config":
                        if "lang" in data:
                            current_language = data["lang"]
                        if "gender" in data:
                            gender_val = data["gender"]
                            if gender_val in ("male", "female"):
                                current_gender = gender_val
                            else:
                                await websocket.send_json({"type": "error", "message": f"Unsupported gender value: {gender_val}"})
                        logger.info(f"WebSocket ID {conn_id} set config context: lang={current_language}, gender={current_gender}")
                    elif frame_type == "end":
                        logger.info(f"WebSocket ID {conn_id} received manual end-of-speech.")
                        lang_hint = current_language or "auto"
                        await execute_voice_pipeline(bytes(audio_buffer), lang_hint, current_gender)
                        audio_buffer = bytearray()
                        vad.speech_detected = False
                        vad.silence_duration_samples = 0
                        vad.last_buffer_len = 0
                except Exception as parse_error:
                    logger.warning(f"Failed to parse WebSocket text frame: {parse_error}")
                    await websocket.send_json({"type": "error", "message": "Invalid text payload format."})

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection ID {conn_id} disconnected cleanly.")
    except Exception as e:
        logger.error(f"WebSocket connection ID {conn_id} failed with error: {e}", exc_info=True)
    finally:
        # 3. Clean up Redis connections set and WebSocket
        try:
            await redis.client.srem(ws_key, conn_id)
        except Exception:
            pass
            
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WebSocket ID {conn_id} closed and resources released.")
