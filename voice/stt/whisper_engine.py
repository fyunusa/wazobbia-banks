import asyncio
import logging
from typing import Optional

import numpy as np
import torch

from voice.stt.base import BaseSTTEngine, TranscriptionResult

logger = logging.getLogger("voice.stt.whisper")


class WhisperSTTEngine(BaseSTTEngine):
    """
    OpenAI Whisper engine for speech recognition.
    
    Implemented as a singleton with lazy-loading of the model.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._pipeline = None
        return cls._instance

    def _get_pipeline(self):
        """Lazily loads the Whisper pipeline on the first use."""
        if self._pipeline is None:
            from transformers import pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            from config.settings import settings
            logger.info(f"Loading Whisper pipeline ({settings.WHISPER_MODEL}) on device: {device}...")
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=settings.WHISPER_MODEL,
                device=device,
                torch_dtype=torch_dtype,
            )
            logger.info(f"Whisper pipeline ({settings.WHISPER_MODEL}) successfully loaded.")
        return self._pipeline

    def _transcribe_sync(self, audio_array: np.ndarray, language_hint: Optional[str]) -> tuple[str, str, Optional[float]]:
        """Performs CPU/GPU bound transcription synchronously."""
        pipe = self._get_pipeline()

        generate_kwargs = {"task": "transcribe"}

        # Language routing & Prompt engineering
        if language_hint == "pcm":
            generate_kwargs["language"] = "en"
            generate_kwargs["prompt"] = "Nigerian Pidgin context: abeg, wetin, naira, bank, GTBank, Zenith, Access, kobo."
        elif language_hint == "en":
            generate_kwargs["language"] = "en"
        elif language_hint:
            # Fallback for other languages if provided
            generate_kwargs["language"] = language_hint

        # Inference
        result = pipe(
            audio_array,
            generate_kwargs=generate_kwargs,
            return_language=True if not language_hint else False
        )

        transcript = result.get("text", "").strip()

        # Handle detected language
        detected_lang = "en"
        if language_hint:
            detected_lang = language_hint
        else:
            raw_lang = result.get("detected_language")
            if raw_lang:
                raw_lang = raw_lang.lower().strip()
                lang_mapping = {
                    "english": "en",
                    "hausa": "ha",
                    "yoruba": "yo",
                    "igbo": "ig",
                    "nigerian pidgin": "pcm",
                    "pidgin": "pcm",
                    "en": "en",
                    "ha": "ha",
                    "yo": "yo",
                    "ig": "ig",
                }
                detected_lang = lang_mapping.get(raw_lang, raw_lang)

        # Whisper pipeline does not natively return a single confidence float in simple format
        confidence = None

        return transcript, detected_lang, confidence

    async def transcribe(self, audio_bytes: bytes, language_hint: Optional[str] = None) -> TranscriptionResult:
        """Preprocesses audio and transcribes it asynchronously using an executor."""
        waveform, duration = self.preprocess_audio(audio_bytes)
        try:
            audio_array = waveform.squeeze().cpu().numpy()
        except RuntimeError:
            # Fallback if NumPy initialization in PyTorch fails (e.g. host test environment mismatch)
            audio_array = np.array(waveform.squeeze().cpu().tolist(), dtype=np.float32)

        loop = asyncio.get_running_loop()
        transcript, detected_lang, confidence = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_array,
            language_hint,
        )

        return TranscriptionResult(
            transcript=transcript,
            detected_language=detected_lang,
            confidence=confidence,
            duration_seconds=duration,
            engine_used="whisper",
        )
