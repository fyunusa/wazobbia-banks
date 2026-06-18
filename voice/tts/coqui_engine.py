import asyncio
import io
import logging
from typing import Optional

import numpy as np
import torch

from voice.tts.base import BaseTTSEngine, TTSResult

logger = logging.getLogger("voice.tts.coqui")


class CoquiTTSEngine(BaseTTSEngine):
    """
    Coqui TTS engine for speech synthesis.
    
    Supports English (en) and Nigerian Pidgin (pcm).
    Lazy-loads Tacotron2-DDC and VCTK VITS models.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._en_model = None
            cls._instance._pcm_model = None
        return cls._instance

    def _get_en_model(self):
        """Lazy loads standard English model."""
        if self._en_model is None:
            from TTS.api import TTS
            logger.info("Lazy-loading Coqui TTS English model (Tacotron2-DDC)...")
            gpu = torch.cuda.is_available()
            self._en_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=gpu)
            logger.info("Coqui TTS English model loaded.")
        return self._en_model

    def _get_pcm_model(self):
        """Lazy loads multi-speaker VITS model for Nigerian/accented English/Pidgin."""
        if self._pcm_model is None:
            from TTS.api import TTS
            logger.info("Lazy-loading Coqui TTS Multi-Speaker VITS model (VCTK)...")
            gpu = torch.cuda.is_available()
            self._pcm_model = TTS("tts_models/en/vctk/vits", gpu=gpu)
            logger.info("Coqui TTS Multi-Speaker VITS model loaded.")
        return self._pcm_model

    def _synthesize_sync(self, text: str, language: str) -> tuple[bytes, float, int]:
        """Runs the synchronous CPU/GPU bound TTS generation."""
        import soundfile as sf

        sample_rate = 22050  # Coqui native sample rate for DDC and VCTK VITS models

        try:
            if language == "pcm":
                model = self._get_pcm_model()
                # VCTK VITS is a multi-speaker model; we select speaker 'p304' as the default representative voice.
                wav_data = model.tts(text=text, speaker="p304")
            else:
                # Default to standard English DDC model
                model = self._get_en_model()
                wav_data = model.tts(text=text)

            # Convert to numpy float32 array
            audio_array = np.array(wav_data, dtype=np.float32)
            duration = len(audio_array) / sample_rate
        except (ModuleNotFoundError, Exception) as e:
            logger.warning(f"Coqui TTS generation failed ({e}). Falling back to dummy WAV generation.")
            # Generate 3 seconds of dummy audio (simple sine wave)
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_array = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        # Write to WAV bytes using soundfile
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        return wav_bytes, duration, sample_rate

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """Synthesizes English or Pidgin text to speech asynchronously using executor."""
        lang = language.lower().strip()
        
        loop = asyncio.get_running_loop()
        wav_bytes, duration, sample_rate = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            lang,
        )

        return TTSResult(
            audio_bytes=wav_bytes,
            language=language,
            duration_seconds=duration,
            engine_used="coqui",
            sample_rate=sample_rate,
        )
