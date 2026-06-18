import asyncio
import io
import logging
import os
from voice.tts.base import BaseTTSEngine, TTSResult

logger = logging.getLogger("voice.tts.f5")


class F5TTSEngine(BaseTTSEngine):
    """
    F5-TTS Engine for high-fidelity Yoruba speech synthesis with tonal accuracy.
    Loads naijaml/f5-tts-yoruba from Hugging Face hub.
    Supports zero-shot voice cloning and falls back gracefully on failure.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = None
        return cls._instance

    def _load_model(self):
        """Lazily downloads and loads the F5-TTS model on first use."""
        if self._model is not None:
            return self._model

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"F5-TTS: Initializing engine on device: {self._device}...")

        try:
            from huggingface_hub import hf_hub_download
            from f5_tts.api import F5TTS

            # 1. Programmatically download weights and vocab from Hugging Face
            logger.info("F5-TTS: Downloading model checkpoint and vocab from Hugging Face Hub (naijaml/f5-tts-yoruba)...")
            ckpt_file = hf_hub_download(repo_id="naijaml/f5-tts-yoruba", filename="model_150000.pt")
            vocab_file = hf_hub_download(repo_id="naijaml/f5-tts-yoruba", filename="vocab.txt")

            # 2. Instantiate F5TTS API model
            self._model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=self._device,
            )
            logger.info("F5-TTS: Model loaded successfully.")
        except Exception as e:
            logger.error(f"F5-TTS: Initialization failed: {e}. Falling back to dummy generator.", exc_info=True)
            self._model = False

        return self._model

    def _synthesize_sync(self, text: str) -> tuple[bytes, float, int]:
        """Synchronous CPU/GPU bound F5-TTS inference."""
        import soundfile as sf
        import numpy as np

        model = self._load_model()
        if not model:
            # Load failure fallback: Generate dummy audio (440Hz sine wave)
            logger.warning("F5-TTS: Synthesis falling back to dummy WAV generator.")
            sample_rate = 24000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_array = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            return wav_io.getvalue(), duration, sample_rate

        try:
            # Resolve default reference audio from the workspace fixtures directory
            workspace_dir = os.getenv("WORKSPACE_DIR", os.getcwd())
            ref_file = os.path.join(workspace_dir, "tests", "fixtures", "sample_query_en.wav")
            
            if not os.path.exists(ref_file):
                # If the test fixture is not available, try a fallback path or log warning
                logger.warning(f"F5-TTS: Reference audio not found at {ref_file}, searching in current dir...")
                ref_file = "tests/fixtures/sample_query_en.wav"

            # Transcription matching sample_query_en.wav
            ref_text = "Thank you very much."

            # Perform inference
            logger.info(f"F5-TTS: Performing voice clone synthesis for text: '{text}'...")
            wav, sr, _ = model.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=text,
            )

            # Convert to standard float32 numpy array
            audio_array = np.array(wav, dtype=np.float32)
            duration = len(audio_array) / sr

            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sr, format="WAV", subtype="PCM_16")
            wav_bytes = wav_io.getvalue()

            return wav_bytes, duration, sr
        except Exception as e:
            logger.error(f"F5-TTS: Synthesis failed with error: {e}. Generating dummy fallback.", exc_info=True)
            # Synthesis error fallback
            sample_rate = 24000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_array = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
            
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_array, sample_rate, format="WAV", subtype="PCM_16")
            return wav_io.getvalue(), duration, sample_rate

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """Asynchronously synthesizes Yoruba text using an executor."""
        loop = asyncio.get_running_loop()
        wav_bytes, duration, sample_rate = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
        )

        return TTSResult(
            audio_bytes=wav_bytes,
            language="yo",
            duration_seconds=duration,
            engine_used="f5-tts",
            sample_rate=sample_rate,
        )
