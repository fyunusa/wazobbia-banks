import asyncio
import logging
from typing import Optional

import numpy as np
import torch

from voice.stt.base import BaseSTTEngine, TranscriptionResult

logger = logging.getLogger("voice.stt.mms")


class MMSSTTEngine(BaseSTTEngine):
    """
    Meta MMS (Massively Multilingual Speech) engine for speech recognition.
    
    Supports Yoruba (yo), Hausa (ha), and Igbo (ig).
    Implemented as a singleton with lazy-loading and adapter swapping.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._processor = None
            cls._instance._current_adapter = None
            cls._instance._device = None
        return cls._instance

    def _load_model(self):
        """Lazily loads the base model and processor on the first use."""
        if self._model is None:
            from transformers import Wav2Vec2ForCTC, AutoProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading MMS base model (facebook/mms-1b-all) on device: {self._device}...")
            
            # facebook/mms-1b-all is loaded via Wav2Vec2ForCTC with default target language 'yor' to initialize the tokenizer/processor vocab
            self._model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/mms-1b-all",
                target_lang="yor",
                ignore_mismatched_sizes=True,
            ).to(self._device)
            self._processor = AutoProcessor.from_pretrained(
                "facebook/mms-1b-all",
                target_lang="yor",
            )
            logger.info("MMS base model successfully loaded.")

    def _transcribe_sync(self, audio_array: np.ndarray, language: str) -> tuple[str, str, Optional[float]]:
        """Performs CPU/GPU bound transcription synchronously."""
        self._load_model()

        # Language adapter routing
        mms_map = {"ha": "hau", "yo": "yor", "ig": "ibo"}
        mms_code = mms_map.get(language)
        if not mms_code:
            raise ValueError(
                f"Language '{language}' is not supported by MMSSTTEngine. Supported: {list(mms_map.keys())}"
            )

        # Load / Swap adapter if needed (keeping model in memory)
        if self._current_adapter != mms_code:
            logger.info(f"Swapping MMS adapter from '{self._current_adapter}' to '{mms_code}'...")
            self._model.load_adapter(mms_code)
            self._processor.tokenizer.set_target_lang(mms_code)
            # Ensure the newly loaded adapter weights are moved to the target device
            self._model.to(self._device)
            self._current_adapter = mms_code

        # Run inference
        inputs = self._processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self._processor.decode(ids)

        # Confidence is not directly output by Wav2Vec2 CTC decoder without extra CTC decoding steps
        confidence = None

        return transcription.strip(), language, confidence

    async def transcribe(self, audio_bytes: bytes, language_hint: Optional[str] = None) -> TranscriptionResult:
        """Preprocesses audio and transcribes it asynchronously using an executor."""
        waveform, duration = self.preprocess_audio(audio_bytes)
        try:
            audio_array = waveform.squeeze().cpu().numpy()
        except RuntimeError:
            # Fallback if NumPy initialization in PyTorch fails (e.g. host test environment mismatch)
            audio_array = np.array(waveform.squeeze().cpu().tolist(), dtype=np.float32)

        # MMS requires a specific target language adapter
        target_language = language_hint
        if not target_language:
            logger.warning("No language hint provided to MMSSTTEngine. Defaulting to 'yo'.")
            target_language = "yo"

        loop = asyncio.get_running_loop()
        transcript, detected_lang, confidence = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_array,
            target_language,
        )

        return TranscriptionResult(
            transcript=transcript,
            detected_language=detected_lang,
            confidence=confidence,
            duration_seconds=duration,
            engine_used="mms",
        )
