import asyncio
import io
import logging
import re
from typing import List

import numpy as np
import torch

from voice.tts.base import BaseTTSEngine, TTSResult

logger = logging.getLogger("voice.tts.mms")


def strip_markdown(text: str) -> str:
    """Strips markdown styling elements to avoid TTS engine reading them aloud."""
    # Strip bold and italics
    text = re.sub(r"\*+\s*([^*]+?)\s*\*+", r"\1", text)
    text = re.sub(r"_+\s*([^_]+?)\s*_+", r"\1", text)
    # Strip links keeping the label text: [link text](url) -> link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip headers: # Header -> Header
    text = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"\1", text)
    # Strip inline code blocks: `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip list item bullet symbols
    text = re.sub(r"^\s*[-\*+]\s+", "", text, flags=re.MULTILINE)
    # Strip blockquote symbols
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Cleans markdown, formats Nigerian currency symbols, and spells out USSD codes."""
    text = strip_markdown(text)
    
    # ₦ -> naira
    text = text.replace("₦", "naira")

    # Spell out USSD code digit-by-digit: e.g. *737# -> star seven three seven hash
    digit_names = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }

    def replace_ussd(match):
        digits = match.group(1)
        spelled = " ".join([digit_names[d] for d in digits])
        return f"star {spelled} hash"

    text = re.sub(r"\*([0-9]+)#", replace_ussd, text)
    return text


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences based on punctuation boundaries followed by spaces."""
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


class MMSTTSEngine(BaseTTSEngine):
    """
    Meta MMS (Massively Multilingual Speech) Text-to-Speech engine.
    
    Loads independent models for Hausa, Yoruba, and Igbo at startup
    to minimize inference latency.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = {}
        self._tokenizers = {}
        
        self._initialized = True
        logger.info("MMSTTSEngine successfully initialized (lazy-loading enabled).")

    def _get_model_and_tokenizer(self, lang: str):
        """Lazily loads the VITS model and tokenizer for the specified language."""
        from transformers import VitsModel, AutoTokenizer

        lang_model_mapping = {
            "ha": "facebook/mms-tts-hau",
            "yo": "facebook/mms-tts-yor",
        }

        if lang not in lang_model_mapping:
            raise ValueError(f"Language '{lang}' not supported by MMSTTSEngine.")

        if lang not in self._models:
            model_id = lang_model_mapping[lang]
            logger.info(f"Lazily loading MMS TTS model for language '{lang}' ({model_id}) on device {self._device}...")
            self._models[lang] = VitsModel.from_pretrained(model_id).to(self._device)
            self._tokenizers[lang] = AutoTokenizer.from_pretrained(model_id)
            logger.info(f"MMS TTS model for '{lang}' loaded successfully.")

        return self._models[lang], self._tokenizers[lang]

    def _synthesize_sentence(self, text: str, lang: str) -> np.ndarray:
        """Synthesizes a single text segment to raw waveform data."""
        model, tokenizer = self._get_model_and_tokenizer(lang)

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).waveform

        try:
            waveform_data = output.squeeze().cpu().numpy()
        except RuntimeError:
            # Fallback if NumPy array conversion in PyTorch fails (e.g. host test environment mismatch)
            waveform_data = np.array(output.squeeze().cpu().tolist(), dtype=np.float32)

        return waveform_data

    def _synthesize_sync(self, text: str, lang: str) -> tuple[bytes, float, int]:
        """Performs pre-processing, sentence-splitting, inference, and concatenation."""
        import soundfile as sf

        cleaned_text = preprocess_text(text)

        # Split text into sentence segments if overall text is longer than 500 chars
        if len(cleaned_text) > 500:
            sentences = split_into_sentences(cleaned_text)
        else:
            sentences = [cleaned_text]

        waveforms = []
        for sentence in sentences:
            if not sentence:
                continue
            wave = self._synthesize_sentence(sentence, lang)
            waveforms.append(wave)

        if not waveforms:
            # Default fallback of 1 second of silence
            waveforms = [np.zeros(16000, dtype=np.float32)]

        # Concatenate sentence waveforms
        concatenated_waveform = np.concatenate(waveforms)
        model, _ = self._get_model_and_tokenizer(lang)
        sample_rate = model.config.sampling_rate
        duration = len(concatenated_waveform) / sample_rate

        # Write to WAV bytes buffer
        wav_io = io.BytesIO()
        sf.write(wav_io, concatenated_waveform, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_io.getvalue()

        return wav_bytes, duration, sample_rate

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """Synthesizes text to speech asynchronously."""
        lang = language.lower().strip()
        if lang not in ["ha", "yo"]:
            raise ValueError(f"Language '{language}' not supported by MMSTTSEngine. Supported: ['ha', 'yo']")

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
            engine_used="mms",
            sample_rate=sample_rate,
        )
