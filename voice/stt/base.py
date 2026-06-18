import io
import os
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field

import torch
import torchaudio

logger = logging.getLogger("voice.stt")


class TranscriptionResult(BaseModel):
    transcript: str
    detected_language: str  # ISO 639-1
    confidence: Optional[float] = None
    duration_seconds: float
    engine_used: str


def guess_audio_extension(audio_bytes: bytes) -> str:
    """Guesses the audio file extension based on magic bytes/headers."""
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
        return ".wav"
    elif audio_bytes.startswith(b"OggS"):
        return ".ogg"
    elif audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    elif audio_bytes.startswith(b"ID3") or (
        len(audio_bytes) > 2
        and audio_bytes[0] == 0xFF
        and (audio_bytes[1] & 0xE0) == 0xE0
    ):
        return ".mp3"
    return ".wav"  # Default fallback


class BaseSTTEngine(ABC):
    """Abstract base class for all Speech-to-Text engines."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language_hint: Optional[str] = None) -> TranscriptionResult:
        """Transcribe raw audio bytes to a TranscriptionResult object."""
        pass

    def preprocess_audio(self, audio_bytes: bytes) -> tuple[torch.Tensor, float]:
        """
        Preprocesses raw audio bytes to a 16kHz mono tensor.
        
        Accepts: WAV, WebM, MP3, OGG.
        Rejects: Audio longer than 60 seconds.
        Normalizes: Amplitude peak scaling before inference.
        
        Returns:
            waveform: torch.Tensor of shape (1, num_samples)
            duration_seconds: float
        """
        # Create temp folder inside workspace if it doesn't exist
        workspace_dir = os.getenv("WORKSPACE_DIR", os.getcwd())
        temp_dir = os.path.join(workspace_dir, ".temp_audio")
        try:
            os.makedirs(temp_dir, exist_ok=True)
        except Exception:
            import tempfile
            temp_dir = os.path.join(tempfile.gettempdir(), "wazobia_temp_audio")
            os.makedirs(temp_dir, exist_ok=True)

        ext = guess_audio_extension(audio_bytes)
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")

        try:
            with open(temp_file_path, "wb") as f:
                f.write(audio_bytes)

            waveform, sample_rate = torchaudio.load(temp_file_path)
        except Exception as e:
            logger.exception(f"Failed to load audio bytes with torchaudio: {e}")
            raise ValueError(f"Failed to load or parse audio format: {e}")
        finally:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as ex:
                    logger.warning(f"Failed to clean up temp file {temp_file_path}: {ex}")

        # Check number of channels and duration
        num_channels, num_frames = waveform.shape
        duration = num_frames / sample_rate

        if duration > 60.0:
            raise ValueError(f"Audio duration of {duration:.2f} seconds exceeds the maximum limit of 60 seconds.")

        # Convert to mono if stereo or multi-channel
        if num_channels > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16000Hz
        if sample_rate != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Normalize audio amplitude (Peak Normalization to scale samples to [-1.0, 1.0])
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return waveform, duration
