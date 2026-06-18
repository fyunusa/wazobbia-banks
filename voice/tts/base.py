from abc import ABC, abstractmethod
from pydantic import BaseModel


class TTSResult(BaseModel):
    audio_bytes: bytes  # WAV format audio
    language: str
    duration_seconds: float
    engine_used: str
    sample_rate: int
    normalized_text: str = ""



class BaseTTSEngine(ABC):
    """Abstract base class for all Text-to-Speech engines."""

    @abstractmethod
    async def synthesize(self, text: str, language: str, gender: str = "female") -> TTSResult:
        """Synthesize text to audio bytes."""
        pass
