import sys
from unittest.mock import MagicMock, patch

# 1. Share or create the mock transformers module to prevent conflicts in full test runs
if "transformers" in sys.modules and isinstance(sys.modules["transformers"], MagicMock):
    mock_transformers = sys.modules["transformers"]
else:
    mock_transformers = MagicMock()
    sys.modules["transformers"] = mock_transformers

# Ensure all needed attributes exist on the shared mock
if not hasattr(mock_transformers, "pipeline"):
    mock_transformers.pipeline = MagicMock()
if not hasattr(mock_transformers, "Wav2Vec2ForCTC"):
    mock_transformers.Wav2Vec2ForCTC = MagicMock()
if not hasattr(mock_transformers, "AutoProcessor"):
    mock_transformers.AutoProcessor = MagicMock()
if not hasattr(mock_transformers, "VitsModel"):
    mock_transformers.VitsModel = MagicMock()
if not hasattr(mock_transformers, "AutoTokenizer"):
    mock_transformers.AutoTokenizer = MagicMock()

mock_pipeline_fn = mock_transformers.pipeline
mock_Wav2Vec2ForCTC = mock_transformers.Wav2Vec2ForCTC
mock_AutoProcessor = mock_transformers.AutoProcessor

import pytest
import numpy as np
import torch
import torchaudio

from voice.stt.base import BaseSTTEngine, TranscriptionResult, guess_audio_extension
from voice.stt.whisper_engine import WhisperSTTEngine
from voice.stt.mms_engine import MMSSTTEngine
from voice.language_detector import LanguageDetector
from voice.normalizer import TranscriptNormalizer


# ==========================================
# 2. TranscriptNormalizer Tests
# ==========================================
def test_transcript_normalizer_english_fillers():
    normalizer = TranscriptNormalizer()
    text = "Hello, erm, I would like, uhh, to open an account like."
    normalized = normalizer.normalize(text, "en")
    assert "erm" not in normalized.lower()
    assert "uhh" not in normalized.lower()
    assert "like" not in normalized.lower()
    
    text_yo = "Mo fe open account like."
    normalized_yo = normalizer.normalize(text_yo, "yo")
    assert "like" in normalized_yo


def test_transcript_normalizer_currency():
    normalizer = TranscriptNormalizer()
    assert normalizer.normalize("I have 5000 naira in my account", "en") == "I have ₦5000 in my account"
    assert normalizer.normalize("What is the cost in naira?", "en") == "What is the cost in ₦?"
    assert normalizer.normalize("I have 50 kobo left", "en") == "I have 50 kobo left"


def test_transcript_normalizer_preservation():
    normalizer = TranscriptNormalizer()
    text = "Please check my GTBank balance using *966#."
    assert normalizer.normalize(text, "en") == "Please check my GTBank balance using *966#."
    
    pidgin_text = "Abeg help me check wetin happen to my transfer."
    assert normalizer.normalize(pidgin_text, "pcm") == "Abeg help me check wetin happen to my transfer."


# ==========================================
# 3. LanguageDetector Tests
# ==========================================
def test_language_detector_text():
    detector = LanguageDetector()
    
    if detector._detector is not None:
        from lingua import Language
        mock_res = MagicMock()
        mock_res.value = 0.85
        
        mock_res.language = Language.ENGLISH
        detector._detector.compute_language_confidence_values = MagicMock(return_value=[mock_res])
        assert detector.detect("english text") == "en"
        
        mock_res.language = Language.YORUBA
        assert detector.detect("yoruba text") == "yo"
        
        mock_res.language = Language.HAUSA
        assert detector.detect("hausa text") == "ha"
        
        mock_res.language = Language.IGBO
        assert detector.detect("igbo text") == "ig"
        
        mock_res.value = 0.5
        assert detector.detect("gibberish text") == "en"
    else:
        with patch("langdetect.detect_langs") as mock_detect:
            mock_lang = MagicMock()
            mock_lang.lang = "yo"
            mock_lang.prob = 0.9
            mock_detect.return_value = [mock_lang]
            assert detector.detect("yoruba text") == "yo"
            
            mock_lang.prob = 0.5
            assert detector.detect("gibberish text") == "en"


def test_language_detector_low_confidence_and_audio():
    detector = LanguageDetector()
    assert detector.detect(text="") == "en"
    assert detector.detect(text="   ") == "en"
    assert detector.detect(audio_bytes=b"dummybytes") == "auto"


# ==========================================
# 4. Audio Preprocessing Tests
# ==========================================
class DummySTTEngine(BaseSTTEngine):
    async def transcribe(self, audio_bytes: bytes, language_hint: str = None) -> TranscriptionResult:
        pass


@patch("torchaudio.load")
def test_audio_preprocessing_success(mock_load):
    dummy_waveform = torch.randn(2, 48000)
    mock_load.return_value = (dummy_waveform, 48000)

    engine = DummySTTEngine()
    waveform, duration = engine.preprocess_audio(b"fake audio data")

    assert duration == 1.0
    assert waveform.shape == (1, 16000)
    assert torch.max(torch.abs(waveform)).item() == pytest.approx(1.0)


@patch("torchaudio.load")
def test_audio_preprocessing_too_long(mock_load):
    dummy_waveform = torch.randn(1, 16000 * 61)
    mock_load.return_value = (dummy_waveform, 16000)

    engine = DummySTTEngine()
    with pytest.raises(ValueError, match="exceeds the maximum limit of 60 seconds"):
        engine.preprocess_audio(b"fake audio data")


def test_guess_audio_extension():
    assert guess_audio_extension(b"RIFF\x00\x00\x00\x00WAVEfmt ") == ".wav"
    assert guess_audio_extension(b"OggS\x00\x02\x00\x00\x00") == ".ogg"
    assert guess_audio_extension(b"\x1a\x45\xdf\xa3\x01\x00") == ".webm"
    assert guess_audio_extension(b"ID3\x03\x00\x00\x00") == ".mp3"
    assert guess_audio_extension(b"unknown format") == ".wav"


# ==========================================
# 5. Whisper Engine Tests
# ==========================================
@pytest.mark.asyncio
@patch("voice.stt.whisper_engine.torch.cuda.is_available", return_value=False)
async def test_whisper_engine_transcribe(mock_cuda):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = {
        "text": " How much is the fee? ",
        "detected_language": "English"
    }
    mock_pipeline_fn.return_value = mock_pipe_instance

    with patch("torchaudio.load") as mock_load:
        mock_load.return_value = (torch.randn(1, 16000), 16000)

        engine = WhisperSTTEngine()
        engine._pipeline = None

        result = await engine.transcribe(b"fake_mp3_data", language_hint="en")

        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "How much is the fee?"
        assert result.detected_language == "en"
        assert result.engine_used == "whisper"
        assert result.duration_seconds == 1.0

        mock_pipe_instance.assert_called_once()
        call_args, call_kwargs = mock_pipe_instance.call_args
        assert call_kwargs["generate_kwargs"]["language"] == "en"


@pytest.mark.asyncio
@patch("voice.stt.whisper_engine.torch.cuda.is_available", return_value=False)
async def test_whisper_engine_pidgin_routing(mock_cuda):
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = {"text": "Wetin be the transfer charge?", "detected_language": "English"}
    mock_pipeline_fn.return_value = mock_pipe_instance

    with patch("torchaudio.load") as mock_load:
        mock_load.return_value = (torch.randn(1, 16000), 16000)

        engine = WhisperSTTEngine()
        engine._pipeline = None

        result = await engine.transcribe(b"fake_ogg_data", language_hint="pcm")

        assert result.transcript == "Wetin be the transfer charge?"
        assert result.detected_language == "pcm"

        call_args, call_kwargs = mock_pipe_instance.call_args
        assert call_kwargs["generate_kwargs"]["language"] == "en"
        assert "Nigerian Pidgin" in call_kwargs["generate_kwargs"]["prompt"]


# ==========================================
# 6. MMS Engine Tests
# ==========================================
@pytest.mark.asyncio
async def test_mms_engine_transcribe_and_adapter_swapping():
    mock_processor = MagicMock()
    mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
    mock_processor.decode.return_value = "kini owo gbigbe"
    mock_AutoProcessor.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_logits = MagicMock()
    mock_logits.logits = torch.randn(1, 50, 100)
    mock_model.return_value = mock_logits
    mock_Wav2Vec2ForCTC.from_pretrained.return_value = mock_model

    with patch("torchaudio.load") as mock_load:
        mock_load.return_value = (torch.randn(1, 16000), 16000)

        engine = MMSSTTEngine()
        engine._model = None
        engine._processor = None
        engine._current_adapter = None

        result = await engine.transcribe(b"fake_wav_data", language_hint="yo")

        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "kini owo gbigbe"
        assert result.detected_language == "yo"
        assert result.engine_used == "mms"
        assert result.duration_seconds == 1.0

        mock_model.load_adapter.assert_called_once_with("yor")
        mock_processor.tokenizer.set_target_lang.assert_called_once_with("yor")

        mock_processor.decode.return_value = "nawa ne kudin"
        result2 = await engine.transcribe(b"fake_wav_data2", language_hint="ha")
        assert result2.transcript == "nawa ne kudin"
        assert result2.detected_language == "ha"

        mock_model.load_adapter.assert_any_call("hau")
        mock_processor.tokenizer.set_target_lang.assert_any_call("hau")
