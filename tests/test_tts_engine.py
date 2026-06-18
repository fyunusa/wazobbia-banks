import sys
from unittest.mock import AsyncMock, MagicMock, patch

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

mock_VitsModel = mock_transformers.VitsModel
mock_AutoTokenizer = mock_transformers.AutoTokenizer
mock_AutoProcessor = mock_transformers.AutoProcessor

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

import pytest
import numpy as np
import torch

from voice.tts.base import BaseTTSEngine, TTSResult
from voice.tts.mms_tts import MMSTTSEngine, strip_markdown, preprocess_text, split_into_sentences
from voice.tts.coqui_engine import CoquiTTSEngine
from voice.tts.f5_engine import F5TTSEngine
from voice.tts.yarngpt_engine import YarnGPTTTSEngine
from voice.tts.router import TTSRouter
from voice.vad import VoiceActivityDetector


# ==========================================
# 2. Text Pre-processing & Normalization Tests
# ==========================================
def test_strip_markdown():
    text = "### Balance\nYour *GTBank* balance is **₦5000**.\nCheck [here](http://gtbank.com) or run `*737#`."
    expected = "Balance\nYour GTBank balance is ₦5000.\nCheck here or run *737#."
    assert strip_markdown(text) == expected


def test_preprocess_text_ussd_and_currency():
    assert preprocess_text("₦1000 transfer fee") == "naira1000 transfer fee"
    assert preprocess_text("Dial *737# now") == "Dial star seven three seven hash now"
    assert preprocess_text("Dial *966# now") == "Dial star nine six six hash now"
    
    mixed = "**Dial** *120*10# to activate *966# abeg."
    assert "star nine six six hash" in preprocess_text(mixed)


def test_split_into_sentences():
    text = "This is sentence one. Sentence two! And sentence three?"
    result = split_into_sentences(text)
    assert len(result) == 3
    assert result[0] == "This is sentence one."
    assert result[1] == "Sentence two!"
    assert result[2] == "And sentence three?"


# ==========================================
# 3. VoiceActivityDetector Tests
# ==========================================
def test_vad_is_speech():
    silence = np.zeros(8000, dtype=np.int16).tobytes()
    sound = np.array([1000, -1000] * 4000, dtype=np.int16).tobytes()

    vad = VoiceActivityDetector(threshold=300.0)
    assert not vad.is_speech(silence)
    assert vad.is_speech(sound)


def test_vad_is_end_of_speech_stateful():
    sample_rate = 16000
    samples_500ms = int(sample_rate * 0.5)
    samples_1_5s = int(sample_rate * 1.5)

    vad = VoiceActivityDetector(threshold=300.0)

    initial_silence = np.zeros(samples_500ms, dtype=np.int16).tobytes()
    assert not vad.is_end_of_speech(initial_silence, sample_rate)
    assert vad.speech_detected is False

    speaking_chunk = np.array([2000, -2000] * int(samples_500ms / 2), dtype=np.int16)
    speaking_buffer = speaking_chunk.tobytes()
    assert not vad.is_end_of_speech(speaking_buffer, sample_rate)
    assert vad.speech_detected is True
    assert vad.silence_duration_samples == 0

    silent_chunk_1s = np.zeros(sample_rate, dtype=np.int16)
    speaking_and_silent = np.concatenate([speaking_chunk, silent_chunk_1s])
    buffer_1s_silence = speaking_and_silent.tobytes()
    
    assert not vad.is_end_of_speech(buffer_1s_silence, sample_rate)
    assert vad.silence_duration_samples == sample_rate

    silent_chunk_2s = np.zeros(sample_rate * 2, dtype=np.int16)
    speaking_and_silent_long = np.concatenate([speaking_chunk, silent_chunk_2s])
    buffer_2s_silence = speaking_and_silent_long.tobytes()

    assert vad.is_end_of_speech(buffer_2s_silence, sample_rate) is True


# ==========================================
# 4. TTS Router Tests
# ==========================================
@pytest.mark.asyncio
@patch("voice.tts.router.F5TTSEngine")
@patch("voice.tts.router.YarnGPTTTSEngine")
@patch("voice.tts.router.CoquiTTSEngine")
async def test_tts_router(mock_coqui_class, mock_yarngpt_class, mock_f5_class):
    mock_coqui = MagicMock()
    mock_yarngpt = MagicMock()
    mock_f5 = MagicMock()
    
    mock_coqui.synthesize = AsyncMock()
    mock_yarngpt.synthesize = AsyncMock()
    mock_f5.synthesize = AsyncMock()
    
    mock_coqui_class.return_value = mock_coqui
    mock_yarngpt_class.return_value = mock_yarngpt
    mock_f5_class.return_value = mock_f5

    router = TTSRouter()

    await router.synthesize("sannu", "ha")
    mock_yarngpt.synthesize.assert_called_once_with("sannu", "ha")

    await router.synthesize("e kaabo", "yo")
    mock_f5.synthesize.assert_called_once_with("e kaabo", "yo")
    
    await router.synthesize("hello", "en")
    mock_coqui.synthesize.assert_called_once_with("hello", "en")

    await router.synthesize("unknown text", "fr")
    mock_coqui.synthesize.assert_any_call("unknown text", "en")



# ==========================================
# 5. TTS Engine Mocked Synthesis Tests
# ==========================================
@pytest.mark.asyncio
@patch("soundfile.write")
async def test_mms_tts_synthesis(mock_sf_write):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_waveform_out = MagicMock()
    mock_waveform_out.waveform = torch.randn(1, 16000)
    mock_model.return_value = mock_waveform_out
    mock_model.config.sampling_rate = 16000
    mock_VitsModel.from_pretrained.return_value = mock_model

    engine = MMSTTSEngine()
    engine._models = {"yo": mock_model}
    engine._tokenizers = {"yo": mock_tokenizer}

    result = await engine.synthesize("kini owo gbigbe", "yo")

    assert isinstance(result, TTSResult)
    assert result.language == "yo"
    assert result.engine_used == "mms"
    assert result.sample_rate == 16000
    assert result.duration_seconds == 1.0
    mock_sf_write.assert_called_once()


@pytest.mark.asyncio
@patch("soundfile.write")
async def test_coqui_tts_synthesis(mock_sf_write):
    mock_tts_instance = MagicMock()
    mock_tts_instance.tts.return_value = [0.0] * 22050
    mock_TTS_class.return_value = mock_tts_instance

    engine = CoquiTTSEngine()
    engine._en_model = None
    engine._pcm_model = None

    result = await engine.synthesize("Please check my GTBank balance", "en")

    assert isinstance(result, TTSResult)
    assert result.language == "en"
    assert result.engine_used == "coqui"
    assert result.sample_rate == 22050
    assert result.duration_seconds == 1.0
    mock_sf_write.assert_called_once()


@pytest.mark.asyncio
@patch("soundfile.write")
@patch("voice.tts.f5_engine.hf_hub_download")
@patch("f5_tts.api.F5TTS")
async def test_f5_tts_synthesis(mock_f5_class, mock_download, mock_sf_write):
    mock_f5_instance = MagicMock()
    mock_f5_instance.infer.return_value = (np.zeros(24000), 24000, None)
    mock_f5_class.return_value = mock_f5_instance
    mock_download.return_value = "mock_path"

    engine = F5TTSEngine()
    engine._model = None

    result = await engine.synthesize("ẹ kaabo", "yo")

    assert isinstance(result, TTSResult)
    assert result.language == "yo"
    assert result.engine_used == "f5-tts"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0
    mock_sf_write.assert_called_once()


@pytest.mark.asyncio
@patch("soundfile.write")
@patch("voice.tts.yarngpt_engine.hf_hub_download")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("voice.tts.yarngpt_engine.AudioTokenizerV2")
async def test_yarngpt_tts_synthesis(mock_tokenizer_class, mock_model_class, mock_download, mock_sf_write):
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.get_audio.return_value = torch.zeros(1, 24000)
    mock_tokenizer_instance.get_codes.return_value = [1, 2, 3]
    mock_tokenizer_class.return_value = mock_tokenizer_instance
    mock_download.return_value = "mock_path"
    
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance

    engine = YarnGPTTTSEngine()
    engine._model = None
    engine._tokenizer = None

    result = await engine.synthesize("sannu", "ha")

    assert isinstance(result, TTSResult)
    assert result.language == "ha"
    assert result.engine_used == "yarngpt"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.0
    mock_sf_write.assert_called_once()
