import io
import os
import uuid
import logging
from fastapi import HTTPException

import torch
import torchaudio
import numpy as np
import soundfile as sf

from voice.stt.base import guess_audio_extension

logger = logging.getLogger("voice.utils")


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, num_channels: int = 1) -> bytes:
    """Wraps raw 16-bit signed PCM audio bytes with a standard 44-byte WAV header."""
    num_samples = len(pcm_data) // 2
    byte_rate = sample_rate * num_channels * 2
    block_align = num_channels * 2
    
    header = bytearray()
    header.extend(b"RIFF")
    header.extend((36 + len(pcm_data)).to_bytes(4, "little"))
    header.extend(b"WAVEfmt ")
    header.extend((16).to_bytes(4, "little"))
    header.extend((1).to_bytes(2, "little"))  # PCM format
    header.extend((num_channels).to_bytes(2, "little"))
    header.extend((sample_rate).to_bytes(4, "little"))
    header.extend((byte_rate).to_bytes(4, "little"))
    header.extend((block_align).to_bytes(2, "little"))
    header.extend((16).to_bytes(2, "little"))  # Bits per sample
    header.extend(b"data")
    header.extend((len(pcm_data)).to_bytes(4, "little"))
    
    return bytes(header) + pcm_data


def validate_and_convert_audio(audio_bytes: bytes) -> bytes:
    """
    Validates audio format and converts it to a standard 16kHz mono WAV format.
    
    Supported formats: WAV, WebM, MP3, OGG, and raw PCM (which will be automatically
    wrapped with a WAV header).
    
    Args:
        audio_bytes: Raw audio data bytes.
        
    Returns:
        WAV-formatted bytes at 16kHz sample rate, mono channel, 16-bit PCM.
    """
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    # Detect if it's raw PCM (no common audio container magic bytes)
    is_raw_pcm = not (
        audio_bytes.startswith(b"RIFF")
        or audio_bytes.startswith(b"OggS")
        or audio_bytes.startswith(b"\x1a\x45\xdf\xa3")
        or audio_bytes.startswith(b"ID3")
        or (
            len(audio_bytes) > 2
            and audio_bytes[0] == 0xFF
            and (audio_bytes[1] & 0xE0) == 0xE0
        )
    )

    if is_raw_pcm:
        logger.info("Raw PCM audio streaming bytes detected. Wrapping with WAV header...")
        audio_bytes = pcm_to_wav(audio_bytes, sample_rate=16000, num_channels=1)

    # 1. Identify format using magic bytes
    ext = guess_audio_extension(audio_bytes)
    if ext not in (".wav", ".mp3", ".ogg", ".webm"):
        logger.warning(f"Unsupported audio format detected via magic bytes: {ext}")
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio format. Only WAV, WebM, MP3, and OGG formats are supported."
        )

    # 2. Save to a temporary file inside the workspace to load with torchaudio safely
    workspace_dir = os.getenv("WORKSPACE_DIR", os.getcwd())
    temp_dir = os.path.join(workspace_dir, ".temp_audio")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), "wazobia_temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
    
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    try:
        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)
        waveform, sample_rate = torchaudio.load(temp_file_path)
    except Exception as e:
        logger.error(f"Failed to load/decode audio file: {e}", exc_info=True)
        raise HTTPException(
            status_code=415,
            detail=f"Invalid or corrupted audio file. Ensure the audio is decodable."
        )
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as ex:
                logger.warning(f"Failed to clean up temporary audio file {temp_file_path}: {ex}")

    # 3. Process waveform (Mono conversion & 16kHz resampling)
    num_channels, num_frames = waveform.shape
    
    # Mono conversion
    if num_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # 16kHz resampling
    if sample_rate != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # 4. Save to WAV bytes using soundfile
    try:
        waveform_data = waveform.squeeze().cpu().numpy()
    except RuntimeError:
        # Fallback if NumPy array conversion in PyTorch fails (e.g. host test environment mismatch)
        waveform_data = np.array(waveform.squeeze().cpu().tolist(), dtype=np.float32)

    wav_io = io.BytesIO()
    sf.write(wav_io, waveform_data, 16000, format="WAV", subtype="PCM_16")
    return wav_io.getvalue()
