import os
import io
import asyncio
import logging
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_fixtures")

def generate_dummy_wav(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    wav_io = io.BytesIO()
    sf.write(wav_io, data, sample_rate, format="WAV", subtype="PCM_16")
    return wav_io.getvalue()

async def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures"))
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "sample_query_en.wav")

    logger.info(f"Generating fixture at: {target_path}")

    # Attempt to synthesize via Coqui TTS
    try:
        # We need to setup PYTHONPATH to import properly
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from voice.tts.coqui_engine import CoquiTTSEngine
        
        logger.info("Initializing CoquiTTSEngine for synthesis...")
        engine = CoquiTTSEngine()
        result = await engine.synthesize("What are the transfer charges for GTBank?", "en")
        
        with open(target_path, "wb") as f:
            f.write(result.audio_bytes)
        logger.info("Successfully generated query using Coqui TTS.")
    except Exception as e:
        logger.warning(f"Coqui TTS generation failed or unavailable ({e}). Falling back to dummy WAV.")
        dummy_bytes = generate_dummy_wav()
        with open(target_path, "wb") as f:
            f.write(dummy_bytes)
        logger.info("Successfully generated dummy WAV query.")

if __name__ == "__main__":
    asyncio.run(main())
