import asyncio
import os
import wave
import numpy as np

async def test_tts(lang, text, filename):
    print(f"=== Synthesizing '{text}' for language '{lang}' ===")
    from voice.tts.router import TTSRouter
    router = TTSRouter()
    try:
        result = await router.synthesize(text, lang)
        print(f"[{lang}] Synthesized successfully using {result.engine_used}!")
        print(f"[{lang}] Duration: {result.duration_seconds:.2f}s, Sample Rate: {result.sample_rate}Hz")
        print(f"[{lang}] Audio bytes size: {len(result.audio_bytes)}")
        
        # Write to static files so it is accessible
        out_dir = "static/audio"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as f:
            f.write(result.audio_bytes)
        print(f"[{lang}] Saved to {out_path}")
        
        # Check audio stats
        with wave.open(out_path, 'rb') as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
            audio = np.frombuffer(frames, dtype=np.int16)
            print(f"[{lang}] Stats: Min={np.min(audio)}, Max={np.max(audio)}, Std={np.std(audio):.2f}, Unique={len(np.unique(audio))}")
    except Exception as e:
        print(f"[{lang}] Synthesis failed: {e}")

async def main():
    # Test Yoruba (MMS)
    await test_tts("yo", "E kaabo si ile ifowopamo gtbank", "test_yo_final.wav")
    
    # Test Hausa (MMS)
    await test_tts("ha", "Sannu da zuwa bankin gtbank", "test_ha_final.wav")
    
    # Test Igbo fallback (Coqui English)
    await test_tts("ig", "Nnoo na gtbank", "test_ig_fallback.wav")
    
    # Test English (Coqui)
    await test_tts("en", "Welcome to GTBank branch", "test_en_final.wav")

if __name__ == "__main__":
    # Run inside the container environment if needed, or locally.
    # We will run this inside the container since it has all weights and models.
    asyncio.run(main())
