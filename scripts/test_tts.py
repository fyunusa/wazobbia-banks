#!/usr/bin/env python
import argparse
import asyncio
import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.tts.router import TTSRouter


async def main():
    parser = argparse.ArgumentParser(description="Test Wazobia Text-to-Speech Engine.")
    parser.add_argument("--text", required=True, help="Text message to synthesize.")
    parser.add_argument("--lang", required=True, choices=["en", "pcm", "ha", "yo", "ig"],
                        help="Language code (en, pcm, ha, yo, ig).")
    parser.add_argument("--output", default="output.wav", help="Destination path for synthesized WAV (default: output.wav).")
    args = parser.parse_args()

    router = TTSRouter()
    print(f"Routing synthesis for language '{args.lang}'...")
    print(f"Synthesizing text: '{args.text}'")
    print("Inference starting... (Please note: first run will download models if not cached)")
    
    try:
        result = await router.synthesize(args.text, args.lang)
    except Exception as e:
        print(f"\nSynthesis failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    print("TTS SYNTHESIS RESULT")
    print("=" * 50)
    print(f"Engine Used:       {result.engine_used}")
    print(f"Language:          {result.language}")
    print(f"Duration:          {result.duration_seconds:.2f} seconds")
    print(f"Sample Rate:       {result.sample_rate} Hz")
    
    # Save WAV file
    try:
        with open(args.output, "wb") as f:
            f.write(result.audio_bytes)
        print(f"Saved audio output to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
