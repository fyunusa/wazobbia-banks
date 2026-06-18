#!/usr/bin/env python
import argparse
import asyncio
import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.stt.whisper_engine import WhisperSTTEngine
from voice.stt.mms_engine import MMSSTTEngine
from voice.normalizer import TranscriptNormalizer


async def main():
    parser = argparse.ArgumentParser(description="Test Wazobia Speech-to-Text Engines.")
    parser.add_argument("--audio", required=True, help="Path to the audio file (WAV, MP3, OGG, WebM).")
    parser.add_argument("--lang", default="auto", choices=["auto", "en", "pcm", "ha", "yo", "ig"],
                        help="Language code hint (default: auto).")
    args = parser.parse_args()

    audio_path = args.audio
    lang_hint = args.lang if args.lang != "auto" else None

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading audio file: {audio_path}...")
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Route to correct engine based on language code
    if lang_hint in ("ha", "yo", "ig"):
        print(f"Routing to MMSSTTEngine for language: {lang_hint}...")
        engine = MMSSTTEngine()
    else:
        # Default to Whisper for English, Pidgin, or Auto-detection
        engine_name = "WhisperSTTEngine" if lang_hint else "WhisperSTTEngine (Auto-detect)"
        print(f"Routing to {engine_name}...")
        engine = WhisperSTTEngine()

    print("Transcribing... (Please note: first run will download models if not cached)")
    try:
        result = await engine.transcribe(audio_bytes, language_hint=lang_hint)
    except Exception as e:
        print(f"\nTranscription failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 50)
    print("STT TRANSCRIPTION RESULT")
    print("=" * 50)
    print(f"Engine Used:       {result.engine_used}")
    print(f"Language Hint:     {args.lang}")
    print(f"Detected Language: {result.detected_language}")
    print(f"Duration:          {result.duration_seconds:.2f} seconds")
    print(f"Raw Transcript:\n{result.transcript}\n")

    # Apply normalization
    normalizer = TranscriptNormalizer()
    normalized_text = normalizer.normalize(result.transcript, result.detected_language)
    print(f"Normalized Transcript:\n{normalized_text}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
