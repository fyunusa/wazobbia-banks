import argparse
import sys
import os
import time
import httpx
from urllib.parse import unquote

API_BASE_URL = os.environ.get("WAZOBIA_API_URL", "http://localhost:8000")

def print_premium_card(institution: str, query: str, language: str, cache_hit: str, latency: str, confidence: str, answer: str, sources: list):
    """Prints a beautiful premium-styled ASCII card displaying query results."""
    border_len = 50
    print("═" * border_len)
    title = f"WAZOBIA AGENT — {institution.upper()} Agent"
    print(title)
    print("═" * border_len)
    print(f"Query     : {query}")
    print(f"Language  : {language}")
    print(f"Cache Hit : {cache_hit}")
    print(f"Latency   : {latency}")
    print(f"Confidence: {confidence}")
    print("─" * border_len)
    print("ANSWER:")
    print(answer)
    print("─" * border_len)
    print("SOURCES:")
    if sources:
        for s in sources:
            print(f" · {s}")
    else:
        print("  (None)")
    print("═" * border_len)

def handle_text_query(args):
    """Executes a standard text query to the RAG endpoint."""
    url = f"{API_BASE_URL}/v1/query"
    payload = {
        "query": args.query,
        "institution_slug": args.institution,
        "language": args.language
    }
    
    start_time = time.time()
    try:
        response = httpx.post(url, json=payload, timeout=30.0)
    except Exception as e:
        print(f"Error connecting to API at {url}: {e}", file=sys.stderr)
        sys.exit(1)
    latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        print(f"Rate limited! Status: 429. Retry-After: {retry_after} seconds.", file=sys.stderr)
        sys.exit(1)
        
    if response.status_code != 200:
        print(f"Query failed with HTTP {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()
    
    # Map language code to human name
    lang_map = {"en": "English", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo", "pcm": "Nigerian Pidgin"}
    lang_name = lang_map.get(data.get("language", args.language), args.language)
    
    print_premium_card(
        institution=args.institution or "General",
        query=args.query,
        language=lang_name,
        cache_hit="Yes" if data.get("cache_hit") else "No",
        latency=f"{latency_ms:,}ms",
        confidence=f"{data.get('confidence', 0.0):.2f}",
        answer=data.get("answer", ""),
        sources=data.get("sources", [])
    )

def handle_voice_query(args):
    """Executes a voice query by uploading an audio file to the voice REST endpoint."""
    if not args.audio:
        print("Error: --audio parameter is required when running in voice mode.", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at {args.audio}", file=sys.stderr)
        sys.exit(1)

    url = f"{API_BASE_URL}/v1/voice/query"
    
    # Read files
    with open(args.audio, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": (os.path.basename(args.audio), audio_bytes, "audio/wav")}
    data = {
        "institution_slug": args.institution,
        "preferred_language": args.language
    }

    start_time = time.time()
    try:
        response = httpx.post(url, files=files, data=data, timeout=30.0)
    except Exception as e:
        print(f"Error connecting to API at {url}: {e}", file=sys.stderr)
        sys.exit(1)
    latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "unknown")
        print(f"Rate limited! Status: 429. Retry-After: {retry_after} seconds.", file=sys.stderr)
        sys.exit(1)

    if response.status_code != 200:
        print(f"Voice query failed with HTTP {response.status_code}: {response.text}", file=sys.stderr)
        sys.exit(1)

    # Decode query details from response headers
    headers = response.headers
    transcript = unquote(headers.get("X-Transcript", "Unknown Query"))
    detected_lang = headers.get("X-Language", args.language)
    institution_slug = headers.get("X-Institution", args.institution)
    confidence = headers.get("X-Confidence", "0.0")
    
    sources_raw = headers.get("X-Sources", "[]")
    import json
    try:
        sources = json.loads(sources_raw)
    except Exception:
        sources = []

    # Map language code to human name
    lang_map = {"en": "English", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo", "pcm": "Nigerian Pidgin"}
    lang_name = lang_map.get(detected_lang, detected_lang)

    print_premium_card(
        institution=institution_slug,
        query=transcript,
        language=lang_name,
        cache_hit="No (Voice Query)",
        latency=f"{latency_ms:,}ms",
        confidence=confidence,
        answer="[Audio Response Generated Successfully]",
        sources=sources
    )

def main():
    parser = argparse.ArgumentParser(description="Wazobia Agent CLI Demo Client")
    parser.add_argument("--institution", type=str, required=True, help="Institution slug (e.g. gtbank, zenith, access)")
    parser.add_argument("--query", type=str, help="Text query to submit (required unless using voice)")
    parser.add_argument("--language", type=str, default="en", help="Language code (en, ha, yo, ig, pcm)")
    parser.add_argument("--voice", action="store_true", help="Enable voice query mode")
    parser.add_argument("--audio", type=str, help="Path to WAV audio file (required in voice mode)")

    args = parser.parse_args()

    if args.voice:
        handle_voice_query(args)
    else:
        if not args.query:
            print("Error: --query parameter is required when running in text mode.", file=sys.stderr)
            sys.exit(1)
        handle_text_query(args)

if __name__ == "__main__":
    main()
