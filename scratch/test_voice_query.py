import httpx
import os
import sys

API_BASE_URL = "http://localhost:8000"

def main():
    audio_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "sample_query_en.wav")
    if not os.path.exists(audio_path):
        print(f"Audio fixture not found at {audio_path}")
        sys.exit(1)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": ("sample_query_en.wav", audio_bytes, "audio/wav")}
    data = {
        "institution_slug": "gtbank",
        "preferred_language": "en",  # Test English audio
        "return_url": "true"         # Request JSON containing URL to audio file
    }

    url = f"{API_BASE_URL}/v1/voice/query"
    print(f"Sending POST request to {url}...")
    try:
        response = httpx.post(url, files=files, data=data, timeout=60.0)
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Response JSON:\n{response.text}")
    print(f"Content Length (bytes): {len(response.content)}")
    if len(response.content) > 100:
        print(f"Audio start bytes (hex): {response.content[:20].hex()}")
        # Save output to scratch/response_yo.wav to inspect
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "response_yo.wav")
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f"Saved response audio to {out_path}")
    else:
        print("Response body is empty or extremely short!")

if __name__ == "__main__":
    main()
