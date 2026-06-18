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
        "preferred_language": "yo",  # Test Yoruba pipeline
        "return_url": "true"         # Request JSON containing URL to audio file
    }

    url = f"{API_BASE_URL}/v1/voice/query"
    print(f"Sending POST request to {url} with Yoruba...")
    try:
        response = httpx.post(url, files=files, data=data, timeout=120.0)
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    print(f"Status Code: {response.status_code}")
    print(f"Response JSON:\n{response.text}")

if __name__ == "__main__":
    main()
