import re


class AudioNormalizer:
    """Normalizes raw audio formats for STT models."""

    def __init__(self) -> None:
        pass

    def normalize(self, audio_data: bytes, sample_rate: int = 16000) -> bytes:
        return audio_data


class TextNormalizer:
    """Normalizes text for TTS output generation."""

    def __init__(self) -> None:
        pass

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        # Normalize currency symbol ₦ to "Naira"
        # e.g., ₦5000 -> 5000 Naira, 5000 ₦ -> 5000 Naira
        text = re.sub(r'₦\s*(\d+)', r'\1 Naira', text)
        text = re.sub(r'(\d+)\s*₦', r'\1 Naira', text)
        text = text.replace('₦', ' Naira ')

        # List of case-insensitive word replacements
        replacements = {
            r'\bgtbank\b': 'G T Bank',
            r'\bgtb\b': 'G T B',
            r'\bcbn\b': 'C B N',
            r'\bussd\b': 'U S S D',
            r'\bopay\b': 'O Pay',
            r'\bngn\b': 'Naira',
            r'\busd\b': 'U S D',
            r'\bfcmb\b': 'F C M B',
            r'\buba\b': 'U B A',
            r'\bwema\b': 'Wema',
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean up any consecutive whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text



class TranscriptNormalizer:
    """
    Normalizes speech transcripts for the Wazobia Agent.
    
    Ensures currency formatting, filler word removal, and preservation of
    named entities, USSD codes, Pidgin vocabulary, and code-switched terms.
    """

    def normalize(self, transcript: str, language: str) -> str:
        if not transcript:
            return ""

        # Remove English filler words (erm, uhh, like) for English/Pidgin
        if language in ("en", "pcm"):
            # Use word boundary boundaries to make sure we don't match substrings of other words
            transcript = re.sub(r'\b(erm|uhh|like)\b', '', transcript, flags=re.IGNORECASE)

        # Normalize currency: "naira" -> "₦", while "kobo" remains as is.
        # Format digits next to naira: "5000 naira" -> "₦5000"
        transcript = re.sub(r'\b(\d+)\s*naira\b', r'₦\1', transcript, flags=re.IGNORECASE)
        # Standalone: "the cost in naira is..." -> "the cost in ₦ is..."
        transcript = re.sub(r'\bnaira\b', '₦', transcript, flags=re.IGNORECASE)

        # Clean up any duplicate punctuation or spaces left behind by filler word removal
        transcript = re.sub(r'\s+([.,!?])', r'\1', transcript)
        transcript = re.sub(r',\s*,', ',', transcript)

        # Strip leading/trailing whitespace, normalize internal whitespace
        transcript = re.sub(r'\s+', ' ', transcript).strip()

        return transcript
