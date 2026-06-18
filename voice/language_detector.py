import logging
from typing import Optional

try:
    from lingua import Language, LanguageDetectorBuilder
    _HAS_LINGUA = True
except ImportError:
    _HAS_LINGUA = False

logger = logging.getLogger("voice.language_detector")


class LanguageDetector:
    """
    Detects language from text or audio.
    
    Uses lingua-language-detector for English, Yoruba, Hausa, and Igbo,
    with a fallback to langdetect.
    """

    def __init__(self) -> None:
        self._detector = None
        if _HAS_LINGUA:
            try:
                languages = [Language.ENGLISH, Language.YORUBA, Language.HAUSA, Language.IGBO]
                self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
            except Exception as e:
                logger.error(f"Failed to initialize lingua detector: {e}")
                self._detector = None

    def detect(self, text: Optional[str] = None, audio_bytes: Optional[bytes] = None) -> str:
        """
        Detects the language of the provided input.
        
        Args:
            text: Text string to detect language from.
            audio_bytes: Raw audio bytes (if text is not provided).
            
        Returns:
            ISO 639-1 language code (e.g. 'en', 'ha', 'yo', 'ig') or 'auto'.
        """
        if text and text.strip():
            cleaned_text = text.strip()
            
            # 1. Try Lingua first if available
            if self._detector is not None:
                try:
                    confidences = self._detector.compute_language_confidence_values(cleaned_text)
                    if confidences:
                        top_result = confidences[0]
                        if top_result.value >= 0.7:
                            iso_map = {
                                Language.ENGLISH: "en",
                                Language.HAUSA: "ha",
                                Language.YORUBA: "yo",
                                Language.IGBO: "ig",
                            }
                            return iso_map.get(top_result.language, "en")
                        else:
                            # Confidence too low, default to English
                            return "en"
                except Exception as e:
                    logger.warning(f"Lingua language detection failed: {e}. Falling back to langdetect.")

            # 2. Fallback to langdetect
            try:
                import langdetect
                # Set seed for deterministic detection
                langdetect.DetectorFactory.seed = 0
                
                langs = langdetect.detect_langs(cleaned_text)
                if langs:
                    top_lang = langs[0]
                    if top_lang.prob >= 0.7:
                        # Map three-letter/two-letter codes if needed
                        lang_map = {
                            "en": "en",
                            "ha": "ha",
                            "yo": "yo",
                            "ig": "ig",
                            "hau": "ha",
                            "yor": "yo",
                            "ibo": "ig",
                        }
                        return lang_map.get(top_lang.lang, "en")
            except Exception as e:
                logger.warning(f"langdetect fallback failed: {e}")

            return "en"  # Default fallback if all fail

        # If audio only, return "auto" (will be determined post-transcription)
        if audio_bytes is not None:
            return "auto"

        return "en"
