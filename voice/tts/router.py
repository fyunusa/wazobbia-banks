import logging
from voice.tts.base import BaseTTSEngine, TTSResult
from voice.tts.f5_engine import F5TTSEngine
from voice.tts.yarngpt_engine import YarnGPTTTSEngine
from voice.tts.coqui_engine import CoquiTTSEngine

logger = logging.getLogger("voice.tts.router")


class TTSRouter:
    """
    Routes speech synthesis requests to the correct TTS engine.
    
    Routes Yoruba to F5TTSEngine (zero-shot DiT), Hausa/Pidgin to YarnGPTTTSEngine
    (Nigerian-accented causal LM), and English/Igbo to CoquiTTSEngine,
    using English as a fallback.
    """

    def __init__(self) -> None:
        self.f5_engine = F5TTSEngine()
        self.yarngpt_engine = YarnGPTTTSEngine()
        self.coqui_engine = CoquiTTSEngine()

    async def synthesize(self, text: str, language: str) -> TTSResult:
        """
        Routes the synthesis request to the appropriate engine.
        
        Args:
            text: Text to synthesize.
            language: Target language code (e.g. 'ha', 'yo', 'ig', 'en', 'pcm').
            
        Returns:
            TTSResult containing the synthesized WAV bytes, duration, and metadata.
        """
        lang = (language or "en").lower().strip()
        
        from voice.normalizer import TextNormalizer
        normalizer = TextNormalizer()
        normalized_text = normalizer.normalize(text, lang)
        logger.info(f"TTS Text Normalization: '{text}' -> '{normalized_text}'")
        
        if lang == "yo":
            logger.info("Routing synthesis to F5TTSEngine (lang: yo)")
            result = await self.f5_engine.synthesize(normalized_text, lang)
        elif lang in ("ha", "pcm"):
            logger.info(f"Routing synthesis to YarnGPTTTSEngine (lang: {lang})")
            result = await self.yarngpt_engine.synthesize(normalized_text, lang)
        elif lang == "en":
            logger.info("Routing synthesis to CoquiTTSEngine (lang: en)")
            result = await self.coqui_engine.synthesize(normalized_text, lang)
        elif lang == "ig":
            logger.warning("Igbo TTS is not supported natively. Falling back to English CoquiTTSEngine.")
            result = await self.coqui_engine.synthesize(normalized_text, "en")
        else:
            logger.warning(f"Unknown language '{language}' requested for TTS. Falling back to English CoquiTTSEngine.")
            result = await self.coqui_engine.synthesize(normalized_text, "en")

        result.normalized_text = normalized_text
        return result


