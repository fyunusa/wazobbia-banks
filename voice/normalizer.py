import re


class AudioNormalizer:
    """Normalizes raw audio formats for STT models."""

    def __init__(self) -> None:
        pass

    def normalize(self, audio_data: bytes, sample_rate: int = 16000) -> bytes:
        return audio_data


import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NormRule:
    """A single normalization rule."""
    pattern: re.Pattern
    replacement: str
    description: str


class TextNormalizer:
    """
    Deterministic pre-TTS text normalizer for Nigerian financial content.

    Handles:
    - Bank name acronyms and brand compounds
    - Regulatory body acronyms (CBN, NDIC, SEC, etc.)
    - Financial operation acronyms (USSD, ATM, POS, NIP, etc.)
    - Identity acronyms (BVN, NIN, KYC)
    - USSD code patterns (*737*6*1#)
    - Naira currency symbol (₦)
    - Markdown artifacts (bold, bullets) that bleed through from LLM

    Language-aware: Hausa/Yoruba/Igbo get currency in local phrasing.
    """

    # ── Nigerian Bank Names ───────────────────────────────────────────────

    BANK_RULES: list[tuple[str, str]] = [
        # Exact brand compounds that TTS reads wrong
        (r"\bGTBank\b",         "G T Bank"),
        (r"\bGT Bank\b",        "G T Bank"),         # already spaced but TTS still slurs
        (r"\bGTB\b",            "G T B"),
        (r"\bFirstBank\b",      "First Bank"),
        (r"\bFirstMonie\b",     "First Monie"),
        (r"\bOPay\b",           "O Pay"),
        (r"\bPalmPay\b",        "Palm Pay"),
        (r"\bMoniePoint\b",     "Monie Point"),
        (r"\bMoiepoint\b",      "Monie Point"),       # common OCR/scrape typo
        (r"\bStanbicIBTC\b",    "Stanbic I B T C"),
        (r"\bStanbic IBTC\b",   "Stanbic I B T C"),
        (r"\bIBTC\b",           "I B T C"),
        (r"\bEcobank\b",        "Eco Bank"),
        (r"\bFCMB\b",           "F C M B"),
        (r"\bUBA\b",            "U B A"),
        (r"\bGTWorld\b",        "G T World"),
        (r"\bZenithBank\b",     "Zenith Bank"),
        (r"\bAccessBank\b",     "Access Bank"),
        (r"\bUnionBank\b",      "Union Bank"),
        (r"\bWemaBank\b",       "Wema Bank"),
        (r"\bKudaBank\b",       "Kuda Bank"),
    ]

    # ── Regulatory & Financial Acronyms ──────────────────────────────────

    ACRONYM_RULES: list[tuple[str, str]] = [
        # Regulatory bodies
        (r"\bCBN\b",            "C B N"),
        (r"\bNDIC\b",           "N D I C"),
        (r"\bSEC\b",            "S E C"),             # Securities & Exchange Commission
        (r"\bEFCC\b",           "E F C C"),
        (r"\bFIRS\b",           "F I R S"),

        # Payment infrastructure
        (r"\bUSSD\b",           "U S S D"),
        (r"\bNIBSS\b",          "N I B S S"),
        (r"\bNIP\b",            "N I P"),             # NIBSS Instant Payment
        (r"\bNEFT\b",           "N E F T"),
        (r"\bRTGS\b",           "R T G S"),

        # Banking channels
        (r"\bATM\b",            "A T M"),
        (r"\bPOS\b",            "P O S"),
        (r"\bOTP\b",            "O T P"),
        (r"\bPIN\b",            "P I N"),

        # Identity & compliance
        (r"\bBVN\b",            "B V N"),
        (r"\bNIN\b",            "N I N"),
        (r"\bKYC\b",            "K Y C"),
        (r"\bAML\b",            "A M L"),

        # Products
        (r"\bAPI\b",            "A P I"),
        (r"\bSME\b",            "S M E"),
        (r"\bHNI\b",            "H N I"),             # High Net Worth Individual
        (r"\bVAS\b",            "V A S"),
        (r"\bSMS\b",            "S M S"),
        (r"\bQR\b",             "Q R"),
    ]

    # ── Markdown cleanup (LLM bleeds formatting into responses) ──────────

    MARKDOWN_RULES: list[tuple[str, str]] = [
        (r"\*\*(.*?)\*\*",      r"\1"),              # **bold** → plain
        (r"\*(.*?)\*",          r"\1"),              # *italic* → plain
        (r"^#{1,6}\s+",         ""),                 # ## Heading → plain
        (r"^[-*+]\s+",          ""),                 # bullet points → plain
        (r"`([^`]+)`",          r"\1"),              # `code` → plain
        (r"\[([^\]]+)\]\([^)]+\)", r"\1"),           # [text](url) → text
        (r"\n{3,}",             "\n\n"),             # excessive newlines
    ]

    def __init__(self):
        self._rules: list[NormRule] = self._compile_rules()
        logger.info(f"TextNormalizer initialized with {len(self._rules)} rules")

    def _compile_rules(self) -> list[NormRule]:
        """Compile all rules into NormRule objects with pre-compiled patterns."""
        rules = []

        # Markdown first — strip formatting before touching content
        for pattern, replacement in self.MARKDOWN_RULES:
            rules.append(NormRule(
                pattern=re.compile(pattern, re.MULTILINE | re.IGNORECASE),
                replacement=replacement,
                description=f"markdown: {pattern}"
            ))

        # Bank names — case-sensitive for brand integrity
        for pattern, replacement in self.BANK_RULES:
            rules.append(NormRule(
                pattern=re.compile(pattern, re.IGNORECASE),
                replacement=replacement,
                description=f"bank: {replacement}"
            ))

        # Acronyms — uppercase match only to avoid clobbering regular words
        # e.g. "pin" in a sentence should NOT become "P I N"
        for pattern, replacement in self.ACRONYM_RULES:
            rules.append(NormRule(
                # \b word boundary, original pattern already uppercase-anchored
                pattern=re.compile(pattern),
                replacement=replacement,
                description=f"acronym: {replacement}"
            ))

        return rules

    # ── Currency normalization ────────────────────────────────────────────

    CURRENCY_TEMPLATES = {
        "en":  "{amount} naira",
        "ha":  "naira {amount}",       # Hausa: naira precedes in some contexts
        "yo":  "naira {amount}",       # Yoruba
        "ig":  "{amount} naira",       # Igbo
        "pcm": "{amount} naira",       # Pidgin
    }

    def _normalize_currency(self, text: str, language: str) -> str:
        """
        ₦5,000  → "5000 naira"
        ₦52.50  → "52 naira 50 kobo"
        NGN5000 → "5000 naira"
        """
        template = self.CURRENCY_TEMPLATES.get(language, "{amount} naira")

        def replace_naira(match: re.Match) -> str:
            raw = match.group(1).replace(",", "")
            try:
                value = float(raw)
                naira = int(value)
                kobo = round((value - naira) * 100)
                if kobo > 0:
                    return f"{naira} naira {kobo} kobo"
                return template.format(amount=naira)
            except ValueError:
                return match.group(0)

        # ₦5,000 or ₦52.50
        text = re.sub(r"₦([\d,]+\.?\d*)", replace_naira, text)

        # NGN5000 or NGN 5,000
        text = re.sub(
            r"\bNGN\s*([\d,]+\.?\d*)",
            lambda m: replace_naira(re.search(r"([\d,]+\.?\d*)", m.group(0))),
            text
        )

        return text

    # ── USSD code normalization ───────────────────────────────────────────

    def _normalize_ussd(self, text: str) -> str:
        """
        *737*6*1#  → "star 7 3 7 star 6 star 1 hash"
        *901#      → "star 9 0 1 hash"

        Strategy: spell out the digits individually for clarity,
        since "star seven three seven" is clearer than "star 737".
        """

        def spell_digits(number: str) -> str:
            return " ".join(list(number))

        def replace_ussd(match: re.Match) -> str:
            full = match.group(0)
            # Strip leading * and trailing #
            inner = full[1:-1]
            # Split on * to get segments
            segments = inner.split("*")
            parts = [f"star {spell_digits(seg)}" for seg in segments]
            return " ".join(parts) + " hash"

        # Match full USSD patterns: *digits(*digits)*..#
        text = re.sub(r"\*[\d*]+#", replace_ussd, text)

        return text

    # ── Number normalization ──────────────────────────────────────────────

    def _normalize_numbers(self, text: str) -> str:
        """
        Large numbers with commas: 1,000,000 → "1 million"
        Percentages stay as-is: 12.5% → "12 point 5 percent"
        """
        # Percentages
        text = re.sub(
            r"(\d+)\.(\d+)%",
            lambda m: f"{m.group(1)} point {m.group(2)} percent",
            text
        )
        text = re.sub(r"(\d+)%", r"\1 percent", text)

        # Large round numbers
        def replace_large(match: re.Match) -> str:
            raw = match.group(0).replace(",", "")
            try:
                n = int(raw)
                if n >= 1_000_000_000:
                    return f"{n // 1_000_000_000} billion"
                if n >= 1_000_000:
                    return f"{n // 1_000_000} million"
                if n >= 1_000:
                    # Keep thousands as-is (e.g. "5,000" → "5000") — TTS handles
                    return raw
                return raw
            except ValueError:
                return match.group(0)

        # Match large numbers with commas
        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", replace_large, text)

        return text

    # ── Main normalize entry point ────────────────────────────────────────

    def normalize(self, text: str, language: str = "en") -> str:
        """
        Full normalization pipeline. Call this on every LLM response
        before passing to any TTS engine.

        Args:
            text:     Raw LLM response text
            language: ISO 639-1 language code (en | ha | yo | ig | pcm)

        Returns:
            Normalized text safe for TTS synthesis
        """
        if not text or not text.strip():
            return text

        original = text

        # 1. Strip markdown artifacts
        # 2. Normalize bank names and acronyms
        for rule in self._rules:
            text = rule.pattern.sub(rule.replacement, text)

        # 3. Currency (language-aware)
        text = self._normalize_currency(text, language)

        # 4. USSD codes
        text = self._normalize_ussd(text)

        # 5. Numbers and percentages
        text = self._normalize_numbers(text)

        # 6. Clean up whitespace artifacts from substitutions
        text = re.sub(r"  +", " ", text)
        text = text.strip()

        if text != original:
            logger.debug(
                "tts_normalization_applied",
                extra={
                    "language": language,
                    "original_length": len(original),
                    "normalized_length": len(text),
                }
            )

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
