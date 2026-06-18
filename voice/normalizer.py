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

    # ── Nigerian Bank Names (Phonetic equivalents for reliable TTS) ─────────

    BANK_RULES: list[tuple[str, str]] = [
        # Commercial & Merchant Banks
        (r"\bGTBank\b",         "gee tee bank"),
        (r"\bGT Bank\b",        "gee tee bank"),
        (r"\bGTB\b",            "gee tee bee"),
        (r"\bFirstBank\b",      "first bank"),
        (r"\bFirst Bank\b",     "first bank"),
        (r"\bFirstMonie\b",     "first money"),
        (r"\bZenithBank\b",     "zenith bank"),
        (r"\bZenith Bank\b",    "zenith bank"),
        (r"\bAccessBank\b",     "access bank"),
        (r"\bAccess Bank\b",    "access bank"),
        (r"\bUBA\b",            "you bee ay"),
        (r"\bUnionBank\b",      "union bank"),
        (r"\bUnion Bank\b",     "union bank"),
        (r"\bWemaBank\b",       "wema bank"),
        (r"\bWema Bank\b",      "wema bank"),
        (r"\bFidelityBank\b",   "fidelity bank"),
        (r"\bFidelity Bank\b",  "fidelity bank"),
        (r"\bFCMB\b",           "eff see em bee"),
        (r"\bStanbicIBTC\b",    "stanbic eye bee tee see"),
        (r"\bStanbic IBTC\b",   "stanbic eye bee tee see"),
        (r"\bIBTC\b",           "eye bee tee see"),
        (r"\bSterlingBank\b",   "sterling bank"),
        (r"\bSterling Bank\b",  "sterling bank"),
        (r"\bPolarisBank\b",    "polaris bank"),
        (r"\bPolaris Bank\b",   "polaris bank"),
        (r"\bKeystoneBank\b",   "keystone bank"),
        (r"\bKeystone Bank\b",  "keystone bank"),
        (r"\bUnityBank\b",      "unity bank"),
        (r"\bUnity Bank\b",     "unity bank"),
        (r"\bEcobank\b",        "eco bank"),
        (r"\bHeritageBank\b",   "heritage bank"),
        (r"\bHeritage Bank\b",  "heritage bank"),
        (r"\bGlobusBank\b",     "globus bank"),
        (r"\bGlobus Bank\b",    "globus bank"),
        (r"\bSunTrustBank\b",   "sun trust bank"),
        (r"\bSunTrust Bank\b",  "sun trust bank"),
        (r"\bProvidusBank\b",   "providus bank"),
        (r"\bProvidus Bank\b",  "providus bank"),
        (r"\bTitanTrustBank\b", "titan trust bank"),
        (r"\bTitan Trust\b",    "titan trust"),
        (r"\bTitan Trust Bank\b", "titan trust bank"),
        (r"\bPremiumTrustBank\b", "premium trust bank"),
        (r"\bPremiumTrust\b",   "premium trust"),
        (r"\bPremium Trust Bank\b", "premium trust bank"),
        (r"\bSignatureBank\b",  "signature bank"),
        (r"\bSignature Bank\b", "signature bank"),
        (r"\bParallexBank\b",   "parallex bank"),
        (r"\bParallex Bank\b",  "parallex bank"),
        (r"\bNovaBank\b",       "nova bank"),
        (r"\bNova Bank\b",      "nova bank"),
        (r"\bNova Commercial Bank\b", "nova bank"),
        (r"\bOptimusBank\b",    "optimus bank"),
        (r"\bOptimus Bank\b",   "optimus bank"),
        (r"\bStandard Chartered\b", "standard chartered"),
        (r"\bCitibank\b",       "citi bank"),
        
        # Merchant Banks
        (r"\bCoronation Merchant Bank\b", "coronation merchant bank"),
        (r"\bCoronation\b",     "coronation"),
        (r"\bFBNQuest\b",       "eff bee enn quest"),
        (r"\bFSDH\b",           "eff ess dee aych"),
        (r"\bGreenwich Merchant Bank\b", "greenwich merchant bank"),
        (r"\bRand Merchant Bank\b", "rand merchant bank"),
        (r"\bRMB\b",            "are em bee"),
        
        # Non-Interest Banks
        (r"\bJaizBank\b",       "jaiz bank"),
        (r"\bJaiz Bank\b",      "jaiz bank"),
        (r"\bTajBank\b",        "taj bank"),
        (r"\bTaj Bank\b",       "taj bank"),
        (r"\bLotusBank\b",      "lotus bank"),
        (r"\bLotus Bank\b",     "lotus bank"),
        (r"\bThe Alternative Bank\b", "the alternative bank"),
        
        # Fintechs, Neo-banks & Microfinance Banks
        (r"\bOPay\b",           "oh pay"),
        (r"\bPalmPay\b",        "pahm pay"),
        (r"\bKudaBank\b",       "kooda bank"),
        (r"\bKuda Bank\b",      "kooda bank"),
        (r"\bKuda\b",           "kooda"),
        (r"\bMoniePoint\b",     "money point"),
        (r"\bMoiepoint\b",      "money point"),
        (r"\bCarbon\b",         "carbon"),
        (r"\bFairMoney\b",      "fair money"),
        (r"\bVFD Microfinance Bank\b", "vee bank"),
        (r"\bVbank\b",          "vee bank"),
        (r"\bV bank\b",         "vee bank"),
        (r"\bALAT\b",           "alat"),
        (r"\bRubies Bank\b",    "rubies bank"),
        (r"\bSparkle\b",        "sparkle"),
        (r"\bEyowo\b",          "eyowo"),
        
        # Payment Service Banks (PSBs)
        (r"\bHope PSBank\b",    "hope p s bank"),
        (r"\b9PSB\b",           "nine p s bee"),
        (r"\bMTN MoMo\b",       "em tee enn momo"),
        (r"\bMTN MoMo PSB\b",   "em tee enn momo p s bee"),
        (r"\bAirtel SmartCash\b", "airtel smart cash"),
        (r"\bSmartCash PSB\b",  "smart cash p s bee"),
    ]


    # ── Regulatory & Financial Acronyms (Phonetic spellings) ──────────────

    ACRONYM_RULES: list[tuple[str, str]] = [
        # Regulatory bodies
        (r"\bCBN\b",            "see bee enn"),
        (r"\bNDIC\b",           "enn dee eye see"),
        (r"\bSEC\b",            "ess ee see"),
        (r"\bEFCC\b",           "ee eff see see"),
        (r"\bFIRS\b",           "eff eye ar ess"),

        # Payment infrastructure
        (r"\bUSSD\b",           "you ess ess dee"),
        (r"\bNIBSS\b",          "nibs"),
        (r"\bNIP\b",            "enn eye pee"),
        (r"\bNEFT\b",           "neft"),
        (r"\bRTGS\b",           "are tee gee ess"),

        # Banking channels
        (r"\bATM\b",            "ay tee em"),
        (r"\bPOS\b",            "pee oh ess"),
        (r"\bOTP\b",            "oh tee pee"),
        (r"\bPIN\b",            "pin"),

        # Identity & compliance
        (r"\bBVN\b",            "bee vee enn"),
        (r"\bNIN\b",            "enn eye enn"),
        (r"\bKYC\b",            "kay wye see"),
        (r"\bAML\b",            "ay em el"),

        # Products
        (r"\bAPI\b",            "ay pee eye"),
        (r"\bSME\b",            "ess em ee"),
        (r"\bHNI\b",            "aych enn eye"),
        (r"\bVAS\b",            "vee ay ess"),
        (r"\bSMS\b",            "ess em ess"),
        (r"\bQR\b",             "cue are"),
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

        # Bank names — case-insensitive for brand integrity
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

        # 1. Run USSD codes first (to prevent markdown rules from stripping asterisks)
        text = self._normalize_ussd(text)

        # 2. Strip markdown artifacts & apply bank names and acronyms
        for rule in self._rules:
            text = rule.pattern.sub(rule.replacement, text)

        # 3. Currency (language-aware)
        text = self._normalize_currency(text, language)

        # 4. Numbers and percentages
        text = self._normalize_numbers(text)

        # 5. Clean up whitespace artifacts from substitutions
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
