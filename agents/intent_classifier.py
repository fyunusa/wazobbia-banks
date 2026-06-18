import re
import logging
from typing import Optional, Dict, List
from openai import AsyncOpenAI
from config.settings import settings
from registry.institutions import list_institutions

logger = logging.getLogger("agents.intent_classifier")

COMPARATIVE_KEYWORDS = ["which bank", "compare", "best bank", "cheapest", "vs", "versus", "cheaper", "difference between"]

INSTITUTION_ALIASES: Dict[str, List[str]] = {
    "gtbank": ["gtbank", "gt bank", "gtb", "guaranty trust", "gt"],
    "zenith": ["zenith", "zenith bank", "zenithbank"],
    "access": ["access", "access bank", "accessbank"],
    "firstbank": ["firstbank", "first bank", "fbn"],
    "uba": ["uba", "united bank for africa"],
    "union": ["union bank", "unionbank"],
    "sterling": ["sterling", "sterling bank"],
    "wema": ["wema", "wema bank"],
    "fidelity": ["fidelity", "fidelity bank"],
    "fcmb": ["fcmb"],
    "stanbic": ["stanbic", "stanbic ibtc", "stanbicibtc"],
    "opay": ["opay"],
    "kuda": ["kuda", "kudabank"],
    "moniepoint": ["moniepoint", "monie point"],
    "palmpay": ["palmpay", "palm pay"],
}


class IntentClassifier:
    """Classifies user queries for intent extraction and target institutions."""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None) -> None:
        self.openai_client = openai_client or AsyncOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY or "mock-key",
        )

    async def is_comparative_query(self, query: str) -> bool:
        """Determines if a query is comparing multiple institutions."""
        query_lower = query.lower()

        # Fast path: check keyword list
        if any(kw in query_lower for kw in COMPARATIVE_KEYWORDS):
            return True

        # Slow path: use LLM to classify ambiguity
        try:
            prompt = f"Analyze the following query and classify if the user wants to compare multiple banks/financial institutions (e.g. comparing fees, USSD codes, features). Reply with 'yes' or 'no' only.\n\nQuery: {query}"
            response = await self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a precise query classification assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0.0,
                extra_body={"repetition_penalty": 1.1},
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )
            result = (response.choices[0].message.content or "").strip().lower()
            return "yes" in result
        except Exception as e:
            logger.error(f"Failed comparative query classification: {e}", exc_info=True)
            return False

    async def extract_institution_slug(self, query: str) -> Optional[str]:
        """Extracts the matching institution slug from a user query using aliases."""
        query_lower = query.lower()

        # Iterate through defined aliases to find a match
        for slug, aliases in INSTITUTION_ALIASES.items():
            for alias in aliases:
                # Use word boundaries to prevent substring overlaps (e.g. matching "kuda" in "kudatest")
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, query_lower):
                    return slug

        return None
