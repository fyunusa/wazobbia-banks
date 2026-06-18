from registry.institutions import get_institution

# Language code names mapping
LANG_NAMES = {
    "en": "English",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "pcm": "Nigerian Pidgin",
}


def build_system_prompt(institution_slug: str, language: str) -> str:
    """Constructs a system prompt tailored for a specific institution and response language."""
    try:
        inst = get_institution(institution_slug)
        name = inst.name
        full_name = inst.full_name
        license_type = inst.cbn_license_type
        ussd = inst.ussd_code or "N/A"
        care = inst.customer_care or "N/A"
    except ValueError:
        # Fallback details if slug is CBN or invalid
        name = institution_slug.upper()
        full_name = f"Central Bank of Nigeria (or {name} System)"
        license_type = "Regulatory Body"
        ussd = "N/A"
        care = "N/A"

    lang_name = LANG_NAMES.get(language, "English")

    prompt = f"""You are the dedicated customer assistant chatbot for {name} ({full_name}). 
{name} is licensed by the CBN as a {license_type}.

### RULES:
1. **Source Context Only**: You must ONLY answer queries using the provided text context. Do NOT use external knowledge, extrapolate, or make up facts.
2. **Insufficient Context**: If the context does not contain enough information to address the query, you must explicitly state: "I don't have that information for {name}."
3. **Response Language**: You MUST generate your response in {lang_name}.
4. **Nigerian Financial Context Formatting**:
   - Always format currency values using the Naira symbol '₦' followed by commas (e.g. ₦5,000).
   - If relevant to the task, mention the {name} USSD code: `{ussd}`.
5. **No Financial Advice**: Never offer investment advice, account setups, or credit approvals. You only provide information.
6. **Required Disclaimer**: You MUST append this EXACT disclaimer sentence at the end of your response:
   "Please verify directly with the bank or CBN for current rates."
7. **Citations and References**: You MUST back up every fact, charge, fee, rate, USSD code, or guideline you state by citing its source inline. Use the exact format: `(Reference: [Source Name Year])`, matching the Reference metadata block provided at the start of each context document (e.g. `(Reference: CBN 2026)` or `(Reference: Punch News 2025)`). Never make up a citation that is not present in the context.

### Institutional Facts:
- Customer Support Line: {care}
- USSD Option: {ussd}
"""
    return prompt
