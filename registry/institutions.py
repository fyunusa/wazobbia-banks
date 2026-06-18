from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field


class ScrapeTarget(BaseModel):
    url: str
    category: Literal["fees", "products", "ussd", "rates", "complaints", "faq", "regulatory", "news"]
    requires_js: bool = False  # True = use Playwright, False = httpx+BS4
    notes: Optional[str] = None


class Institution(BaseModel):
    slug: str
    name: str
    full_name: str
    cbn_license_type: Literal[
        "Commercial Bank",
        "Merchant Bank",
        "Microfinance Bank",
        "Fintech/MFB",
        "Mobile Money Operator",
    ]
    ussd_code: Optional[str] = None
    customer_care: Optional[str] = None
    hq: str
    website: str
    scrape_targets: List[ScrapeTarget]
    logo_slug: str
    active: bool = True


# Supported Nigerian Financial Institutions Registry
INSTITUTIONS: Dict[str, Institution] = {
    "gtbank": Institution(
        slug="gtbank",
        name="GTBank",
        full_name="Guaranty Trust Bank Limited",
        cbn_license_type="Commercial Bank",
        ussd_code="*737#",
        customer_care="+234 1 448 0000",
        hq="Lagos, Nigeria",
        website="https://www.gtbank.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.gtbank.com/help-centre/bank-charges",
                category="fees",
                requires_js=False,
                notes="GTBank tariff and charges information",
            ),
            ScrapeTarget(
                url="https://www.gtbank.com/help-centre/faqs",
                category="faq",
                requires_js=True,
                notes="Frequently asked questions",
            ),
            ScrapeTarget(
                url="https://www.gtbank.com/personal-banking/cards",
                category="products",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.gtbank.com/personal-banking/loans-and-advances",
                category="products",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.gtbank.com/personal-banking/accounts",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="gtbank",
        active=True,
    ),
    "zenith": Institution(
        slug="zenith",
        name="Zenith Bank",
        full_name="Zenith Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*966#",
        customer_care="+234 1 278 7000",
        hq="Lagos, Nigeria",
        website="https://www.zenithbank.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.zenithbank.com/customer-service/tariff-guide",
                category="fees",
                requires_js=False,
                notes="Zenith Bank charges and tariff booklet",
            ),
            ScrapeTarget(
                url="https://www.zenithbank.com/customer-service/faq",
                category="faq",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.zenithbank.com/personal-banking/cards",
                category="products",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.zenithbank.com/personal-banking/loans",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="zenith",
        active=True,
    ),
    "access": Institution(
        slug="access",
        name="Access Bank",
        full_name="Access Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*901#",
        customer_care="+234 1 271 2005",
        hq="Lagos, Nigeria",
        website="https://www.accessbankplc.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.accessbankplc.com/pages/tariff-guide.aspx",
                category="fees",
                requires_js=False,
                notes="Access bank commercial charges",
            ),
            ScrapeTarget(
                url="https://www.accessbankplc.com/personal/cards",
                category="products",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.accessbankplc.com/personal/loans",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="access",
        active=True,
    ),
    "firstbank": Institution(
        slug="firstbank",
        name="FirstBank",
        full_name="First Bank of Nigeria Limited",
        cbn_license_type="Commercial Bank",
        ussd_code="*894#",
        customer_care="+234 1 448 5500",
        hq="Lagos, Nigeria",
        website="https://www.firstbanknigeria.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.firstbanknigeria.com/personal/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="firstbank",
        active=True,
    ),
    "uba": Institution(
        slug="uba",
        name="UBA",
        full_name="United Bank for Africa PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*919#",
        customer_care="+234 1 280 8822",
        hq="Lagos, Nigeria",
        website="https://www.ubagroup.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.ubagroup.com/nigeria/help-center/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="uba",
        active=True,
    ),
    "union": Institution(
        slug="union",
        name="Union Bank",
        full_name="Union Bank of Nigeria PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*826#",
        customer_care="+234 1 271 6816",
        hq="Lagos, Nigeria",
        website="https://www.unionbankng.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.unionbankng.com/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="union",
        active=True,
    ),
    "sterling": Institution(
        slug="sterling",
        name="Sterling Bank",
        full_name="Sterling Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*822#",
        customer_care="+234 1 448 4481",
        hq="Lagos, Nigeria",
        website="https://sterling.ng",
        scrape_targets=[
            ScrapeTarget(
                url="https://sterling.ng/help/tariffs",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="sterling",
        active=True,
    ),
    "wema": Institution(
        slug="wema",
        name="Wema Bank",
        full_name="Wema Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*945#",
        customer_care="+234 1 277 8600",
        hq="Lagos, Nigeria",
        website="https://www.wemabank.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.wemabank.com/help/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="wema",
        active=True,
    ),
    "fidelity": Institution(
        slug="fidelity",
        name="Fidelity Bank",
        full_name="Fidelity Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*770#",
        customer_care="+234 1 448 5252",
        hq="Lagos, Nigeria",
        website="https://www.fidelitybank.ng",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.fidelitybank.ng/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="fidelity",
        active=True,
    ),
    "fcmb": Institution(
        slug="fcmb",
        name="FCMB",
        full_name="First City Monument Bank Limited",
        cbn_license_type="Commercial Bank",
        ussd_code="*329#",
        customer_care="+234 1 270 8900",
        hq="Lagos, Nigeria",
        website="https://www.fcmb.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.fcmb.com/tariff-guide",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="fcmb",
        active=True,
    ),
    "stanbic": Institution(
        slug="stanbic",
        name="Stanbic IBTC",
        full_name="Stanbic IBTC Bank PLC",
        cbn_license_type="Commercial Bank",
        ussd_code="*909#",
        customer_care="+234 1 271 0123",
        hq="Lagos, Nigeria",
        website="https://www.stanbicibtcbank.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.stanbicibtcbank.com/nigeriabank/personal/our-pricing-and-tariffs",
                category="fees",
                requires_js=False,
            ),
        ],
        logo_slug="stanbic",
        active=True,
    ),
    "opay": Institution(
        slug="opay",
        name="OPay",
        full_name="OPay Digital Services Limited",
        cbn_license_type="Mobile Money Operator",
        ussd_code="*955#",
        customer_care="+234 1 888 8329",
        hq="Lagos, Nigeria",
        website="https://www.opayweb.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://www.opayweb.com/pricing",
                category="fees",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://www.opayweb.com/save",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="opay",
        active=True,
    ),
    "kuda": Institution(
        slug="kuda",
        name="Kuda",
        full_name="Kuda Microfinance Bank",
        cbn_license_type="Fintech/MFB",
        ussd_code="*506#",
        customer_care="+234 1 633 5832",
        hq="Lagos, Nigeria",
        website="https://kudabank.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://kudabank.com/pricing",
                category="fees",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://kudabank.com/save",
                category="products",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://kudabank.com/spend",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="kuda",
        active=True,
    ),
    "moniepoint": Institution(
        slug="moniepoint",
        name="Moniepoint",
        full_name="Moniepoint Microfinance Bank",
        cbn_license_type="Fintech/MFB",
        ussd_code="*5573#",
        customer_care="+234 1 888 9900",
        hq="Lagos, Nigeria",
        website="https://moniepoint.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://moniepoint.com/pricing",
                category="fees",
                requires_js=False,
            ),
            ScrapeTarget(
                url="https://moniepoint.com/personal",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="moniepoint",
        active=True,
    ),
    "palmpay": Institution(
        slug="palmpay",
        name="PalmPay",
        full_name="PalmPay Digital Services Limited",
        cbn_license_type="Mobile Money Operator",
        ussd_code="*656#",
        customer_care="+234 1 888 6888",
        hq="Lagos, Nigeria",
        website="https://palmpay.com",
        scrape_targets=[
            ScrapeTarget(
                url="https://palmpay.com/help-pricing",
                category="fees",
                requires_js=True,
                notes="Requires JS validation sometimes for pricing table",
            ),
            ScrapeTarget(
                url="https://palmpay.com/personal",
                category="products",
                requires_js=False,
            ),
        ],
        logo_slug="palmpay",
        active=True,
    ),
}


def get_institution(slug: str) -> Institution:
    """Retrieves an institution by its slug. Raises ValueError if not found."""
    if slug not in INSTITUTIONS:
        raise ValueError(f"Institution with slug '{slug}' not found in registry.")
    return INSTITUTIONS[slug]


def list_institutions(active_only: bool = True) -> List[Institution]:
    """Returns a list of all registered institutions, optionally filtering for active ones."""
    if active_only:
        return [inst for inst in INSTITUTIONS.values() if inst.active]
    return list(INSTITUTIONS.values())


# Languages configuration dictionary
LANGUAGE_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "en": {
        "name": "English",
        "mms_stt_code": None,
        "mms_tts_model": None,
        "whisper_lang": "en",
    },
    "ha": {
        "name": "Hausa",
        "mms_stt_code": "hau",
        "mms_tts_model": "facebook/mms-tts-hau",
        "whisper_lang": None,
    },
    "yo": {
        "name": "Yoruba",
        "mms_stt_code": "yor",
        "mms_tts_model": "facebook/mms-tts-yor",
        "whisper_lang": None,
    },
    "ig": {
        "name": "Igbo",
        "mms_stt_code": "ibo",
        "mms_tts_model": None,
        "whisper_lang": None,
    },
    "pcm": {
        "name": "Nigerian Pidgin",
        "mms_stt_code": None,
        "mms_tts_model": None,
        "whisper_lang": "en",
    },
}
