import io
import re
import unicodedata
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import trafilatura
import pdfplumber
from langdetect import detect
from ingestion.scrapers.base_scraper import RawDocument

logger = logging.getLogger("ingestion.processors.cleaner")

CSS_BLOCKLIST = [
    "nav", "footer", "header", "aside", "noscript", "script", "style",
    ".cookie-banner", ".cookie-consent", ".cookie-policy", "#cookie-consent",
    ".ads", ".ad-container", ".navigation", ".menu", ".footer-links"
]


class CleanedDocument(BaseModel):
    url: str
    category: str
    sub_category: Optional[str] = None
    scraped_at: datetime
    institution_slug: str
    http_status: int
    content_type: str
    clean_text: str
    title: Optional[str] = None
    extracted_tables: List[str] = Field(default_factory=list)
    word_count: int
    language_detected: str


class DocumentCleaner:
    """Cleans raw HTML or PDF documents, normalizes text content, and extracts tables."""

    def __init__(self) -> None:
        pass

    def _format_amount(self, amount_str: str) -> str:
        """Helper to format numeric string with commas."""
        cleaned = amount_str.replace(",", "")
        try:
            if "." in cleaned:
                parts = cleaned.split(".")
                val = int(parts[0])
                dec = parts[1][:2]
                return f"{val:,}.{dec}"
            else:
                val = int(cleaned)
                return f"{val:,}"
        except ValueError:
            return amount_str

    def normalize_text(self, text: str) -> str:
        """Normalizes unicode, white spaces, Naira currency format, and USSD codes."""
        if not text:
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Normalize Naira values: e.g. "NGN5000", "5000 naira", "N5,000" -> "₦5,000"
        # 1. Prefix NGN / N with optional space: (NGN|N)\s*(\d+)
        pattern_prefix = re.compile(r'\b(?:NGN|N)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b', re.IGNORECASE)
        text = pattern_prefix.sub(lambda m: f"₦{self._format_amount(m.group(1))}", text)

        # 2. Suffix with space and naira/NGN: (\d+)\s*(naira|NGN)
        pattern_suffix = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:naira|NGN)\b', re.IGNORECASE)
        text = pattern_suffix.sub(lambda m: f"₦{self._format_amount(m.group(1))}", text)

        # 3. Format raw ₦5000 -> ₦5,000
        pattern_naira = re.compile(r'₦\s*(\d+(?:,\d{3})*(?:\.\d+)?)')
        text = pattern_naira.sub(lambda m: f"₦{self._format_amount(m.group(1))}", text)

        # Normalize USSD codes: format to *XXX# with no spaces
        ussd_pattern = re.compile(r'\*\s*(\d+(?:\*\d+)*)\s*#')
        text = ussd_pattern.sub(lambda m: f"*{m.group(1)}#", text)

        # Remove excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        return text.strip()

    def _table_to_markdown(self, table: List[List[Optional[str]]]) -> str:
        """Serializes a extracted table grid into a Markdown table string."""
        if not table or not table[0]:
            return ""

        # Clean cell values
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                val = str(cell or "").strip().replace("\n", " ").replace("|", "\\|")
                cleaned_row.append(val)
            cleaned_table.append(cleaned_row)

        cols = len(cleaned_table[0])
        headers = cleaned_table[0]
        separators = ["---"] * cols

        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separators) + " |"
        ]

        for row in cleaned_table[1:]:
            md_lines.append("| " + " | ".join(row) + " |")

        return "\n".join(md_lines)

    async def clean(self, doc: RawDocument) -> CleanedDocument:
        """Dispatches cleaning logic depending on document content_type."""
        clean_text = ""
        title = None
        extracted_tables: List[str] = []

        if doc.content_type == "html":
            raw_html = doc.raw_html or ""
            # 1. Try Trafilatura main content extraction
            try:
                extracted = trafilatura.extract(raw_html)
                if extracted:
                    clean_text = extracted
            except Exception as e:
                logger.warning(f"Trafilatura failed extraction for {doc.url}: {e}")

            # 2. Fallback to BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")

            # Extract Title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            if not clean_text:
                # Strip blocklisted elements
                for selector in CSS_BLOCKLIST:
                    for element in soup.select(selector):
                        element.decompose()
                clean_text = soup.get_text(separator="\n")

        elif doc.content_type == "pdf":
            pdf_bytes = doc.pdf_bytes or b""
            text_pages = []
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    # Attempt to extract metadata title
                    title = pdf.metadata.get("Title")

                    for index, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(page_text)

                        # Extract tables
                        tables = page.extract_tables()
                        for table in tables:
                            md_table = self._class_table_md = self._table_to_markdown(table)
                            if md_table:
                                extracted_tables.append(md_table)

                clean_text = "\n\n".join(text_pages)
                # Fallback title
                if not title and text_pages:
                    first_line = text_pages[0].split("\n")[0].strip()
                    title = first_line[:100] if first_line else None
            except Exception as e:
                logger.error(f"pdfplumber extraction failed for {doc.url}: {e}", exc_info=True)

        else:
            # text content_type
            clean_text = doc.raw_text or ""
            title = clean_text.split("\n")[0][:100] if clean_text else None

        # Normalize text patterns (currencies, USSDs, spaces)
        clean_text = self.normalize_text(clean_text)

        # Detect language
        try:
            language_detected = detect(clean_text)
        except Exception:
            language_detected = "en"

        word_count = len(clean_text.split())

        return CleanedDocument(
            url=doc.url,
            category=doc.category,
            sub_category=doc.sub_category,
            scraped_at=doc.scraped_at,
            institution_slug=doc.institution_slug,
            http_status=doc.http_status,
            content_type=doc.content_type,
            clean_text=clean_text,
            title=title or "Untitled Document",
            extracted_tables=extracted_tables,
            word_count=word_count,
            language_detected=language_detected,
        )
