import logging
import urllib.parse
from typing import List
from bs4 import BeautifulSoup
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument

logger = logging.getLogger("ingestion.scrapers.cbn_regulatory")

CBN_CIRCULAR_URLS = [
    "https://www.cbn.gov.ng/documents/circulars.html",
    "https://www.cbn.gov.ng/documents/circulars.asp",
    "https://www.cbn.gov.ng/supervision/bsd_circulars.asp",
    "https://www.cbn.gov.ng/tradeexchange/circulars.asp",
    "https://www.cbn.gov.ng/ofisd/circulars.asp",
]
FEE_KEYWORDS = {"tariff", "fee", "charge", "guide", "pricing", "penalty", "circular", "directive", "rate", "interest", "commission", "fx", "forex", "aml", "cft", "payment"}


class CBNRegulatoryScraper(BaseScraper):
    """Scrapes CBN public circulars and pulls banking tariff-related PDFs."""

    slug = "cbn"

    async def scrape(self) -> List[RawDocument]:
        results = []
        pdf_links = []

        for index_url in CBN_CIRCULAR_URLS:
            logger.info(f"Starting CBN circulars scrape from {index_url}")

            # Fetch circulars landing page
            html_content, status = await self.fetch_html(index_url, use_playwright=False)
            if status != 200 or not html_content:
                logger.warning(f"Failed to fetch CBN circulars index page {index_url}, status: {status}")
                continue

            # Parse links using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            for anchor in soup.find_all("a", href=True):
                href = anchor["href"].strip()
                text = anchor.get_text(strip=True).lower()

                if href.lower().endswith(".pdf"):
                    href_lower = href.lower()
                    
                    # 1. Relevance filter
                    is_relevant = any(kw in text for kw in FEE_KEYWORDS) or any(
                        kw in href_lower for kw in FEE_KEYWORDS
                    )
                    
                    # 2. Freshness filter (2024, 2025, or 2026)
                    is_current = any(yr in text for yr in ["2024", "2025", "2026"]) or any(
                        yr in href_lower for yr in ["/2024/", "/2025/", "/2026/", "2024", "2025", "2026"]
                    )

                    if is_relevant and is_current:
                        absolute_url = urllib.parse.urljoin(index_url, href)
                        pdf_links.append((absolute_url, anchor.get_text(strip=True)))

        # Deduplicate links
        pdf_links = list(set(pdf_links))
        logger.info(f"Found {len(pdf_links)} relevant PDF circular links on CBN circulars pages.")

        # Limit to top 15 most recent
        for pdf_url, title in pdf_links[:15]:
            try:
                pdf_bytes, pdf_status = await self.fetch_pdf(pdf_url)
                if pdf_status == 200 and pdf_bytes:
                    doc = RawDocument(
                        url=pdf_url,
                        pdf_bytes=pdf_bytes,
                        category="regulatory",
                        sub_category="circular",
                        institution_slug=self.slug,
                        http_status=pdf_status,
                        content_type="pdf",
                    )
                    results.append(doc)
                    logger.info(f"Successfully scraped CBN circular PDF: {pdf_url}")
                else:
                    logger.warning(f"Failed to download CBN circular PDF {pdf_url} - Status: {pdf_status}")
            except Exception as e:
                logger.error(f"Failed to fetch CBN PDF circular {pdf_url}: {e}", exc_info=True)

        return results
