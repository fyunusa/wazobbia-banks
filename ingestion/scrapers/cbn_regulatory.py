import logging
import urllib.parse
from typing import List
from bs4 import BeautifulSoup
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument

logger = logging.getLogger("ingestion.scrapers.cbn_regulatory")

CBN_CIRCULARS_URL = "https://www.cbn.gov.ng/documents/circulars.html"
FEE_KEYWORDS = {"tariff", "fee", "charge", "guide", "pricing", "penalty", "circular", "directive"}


class CBNRegulatoryScraper(BaseScraper):
    """Scrapes CBN public circulars and pulls banking tariff-related PDFs."""

    slug = "cbn"

    async def scrape(self) -> List[RawDocument]:
        results = []
        logger.info(f"Starting CBN circulars scrape from {CBN_CIRCULARS_URL}")

        # Fetch circulars landing page (trying regular httpx first)
        html_content, status = await self.fetch_html(CBN_CIRCULARS_URL, use_playwright=False)
        if status != 200 or not html_content:
            logger.error(f"Failed to fetch CBN circulars index page, status: {status}")
            return results

        # Parse links using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        pdf_links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            text = anchor.get_text(strip=True).lower()

            if href.lower().endswith(".pdf"):
                # Check if the circular is tariff/fee/charge related
                href_lower = href.lower()
                is_relevant = any(kw in text for kw in FEE_KEYWORDS) or any(
                    kw in href_lower for kw in FEE_KEYWORDS
                )

                if is_relevant:
                    absolute_url = urllib.parse.urljoin(CBN_CIRCULARS_URL, href)
                    pdf_links.append((absolute_url, anchor.get_text(strip=True)))

        # Deduplicate links
        pdf_links = list(set(pdf_links))
        logger.info(f"Found {len(pdf_links)} relevant PDF circular links on CBN circulars page.")

        # Limit to top 5 most recent to prevent abuse during local test runs
        for pdf_url, title in pdf_links[:5]:
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
