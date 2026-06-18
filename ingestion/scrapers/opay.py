import logging
from typing import List
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument
from registry.institutions import get_institution

logger = logging.getLogger("ingestion.scrapers.opay")


class OPayScraper(BaseScraper):
    """Scraper implementation for OPay Digital Services."""

    slug = "opay"

    async def scrape(self) -> List[RawDocument]:
        results = []
        try:
            inst = get_institution(self.slug)
        except ValueError as e:
            logger.error(f"Registry entry not found for {self.slug}: {e}")
            return results

        for target in inst.scrape_targets:
            try:
                if target.url.lower().endswith(".pdf"):
                    pdf_bytes, status = await self.fetch_pdf(target.url)
                    doc = RawDocument(
                        url=target.url,
                        pdf_bytes=pdf_bytes if status == 200 else None,
                        category=target.category,
                        institution_slug=self.slug,
                        http_status=status,
                        content_type="pdf",
                    )
                else:
                    html, status = await self.fetch_html(
                        target.url, use_playwright=target.requires_js
                    )
                    doc = RawDocument(
                        url=target.url,
                        raw_html=html if status == 200 else None,
                        category=target.category,
                        institution_slug=self.slug,
                        http_status=status,
                        content_type="html",
                    )
                results.append(doc)
            except Exception as e:
                logger.error(f"Failed to scrape {target.url} for OPay: {e}", exc_info=True)

        return results
