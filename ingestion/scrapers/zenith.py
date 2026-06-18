import logging
from typing import List
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument
from registry.institutions import get_institution

logger = logging.getLogger("ingestion.scrapers.zenith")


class ZenithScraper(BaseScraper):
    """Scraper implementation for Zenith Bank."""

    slug = "zenith"

    async def scrape(self) -> List[RawDocument]:
        results = []
        try:
            inst = get_institution(self.slug)
        except ValueError as e:
            logger.error(f"Registry entry not found for {self.slug}: {e}")
            return results

        # 1. Scrape targets with crawling
        for target in inst.scrape_targets:
            try:
                docs = await self.scrape_target_with_crawl(target)
                results.extend(docs)
            except Exception as e:
                logger.error(f"Failed to scrape {target.url} for Zenith: {e}", exc_info=True)

        # 2. Scrape news articles
        try:
            news_docs = await self.fetch_news_articles(inst.name, limit=3)
            results.extend(news_docs)
        except Exception as e:
            logger.error(f"Failed to fetch news for Zenith: {e}", exc_info=True)

        return results
