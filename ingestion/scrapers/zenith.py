import logging
from typing import List
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument
from ingestion.processors.json_reader import load_bank_data
from registry.institutions import get_institution

logger = logging.getLogger("ingestion.scrapers.zenith")


class ZenithScraper(BaseScraper):
    """Scraper implementation for Zenith Bank.
    
    Supports two modes:
    1. JSON mode: Reads pre-collected data from data/zenith/ directory
    2. Live mode: Scrapes live from Zenith website
    
    Defaults to JSON mode if data exists, falls back to live scraping.
    """

    slug = "zenith"

    async def scrape(self) -> List[RawDocument]:
        """Load Zenith data from pre-collected JSON files only.
        
        No live scraping - data must be collected via scripts/collect_bank_data.py
        """
        json_data = load_bank_data(self.slug)
        
        if json_data:
            logger.info(f"✓ Loaded {len(json_data)} documents from JSON for {self.slug}")
            return json_data
        
        logger.error(f"❌ No JSON data found for {self.slug}")
        logger.error(f"   Please run: python scripts/collect_bank_data.py zenith")
        return []
