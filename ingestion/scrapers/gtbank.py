import logging
from typing import List
from pathlib import Path
from ingestion.scrapers.base_scraper import BaseScraper, RawDocument
from ingestion.processors.json_reader import load_bank_data
from registry.institutions import get_institution

logger = logging.getLogger("ingestion.scrapers.gtbank")


class GTBankScraper(BaseScraper):
    """Scraper implementation for Guaranty Trust Bank.
    
    Supports two modes:
    1. JSON mode: Reads pre-collected data from data/gtbank/ directory
    2. Live mode: Scrapes live from GTBank website
    
    Defaults to JSON mode if data exists, falls back to live scraping.
    """

    slug = "gtbank"

    async def scrape(self) -> List[RawDocument]:
        """Load GTBank data from pre-collected JSON files only.
        
        No live scraping - data must be collected via scripts/deep_crawl_gtbank.py
        """
        json_data = load_bank_data(self.slug)
        
        if json_data:
            logger.info(f"✓ Loaded {len(json_data)} documents from JSON for {self.slug}")
            return json_data
        
        logger.error(f"❌ No JSON data found for {self.slug}")
        logger.error(f"   Please run: python scripts/deep_crawl_gtbank.py")
        return []
