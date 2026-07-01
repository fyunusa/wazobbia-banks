#!/usr/bin/env python3
"""
Local scraper debugging script - test one bank at a time.

Usage:
    python scripts/debug_scraper.py gtbank     # Debug GTBank
    python scripts/debug_scraper.py zenith     # Debug Zenith
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.scrapers.gtbank import GTBankScraper
from ingestion.scrapers.zenith import ZenithScraper
from ingestion.scrapers.opay import OPayScraper
from ingestion.scrapers.kuda import KudaScraper
from config.scraper_configs import load_scraper_config, get_keywords_for_category

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_scraper")

SCRAPERS = {
    "gtbank": GTBankScraper,
    "zenith": ZenithScraper,
    "opay": OPayScraper,
    "kuda": KudaScraper,
}


async def debug_scraper(slug: str):
    """Run scraper in debug mode with verbose logging."""
    if slug not in SCRAPERS:
        print(f"❌ Unknown scraper: {slug}")
        print(f"Available: {', '.join(SCRAPERS.keys())}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 Debugging {slug.upper()} Scraper")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_scraper_config(slug)
    print(f"📋 Config loaded: {len(config['scrape_targets'])} targets")
    print(f"   Max sub-links: {config.get('max_sub_links', 50)}")
    print(f"   Rate limit: {config['rate_limit']}\n")
    
    # Run scraper
    scraper = SCRAPERS[slug]()
    
    try:
        logger.info(f"Starting scrape for {slug}...")
        results = await scraper.scrape()
        
        print(f"\n{'='*60}")
        print(f"✅ Scraping Complete for {slug.upper()}")
        print(f"{'='*60}")
        print(f"Total documents: {len(results)}")
        
        # Group by category
        by_category = {}
        for doc in results:
            cat = doc.category
            by_category.setdefault(cat, []).append(doc)
        
        for category, docs in sorted(by_category.items()):
            print(f"\n📚 {category.upper()}: {len(docs)} documents")
            for i, doc in enumerate(docs[:10], 1):
                url = doc.url[:70] + "..." if len(doc.url) > 70 else doc.url
                html_size = len(doc.raw_html or "") if doc.raw_html else 0
                text_size = len(doc.raw_text or "") if doc.raw_text else 0
                size_str = f"{html_size} HTML" if html_size else f"{text_size} text"
                print(f"   {i}. {url}")
                print(f"      → {size_str} bytes, status: {doc.http_status}")
            if len(docs) > 10:
                print(f"   ... and {len(docs) - 10} more")
        
        # Summary
        html_total = sum(len(d.raw_html or '') for d in results)
        text_total = sum(len(d.raw_text or '') for d in results)
        total_bytes = html_total + text_total
        print(f"\n📊 Summary:")
        print(f"   Total URLs scraped: {len(results)}")
        print(f"   Categories found: {len(by_category)}")
        print(f"   Total HTML bytes: {html_total:,}")
        print(f"   Total text bytes: {text_total:,}")
        print(f"   Combined size: {total_bytes:,} bytes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Scraper error:")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_scraper.py <bank_slug>")
        print(f"Available: {', '.join(SCRAPERS.keys())}")
        sys.exit(1)
    
    slug = sys.argv[1].lower()
    asyncio.run(debug_scraper(slug))
