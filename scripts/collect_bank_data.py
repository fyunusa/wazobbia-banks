"""
Data Collection Script - Uses Playwright to scrape real bank data
and stores it as JSON files organized by bank directory.

Usage:
  python scripts/collect_bank_data.py gtbank    # Collect GTBank data
  python scripts/collect_bank_data.py all       # Collect all banks
  python scripts/collect_bank_data.py --list    # List available banks
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.scraper_configs import load_scraper_config, list_all_configs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("collect_bank_data")

# Data directory structure
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class PlaywrightDataCollector:
    """Collects bank data using Playwright and stores as JSON."""
    
    def __init__(self, bank_slug: str):
        self.bank_slug = bank_slug
        self.config = load_scraper_config(bank_slug)
        self.bank_data_dir = DATA_DIR / bank_slug
        self.bank_data_dir.mkdir(exist_ok=True)
        self.collected_data: List[Dict[str, Any]] = []
        
    async def collect(self):
        """Main collection method - fetches all targets and stores as JSON."""
        from playwright.async_api import async_playwright
        import asyncio
        
        logger.info(f"🚀 Starting data collection for {self.bank_slug}")
        logger.info(f"📋 Found {len(self.config['scrape_targets'])} targets")
        
        async with async_playwright() as p:
            # Launch with visible browser for better success rate
            logger.info("🌐 Launching browser...")
            browser = await p.chromium.launch(
                headless=False,  # Show browser window
                slow_mo=500  # Slow down interactions for stability
            )
            
            for i, target in enumerate(self.config['scrape_targets'], 1):
                logger.info(f"\n[{i}/{len(self.config['scrape_targets'])}] Processing target...")
                await self._collect_target(browser, target)
                # Add delay between requests to avoid rate limiting
                await asyncio.sleep(2)
            
            logger.info("\n✓ Closing browser...")
            await browser.close()
        
        # Save to JSON
        self._save_to_json()
        
        logger.info(f"\n✅ Data collection complete for {self.bank_slug}")
        logger.info(f"   📊 Collected: {len(self.collected_data)} documents")
        logger.info(f"   💾 Saved to: {self.bank_data_dir}/")
    
    async def _collect_target(self, browser, target):
        """Collect data for a single target URL with retries."""
        url = target['url']
        category = target.get('category', 'OTHER')
        
        logger.info(f"   URL: {url}")
        logger.info(f"   Category: {category}")
        
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                page = await browser.new_page(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                
                try:
                    # Try different wait conditions
                    logger.info(f"   Attempt {attempt}/{max_retries}: Loading page...")
                    response = await page.goto(
                        url,
                        wait_until="domcontentloaded",  # Faster than networkidle
                        timeout=45000  # 45 seconds
                    )
                    
                    # Wait a bit for dynamic content
                    import asyncio
                    await asyncio.sleep(2)
                    
                    if response:
                        status = response.status
                        logger.info(f"   Response status: {status}")
                        
                        if status == 200:
                            # Get full HTML
                            content = await page.content()
                            
                            # Get text content
                            try:
                                text = await page.evaluate(
                                    """() => {
                                        return document.body.innerText || document.documentElement.innerText;
                                    }"""
                                )
                            except:
                                text = ""
                            
                            html_size = len(content)
                            text_size = len(text)
                            
                            # Only save if we got meaningful content
                            if html_size > 500:  # Reasonable minimum for real content
                                self.collected_data.append({
                                    "url": url,
                                    "category": category,
                                    "status": 200,
                                    "content_type": "text/html",
                                    "html": content,
                                    "text": text,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                logger.info(f"   ✅ SUCCESS: {html_size:,} bytes HTML, {text_size:,} bytes text")
                                return
                            else:
                                logger.warning(f"   ⚠️  Content too small ({html_size} bytes), might be error page")
                        else:
                            logger.warning(f"   ⚠️  HTTP {status} - {response.status_text}")
                    else:
                        logger.warning(f"   ⚠️  No response object")
                
                except asyncio.TimeoutError:
                    logger.warning(f"   ⏱️  Timeout on attempt {attempt}")
                except Exception as page_error:
                    logger.warning(f"   ❌ Page error: {type(page_error).__name__}: {str(page_error)[:80]}")
                
                finally:
                    try:
                        await page.close()
                    except:
                        pass
                
                # Wait before retry
                if attempt < max_retries:
                    import asyncio
                    wait_time = 5 * attempt  # Exponential backoff: 5s, 10s, 15s
                    logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"   ❌ Browser error: {e}")
        
        logger.warning(f"   ❌ FAILED after {max_retries} attempts: {url}")
    
    def _save_to_json(self):
        """Save collected data to JSON files (one per category)."""
        by_category: Dict[str, List[Dict]] = {}
        
        for doc in self.collected_data:
            category = doc['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(doc)
        
        # Save each category as a separate JSON file
        for category, docs in by_category.items():
            filename = self.bank_data_dir / f"{category.lower()}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(docs, f, indent=2, ensure_ascii=False)
            logger.info(f"   Saved: {filename} ({len(docs)} documents)")
        
        # Also save index file with summary
        index = {
            "bank": self.bank_slug,
            "timestamp": datetime.utcnow().isoformat(),
            "total_documents": len(self.collected_data),
            "categories": {
                cat: len(docs) 
                for cat, docs in by_category.items()
            },
            "files": {
                cat: f"{category.lower()}.json"
                for cat in by_category.keys()
            }
        }
        
        index_file = self.bank_data_dir / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        logger.info(f"   Saved: {index_file}")


def print_bank_list():
    """Print available bank configs."""
    configs = list_all_configs()
    print("\nAvailable bank configurations:")
    for config in configs:
        bank_slug = config.replace('.json', '')
        print(f"  • {bank_slug}")
    print()


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    
    if command == "--list":
        print_bank_list()
        return
    
    if command == "all":
        configs = list_all_configs()
        banks = [cfg.replace('.json', '') for cfg in configs]
    else:
        banks = [command]
    
    for bank in banks:
        collector = PlaywrightDataCollector(bank)
        await collector.collect()
        print()


if __name__ == "__main__":
    asyncio.run(main())
