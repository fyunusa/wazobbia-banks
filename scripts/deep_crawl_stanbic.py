"""Specialized Deep Crawler for Stanbic IBTC"""
import asyncio, json, logging, aiohttp
from pathlib import Path
from datetime import datetime
from typing import Set
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deep_crawl_stanbic")

BANK_HOME = "https://www.stanbicibtc.com/"
DATA_DIR = Path("data/stanbic")
FILES_DIR = DATA_DIR / "files"
TIMEOUT = 60000

FILES_DIR.mkdir(parents=True, exist_ok=True)

class StanbicDeepCrawler:
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.page_queue = [BANK_HOME]
        self.page_count = 0
    
    async def crawl(self):
        from playwright.async_api import async_playwright
        logger.info(f"🚀 Starting deep crawl of STANBIC IBTC")
        logger.info(f"🔗 Homepage: {BANK_HOME}")
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)        
        # Load previously crawled URLs to resume from where we left off
        self._load_existing_visited_urls()
                async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page_count = self.page_count  # Start from previously loaded count
            
            while self.page_queue:
                url = self.page_queue.pop(0)
                if url in self.visited_urls:
                    continue
                
                page_count += 1
                logger.info(f"\n📄 [{page_count}] {url}")
                self.visited_urls.add(url)
                
                try:
                    context = await browser.new_context(viewport={"width": 1920, "height": 1080}, locale="en-NG", timezone_id="Africa/Lagos")
                    page = await context.new_page()
                    
                    try:
                        await page.goto(url, wait_until="load", timeout=TIMEOUT)
                    except:
                        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
                    
                    html = await page.content()
                    page_data = {"url": url, "html": html, "status": 200}
                    self._save_page(page_data)
                    logger.info(f"   ✓ {len(html)} bytes")
                    
                    links = await page.evaluate("""() => {const urls = new Set(); document.querySelectorAll('a[href]').forEach(a => {const h = a.getAttribute('href'); if(h && !h.startsWith('#')) urls.add(h);}); return Array.from(urls);}""")
                    for link in links:
                        try:
                            abs_url = urljoin(url, link)
                            if urlparse(abs_url).netloc == urlparse(BANK_HOME).netloc and abs_url not in self.visited_urls and abs_url not in self.page_queue:
                                self.page_queue.append(abs_url)
                        except:
                            pass
                    
                    await context.close()
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"   ✗ {type(e).__name__}")
            
            await browser.close()
        
        with open(DATA_DIR / "index.json", "w") as f:
            json.dump({"bank": "Stanbic IBTC", "total_pages": self.page_count}, f)
        logger.info(f"\n✅ Complete! Pages: {self.page_count}")
    
    def _load_existing_visited_urls(self):
        """Load all previously crawled URLs from pages.json to resume crawling"""
        output_file = DATA_DIR / "pages.json"
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    pages = json.load(f)
                    for page in pages:
                        if "url" in page:
                            self.visited_urls.add(page["url"])
                            self.page_count += 1
                    
                    if self.page_count > 0:
                        logger.info(f"📂 Loaded {self.page_count} previously crawled URLs from cache")
                        logger.info(f"📊 Resuming from page {self.page_count + 1}...")
            except Exception as e:
                logger.warning(f"Failed to load {output_file}: {e}")
    
    def _save_page(self, page_data):
        """Incrementally save each page to JSON"""
        output_file = DATA_DIR / "pages.json"
        pages = []
        if output_file.exists():
            with open(output_file, "r") as f:
                pages = json.load(f)
        pages.append(page_data)
        with open(output_file, "w") as f:
            json.dump(pages, f)
        self.page_count += 1

async def main():
    await StanbicDeepCrawler().crawl()

if __name__ == "__main__":
    asyncio.run(main())
