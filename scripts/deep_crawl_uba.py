"""
Specialized Deep Crawler for UBA (Union Bank of Africa)
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deep_crawl_uba")

BANK_HOME = "https://www.ubagroup.com"
DATA_DIR = Path("data/uba")
TIMEOUT = 60000
SAME_ORIGIN_ONLY = True

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
}

CATEGORIES = {
    "personal": ["personal", "retail", "individual", "savings", "accounts"],
    "business": ["business", "sme", "commercial"],
    "corporate": ["corporate", "investor", "about", "news", "press"],
    "investment": ["investment", "financial", "shareholder"],
    "support": ["support", "faq", "help", "customer"],
    "digital": ["digital", "app", "online", "mobile"],
}

class UbaDeepCrawler:
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.page_queue: List[str] = [BANK_HOME]
        self.failed_urls = {"timeout": [], "error": [], "http_error": []}
        self.page_count = 0
        self.category_counts = {}
    
    async def crawl(self):
        from playwright.async_api import async_playwright
        
        logger.info(f"🚀 Starting deep crawl of UBA")
        logger.info(f"🔗 Homepage: {BANK_HOME}")
        logger.info(f"📁 Output: {DATA_DIR}")
        
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
                logger.info(f"\n📄 [{page_count}] Visiting: {url}")
                self.visited_urls.add(url)
                
                try:
                    context = await browser.new_context(viewport={"width": 1920, "height": 1080}, locale="en-NG", timezone_id="Africa/Lagos")
                    page = await context.new_page()
                    await page.set_extra_http_headers(BROWSER_HEADERS)
                    
                    try:
                        response = await page.goto(url, wait_until="load", timeout=TIMEOUT)
                        await asyncio.sleep(1)
                    except asyncio.TimeoutError:
                        logger.warning(f"   ⏱️  Timeout on load, trying domcontentloaded...")
                        try:
                            response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
                            await asyncio.sleep(1)
                        except asyncio.TimeoutError:
                            logger.error(f"   ✗ Still timeout, skipping")
                            self.failed_urls["timeout"].append(url)
                            await context.close()
                            await asyncio.sleep(1)
                            continue
                    
                    if response and response.status == 200:
                        page_data = await self._capture_page_data(page, url)
                        if page_data:
                            self._save_page(page_data)
                            logger.info(f"   ✓ Captured: {len(page_data['html'])} bytes")
                        
                        new_urls = await self._extract_links(page, url)
                        queued_count = 0
                        for new_url in new_urls:
                            if new_url not in self.visited_urls and new_url not in self.page_queue:
                                self.page_queue.append(new_url)
                                queued_count += 1
                        
                        if queued_count > 0:
                            logger.info(f"   🔗 Found and queued {queued_count} new links")
                    else:
                        status = response.status if response else 0
                        if status not in [404, 301, 302]:
                            logger.error(f"   ✗ HTTP {status}")
                        self.failed_urls["http_error"].append((url, status))
                    
                    await context.close()
                    await asyncio.sleep(0.5)
                
                except Exception as e:
                    logger.error(f"   ✗ Error: {type(e).__name__}: {str(e)[:80]}")
                    self.failed_urls["error"].append(url)
            
            await browser.close()
        
        self._save_to_json()
        self._print_summary()
    
    async def _capture_page_data(self, page, url: str) -> Dict[str, Any]:
        try:
            html = await page.content()
            try:
                text = await page.evaluate("document.body.innerText || ''")
            except:
                text = ""
            try:
                title = await page.evaluate("document.title || ''")
            except:
                title = ""
            try:
                heading = await page.evaluate("document.querySelector('h1')?.innerText || ''")
            except:
                heading = ""
            
            return {
                "url": url, "title": title, "heading": heading, "category": self._categorize_url(url),
                "html": html, "text": text, "status": 200, "timestamp": datetime.utcnow().isoformat(), "content_type": "text/html"
            }
        except Exception as e:
            logger.error(f"Failed to capture page: {e}")
            return None
    
    async def _extract_links(self, page, base_url: str) -> List[str]:
        try:
            links = await page.evaluate("""() => {
                const urls = new Set();
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.getAttribute('href');
                    if (href && !href.startsWith('#') && !href.startsWith('javascript')) urls.add(href);
                });
                return Array.from(urls);
            }""")
            
            absolute_urls = []
            for link in links:
                try:
                    absolute_url = urljoin(base_url, link)
                    if SAME_ORIGIN_ONLY:
                        parsed = urlparse(absolute_url)
                        bank_domain = urlparse(BANK_HOME).netloc
                        if parsed.netloc != bank_domain:
                            continue
                    if not self._should_skip_url(absolute_url):
                        absolute_urls.append(absolute_url)
                except:
                    continue
            
            return absolute_urls
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
            return []
    
    def _categorize_url(self, url: str) -> str:
        url_lower = url.lower()
        for category, keywords in CATEGORIES.items():
            for keyword in keywords:
                if keyword in url_lower:
                    return category
        return "other"
    
    def _should_skip_url(self, url: str) -> bool:
        skip_patterns = ["/login", "/admin", "/api/", "/static/", "/assets/", ".pdf", ".jpg", ".png", ".css", ".js", "javascript:", "mailto:", "tel:"]
        return any(pattern in url.lower() for pattern in skip_patterns)
    
    def _load_existing_visited_urls(self):
        """Load all previously crawled URLs from existing JSON files to resume crawling"""
        total_loaded = 0
        for category in CATEGORIES.keys():
            filename = DATA_DIR / f"{category.lower()}.json"
            if filename.exists():
                try:
                    with open(filename, "r") as f:
                        pages = json.load(f)
                        for page in pages:
                            if "url" in page:
                                self.visited_urls.add(page["url"])
                                total_loaded += 1
                                self.page_count += 1
                                if category not in self.category_counts:
                                    self.category_counts[category] = 0
                                self.category_counts[category] += 1
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        if total_loaded > 0:
            logger.info(f"📂 Loaded {total_loaded} previously crawled URLs from cache")
            logger.info(f"📊 Resuming from page {total_loaded + 1}...")
    
    def _save_page(self, page_data: Dict[str, Any]):
        """Incrementally save each page to its category JSON file"""
        category = page_data["category"]
        filename = DATA_DIR / f"{category.lower()}.json"
        pages = []
        if filename.exists():
            with open(filename, "r") as f:
                pages = json.load(f)
        pages.append(page_data)
        with open(filename, "w") as f:
            json.dump(pages, f, indent=2)
        self.page_count += 1
        if category not in self.category_counts:
            self.category_counts[category] = 0
        self.category_counts[category] += 1
    
    def _save_to_json(self):
        index = {"bank": "UBA", "homepage": BANK_HOME, "crawl_time": datetime.utcnow().isoformat(),
                 "total_pages": self.page_count, "categories": self.category_counts,
                 "failed_urls": {"timeout": len(self.failed_urls["timeout"]), "error": len(self.failed_urls["error"]), "http_error": len(self.failed_urls["http_error"])}}
        
        with open(DATA_DIR / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"   📑 Saved index: {DATA_DIR / 'index.json'}")
    
    def _print_summary(self):
        total_pages = self.page_count
        total_errors = len(self.failed_urls["timeout"]) + len(self.failed_urls["error"]) + len(self.failed_urls["http_error"])
        total_attempts = total_pages + total_errors
        success_rate = (total_pages / total_attempts * 100) if total_attempts > 0 else 0
        
        logger.info("")
        logger.info("✅ Crawling complete!")
        logger.info(f"   📊 Total pages: {total_pages}")
        logger.info(f"   📁 Saved to: {DATA_DIR}/")
        if self.failed_urls["timeout"]:
            logger.info(f"   ⏱️  Timeouts: {len(self.failed_urls['timeout'])}")
        if self.failed_urls["error"]:
            logger.info(f"   ❌ Errors: {len(self.failed_urls['error'])}")
        logger.info(f"   📈 Success rate: {success_rate:.1f}%")

async def main():
    crawler = UbaDeepCrawler()
    await crawler.crawl()

if __name__ == "__main__":
    asyncio.run(main())
