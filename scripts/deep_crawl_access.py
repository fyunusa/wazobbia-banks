"""
Specialized Deep Crawler for Access Bank - Handles slow-loading pages

Access Bank's website has many slow-loading pages that need extended timeouts
and special handling. This crawler is optimized for their infrastructure.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any
from urllib.parse import urljoin, urlparse
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deep_crawl_access")

# Configuration
BANK_HOME = "https://www.accessbankplc.com"
DATA_DIR = Path("data/access")
TIMEOUT = 180000  # 180 seconds (3 minutes) for very slow homepage
SAME_ORIGIN_ONLY = True

# Browser headers for Access Bank
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# URL categorization patterns
CATEGORIES = {
    "personal": ["personal", "savings", "accounts", "card", "credit"],
    "business": ["business", "sme", "commercial", "emerging"],
    "corporate": ["corporate", "investor", "about", "governance", "award", "history", "profile"],
    "investment": ["investment", "financial", "shareholder", "investor"],
    "support": ["help", "faq", "support", "contact", "customer-support", "whistle"],
    "digital": ["digital", "app", "online", "mobile", "access-more"],
    "careers": ["careers", "recruitment", "job", "working"],
}


class AccessBankDeepCrawler:
    """Specialized crawler for Access Bank with extended timeouts."""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.collected_pages: List[Dict[str, Any]] = []
        self.page_queue: List[str] = [BANK_HOME]
        self.sections: Dict[str, List[Dict]] = {}
        self.failed_urls = {"timeout": [], "error": [], "http_error": []}
    
    async def crawl(self):
        """Start the crawling process."""
        from playwright.async_api import async_playwright
        
        logger.info(f"🚀 Starting deep crawl of ACCESS BANK (Specialized)")
        logger.info(f"🔗 Homepage: {BANK_HOME}")
        logger.info(f"📁 Output: {DATA_DIR}")
        logger.info(f"⏱️  Extended timeout: {TIMEOUT}ms (180 seconds) - domcontentloaded strategy")
        
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
                    context = await browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        locale="en-NG",
                        timezone_id="Africa/Lagos",
                    )
                    page = await context.new_page()
                    await page.set_extra_http_headers(BROWSER_HEADERS)
                    
                    # Navigate with extended timeout and retries
                    response = None
                    try:
                        response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
                        # Extra wait for dynamic content
                        await asyncio.sleep(2)
                    except asyncio.TimeoutError:
                        logger.warning(f"   ⏱️  Timeout on domcontentloaded, trying with no wait...")
                        try:
                            response = await page.goto(url, wait_until="commit", timeout=TIMEOUT)
                            await asyncio.sleep(3)
                        except asyncio.TimeoutError:
                            logger.error(f"   ✗ Still timeout after retry, skipping")
                            self.failed_urls["timeout"].append(url)
                            await context.close()
                            await asyncio.sleep(2)
                            continue
                    
                    # Check response status
                    if response and response.status == 200:
                        # Capture page data
                        page_data = await self._capture_page_data(page, url)
                        if page_data:
                            self.collected_pages.append(page_data)
                            logger.info(f"   ✓ Captured: {len(page_data['html'])} bytes")
                        
                        # Find new links to crawl
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
                        logger.error(f"   ✗ HTTP {status}")
                        self.failed_urls["http_error"].append((url, status))
                    
                    await context.close()
                    
                    # Staggered delay between requests
                    await asyncio.sleep(1)
                
                except Exception as e:
                    logger.error(f"   ✗ Unexpected error: {type(e).__name__}: {str(e)[:80]}")
                    self.failed_urls["error"].append(url)
            
            await browser.close()
        
        # Save collected data
        self._save_to_json()
        self._print_summary()
    
    async def _capture_page_data(self, page, url: str) -> Dict[str, Any]:
        """Capture HTML, text, and metadata from a page."""
        try:
            html = await page.content()
            
            # Extract text
            try:
                text = await page.evaluate("document.body.innerText || ''")
            except:
                text = ""
            
            # Extract title
            try:
                title = await page.evaluate("document.title || ''")
            except:
                title = ""
            
            # Extract first heading
            try:
                heading = await page.evaluate("document.querySelector('h1')?.innerText || ''")
            except:
                heading = ""
            
            # Categorize URL
            category = self._categorize_url(url)
            
            return {
                "url": url,
                "title": title,
                "heading": heading,
                "category": category,
                "html": html,
                "text": text,
                "status": 200,
                "timestamp": datetime.utcnow().isoformat(),
                "content_type": "text/html"
            }
        
        except Exception as e:
            logger.error(f"Failed to capture page: {e}")
            return None
    
    async def _extract_links(self, page, base_url: str) -> List[str]:
        """Extract all clickable links from the page."""
        try:
            links = await page.evaluate("""
                () => {
                    const urls = new Set();
                    
                    // Get <a> links
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href && !href.startsWith('#') && !href.startsWith('javascript')) {
                            urls.add(href);
                        }
                    });
                    
                    // Get button onclick links
                    document.querySelectorAll('button[onclick]').forEach(btn => {
                        const onclick = btn.getAttribute('onclick');
                        if (onclick && onclick.includes('window.location')) {
                            const match = onclick.match(/'([^']+)'|"([^"]+)"/);
                            if (match) urls.add(match[1] || match[2]);
                        }
                    });
                    
                    return Array.from(urls);
                }
            """)
            
            # Convert relative URLs to absolute
            absolute_urls = []
            for link in links:
                try:
                    absolute_url = urljoin(base_url, link)
                    
                    # Filter: same origin only
                    if SAME_ORIGIN_ONLY:
                        parsed = urlparse(absolute_url)
                        bank_domain = urlparse(BANK_HOME).netloc
                        if parsed.netloc != bank_domain:
                            continue
                    
                    # Filter: no common excludes
                    if not self._should_skip_url(absolute_url):
                        absolute_urls.append(absolute_url)
                except:
                    continue
            
            return absolute_urls
        
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
            return []
    
    def _categorize_url(self, url: str) -> str:
        """Categorize URL using keyword patterns."""
        url_lower = url.lower()
        
        for category, keywords in CATEGORIES.items():
            for keyword in keywords:
                if keyword in url_lower:
                    return category
        
        return "other"
    
    def _should_skip_url(self, url: str) -> bool:
        """Determine if URL should be skipped."""
        skip_patterns = [
            "/login", "/admin", "/api/", "/static/", "/assets/", "/cdn",
            ".pdf", ".jpg", ".png", ".css", ".js", ".zip", ".exe",
            "javascript:", "mailto:", "tel:", "/search", "/signup",
            "/register", "/download", ".mp4", ".mov", ".avi"
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
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
    
    def _save_to_json(self):
        """Save collected pages grouped by category to JSON files."""
        # Group by category
        by_category = {}
        for page in self.collected_pages:
            category = page["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(page)
        
        # Save each category to a file
        for category, pages in by_category.items():
            filename = DATA_DIR / f"{category.lower()}.json"
            with open(filename, "w") as f:
                json.dump(pages, f, indent=2)
            logger.info(f"   💾 Saved: {filename} ({len(pages)} pages)")
        
        # Create index
        index = {
            "bank": "Access Bank",
            "homepage": BANK_HOME,
            "crawl_time": datetime.utcnow().isoformat(),
            "total_pages": len(self.collected_pages),
            "categories": {cat: len(pages) for cat, pages in by_category.items()},
            "failed_urls": {
                "timeout": len(self.failed_urls["timeout"]),
                "error": len(self.failed_urls["error"]),
                "http_error": len(self.failed_urls["http_error"])
            }
        }
        
        index_file = DATA_DIR / "index.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"   📑 Saved index: {index_file}")
    
    def _print_summary(self):
        """Print crawling summary."""
        total_pages = len(self.collected_pages)
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
    crawler = AccessBankDeepCrawler()
    await crawler.crawl()


if __name__ == "__main__":
    asyncio.run(main())
