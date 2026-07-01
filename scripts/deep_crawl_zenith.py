"""
Deep Crawler for Zenith Bank - Handles anti-bot protection with sophisticated headers.

This crawler:
1. Uses realistic browser headers to avoid detection
2. Adds realistic delays and behavior patterns
3. Navigates all pages and interactive elements
4. Captures content from each page
5. Stores all content in JSON organized by section

Usage:
  python scripts/deep_crawl_zenith.py
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("deep_crawl_zenith")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data" / "zenith"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Configuration
ZENITH_HOME = "https://www.zenithbank.com"
TIMEOUT = 30000
SAME_ORIGIN_ONLY = True

# Anti-detection headers
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


class ZenithDeepCrawler:
    """Deep crawler specialized for Zenith Bank with anti-detection."""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.collected_pages: List[Dict[str, Any]] = []
        self.page_queue: List[str] = [ZENITH_HOME]
        self.failed_urls = {"timeout": [], "error": [], "http_error": []}
    
    async def crawl(self):
        """Start the crawling process."""
        from playwright.async_api import async_playwright
        
        logger.info(f"🚀 Starting deep crawl of ZENITH BANK")
        logger.info(f"🔗 Homepage: {ZENITH_HOME}")
        logger.info(f"📁 Output: {DATA_DIR}")
        
        # Load previously crawled URLs to resume from where we left off
        self._load_existing_visited_urls()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
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
                        locale="en-US",
                        timezone_id="America/New_York",
                        java_script_enabled=True,
                    )
                    page = await context.new_page()
                    
                    # Masquerade as real browser
                    await page.add_init_script("""
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => false,
                        });
                    """)
                    
                    # Set realistic headers
                    await page.set_extra_http_headers(BROWSER_HEADERS)
                    
                    # Navigate with anti-detection measures
                    try:
                        response = await page.goto(
                            url,
                            wait_until="load",  # Wait for full page load, not just network idle
                            timeout=TIMEOUT
                        )
                        
                        # Wait extra time for Imperva/bot protection to complete
                        await asyncio.sleep(3)
                        
                        # Scroll down to trigger lazy loading
                        await page.evaluate("window.scrollBy(0, window.innerHeight)")
                        await asyncio.sleep(1)
                        
                    except asyncio.TimeoutError:
                        logger.error(f"   ✗ Error: Timeout 30000ms exceeded")
                        self.failed_urls["timeout"].append(url)
                        await context.close()
                        await asyncio.sleep(2)  # Longer delay after timeout
                        continue
                    except Exception as nav_error:
                        logger.error(f"   ✗ Error: {type(nav_error).__name__}: {str(nav_error)[:80]}")
                        self.failed_urls["error"].append(url)
                        await context.close()
                        await asyncio.sleep(1)
                        continue
                    
                    # Check response status
                    if response:
                        status = response.status
                        if status == 200:
                            # Capture page data
                            page_data = await self._capture_page_data(page, url)
                            if page_data:
                                title = page_data.get('title', '')
                                logger.info(f"   ✓ Captured: {len(page_data['html'])} bytes | Title: {title[:50]}")
                                self.collected_pages.append(page_data)
                            
                            # Find new links
                            new_urls = await self._extract_links(page, url)
                            logger.info(f"   🔗 Found {len(new_urls)} links on page")
                            queued_count = 0
                            for new_url in new_urls:
                                if new_url not in self.visited_urls and new_url not in self.page_queue:
                                    self.page_queue.append(new_url)
                                    queued_count += 1
                                    if queued_count <= 5:  # Show first 5 queued links
                                        logger.info(f"   → Queued: {new_url}")
                            if queued_count > 5:
                                logger.info(f"   → ... and {queued_count - 5} more links")
                        else:
                            logger.error(f"   ✗ HTTP {status}")
                            self.failed_urls["http_error"].append((url, status))
                    else:
                        logger.error(f"   ✗ No response")
                        self.failed_urls["error"].append(url)
                    
                    await context.close()
                    
                    # Random delay to avoid detection (2-5 seconds)
                    delay = 2 + (hash(url) % 30) / 10
                    await asyncio.sleep(delay)
                
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
            
            # Extract text with better parsing
            try:
                text = await page.evaluate("""
                    () => {
                        const body = document.body;
                        return body ? body.innerText : '';
                    }
                """)
            except:
                text = ""
            
            # Extract title
            try:
                title = await page.title()
            except:
                title = ""
            
            # Extract main heading
            try:
                heading = await page.evaluate("""
                    () => {
                        const h1 = document.querySelector('h1');
                        const h2 = document.querySelector('h2');
                        return (h1 || h2)?.innerText || '';
                    }
                """)
            except:
                heading = ""
            
            # Extract meta description
            try:
                meta_desc = await page.evaluate("""
                    () => {
                        const meta = document.querySelector('meta[name="description"]');
                        return meta ? meta.getAttribute('content') : '';
                    }
                """)
            except:
                meta_desc = ""
            
            category = self._categorize_url(url)
            
            return {
                "url": url,
                "title": title,
                "heading": heading,
                "description": meta_desc,
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
                    
                    // Get all links from <a> tags
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href && 
                            !href.startsWith('#') && 
                            !href.startsWith('javascript') &&
                            !href.startsWith('mailto') &&
                            !href.startsWith('tel') &&
                            href.trim().length > 0) {
                            urls.add(href);
                        }
                    });
                    
                    // Get links from onclick handlers
                    document.querySelectorAll('[onclick*="location"]').forEach(el => {
                        const onclick = el.getAttribute('onclick');
                        if (onclick) {
                            const matches = onclick.match(/'([^']+)'|"([^"]+)"/g);
                            if (matches) {
                                matches.forEach(match => {
                                    const url = match.replace(/['\"]/g, '');
                                    if (!url.startsWith('javascript') && url.includes('/')) {
                                        urls.add(url);
                                    }
                                });
                            }
                        }
                    });
                    
                    // Get data-href attributes (some SPAs use this)
                    document.querySelectorAll('[data-href]').forEach(el => {
                        const href = el.getAttribute('data-href');
                        if (href && href.startsWith('/')) {
                            urls.add(href);
                        }
                    });
                    
                    return Array.from(urls);
                }
            """)
            
            # Convert relative to absolute and filter
            absolute_urls = []
            for link in links:
                try:
                    absolute_url = urljoin(base_url, link)
                    
                    # Same origin check
                    if SAME_ORIGIN_ONLY:
                        parsed = urlparse(absolute_url)
                        zenith_domain = urlparse(ZENITH_HOME).netloc
                        if parsed.netloc != zenith_domain:
                            continue
                    
                    # Skip bad URLs
                    if not self._should_skip_url(absolute_url):
                        absolute_urls.append(absolute_url)
                except:
                    continue
            
            return absolute_urls
        
        except Exception as e:
            logger.error(f"Failed to extract links: {e}")
            return []
    
    def _categorize_url(self, url: str) -> str:
        """Categorize URL by path pattern."""
        path = urlparse(url).path.lower()
        
        categories = {
            "personal": ["personal", "retail", "individuals", "savings", "accounts"],
            "business": ["business", "sme", "commercial", "enterprise"],
            "corporate": ["corporate", "investor", "about", "press", "news", "csr"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer", "contact"],
            "digital": ["digital", "mobile", "online", "app", "web"],
            "products": ["products", "services"],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in path:
                    return category.upper()
        
        return "OTHER"
    
    def _should_skip_url(self, url: str) -> bool:
        """Determine if URL should be skipped."""
        skip_patterns = [
            "/login", "/admin", "/api/", "/static/", "/assets/", "/cdn",
            ".pdf", ".jpg", ".png", ".css", ".js", ".zip", ".exe",
            "javascript:", "mailto:", "tel:", "/search", "/signup",
            "/register", "/download", ".mp4", ".mov", ".avi", ".doc",
            ".xlsx", ".pptx"
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    def _load_existing_visited_urls(self):
        """Load all previously crawled URLs from existing JSON files to resume crawling"""
        categories = ["PERSONAL", "BUSINESS", "CORPORATE", "INVESTMENT", "SUPPORT", "DIGITAL", "PRODUCTS", "OTHER"]
        total_loaded = 0
        
        for category in categories:
            filename = DATA_DIR / f"{category.lower()}.json"
            if filename.exists():
                try:
                    with open(filename, "r") as f:
                        pages = json.load(f)
                        for page in pages:
                            if "url" in page:
                                self.visited_urls.add(page["url"])
                                total_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
        
        if total_loaded > 0:
            logger.info(f"📂 Loaded {total_loaded} previously crawled URLs from cache")
            logger.info(f"📊 Resuming from page {total_loaded + 1}...")
    
    def _save_to_json(self):
        """Save collected pages grouped by category to JSON files."""
        by_category = {}
        for page in self.collected_pages:
            category = page["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(page)
        
        # Save each category
        for category, pages in by_category.items():
            filename = DATA_DIR / f"{category.lower()}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
            logger.info(f"   💾 Saved: {filename} ({len(pages)} pages)")
        
        # Save index
        index_data = {
            "bank": "zenith",
            "crawl_date": datetime.utcnow().isoformat(),
            "total_pages": len(self.collected_pages),
            "categories": {
                category: len(pages)
                for category, pages in by_category.items()
            },
            "failed": {
                "timeouts": len(self.failed_urls["timeout"]),
                "errors": len(self.failed_urls["error"]),
                "http_errors": len(self.failed_urls["http_error"])
            }
        }
        
        index_file = DATA_DIR / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"   📑 Saved index: {index_file}")
    
    def _print_summary(self):
        """Print crawl summary."""
        logger.info(f"\n✅ Crawling complete!")
        logger.info(f"   📊 Total pages: {len(self.collected_pages)}")
        logger.info(f"   📁 Saved to: {DATA_DIR}/")
        
        if self.failed_urls["timeout"]:
            logger.info(f"   ⏱️  Timeouts: {len(self.failed_urls['timeout'])}")
        if self.failed_urls["error"]:
            logger.info(f"   ❌ Errors: {len(self.failed_urls['error'])}")
        if self.failed_urls["http_error"]:
            logger.info(f"   📡 HTTP errors: {len(self.failed_urls['http_error'])}")
        
        success_rate = (len(self.collected_pages) / len(self.visited_urls) * 100) if self.visited_urls else 0
        logger.info(f"   📈 Success rate: {success_rate:.1f}%")


async def main():
    """Main entry point."""
    crawler = ZenithDeepCrawler()
    await crawler.crawl()


if __name__ == "__main__":
    asyncio.run(main())
