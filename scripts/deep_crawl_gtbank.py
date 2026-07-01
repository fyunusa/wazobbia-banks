"""
Deep Crawler for GTBank - Navigates all pages and interactive elements.

This crawler:
1. Starts at GTBank homepage
2. Finds all clickable elements (links, buttons)
3. Clicks and captures content from each page
4. Continues recursively through the site
5. Stores all content in JSON organized by section

Usage:
  python scripts/deep_crawl_gtbank.py
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("deep_crawl_gtbank")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DATA_DIR = Path(__file__).parent.parent / "data" / "gtbank"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Configuration
GTBANK_HOME = "https://www.gtbank.com"
# Crawl until queue exhausted (no artificial limit)
TIMEOUT = 30000  # 30 seconds per page
SAME_ORIGIN_ONLY = True  # Only crawl GTBank domain


class DeepCrawler:
    """Deep crawler that navigates GTBank website comprehensively."""
    
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.collected_pages: List[Dict[str, Any]] = []
        self.page_queue: List[str] = [GTBANK_HOME]
        self.sections: Dict[str, List[Dict]] = {}
    
    async def crawl(self):
        """Start the crawling process."""
        from playwright.async_api import async_playwright
        
        logger.info(f"🚀 Starting deep crawl of GTBank")
        logger.info(f"🔗 Homepage: {GTBANK_HOME}")
        
        # Load previously crawled URLs to resume from where we left off
        self._load_existing_visited_urls()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, slow_mo=500)
            
            # Initialize counter from previously loaded URLs
            current_count = self.page_count
            
            while self.page_queue:
                url = self.page_queue.pop(0)
                
                if url in self.visited_urls:
                    continue
                
                current_count += 1
                logger.info(f"\n📄 [{current_count}] Visiting: {url}")
                self.visited_urls.add(url)
                
                try:
                    page = await browser.new_page(
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    )
                    
                    # Navigate to page
                    response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
                    
                    if response and response.status == 200:
                        # Capture page data
                        page_data = await self._capture_page_data(page, url)
                        if page_data:
                            self.collected_pages.append(page_data)
                            logger.info(f"   ✓ Captured: {len(page_data['html'])} bytes")
                        
                        # Find new links to crawl
                        new_urls = await self._extract_links(page, url)
                        for new_url in new_urls:
                            if new_url not in self.visited_urls and new_url not in self.page_queue:
                                self.page_queue.append(new_url)
                                logger.info(f"   → Queued: {new_url}")
                    else:
                        status = response.status if response else 0
                        logger.warning(f"   ✗ HTTP {status}")
                    
                    await page.close()
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                
                except Exception as e:
                    logger.error(f"   ✗ Error: {type(e).__name__}: {str(e)[:100]}")
            
            await browser.close()
        
        # Save collected data
        self._save_to_json()
        
        logger.info(f"\n✅ Crawling complete!")
        logger.info(f"   📊 Total pages: {len(self.collected_pages)}")
        logger.info(f"   📁 Saved to: {DATA_DIR}/")
    
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
                title = await page.evaluate("document.title")
            except:
                title = ""
            
            # Extract main heading if exists
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
            
            # Categorize by URL pattern
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
                    
                    // Get links
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.getAttribute('href');
                        if (href && !href.startsWith('#') && !href.startsWith('javascript')) {
                            urls.add(href);
                        }
                    });
                    
                    // Get button onclick links (some buttons navigate)
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
                        gtbank_domain = urlparse(GTBANK_HOME).netloc
                        if parsed.netloc != gtbank_domain:
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
        """Categorize URL by path pattern."""
        path = urlparse(url).path.lower()
        
        if "personal" in path:
            return "PERSONAL"
        elif "business" in path:
            return "BUSINESS"
        elif "corporate" in path:
            return "CORPORATE"
        elif "help" in path or "faq" in path or "support" in path:
            return "SUPPORT"
        elif "about" in path or "contact" in path:
            return "ABOUT"
        elif "invest" in path or "investment" in path:
            return "INVESTMENT"
        elif "digital" in path:
            return "DIGITAL"
        elif "services" in path:
            return "SERVICES"
        elif "products" in path:
            return "PRODUCTS"
        else:
            return "OTHER"
    
    def _should_skip_url(self, url: str) -> bool:
        """Determine if URL should be skipped."""
        skip_patterns = [
            "/login",
            "/admin",
            "/api/",
            "/static/",
            "/assets/",
            "/cdn",
            ".pdf",
            ".jpg",
            ".png",
            ".css",
            ".js",
            "javascript:",
            "mailto:",
            "tel:",
            "/search",
            "/download",
        ]
        
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in skip_patterns)
    
    def _load_existing_visited_urls(self):
        """Load all previously crawled URLs from existing JSON files to resume crawling"""
        categories = ["PERSONAL", "BUSINESS", "CORPORATE", "SUPPORT", "ABOUT", "INVESTMENT", "DIGITAL", "SERVICES", "PRODUCTS", "OTHER"]
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
        """Save collected pages to JSON organized by category."""
        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for page in self.collected_pages:
            category = page.get("category", "OTHER")
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
        index = {
            "bank": "gtbank",
            "timestamp": datetime.utcnow().isoformat(),
            "total_pages": len(self.collected_pages),
            "categories": {cat: len(pages) for cat, pages in by_category.items()},
            "files": {cat: f"{cat.lower()}.json" for cat in by_category.keys()}
        }
        
        index_file = DATA_DIR / "index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        logger.info(f"   📑 Saved index: {index_file}")


async def main():
    """Main entry point."""
    crawler = DeepCrawler()
    await crawler.crawl()


if __name__ == "__main__":
    asyncio.run(main())
