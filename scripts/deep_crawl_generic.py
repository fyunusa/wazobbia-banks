"""
Generic Deep Crawler - Works for any financial institution website.

This crawler:
1. Takes a bank configuration (homepage, output dir, categorization rules)
2. Navigates all pages and interactive elements
3. Captures content from each page
4. Stores all content in JSON organized by section

Usage:
  python scripts/deep_crawl_generic.py gtbank
  python scripts/deep_crawl_generic.py zenith
  python scripts/deep_crawl_generic.py kuda
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

logger = logging.getLogger("deep_crawl")

# Configuration for each bank
BANK_CONFIG = {
    "gtbank": {
        "home": "https://www.gtbank.com",
        "output_dir": "data/gtbank",
        "categories": {
            "personal": ["personal", "savings", "accounts"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "governance", "csr"],
            "investment": ["investor", "financial", "shareholder"],
            "support": ["help", "faq", "support", "contact"],
            "about": ["about", "company", "covid", "news"],
            "digital": ["digital", "app", "online", "digital-banking"],
        }
    },
    "zenith": {
        "home": "https://www.zenithbank.com",
        "output_dir": "data/zenith",
        "categories": {
            "personal": ["personal", "retail", "individuals", "savings", "accounts"],
            "business": ["business", "sme", "corporate-banking", "commercial"],
            "corporate": ["corporate", "investor", "about", "press", "news", "csr"],
            "investment": ["investment", "financial", "shareholder", "investor"],
            "support": ["support", "faq", "help", "customer-care"],
            "digital": ["digital", "mobile", "online", "digital-banking"],
            "products": ["products", "services"],
        }
    },
    "kuda": {
        "home": "https://www.kudabank.com",
        "output_dir": "data/kuda",
        "categories": {
            "personal": ["personal", "individual", "savings"],
            "business": ["business", "sme"],
            "features": ["features", "product"],
            "support": ["support", "help", "faq"],
            "about": ["about", "company", "why-kuda"],
            "blog": ["blog", "news", "press"],
        }
    },
    "opay": {
        "home": "https://www.opayweb.com",
        "output_dir": "data/opay",
        "categories": {
            "personal": ["personal", "individual", "savings"],
            "business": ["business", "sme"],
            "features": ["features", "product", "services"],
            "support": ["support", "help", "faq"],
            "about": ["about", "company", "why-opay"],
            "blog": ["blog", "news", "press"],
        }
    },
    "access": {
        "home": "https://www.accessbankplc.com",
        "output_dir": "data/access",
        "categories": {
            "personal": ["personal", "retail", "individuals"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online"],
        }
    },
    "uba": {
        "home": "https://www.ubagroup.com",
        "output_dir": "data/uba",
        "categories": {
            "personal": ["personal", "retail", "individual", "savings"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online", "mobile"],
        }
    },
    "firstbank": {
        "home": "https://www.firstbanknigeria.com",
        "output_dir": "data/firstbank",
        "categories": {
            "personal": ["personal", "retail", "individual", "savings"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online", "mobile"],
        }
    },
    "taj": {
        "home": "https://www.tajbank.com",
        "output_dir": "data/taj",
        "categories": {
            "personal": ["personal", "retail", "individual", "savings"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online"],
        }
    },
    "fcmb": {
        "home": "https://www.fcmb.com",
        "output_dir": "data/fcmb",
        "categories": {
            "personal": ["personal", "retail", "individual", "savings"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online"],
        }
    },
    "sterling": {
        "home": "https://www.sterlingbankng.com",
        "output_dir": "data/sterling",
        "categories": {
            "personal": ["personal", "retail", "individual", "savings"],
            "business": ["business", "sme", "commercial"],
            "corporate": ["corporate", "investor", "about", "news"],
            "investment": ["investment", "financial", "shareholder"],
            "support": ["support", "faq", "help", "customer"],
            "digital": ["digital", "app", "online"],
        }
    }
}

# Global configuration
CONFIG = {}
DATA_DIR = None
BANK_HOME = None
TIMEOUT = 60000  # Increased to 60 seconds for slow sites like Access Bank
SAME_ORIGIN_ONLY = True


class GenericDeepCrawler:
    """Deep crawler that works with any financial institution website."""
    
    def __init__(self, bank: str):
        self.bank = bank
        self.visited_urls: Set[str] = set()
        self.collected_pages: List[Dict[str, Any]] = []
        self.page_queue: List[str] = [BANK_HOME]
        self.sections: Dict[str, List[Dict]] = {}
        self.failed_urls = {"timeout": [], "error": [], "http_error": []}
    
    async def crawl(self):
        """Start the crawling process."""
        from playwright.async_api import async_playwright
        
        logger.info(f"🚀 Starting deep crawl of {self.bank.upper()}")
        logger.info(f"🔗 Homepage: {BANK_HOME}")
        logger.info(f"📁 Output: {DATA_DIR}")
        
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
                    page = await browser.new_page(
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    )
                    
                    # Navigate to page with timeout
                    try:
                        response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.error(f"   ✗ Error: Timeout 30000ms exceeded")
                        self.failed_urls["timeout"].append(url)
                        await page.close()
                        continue
                    except Exception as nav_error:
                        logger.error(f"   ✗ Error: {type(nav_error).__name__}: {str(nav_error)[:80]}")
                        self.failed_urls["error"].append(url)
                        await page.close()
                        continue
                    
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
                        logger.error(f"   ✗ HTTP {status}")
                        self.failed_urls["http_error"].append((url, status))
                    
                    await page.close()
                    
                    # Small delay between requests
                    await asyncio.sleep(0.5)
                
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
        """Categorize URL using bank-specific patterns."""
        path = urlparse(url).path.lower()
        
        categories = BANK_CONFIG[self.bank].get("categories", {})
        
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
            "/register", "/download", ".mp4", ".mov", ".avi"
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return True
        
        return False
    
    def _load_existing_visited_urls(self):
        """Load all previously crawled URLs from existing JSON files to resume crawling"""
        categories = list(CONFIG.get("categories", {}).keys())
        categories.extend(["other"])  # Include other category
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
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
            logger.info(f"   💾 Saved: {filename} ({len(pages)} pages)")
        
        # Save index
        index_data = {
            "bank": self.bank,
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


def main():
    global CONFIG, DATA_DIR, BANK_HOME
    
    # Get bank from command line
    bank = sys.argv[1].lower() if len(sys.argv) > 1 else "gtbank"
    
    if bank not in BANK_CONFIG:
        print(f"❌ Unknown bank: {bank}")
        print(f"Available: {', '.join(BANK_CONFIG.keys())}")
        sys.exit(1)
    
    CONFIG = BANK_CONFIG[bank]
    BANK_HOME = CONFIG["home"]
    DATA_DIR = Path(__file__).parent.parent / CONFIG["output_dir"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run crawler
    crawler = GenericDeepCrawler(bank)
    asyncio.run(crawler.crawl())


if __name__ == "__main__":
    main()
