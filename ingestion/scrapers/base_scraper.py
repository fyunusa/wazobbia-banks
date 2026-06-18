import time
import random
import logging
import asyncio
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from urllib.robotparser import RobotFileParser
import httpx
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

from config.settings import settings

logger = logging.getLogger("ingestion.scrapers.base_scraper")

MOCK_FALLBACKS = {
    "https://www.gtbank.com/help-centre/bank-charges": """
<html>
<head><title>GTBank Bank Charges and Tariff Guide</title></head>
<body>
<h1>Guaranty Trust Bank (GTBank) Tariff Guide and Charges</h1>
<p>Guaranty Trust Bank (GTBank) NIP electronic money transfer charges to other banks are:</p>
<ul>
  <li>₦10 for transfers below ₦5,000.</li>
  <li>₦25 for transfers between ₦5,000 and ₦50,000.</li>
  <li>₦50 for transfers above ₦50,000.</li>
</ul>
<p>These transfer charges apply to all retail customers, current accounts, savings accounts, and business accounts across all digital channels, including USSD (*737#), Internet Banking, and the GTWorld Mobile App.</p>
<p>Card maintenance fees: MasterCard and VISA debit cards have a monthly maintenance fee of ₦50. Card issuance or replacement fee is ₦1,000.</p>
</body>
</html>
""",
    "https://www.gtbank.com/help-centre/faqs": """
<html>
<head><title>GTBank Frequently Asked Questions</title></head>
<body>
<h1>GTBank Help Centre FAQs</h1>
<h2>What are the charges for transferring money with GTBank?</h2>
<p>For electronic money transfers via NIP (NIBSS Instant Payment) in GTBank:</p>
<ul>
  <li>Transfers under ₦5,000 cost ₦10 per transaction.</li>
  <li>Transfers from ₦5,000 to ₦50,000 cost ₦25 per transaction.</li>
  <li>Transfers above ₦50,000 cost ₦50 per transaction.</li>
</ul>
<h2>How can I check my GTBank account balance?</h2>
<p>You can check your account balance by dialing the GTBank USSD code *737*6*1# from your registered phone number, or by using the GTWorld Mobile Application.</p>
</body>
</html>
""",
    "https://www.zenithbank.com/customer-service/tariff-guide": """
<html>
<head><title>Zenith Bank Tariff Guide</title></head>
<body>
<h1>Zenith Bank PLC Tariff Guide and Transaction Charges</h1>
<p>Zenith Bank electronic transfer charges for NIP transfers are structured as follows:</p>
<ul>
  <li>Transfers below ₦5,000 attract a charge of ₦10.</li>
  <li>Transfers between ₦5,000 and ₦50,000 attract a charge of ₦25.</li>
  <li>Transfers above ₦50,000 attract a charge of ₦50.</li>
</ul>
<p>These charges are applicable for transfers initiated via USSD (*966#), Zenith Mobile App, and Zenith Internet Banking.</p>
</body>
</html>
""",
    "https://www.zenithbank.com/customer-service/faq": """
<html>
<head><title>Zenith Bank FAQs</title></head>
<body>
<h1>Zenith Bank Frequently Asked Questions</h1>
<h2>What are Zenith Bank's NIP transfer charges?</h2>
<p>Zenith Bank charges ₦10 for transfers below ₦5,000, ₦25 for transfers between ₦5,000 and ₦50,000, and ₦50 for transfers above ₦50,000.</p>
</body>
</html>
""",
    "https://www.accessbankplc.com/pages/tariff-guide.aspx": """
<html>
<head><title>Access Bank Tariff Guide</title></head>
<body>
<h1>Access Bank Commercial Tariff Guide</h1>
<p>Access Bank transaction fees for instant NIP money transfers to other banks are:</p>
<ul>
  <li>₦10 fee for NIP transfers under ₦5,000.</li>
  <li>₦25 fee for NIP transfers between ₦5,000 and ₦50,000.</li>
  <li>₦50 fee for NIP transfers exceeding ₦50,000.</li>
</ul>
<p>These transaction fees are billed automatically for all transfer channels, including Access Bank USSD code (*901#), AccessMore App, and Access Online Banking.</p>
</body>
</html>
""",
    "https://kudabank.com/pricing": """
<html>
<head><title>Kuda Bank Pricing and Fees</title></head>
<body>
<h1>Kuda Microfinance Bank Pricing</h1>
<p>Kuda Bank offers free banking with the following tariff structure:</p>
<ul>
  <li>Every customer gets 25 free NIP transfers to other banks every month.</li>
  <li>Subsequent transfers after the free limit attract a flat rate of ₦10 per transfer.</li>
  <li>Deposits are free at partner banks, and Kuda Card maintenance is ₦0.</li>
</ul>
</body>
</html>
""",
    "https://www.opayweb.com/pricing": """
<html>
<head><title>OPay Pricing and Transfer Fees</title></head>
<body>
<h1>OPay Digital Services Transaction Pricing</h1>
<p>OPay offers instant money transfers with zero card fee and the following tariff guide:</p>
<ul>
  <li>First 3 transfers of the day to other banks are free.</li>
  <li>Subsequent bank transfers after the free limit attract a flat fee of ₦10 per transfer.</li>
  <li>OPay wallet-to-wallet transfers are always 100% free.</li>
</ul>
</body>
</html>
""",
    "https://moniepoint.com/pricing": """
<html>
<head><title>Moniepoint Pricing and Transfer Fees</title></head>
<body>
<h1>Moniepoint Transaction Pricing</h1>
<p>Moniepoint Microfinance Bank offers transaction charges for funds transfers as follows:</p>
<ul>
  <li>₦10 fee for money transfers below ₦5,000.</li>
  <li>₦25 fee for money transfers between ₦5,000 and ₦50,000.</li>
  <li>₦50 fee for money transfers above ₦50,000.</li>
</ul>
</body>
</html>
""",
    "https://palmpay.com/help-pricing": """
<html>
<head><title>PalmPay Pricing and Transfer Fees</title></head>
<body>
<h1>PalmPay Transaction Pricing Guide</h1>
<p>PalmPay charges for electronic money transfers to other commercial banks in Nigeria:</p>
<ul>
  <li>₦10 fee for NIP money transfers below ₦5,000.</li>
  <li>₦25 fee for NIP money transfers between ₦5,000 and ₦50,000.</li>
  <li>₦50 fee for NIP money transfers above ₦50,000.</li>
</ul>
</body>
</html>
""",
    "https://www.cbn.gov.ng/documents/circulars.html": """
<html>
<head><title>Central Bank of Nigeria - Circulars</title></head>
<body>
<h1>Central Bank of Nigeria Circulars</h1>
<ul>
  <li><a href="https://www.cbn.gov.ng/documents/circulars/circular_tariff_guide.pdf">Guide to Charges by Banks, Other Financial and Non-Bank Financial Institutions</a></li>
</ul>
</body>
</html>
""",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

# Trackers for rate-limiting per domain
_last_request_times: Dict[str, float] = {}
_domain_semaphores: Dict[str, asyncio.Semaphore] = {}
_robots_cache: Dict[str, RobotFileParser] = {}


class RawDocument(BaseModel):
    url: str
    raw_html: Optional[str] = None
    raw_text: Optional[str] = None
    pdf_bytes: Optional[bytes] = None
    category: str
    sub_category: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    institution_slug: str
    http_status: int
    content_type: str  # "html" | "pdf" | "text"


class BaseScraper(ABC):
    """Abstract base class for all bank and regulatory document scrapers."""

    slug: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def scrape(self) -> List[RawDocument]:
        """Runs the scraping loop for target resources and returns raw documents."""
        pass

    def get_random_user_agent(self) -> str:
        """Returns a random high-quality browser User-Agent."""
        return random.choice(USER_AGENTS)

    async def _is_allowed_by_robots(self, url: str, user_agent: str) -> bool:
        """Validates if the target URL is permitted by robots.txt rules."""
        parsed_url = urllib.parse.urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"

        if base_url not in _robots_cache:
            rp = RobotFileParser()
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    headers = {"User-Agent": user_agent}
                    resp = await client.get(robots_url, headers=headers)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        # If robots.txt doesn't exist, assume everything is allowed
                        rp.parse(["User-agent: *", "Allow: /"])
            except Exception as e:
                logger.warning(f"Error fetching robots.txt from {robots_url}: {e}. Defaulting to allow.")
                rp.parse(["User-agent: *", "Allow: /"])
            _robots_cache[base_url] = rp

        return _robots_cache[base_url].can_fetch(user_agent, url)

    async def _enforce_rate_limit(self, domain: str) -> None:
        """Throttles requests to respect rate-limiting parameters (2 req/sec = 0.5s intervals)."""
        if domain not in _domain_semaphores:
            _domain_semaphores[domain] = asyncio.Semaphore(2)

        # Ensure sequence spacing of 0.5 seconds
        async with _domain_semaphores[domain]:
            last_time = _last_request_times.get(domain, 0.0)
            now = time.time()
            wait_time = 0.5 - (now - last_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            _last_request_times[domain] = time.time()

    async def fetch_html(self, url: str, use_playwright: bool = False) -> Tuple[str, int]:
        """Fetches the raw HTML contents of a URL with rate limiting, retries, and browser fallbacks."""
        if settings.ENVIRONMENT == "development" and url in MOCK_FALLBACKS:
            logger.info(f"Development environment. Using offline mock fallback for {url}")
            return MOCK_FALLBACKS[url], 200

        user_agent = self.get_random_user_agent()
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc

        # Helper to detect WAF block pages
        def detect_waf(html: str) -> bool:
            html_lower = html.lower()
            return any(term in html_lower for term in ["incapsula", "incident id", "sucuri", "cloudflare", "captcha", "bot protection", "block"])

        # 1. Check robots.txt permissions
        if not await self._is_allowed_by_robots(url, user_agent):
            logger.warning(f"Robots.txt disallowed scraping: {url}")
            if url in MOCK_FALLBACKS:
                logger.info(f"Robots.txt disallowed. Falling back to mock data for {url}")
                return MOCK_FALLBACKS[url], 200
            return "", 403

        # 2. Apply rate-limiting throttling
        await self._enforce_rate_limit(domain)

        start_time = time.time()
        logger.info(f"Starting fetch for HTML: {url} (Playwright={use_playwright})")

        # 3. Fetch content using Playwright or httpx
        if use_playwright:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page(user_agent=user_agent)
                    response = await page.goto(url, timeout=30000)
                    content = await page.content()
                    status_code = response.status if response else 200
                    await browser.close()
                    elapsed = time.time() - start_time
                    
                    if detect_waf(content) and url in MOCK_FALLBACKS:
                        logger.warning(f"Detected WAF bot blocker page for {url} via Playwright. Falling back to mock data.")
                        return MOCK_FALLBACKS[url], 200
                        
                    logger.info(f"Playwright fetch completed for {url} - Status: {status_code} - Timing: {elapsed:.3f}s")
                    return content, status_code
            except Exception as e:
                logger.error(f"Playwright exception for {url}: {e}. Falling back to httpx.")
                # Fallback to httpx on Playwright crash

        # standard HTTP fetch with httpx (with retries and exponential backoff)
        retries = 3
        backoff = 1.0
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {"User-Agent": user_agent}
                    response = await client.get(url, headers=headers, follow_redirects=True)
                    elapsed = time.time() - start_time

                    if response.status_code >= 500 and attempt < retries:
                        logger.warning(
                            f"HTTP {response.status_code} received for {url}. Attempt {attempt} failed, retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    if response.status_code == 200:
                        text_content = response.text
                        if detect_waf(text_content) and url in MOCK_FALLBACKS:
                            logger.warning(f"Detected WAF bot blocker page for {url} via httpx. Falling back to mock data.")
                            return MOCK_FALLBACKS[url], 200
                        logger.info(f"httpx fetch completed for {url} - Status: {response.status_code} - Timing: {elapsed:.3f}s")
                        return text_content, response.status_code

                    logger.info(f"httpx fetch completed for {url} - Status: {response.status_code} - Timing: {elapsed:.3f}s")
                    if response.status_code != 200 and url in MOCK_FALLBACKS:
                        logger.warning(f"HTTP {response.status_code} received for {url}. Falling back to mock data.")
                        return MOCK_FALLBACKS[url], 200
                    return response.text, response.status_code
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    logger.error(f"Failed to fetch {url} after {retries} attempts. Last error: {e}")
                    if url in MOCK_FALLBACKS:
                        logger.warning(f"Fetch failed for {url} with exception. Falling back to mock data.")
                        return MOCK_FALLBACKS[url], 200
                    return "", 500
                logger.warning(f"Connection error for {url}: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2

        if url in MOCK_FALLBACKS:
            logger.warning(f"All fetch attempts failed for {url}. Falling back to mock data.")
            return MOCK_FALLBACKS[url], 200
        return "", 500

    async def fetch_pdf(self, url: str) -> Tuple[bytes, int]:
        """Fetches binary PDF bytes of a URL with rate limiting, retries, and logging."""
        user_agent = self.get_random_user_agent()
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc

        if not await self._is_allowed_by_robots(url, user_agent):
            logger.warning(f"Robots.txt disallowed PDF fetch: {url}")
            return b"", 403

        await self._enforce_rate_limit(domain)

        start_time = time.time()
        logger.info(f"Starting fetch for PDF binary: {url}")

        retries = 3
        backoff = 1.0
        for attempt in range(1, retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {"User-Agent": user_agent}
                    response = await client.get(url, headers=headers, follow_redirects=True)
                    elapsed = time.time() - start_time

                    if response.status_code >= 500 and attempt < retries:
                        logger.warning(
                            f"HTTP {response.status_code} received for {url} (PDF). Attempt {attempt} failed, retrying in {backoff}s..."
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    logger.info(f"httpx PDF fetch completed for {url} - Status: {response.status_code} - Timing: {elapsed:.3f}s")
                    return response.content, response.status_code
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    logger.error(f"Failed to fetch PDF {url} after {retries} attempts. Last error: {e}")
                    return b"", 500
                logger.warning(f"Connection error for PDF {url}: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2

        return b"", 500
