"""Load bank-specific scraper configs from JSON files."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("config.scraper_configs")

CONFIG_DIR = Path(__file__).parent / "scrapers"


def load_scraper_config(slug: str) -> Dict[str, Any]:
    """Load JSON config for a specific bank scraper.
    
    Args:
        slug: Bank identifier (gtbank, zenith, opay, kuda)
        
    Returns:
        Config dict with targets, keywords, rate limits, etc
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    config_path = CONFIG_DIR / f"{slug}.json"
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        # Return minimal config for backwards compatibility
        return {
            "slug": slug,
            "name": slug.title(),
            "scrape_targets": [],
            "keywords": {},
            "exclude_patterns": ["/admin", "/login"],
            "rate_limit": {"requests_per_second": 1, "jitter_ms": 500},
            "max_sub_links": 50,
        }
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded scraper config for {slug}: {len(config.get('scrape_targets', []))} targets")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {config_path}: {e}")
        raise


def list_all_configs() -> List[str]:
    """List all available scraper configs."""
    return [f.stem for f in CONFIG_DIR.glob("*.json")]


def get_keywords_for_category(config: Dict, category: str) -> List[str]:
    """Get keywords for a specific category from config.
    
    Args:
        config: Loaded config dict
        category: Category name (fees, faq, products, etc)
        
    Returns:
        List of keyword strings
    """
    return config.get("keywords", {}).get(category, [])


def get_exclude_patterns(config: Dict) -> List[str]:
    """Get URL exclusion patterns from config."""
    return config.get("exclude_patterns", [])


def get_rate_limit(config: Dict) -> Dict[str, float]:
    """Get rate limit settings from config."""
    return config.get("rate_limit", {"requests_per_second": 1, "jitter_ms": 500})
