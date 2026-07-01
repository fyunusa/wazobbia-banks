"""
JSON Data Reader - Reads pre-collected bank data from JSON files.

In production, instead of scraping live, we read from pre-collected
JSON files stored in data/{bank_slug}/ directory.

Usage:
  from ingestion.processors.json_reader import load_bank_data
  
  docs = load_bank_data('gtbank')  # Returns list of RawDocument objects
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from store.models import RawDocument

logger = logging.getLogger("ingestion.processors.json_reader")

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_bank_data(bank_slug: str) -> List[RawDocument]:
    """Load pre-collected bank data from JSON files.
    
    Args:
        bank_slug: Bank identifier (e.g., 'gtbank', 'zenith')
        
    Returns:
        List of RawDocument objects ready for ingestion
    """
    bank_dir = DATA_DIR / bank_slug
    
    if not bank_dir.exists():
        logger.warning(f"No data directory for {bank_slug}: {bank_dir}")
        return []
    
    documents: List[RawDocument] = []
    
    # Load all JSON files in the bank directory
    for json_file in sorted(bank_dir.glob("*.json")):
        if json_file.name == "index.json":
            continue  # Skip index file
        
        logger.info(f"Loading {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # data is a list of document dicts
            if isinstance(data, list):
                for item in data:
                    doc = RawDocument(
                        url=item.get('url'),
                        raw_html=item.get('html', ''),
                        raw_text=item.get('text', ''),
                        category=item.get('category', 'OTHER'),
                        http_status=item.get('status', 200),
                        content_type=item.get('content_type', 'text/html'),
                    )
                    documents.append(doc)
                
                logger.info(f"  Loaded {len(data)} documents from {json_file.name}")
        
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"✓ Total documents loaded for {bank_slug}: {len(documents)}")
    return documents


def get_available_banks() -> List[str]:
    """Get list of banks that have pre-collected data."""
    if not DATA_DIR.exists():
        return []
    
    banks = [
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "index.json").exists()
    ]
    return sorted(banks)


def print_bank_summary(bank_slug: str) -> Optional[dict]:
    """Print summary of collected data for a bank."""
    bank_dir = DATA_DIR / bank_slug
    index_file = bank_dir / "index.json"
    
    if not index_file.exists():
        logger.warning(f"No index for {bank_slug}")
        return None
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)
        
        print(f"\n📊 {bank_slug.upper()} Data Summary")
        print(f"   Timestamp: {index.get('timestamp')}")
        print(f"   Total Documents: {index.get('total_documents')}")
        print(f"   Categories:")
        
        for category, count in index.get('categories', {}).items():
            print(f"      • {category}: {count} documents")
        
        return index
    except Exception as e:
        logger.error(f"Error reading index: {e}")
        return None
