import re
import uuid
import hashlib
import logging
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
import tiktoken
from registry.institutions import get_institution
from ingestion.processors.cleaner import CleanedDocument

logger = logging.getLogger("ingestion.processors.chunker")


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    chunk_index: int
    total_chunks: int
    institution_slug: str
    category: str
    sub_category: Optional[str] = None
    source_url: str
    title: Optional[str] = None
    scraped_at: datetime
    content_hash: str


class SemanticChunker:
    """Chunks documents into sentence-boundary-aware blocks of target token lengths."""

    def __init__(self, target_size: int = 512, overlap: int = 64) -> None:
        self.target_size = target_size
        self.overlap = overlap
        # cl100k_base token encoding for text-embedding-3-small
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _get_sentences(self, text: str) -> List[str]:
        # Split on sentence boundaries (. ? !) followed by spaces
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, doc: CleanedDocument) -> List[Chunk]:
        """Splits CleanedDocument text and tables into chunks with prepended metadata."""
        chunks: List[Chunk] = []

        # Resolve institution display name
        try:
            inst_name = get_institution(doc.institution_slug).name
        except Exception:
            inst_name = doc.institution_slug.upper()

        header_prefix = f"[{inst_name} | {doc.category.title()}]\n\n"
        prefix_tokens = self._token_count(header_prefix)

        # 1. Chunk Clean Text
        sentences = self._get_sentences(doc.clean_text)
        sentence_data = [(s, self._token_count(s)) for s in sentences]

        current_chunk_sentences: List[str] = []
        current_tokens = 0
        chunk_texts: List[str] = []

        current_chunk_sentences = []
        current_tokens = 0
        chunk_texts = []

        for sentence, tokens in sentence_data:
            if current_chunk_sentences and (current_tokens + tokens + prefix_tokens > self.target_size):
                chunk_texts.append(" ".join(current_chunk_sentences))
                
                # Backtrack to implement overlap window
                overlap_sentences = []
                overlap_tokens = 0
                for s_prev in reversed(current_chunk_sentences):
                    s_prev_tokens = self._token_count(s_prev)
                    if overlap_tokens + s_prev_tokens <= self.overlap:
                        overlap_sentences.insert(0, s_prev)
                        overlap_tokens += s_prev_tokens
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk_sentences.append(sentence)
            current_tokens += tokens

        if current_chunk_sentences:
            chunk_texts.append(" ".join(current_chunk_sentences))

        # Filter out chunks that are too small (< 50 tokens)
        valid_texts = []
        for text in chunk_texts:
            if self._token_count(text) >= 50:
                valid_texts.append(text)

        # 2. Add Table chunks (One table = One atomic chunk)
        table_chunks: List[str] = []
        for table in doc.extracted_tables:
            table_chunks.append(table)

        all_chunk_contents = valid_texts + table_chunks
        total_chunks = len(all_chunk_contents)

        for idx, content in enumerate(all_chunk_contents):
            # Prepend context header
            final_content = f"{header_prefix}{content}"
            content_hash = hashlib.sha256(final_content.encode("utf-8")).hexdigest()

            chunks.append(
                Chunk(
                    content=final_content,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    institution_slug=doc.institution_slug,
                    category=doc.category,
                    sub_category=doc.sub_category,
                    source_url=doc.url,
                    title=doc.title,
                    scraped_at=doc.scraped_at,
                    content_hash=content_hash,
                )
            )

        return chunks
