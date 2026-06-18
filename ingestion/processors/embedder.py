import asyncio
import logging
from typing import List
import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer, AutoModel
import torch

from config.settings import settings
from store.qdrant_client import KnowledgePoint
from ingestion.processors.chunker import Chunk

logger = logging.getLogger("ingestion.processors.embedder")


class Embedder:
    """Generates dense vector representations for document chunks using configured backends."""

    def __init__(self) -> None:
        self.backend = settings.EMBEDDING_BACKEND.lower()
        self.model_name = settings.EMBEDDING_MODEL

        # Initialize local BGE models only if selected to save startup memory/time
        self.bge_tokenizer = None
        self.bge_model = None

        self.use_offline_fallback = False
        if self.backend == "bge":
            logger.info("Initializing offline BGE-M3 embedding model...")
            self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            self.bge_model = AutoModel.from_pretrained("BAAI/bge-m3")
            self.bge_model.eval()
        elif self.backend == "openai":
            logger.info("Using OpenAI embedding client...")
            api_key = settings.OPENAI_API_KEY
            base_url = None
            if (not api_key or api_key.startswith("mock") or api_key == "your_openai_api_key_here") and "ollama" in settings.LLM_BASE_URL:
                logger.info(f"OpenAI key is mock/empty. Routing OpenAI embeddings to local Ollama at {settings.LLM_BASE_URL}")
                api_key = "mock-key-for-local"
                base_url = settings.LLM_BASE_URL
            
            self.openai_client = AsyncOpenAI(
                api_key=api_key or "mock-key",
                base_url=base_url,
            )
        elif self.backend == "cohere":
            api_key = settings.COHERE_API_KEY
            if (not api_key or api_key.startswith("mock") or api_key == "your_cohere_api_key_here") and "ollama" in settings.LLM_BASE_URL:
                logger.info(f"Cohere key is mock/empty. Routing Cohere embeddings to local Ollama at {settings.LLM_BASE_URL}")
                self.use_offline_fallback = True
                self.openai_client = AsyncOpenAI(
                    api_key="mock-key-for-local",
                    base_url=settings.LLM_BASE_URL,
                )

    async def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Invokes OpenAI API with exponential backoff retry on 429 rate limits."""
        backoff = 1.0
        for attempt in range(4):
            try:
                response = await self.openai_client.embeddings.create(
                    input=texts,
                    model=self.model_name,
                )
                # Log tokens consumed
                usage = response.usage
                logger.info(
                    f"OpenAI Embeddings generated. Prompt tokens: {usage.prompt_tokens}, total tokens: {usage.total_tokens}"
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                # Detect rate limit errors
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower():
                    logger.warning(f"OpenAI rate limit hit. Retrying in {backoff:.2f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                logger.error(f"OpenAI embedding failure: {e}")
                raise

        raise Exception("Failed to generate OpenAI embeddings after multiple retries due to rate limits.")

    async def _embed_cohere_batch(self, texts: List[str]) -> List[List[float]]:
        """Invokes Cohere API via httpx client."""
        if self.use_offline_fallback:
            try:
                response = await self.openai_client.embeddings.create(
                    input=texts,
                    model="mxbai-embed-large",
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"Offline fallback embedding failure: {e}")
                raise

        backoff = 1.0
        for attempt in range(4):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {
                        "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "texts": texts,
                        "model": self.model_name,
                        "input_type": "search_document",
                        "embedding_types": ["float"]
                    }
                    resp = await client.post("https://api.cohere.com/v1/embed", json=payload, headers=headers)
                    if resp.status_code == 429:
                        logger.warning(f"Cohere rate limit hit. Retrying in {backoff:.2f}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    # Cohere returns a dictionary with 'embeddings' key containing lists or dicts
                    embeddings = data.get("embeddings", {})
                    if isinstance(embeddings, dict) and "float" in embeddings:
                        return embeddings["float"]
                    return embeddings
            except Exception as e:
                logger.error(f"Cohere embedding failure: {e}")
                raise

        raise Exception("Failed to generate Cohere embeddings after multiple retries.")

    def _embed_bge_batch(self, texts: List[str]) -> List[List[float]]:
        """Inferences offline BGE-M3 local model on CPU."""
        if not self.bge_model or not self.bge_tokenizer:
            raise RuntimeError("BGE-M3 model components not initialized.")

        with torch.no_grad():
            encoded_input = self.bge_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            model_output = self.bge_model(**encoded_input)
            # Perform Mean Pooling
            token_embeddings = model_output[0]
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            # L2 Normalize
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.tolist()

    async def embed_chunks(self, chunks: List[Chunk]) -> List[KnowledgePoint]:
        """Batches and embeds a list of chunks, mapping them into KnowledgePoints."""
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        batch_size = 100
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            logger.info(f"Processing embedding batch of size {len(batch_texts)} using backend: {self.backend}")

            if self.backend == "openai":
                batch_embeds = await self._embed_openai_batch(batch_texts)
            elif self.backend == "cohere":
                batch_embeds = await self._embed_cohere_batch(batch_texts)
            elif self.backend == "bge":
                # CPU-bound inference run in system executors to prevent blockages
                loop = asyncio.get_running_loop()
                batch_embeds = await loop.run_in_executor(
                    None, self._embed_bge_batch, batch_texts
                )
            else:
                raise ValueError(f"Unsupported embedding backend config: {self.backend}")

            embeddings.extend(batch_embeds)

        knowledge_points = []
        for chunk, vector in zip(chunks, embeddings):
            # Resolve display name
            try:
                inst_name = get_institution(chunk.institution_slug).name
            except Exception:
                inst_name = chunk.institution_slug.upper()

            knowledge_points.append(
                KnowledgePoint(
                    id=chunk.id,
                    vector=vector,
                    institution_slug=chunk.institution_slug,
                    institution_name=inst_name,
                    category=chunk.category,
                    sub_category=chunk.sub_category,
                    content=chunk.content,
                    source_url=chunk.source_url,
                    scraped_at=chunk.scraped_at,
                    content_hash=chunk.content_hash,
                    is_verified=False,
                    language="en",
                )
            )

        return knowledge_points

    async def embed_query(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a single query string."""
        if self.backend == "openai":
            res = await self._embed_openai_batch([text])
            return res[0]
        elif self.backend == "cohere":
            if self.use_offline_fallback:
                try:
                    response = await self.openai_client.embeddings.create(
                        input=[text],
                        model="mxbai-embed-large",
                    )
                    return response.data[0].embedding
                except Exception as e:
                    logger.error(f"Offline fallback query embedding failure: {e}")
                    raise

            backoff = 1.0
            for attempt in range(4):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        headers = {
                            "Authorization": f"Bearer {settings.COHERE_API_KEY}",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "texts": [text],
                            "model": self.model_name,
                            "input_type": "search_query",
                            "embedding_types": ["float"]
                        }
                        resp = await client.post("https://api.cohere.com/v1/embed", json=payload, headers=headers)
                        if resp.status_code == 429:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        embeddings = data.get("embeddings", {})
                        if isinstance(embeddings, dict) and "float" in embeddings:
                            return embeddings["float"][0]
                        return embeddings[0]
                except Exception as e:
                    logger.error(f"Cohere embedding query failure: {e}")
                    raise
            raise Exception("Failed to generate Cohere query embedding after retries.")
        elif self.backend == "bge":
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(None, self._embed_bge_batch, [text])
            return res[0]
        else:
            raise ValueError(f"Unsupported embedding backend config: {self.backend}")
