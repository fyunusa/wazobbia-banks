import asyncio
import logging
from store.qdrant_client import QdrantStore
from ingestion.processors.embedder import Embedder
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)

async def test_search():
    qdrant = QdrantStore()
    embedder = Embedder()
    
    filter_cond = models.Filter(
        must=[
            models.FieldCondition(
                key="institution_slug",
                match=models.MatchValue(value="gtbank")
            )
        ]
    )

    queries = [
        "What are the charges for transferring money with GTBank?",
        "What is the cost of sending money through GTBank?",
        "How can I check my GTBank account balance?"
    ]
    
    for query in queries:
        print(f"\n==============================================")
        print(f"Running search for: '{query}'")
        query_vector = await embedder.embed_query(query)
        
        # Query WITHOUT threshold to see raw similarity score
        hits = await qdrant.client.query_points(
            collection_name=qdrant.collection_name,
            query=query_vector,
            query_filter=filter_cond,
            limit=5,
            with_payload=True
        )
        print(f"Total search hits found: {len(hits.points)}")
        for i, hit in enumerate(hits.points):
            print(f"Hit {i}: Score={hit.score:.4f}, Source={hit.payload.get('source_url')}")
            print(f"Content: {hit.payload.get('content')[:180]}...")
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_search())
