import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.rag_query import RAGQueryEngine, QueryRequest, QueryResponse
from store.qdrant_client import SearchResult
from agents.intent_classifier import IntentClassifier
from agents.orchestrator import MultiAgentOrchestrator, OrchestratorResponse


@pytest.mark.asyncio
async def test_rag_returns_answer():
    # Mock dependencies
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    # Mock return values
    redis_store.get_cached_query = AsyncMock(return_value=None)
    redis_store.cache_query_result = AsyncMock()
    redis_store.build_cache_key = MagicMock(return_value="mock_cache_key")

    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

    # Mock Qdrant returning 3 results
    mock_hits = [
        SearchResult(
            id="1",
            score=0.85,
            content="GTBank transfer fees are ₦10 for under ₦5,000.",
            source_url="https://gtbank.com/fees",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        ),
        SearchResult(
            id="2",
            score=0.75,
            content="GTBank monthly charge is ₦50.",
            source_url="https://gtbank.com/fees",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        ),
        SearchResult(
            id="3",
            score=0.60,
            content="GTBank USSD charge is ₦6.98.",
            source_url="https://gtbank.com/fees",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        ),
    ]
    qdrant.hybrid_search = AsyncMock(return_value=mock_hits)

    # Mock LLM Chat Completion
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "According to GTBank context, transfer charges are ₦10."
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis_store,
        openai_client=openai_client,
    )

    request = QueryRequest(
        query="What are GTBank charges?",
        institution_slug="gtbank",
        language="en",
    )

    response = await engine.query(request)

    assert isinstance(response, QueryResponse)
    assert "GTBank" in response.answer
    assert "₦10" in response.answer
    assert response.cache_hit is False
    assert len(response.sources) == 1
    assert "https://gtbank.com/fees" in response.sources

    # Verify calls
    redis_store.get_cached_query.assert_called_once()
    embedder.embed_query.assert_called_once_with("What are GTBank charges?")
    qdrant.hybrid_search.assert_called_once()
    openai_client.chat.completions.create.assert_called_once()
    redis_store.cache_query_result.assert_called_once()


@pytest.mark.asyncio
async def test_rag_empty_qdrant():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    redis_store.get_cached_query = AsyncMock(return_value=None)
    redis_store.build_cache_key = MagicMock(return_value="mock_cache_key")
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)
    
    # Mock Qdrant returning empty list
    qdrant.hybrid_search = AsyncMock(return_value=[])

    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis_store,
        openai_client=openai_client,
    )

    request = QueryRequest(
        query="How do I get loans?",
        institution_slug="gtbank",
        language="en",
    )

    response = await engine.query(request)

    assert "I don't have information about that for GTBank." in response.answer
    assert response.sources == []
    # Verify no LLM call was made for generation
    openai_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_rag_cache_hit():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    # Mock Redis returning cached response
    cached_response = {
        "answer": "This is a cached answer containing GTBank.",
        "sources": ["https://cached.url"],
        "institution_slug": "gtbank",
        "language": "en",
        "confidence": 0.9,
        "cache_hit": True,
        "latency_ms": 5,
    }
    redis_store.get_cached_query = AsyncMock(return_value=json.dumps(cached_response))
    redis_store.build_cache_key = MagicMock(return_value="mock_cache_key")

    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis_store,
        openai_client=openai_client,
    )

    request = QueryRequest(
        query="What are the charges?",
        institution_slug="gtbank",
        language="en",
    )

    response = await engine.query(request)

    assert response.cache_hit is True
    assert "cached answer" in response.answer
    assert "https://cached.url" in response.sources

    # Verify embedder and Qdrant/LLM are skipped
    redis_store.get_cached_query.assert_called_once()
    embedder.embed_query.assert_not_called()
    qdrant.hybrid_search.assert_not_called()
    openai_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_rag_yo_translation():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    redis_store.get_cached_query = AsyncMock(return_value=None)
    redis_store.cache_query_result = AsyncMock()
    redis_store.build_cache_key = MagicMock(return_value="mock_cache_key")
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

    mock_hits = [
        SearchResult(
            id="1",
            score=0.80,
            content="Transfer charges are ₦10.",
            source_url="https://gtbank.com",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        )
    ]
    qdrant.hybrid_search = AsyncMock(return_value=mock_hits)

    # Mock multiple LLM completions:
    # 1. Translate Query from Yoruba to English
    # 2. Generate English response from Context
    # 3. Translate English response back to Yoruba
    mock_translations = [
        "What are transfer fees?",  # Query translation
        "The transfer fee is ₦10.",  # Response generation
        "Owo gbigbe jẹ ₦10.",  # Final translation
    ]
    
    completions = []
    for trans in mock_translations:
        comp = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = trans
        choice.message = msg
        comp.choices = [choice]
        completions.append(comp)

    openai_client.chat.completions.create = AsyncMock(side_calls=completions)
    # Use side_effect to return list elements sequentially
    openai_client.chat.completions.create.side_effect = completions

    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis_store,
        openai_client=openai_client,
    )

    request = QueryRequest(
        query="Kini owo gbigbe?",
        institution_slug="gtbank",
        language="yo",
    )

    response = await engine.query(request)

    assert response.language == "yo"
    assert "Owo gbigbe" in response.answer

    # Verify chat completion called 3 times (translation in, inference, translation out)
    assert openai_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_rag_low_score_qdrant():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    redis_store.get_cached_query = AsyncMock(return_value=None)
    redis_store.build_cache_key = MagicMock(return_value="mock_cache_key")
    embedder.embed_query = AsyncMock(return_value=[0.1] * 1024)

    # Mock Qdrant returning low scores (all < 0.35)
    mock_hits = [
        SearchResult(
            id="1",
            score=0.32,
            content="Transfer charges are ₦10.",
            source_url="https://gtbank.com",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        ),
        SearchResult(
            id="2",
            score=0.15,
            content="GTBank monthly charge is ₦50.",
            source_url="https://gtbank.com",
            category="fees",
            institution_slug="gtbank",
            institution_name="GTBank",
        )
    ]
    qdrant.hybrid_search = AsyncMock(return_value=mock_hits)

    engine = RAGQueryEngine(
        qdrant=qdrant,
        embedder=embedder,
        redis_store=redis_store,
        openai_client=openai_client,
    )

    request = QueryRequest(
        query="What are the charges?",
        institution_slug="gtbank",
        language="en",
    )

    response = await engine.query(request)

    assert "I don't have information about that for GTBank." in response.answer
    assert response.sources == []
    # Verify LLM generation is skipped
    openai_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_intent_classifier_is_comparative_query():
    # Test keywords (fast path)
    classifier = IntentClassifier(openai_client=MagicMock())
    assert await classifier.is_comparative_query("compare GTBank vs Zenith") is True
    assert await classifier.is_comparative_query("which bank has best transfer fees") is True
    assert await classifier.is_comparative_query("what are gtbank transfer fees") is False

    # Test LLM classification fallback (slow path)
    openai_client = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "yes"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    classifier_llm = IntentClassifier(openai_client=openai_client)
    # This query doesn't match comparative keywords, so it should trigger LLM call
    assert await classifier_llm.is_comparative_query("Is it better to open an account with them?") is True
    openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_intent_classifier_extract_institution_slug():
    classifier = IntentClassifier(openai_client=MagicMock())

    # Exact and alias matches
    assert await classifier.extract_institution_slug("How do I transfer with GT?") == "gtbank"
    assert await classifier.extract_institution_slug("What are GTBank transfer fees?") == "gtbank"
    assert await classifier.extract_institution_slug("How to open access bank account") == "access"
    
    # Negative match
    assert await classifier.extract_institution_slug("Tell me about Konga pay") is None


@pytest.mark.asyncio
async def test_orchestrator_route_single_explicit():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    mock_resp = QueryResponse(
        answer="Direct answer for GTBank.",
        sources=["https://gtbank.com"],
        institution_slug="gtbank",
        language="en",
        confidence=0.9,
        cache_hit=False,
        latency_ms=10,
    )

    orchestrator = MultiAgentOrchestrator(qdrant, embedder, redis_store, openai_client)
    orchestrator.query_engine.query = AsyncMock(return_value=mock_resp)

    res = await orchestrator.route("What are the fees?", "en", "gtbank")
    assert isinstance(res, OrchestratorResponse)
    assert res.response_type == "single_agent"
    assert len(res.answers) == 1
    assert res.answers[0].answer == "Direct answer for GTBank."
    assert res.institutions_queried == ["gtbank"]


@pytest.mark.asyncio
async def test_orchestrator_route_single_implicit():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    mock_resp = QueryResponse(
        answer="Direct answer for GTBank.",
        sources=["https://gtbank.com"],
        institution_slug="gtbank",
        language="en",
        confidence=0.9,
        cache_hit=False,
        latency_ms=10,
    )

    orchestrator = MultiAgentOrchestrator(qdrant, embedder, redis_store, openai_client)
    orchestrator.query_engine.query = AsyncMock(return_value=mock_resp)

    res = await orchestrator.route("What are GT transfer fees?", "en", None)
    assert res.response_type == "single_agent"
    assert len(res.answers) == 1
    assert res.answers[0].institution_slug == "gtbank"
    assert res.institutions_queried == ["gtbank"]


@pytest.mark.asyncio
async def test_orchestrator_route_comparative():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    orchestrator = MultiAgentOrchestrator(qdrant, embedder, redis_store, openai_client)
    orchestrator.intent_classifier.is_comparative_query = AsyncMock(return_value=True)
    orchestrator.intent_classifier.extract_institution_slug = AsyncMock(return_value=None)

    async def mock_query(req):
        return QueryResponse(
            answer=f"Answer for {req.institution_slug}.",
            sources=[f"https://{req.institution_slug}.com"],
            institution_slug=req.institution_slug,
            language=req.language,
            confidence=0.8,
            cache_hit=False,
            latency_ms=5,
        )
    orchestrator.query_engine.query = AsyncMock(side_effect=mock_query)

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "| Bank | Table |\n|---|---|\n| GT | info |"
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    openai_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    res = await orchestrator.route("compare transfer fees across all banks", "en", None)
    assert res.response_type == "comparative"
    assert len(res.answers) > 0
    assert "GT | info" in res.synthesis
    assert "gtbank" in res.institutions_queried


@pytest.mark.asyncio
async def test_orchestrator_route_out_of_scope():
    qdrant = MagicMock()
    embedder = MagicMock()
    redis_store = MagicMock()
    openai_client = MagicMock()

    orchestrator = MultiAgentOrchestrator(qdrant, embedder, redis_store, openai_client)
    orchestrator.intent_classifier.is_comparative_query = AsyncMock(return_value=False)
    orchestrator.intent_classifier.extract_institution_slug = AsyncMock(return_value=None)

    res = await orchestrator.route("Tell me a story about Nigeria.", "en", None)
    assert res.response_type == "out_of_scope"
    assert res.answers == []
    assert res.synthesis is None
