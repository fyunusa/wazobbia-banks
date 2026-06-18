import pytest
from datetime import datetime
from ingestion.scrapers.base_scraper import RawDocument
from ingestion.processors.cleaner import DocumentCleaner, CleanedDocument
from ingestion.processors.chunker import SemanticChunker, Chunk


@pytest.mark.asyncio
async def test_clean_html_snippet():
    cleaner = DocumentCleaner()

    raw_html = """
    <html>
        <head><title>GTBank Help Centre</title></head>
        <body>
            <nav>Home | Accounts | Loans</nav>
            <h1>GTBank Guides</h1>
            <p>Our transfer charge is N5000 and the daily limit is NGN 10000.</p>
            <p>You can dial * 737 # to get started. Other charges include 500 naira maintenance fee.</p>
            <footer>Copyright 2026 GTBank</footer>
        </body>
    </html>
    """
    raw_doc = RawDocument(
        url="https://www.gtbank.com/test",
        raw_html=raw_html,
        category="fees",
        institution_slug="gtbank",
        http_status=200,
        content_type="html",
        scraped_at=datetime.utcnow(),
    )

    cleaned = await cleaner.clean(raw_doc)

    # Verify formatting and normalizations
    assert cleaned.title == "GTBank Help Centre"
    assert "Home | Accounts" not in cleaned.clean_text  # nav was removed
    assert "Copyright 2026" not in cleaned.clean_text  # footer was removed
    
    # Currency normalizations
    assert "₦5,000" in cleaned.clean_text
    assert "₦10,000" in cleaned.clean_text
    assert "₦500" in cleaned.clean_text
    
    # USSD normalizations
    assert "*737#" in cleaned.clean_text


def test_chunk_2000_word_document():
    # Construct a 2000-word paragraph
    base_sentence = "This is a sentence containing valuable information about banking fees in Nigeria. "
    long_text = base_sentence * 250  # ~2000 words

    cleaned_doc = CleanedDocument(
        url="https://www.gtbank.com/test",
        category="fees",
        scraped_at=datetime.utcnow(),
        institution_slug="gtbank",
        http_status=200,
        content_type="html",
        clean_text=long_text,
        title="Tariffs Guide",
        extracted_tables=[],
        word_count=2000,
        language_detected="en",
    )

    chunker = SemanticChunker(target_size=200, overlap=30)
    chunks = chunker.chunk(cleaned_doc)

    assert len(chunks) > 1
    # Check that prepended headers exist
    for chunk in chunks:
        assert "[GTBank | Fees]" in chunk.content
        # Ensure minimum size validation works
        assert chunker._token_count(chunk.content) >= 50
        # Check boundary target limits are respected
        assert chunker._token_count(chunk.content) <= 250


def test_verify_chunk_overlap():
    base_text = (
        "Sentence one of the document containing fees guidelines. " * 6 +
        "Sentence two of the document describing tariff schedules. " * 6 +
        "Sentence three of the document listing customer support contact lines. " * 6 +
        "Sentence four of the document outlining the refund policy for failed transactions. " * 6 +
        "Sentence five of the document providing mobile banking codes. " * 6 +
        "Sentence six of the document detailing cash withdrawal limits." * 6
    )
    cleaned_doc = CleanedDocument(
        url="https://www.gtbank.com/test",
        category="fees",
        scraped_at=datetime.utcnow(),
        institution_slug="gtbank",
        http_status=200,
        content_type="html",
        clean_text=base_text,
        title="Overlap Title",
        extracted_tables=[],
        word_count=360,
        language_detected="en",
    )

    # Set target size to 100, and overlap to 45 (minimum chunk size is 50, so this forces split and preserves chunks)
    chunker = SemanticChunker(target_size=100, overlap=45)
    chunks = chunker.chunk(cleaned_doc)

    assert len(chunks) >= 2
    # Verify that some sentence content overlaps between chunk 0 and chunk 1
    chunk_0_text = chunks[0].content
    chunk_1_text = chunks[1].content

    # Overlapped sentences must appear in both chunks
    has_overlap = False
    for sentence in [
        "Sentence one", "Sentence two", "Sentence three", "Sentence four", "Sentence five", "Sentence six"
    ]:
        if sentence in chunk_0_text and sentence in chunk_1_text:
            has_overlap = True
            break
    
    assert has_overlap, "No overlap detected between chunks."


def test_verify_dedup_via_content_hash():
    text = "Important regulatory circular content regarding bank limits."
    
    cleaned_doc1 = CleanedDocument(
        url="https://www.cbn.gov.ng/test",
        category="regulatory",
        scraped_at=datetime.utcnow(),
        institution_slug="cbn",
        http_status=200,
        content_type="html",
        clean_text=text,
        title="CBN Circular",
        extracted_tables=[],
        word_count=10,
        language_detected="en",
    )
    cleaned_doc2 = CleanedDocument(
        url="https://www.cbn.gov.ng/test",
        category="regulatory",
        scraped_at=datetime.utcnow(),
        institution_slug="cbn",
        http_status=200,
        content_type="html",
        clean_text=text,
        title="CBN Circular",
        extracted_tables=[],
        word_count=10,
        language_detected="en",
    )

    chunker = SemanticChunker()
    chunks1 = chunker.chunk(cleaned_doc1)
    chunks2 = chunker.chunk(cleaned_doc2)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        # The hashes must match for identical content
        assert c1.content_hash == c2.content_hash
