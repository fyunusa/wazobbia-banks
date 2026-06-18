from prometheus_client import Counter, Histogram, Gauge

# RAG Query metrics
wazobia_queries_total = Counter(
    "wazobia_queries_total",
    "Total count of queries processed by the RAG engine",
    ["institution", "language", "cache_hit"]
)

wazobia_query_latency_seconds = Histogram(
    "wazobia_query_latency_seconds",
    "Latency of RAG engine query processing in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# Voice requests metrics
wazobia_voice_requests_total = Counter(
    "wazobia_voice_requests_total",
    "Total count of voice queries processed",
    ["stt_engine", "tts_engine", "language"]
)

# Ingestion/Scrape runs metrics
wazobia_scrape_runs_total = Counter(
    "wazobia_scrape_runs_total",
    "Total count of scrape runs executed",
    ["institution", "status"]
)

# Vector DB counts
wazobia_qdrant_points_total = Gauge(
    "wazobia_qdrant_points_total",
    "Total count of active points in Qdrant collection per institution",
    ["institution"]
)
