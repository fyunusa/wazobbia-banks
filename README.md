# Wazobia Agent

Wazobia Agent is a production-grade FastAPI application engineered for RAG-driven institutional query and voice operations in Nigeria/Africa.

## Project Structure

```
wazobia-agent/ (wazobia-rag/)
├── api/                  # FastAPI Application
│   ├── routers/          # API Route Controllers
│   ├── dependencies.py   # DI Providers
│   └── main.py           # Application Entrypoint
├── agents/               # AI Orchestrator & RAG Agents
├── voice/                # Audio, Speech-To-Text & Text-To-Speech
├── ingestion/            # Scrapers, Cleaners, Chunkers & Embeddings
├── store/                # Database clients (Postgres, Redis, Qdrant)
├── registry/             # Institutional definitions
├── config/               # Settings & JSON Logging Config
└── tests/                # Testing Suite
```

## Getting Started

### Prerequisites

Ensure you have the following installed on your system:
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for ultra-fast dependency management)
- Docker and Docker Compose

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to configure database credentials, API keys (e.g., `OPENAI_API_KEY`, `COHERE_API_KEY`), and set `ADMIN_API_KEY` to the bcrypt hash of your admin secret.

### Database Setup & Migrations

1. Spin up target databases via docker-compose:
   ```bash
   docker compose up -d
   ```
2. Run database migrations to initialize tables (`ScrapeJob` and `QueryLog`):
   ```bash
   uv run alembic upgrade head
   ```

### Running the Application

1. Spin up the FastAPI API server:
   ```bash
   uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Spin up the Celery worker and Celery Beat scheduler in separate terminal sessions (ensure Redis is running):
   ```bash
   # Run Celery Worker
   uv run celery -A ingestion.tasks worker --loglevel=info -c 4

   # Run Celery Beat
   uv run celery -A ingestion.tasks beat --loglevel=info
   ```

### Ingestion

To trigger your first scraping and ingestion run for a registered institution (e.g., GTBank), invoke the admin endpoint:
```bash
curl -X POST http://localhost:8000/v1/institutions/gtbank/ingest \
  -H "X-API-Key: test-admin-secret-key-123"
```

### Running Tests

Execute the test suite with coverage reporting:
```bash
uv run pytest tests/ -v --cov
```
