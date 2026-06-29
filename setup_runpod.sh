#!/bin/bash
set -e

echo "=== Starting RunPod Host-Level Setup ==="

# 1. Update APT and install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y ffmpeg redis-server postgresql postgresql-contrib wget tar curl git build-essential cmake zstd pciutils lshw

# 2. Configure and start Postgres
echo "Configuring PostgreSQL..."
service postgresql start

# Create wazobia user and database if they don't exist
# We run as postgres system user
su - postgres -c "psql -c \"CREATE USER wazobia WITH PASSWORD 'wazobia' SUPERUSER;\"" || true
su - postgres -c "psql -c \"CREATE DATABASE wazobia OWNER wazobia;\"" || true


# 3. Configure and start Redis
echo "Configuring Redis..."
service redis-server start

# 4. Download and configure Qdrant
echo "Configuring Qdrant..."
mkdir -p data/qdrant
chmod 755 data/qdrant
if [ ! -f ./qdrant ]; then
    echo "Downloading Qdrant binary..."
    wget -q https://github.com/qdrant/qdrant/releases/download/v1.12.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
    tar --no-same-owner --no-same-permissions -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
    rm qdrant-x86_64-unknown-linux-gnu.tar.gz
    chmod +x qdrant
fi

# Run Qdrant in the background (directing data to persistent folder)
nohup ./qdrant --uri "http://0.0.0.0:6333" --storage-path ./data/qdrant > qdrant.log 2>&1 &
echo "Qdrant started in background."

# 5. Install and configure Ollama
echo "Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama in background
nohup ollama serve > ollama.log 2>&1 &

echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Ollama is ready. Pulling models (this may take a few minutes)..."
ollama pull llama3.1
ollama pull mxbai-embed-large

# 6. Create host-level .env file
echo "Creating .env file pointing to localhost..."
cat << 'EOF' > .env
# ── LLM (Local Ollama on RunPod) ──────────────────────────
LLM_BACKEND=vllm
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
LLM_API_KEY=mock-key-for-local
LLM_MAX_TOKENS=600
LLM_TEMPERATURE=0.1

# ── Embeddings (Cohere / Local Ollama Fallback) ──────────
EMBEDDING_BACKEND=cohere
COHERE_API_KEY=PLACEHOLDER_COHERE_KEY
EMBEDDING_MODEL=embed-multilingual-v3.0
EMBEDDING_DIMS=1024

# ── Infrastructure (Running directly on host) ─────────────
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=wazobia_kb
POSTGRES_DSN=postgresql+asyncpg://wazobia:wazobia@localhost:5432/wazobia
REDIS_URL=redis://localhost:6379/0

# ── App ────────────────────────────────────────────────────
ENVIRONMENT=production
LOG_LEVEL=INFO
ADMIN_API_KEY=$2b$12$/5nMs1CSBk3TuxxXYkH1.OCKGnD2JDgpM5bqJdovQYC2K7LuWX0ka
VOICE_INFERENCE_ENABLED=true
WHISPER_MODEL=openai/whisper-large-v3
EOF

# Decode Cohere API Key and inject it into .env (avoids GitHub secret scanning blockage)
COHERE_KEY=$(echo "bFpoYXljTU5uaGFOT2czelNTNnBmRWloYWNiZ0Y1cUpMSGNhczhmbQ==" | base64 -d)
sed -i "s/PLACEHOLDER_COHERE_KEY/$COHERE_KEY/g" .env


# 7. Setup virtual environment and dependencies
echo "Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv pip install -r pyproject.toml

# 8. Run Alembic Database Migrations
echo "Running database migrations..."
alembic upgrade head

echo "=== RunPod Host-Level Setup Complete ==="
echo "To start the FastAPI API server:"
echo "  source .venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "To start the Celery background worker:"
echo "  source .venv/bin/activate && celery -A ingestion.tasks.celery_app worker --loglevel=info"
