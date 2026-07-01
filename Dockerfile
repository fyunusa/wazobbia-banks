# Use a multi-stage build to keep the image size small
FROM python:3.11-slim-bookworm AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install build dependencies to compile native extensions (like llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=300

# Copy dependency specifications
COPY pyproject.toml ./

ARG PIP_INDEX_ARGS=""
# Install dependencies into system Python (since it's a isolated container)
RUN uv pip install --system $PIP_INDEX_ARGS --requirements pyproject.toml

# Production stage
FROM python:3.11-slim-bookworm AS runner

# Create a non-root user and group
RUN groupadd -g 1000 wazobia && \
    useradd -m -u 1000 -g wazobia wazobia

# Install ffmpeg for torchcodec/torchaudio decoding and Playwright dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libx11-6 \
    libxcb1 \
    libxext6 \
    libxfixes3 \
    libxdamage1 \
    libxcomposite1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libxkbcommon0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV NUMBA_CACHE_DIR=/tmp
ENV COQUI_TOS_AGREED=1
ENV MPLCONFIGDIR=/tmp

# Copy system dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY --chown=wazobia:wazobia . .

# Expose port
EXPOSE 8000

# Switch to non-root user
USER wazobia

# Docker healthcheck using Python's built-in urllib
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
