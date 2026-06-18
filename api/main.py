import uuid
import logging
import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from config.logging import setup_logging, request_id_context
from api.routers import query, voice, institutions, health
from store.qdrant_client import QdrantStore
from store.redis_client import RedisStore
from store.postgres import engine
from ingestion.processors.embedder import Embedder

logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logging configuration on startup
    setup_logging()
    logger.info("Starting Wazobia Agent FastAPI application", extra={"environment": settings.ENVIRONMENT})

    # Instantiate stores and embedder once
    qdrant_store = QdrantStore()
    redis_store = RedisStore()
    embedder = Embedder()

    # Save to app state for dependency injection access
    app.state.qdrant_store = qdrant_store
    app.state.redis_store = redis_store
    app.state.embedder = embedder

    # Ensure collection exists in Qdrant
    await qdrant_store.ensure_collection()

    # Run migrations with a distributed lock in Redis (max 3 retry attempts)
    lock_key = "wazobia:migration_lock"
    acquired = False
    
    for attempt in range(1, 4):
        try:
            res = await redis_store.client.set(lock_key, "locked", nx=True, ex=30)
            if res:
                acquired = True
                break
        except Exception as lock_err:
            logger.warning(f"Failed to check/acquire migration lock (Attempt {attempt}/3): {lock_err}")
        await asyncio.sleep(2)
        
    if acquired:
        try:
            logger.info("Acquired migration lock. Running alembic migrations...")
            import subprocess
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["alembic", "upgrade", "head"],
                    check=True,
                    capture_output=True,
                    text=True
                )
            )
            logger.info("Alembic migrations completed successfully.")
        except Exception as mig_err:
            logger.error(f"Failed to run alembic migrations: {mig_err}")
            if settings.ENVIRONMENT == "production":
                raise mig_err
        finally:
            try:
                await redis_store.client.delete(lock_key)
            except Exception:
                pass
    else:
        logger.info("Migration lock is held by another instance. Waiting for completion...")
        for attempt in range(1, 4):
            try:
                exists = await redis_store.client.exists(lock_key)
                if not exists:
                    break
            except Exception:
                pass
            await asyncio.sleep(5)
        logger.info("Migration sync check finished. Proceeding startup.")

    yield

    # Clean up client connections on shutdown
    await qdrant_store.close()
    await redis_store.close()
    await engine.dispose()
    logger.info("Stopping Wazobia Agent FastAPI application")


app = FastAPI(
    title="Wazobia Agent",
    description="Production-grade API for Nigerian institutional query and voice operations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os

# Ensure static/audio directory exists for saving generated audio files
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request ID Context & Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_context.set(request_id)
    
    start_time = time.time()
    user_ip = request.client.host if request.client else "127.0.0.1"
    
    try:
        response = await call_next(request)
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Log structured request details
        logger.info(
            f"Request {request.method} {request.url.path} handled with status {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "user_ip": user_ip,
            }
        )
        
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.exception(
            f"Request {request.method} {request.url.path} failed: {e}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "latency_ms": latency_ms,
                "request_id": request_id,
                "user_ip": user_ip,
            }
        )
        raise e
    finally:
        request_id_context.reset(token)


# Global Exception Handlers (RFC 7807 Problem Details)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    content = {
        "type": f"https://wazobia.agent/errors/{exc.status_code}",
        "title": exc.detail or "HTTP Error",
        "status": exc.status_code,
        "detail": exc.detail,
        "instance": request.url.path,
    }
    headers = {"Content-Type": "application/problem+json"}
    if exc.headers:
        headers.update(exc.headers)
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    content = {
        "type": "https://wazobia.agent/errors/422",
        "title": "Unprocessable Entity (Validation Error)",
        "status": 422,
        "detail": exc.errors(),
        "instance": request.url.path,
    }
    return JSONResponse(
        status_code=422,
        content=content,
        headers={"Content-Type": "application/problem+json"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("An unhandled exception occurred during request handling")
    content = {
        "type": "https://wazobia.agent/errors/500",
        "title": "Internal Server Error",
        "status": 500,
        "detail": "An unexpected error occurred. Please contact system administrator.",
        "instance": request.url.path,
    }
    return JSONResponse(
        status_code=500,
        content=content,
        headers={"Content-Type": "application/problem+json"},
    )


# Mount health check router directly to root (no /v1 prefix)
app.include_router(health.router)

# Mount all other routers with /v1 prefix
app.include_router(query.router, prefix="/v1")
app.include_router(voice.router, prefix="/v1")
app.include_router(institutions.router, prefix="/v1")
