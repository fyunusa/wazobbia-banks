from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # API Keys
    OPENAI_API_KEY: str = Field(default="")

    # LLM (Self-hosted vLLM on RunPod or Local Ollama)
    LLM_BACKEND: str = Field(default="vllm")
    LLM_BASE_URL: str = Field(default="http://localhost:8001/v1")
    LLM_MODEL: str = Field(default="casperhansen/llama-3.1-8b-instruct-awq")
    LLM_API_KEY: str = Field(default="")
    LLM_MAX_TOKENS: int = Field(default=600)
    LLM_TEMPERATURE: float = Field(default=0.1)

    # Embeddings (Cohere / OpenAI)
    EMBEDDING_BACKEND: str = Field(default="cohere")
    COHERE_API_KEY: str = Field(default="")
    EMBEDDING_MODEL: str = Field(default="embed-multilingual-v3.0")
    EMBEDDING_DIMS: int = Field(default=1024)

    # Qdrant
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_COLLECTION: str = Field(default="wazobia_kb")

    # Databases
    POSTGRES_DSN: str = Field(
        default="postgresql+asyncpg://wazobia:wazobia@localhost:5432/wazobia"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Security & App Config
    ADMIN_API_KEY: str = Field(default="$2b$12$/5nMs1CSBk3TuxxXYkH1.OCKGnD2JDgpM5bqJdovQYC2K7LuWX0ka")
    VOICE_INFERENCE_ENABLED: bool = Field(default=True)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    WHISPER_MODEL: str = Field(default="openai/whisper-large-v3")


settings = Settings()
