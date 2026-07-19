from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Reviews Analytics Service"
    app_env: str = "local"
    api_prefix: str = "/api/v1"

    postgres_dsn: str | None = None

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "wb_reviews_bge_m3"

    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_timeout_seconds: int = 30
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_provider: str = "bge_m3"
    embedding_model_name: str = "BAAI/bge-m3"
    runpod_api_key: str | None = None
    runpod_endpoint_id: str | None = None
    runpod_base_url: str = "https://api.runpod.ai/v2"
    runpod_wait_ms: int = 120000
    runpod_embedding_input_key: str = "text"
    chat_parser_mode: str = "auto"
    chat_parser_timeout_seconds: int = 8
    rag_validate_candidates: bool = True
    rag_validation_limit: int = 30
    rag_top_k: int = 80
    rag_embedding_timeout_seconds: int = 30
    rag_qdrant_timeout_seconds: int = 15
    rag_validation_timeout_seconds: int = 20
    rag_keyword_rescue_limit: int = 10
    rag_keyword_rescue_max_scan: int = 1500
    rag_warmup_on_startup: bool = False
    cache_ttl_seconds: int = 900
    min_total_reviews_for_product_risk: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
