"""
Centralized configuration. All tunables that let this system scale from
50 tools to 1000+ tools without code changes live here.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"

    # Observability (LangSmith / OpenTelemetry)
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "scalable-agentic-system"

    # Storage
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/agentic_system"
    redis_url: str = "redis://localhost:6379/0"

    # Tool Router — THE key scaling knob.
    # We never send more than top_k_tools to the LLM's context, regardless
    # of whether the registry holds 50 tools or 5000.
    top_k_tools: int = 8
    embedding_model: str = "text-embedding-3-small"
    router_similarity_floor: float = 0.55  # below this, tool is dropped even if top-K

    # PayPal (example integration; pattern generalizes to any REST API)
    paypal_client_id: str = ""
    paypal_client_secret: str = ""
    paypal_base_url: str = "https://api-m.sandbox.paypal.com"

    # Security
    jwt_secret: str = "change_me"

    class Config:
        env_file = ".env"


settings = Settings()
