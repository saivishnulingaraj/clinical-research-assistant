"""Central application settings.

All configuration is read from environment variables (optionally via a ``.env``
file). Nothing provider-specific — especially the Bedrock model ID and AWS
region — is ever hard-coded; those are supplied through ``CRA_*`` env vars.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…/clinical-research-assistant). settings.py lives at
# src/cra/config/settings.py, so the root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime configuration for the service.

    Values are sourced from the environment with the ``CRA_`` prefix, e.g.
    ``CRA_BEDROCK_MODEL_ID``. See ``.env.example`` for the full list.
    """

    model_config = SettingsConfigDict(
        env_prefix="CRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM / Bedrock ---------------------------------------------------
    # Which backend to use: "bedrock" (real) or "fake" (deterministic, for
    # tests and CI). Defaults to "fake" so nothing accidentally calls AWS.
    llm_provider: str = Field(default="fake")
    # Bedrock model identifier, e.g. "amazon.nova-pro-v1:0". Never hard-coded
    # in application logic — only ever read from here.
    bedrock_model_id: str = Field(default="amazon.nova-pro-v1:0")
    aws_region: str = Field(default="us-east-1")
    # Pricing used to estimate cost (USD per 1K tokens). These are configurable
    # because Bedrock pricing changes; they are estimates, not billing truth.
    price_per_1k_input_usd: float = Field(default=0.0008)
    price_per_1k_output_usd: float = Field(default=0.0032)

    # ---- OpenAI-compatible provider -------------------------------------
    # Used when llm_provider == "openai_compatible". Works with any backend
    # exposing the OpenAI Chat Completions API (AshnaAI, OpenRouter, Groq,
    # Together, a local vLLM/Ollama, etc.). Base URL, key, and model come from
    # env; nothing is hard-coded. The API key must NEVER be committed.
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="")
    # Seconds to wait for the HTTP call before giving up.
    openai_timeout_s: float = Field(default=60.0)

    # ---- LLM limits & caching -------------------------------------------
    max_llm_calls_per_request: int = Field(default=8)
    llm_cache_size: int = Field(default=256)
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)

    # ---- Embeddings & vector store --------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index_dir: Path = Field(default=REPO_ROOT / "data" / "index")

    # ---- Chunking --------------------------------------------------------
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=120)

    # ---- Retrieval -------------------------------------------------------
    retrieval_top_k: int = Field(default=4)
    max_sub_queries: int = Field(default=3)
    # Below this mean similarity score the respond agent returns
    # "insufficient evidence" rather than guessing.
    confidence_threshold: float = Field(default=0.30)

    # ---- API / data ------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")
    log_level: str = Field(default="INFO")

    def ensure_dirs(self) -> None:
        """Create directories the service writes to, if missing."""
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the whole process shares one configuration object. Tests that
    need to override values should construct ``Settings(...)`` directly or call
    ``get_settings.cache_clear()`` after mutating the environment.
    """
    return Settings()
