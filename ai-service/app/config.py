"""Runtime configuration, read from environment variables or ai-service/.env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM: any OpenAI-compatible endpoint (DeepSeek, OpenAI, Azure OpenAI via compatible URL) or Anthropic.
    llm_provider: Literal["deepseek", "openai", "anthropic", "fake"] = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_api_key: str = ""
    llm_base_url: str | None = "https://api.deepseek.com"
    llm_timeout_s: float = 60.0

    # Backend (single source of truth for data and booking rules)
    backend_url: str = "http://localhost:3000/api"

    # How the agent calls its tools: plain REST, or through the bundled MCP server (stdio).
    tool_transport: Literal["rest", "mcp"] = "rest"

    # Conversation state (LangGraph checkpointer)
    checkpoint_db: str = "data/checkpoints.sqlite3"
    history_turns: int = 8

    # Pin "today" for demos and tests (YYYY-MM-DD). Empty = real date in the clinic timezone.
    clinic_today: str | None = None
    clinic_timezone: str = "America/New_York"

    # Optional Langfuse tracing (enabled only when both keys are present)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
