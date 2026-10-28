"""Runtime configuration, read from environment variables or ai-service/.env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ai-service/.env is found regardless of the directory uvicorn is started from
    model_config = SettingsConfigDict(env_file=SERVICE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

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
    checkpoint_db: str = str(SERVICE_DIR / "data" / "checkpoints.sqlite3")
    history_turns: int = 8  # cap on turns in the short-term window (on top of the token budget)

    # Memory (app/memory.py)
    short_term_max_tokens: int = 1200  # token budget for the recent-turns window; overflow is folded into a summary
    memory_enabled: bool = True  # long-term memory across sessions
    memory_db: str = str(SERVICE_DIR / "data" / "memory.sqlite3")
    memory_tenant: str = "default"  # clinic group; namespaces are (tenant, patient)
    memory_retention_days: int = 365

    # Pin "today" for demos and tests (YYYY-MM-DD). Empty = real date in the clinic timezone.
    clinic_today: str | None = None
    clinic_timezone: str = "America/New_York"

    # Optional Langfuse tracing (enabled only when both keys are present)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None
    langfuse_environment: str = "development"  # Langfuse environment filter (development / staging / production / eval)
    langfuse_user_salt: str = ""  # salt for the pseudonymous user id in traces
    app_release: str | None = None  # shown as "release" in Langfuse (e.g. git sha)
    # Evaluation only: return the facts behind each answer so a judge sees exactly what the model saw. Never in prod.
    eval_expose_grounding: bool = False

    # Speech-to-text for the consultation summarizer (faster-whisper; tiny / base / small / medium)
    whisper_model: str = "small"
    whisper_preload: bool = True  # load the model at startup in the background (first start downloads it)


@lru_cache
def get_settings() -> Settings:
    return Settings()
