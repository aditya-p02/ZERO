# core/config.py
# Centralized settings for ZERO's existing runtime configuration.

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    user_name: str
    groq_api_key: str | None
    elevenlabs_api_key: str | None
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    ollama_base_url: str
    ollama_model: str
    tesseract_path: str
    log_level: str

    # Per-role Groq models
    groq_chat_model: str
    groq_router_model: str
    groq_memory_model: str
    groq_research_model: str
    groq_code_model: str
    groq_automation_model: str
    groq_screen_model: str
    groq_transcribe_model: str


settings = Settings(
    user_name=os.getenv("USER_NAME", "Aditya"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
    postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
    postgres_port=_int_env("POSTGRES_PORT", 5433),
    postgres_db=os.getenv("POSTGRES_DB", "zero_memory"),
    postgres_user=os.getenv("POSTGRES_USER", "zero"),
    postgres_password=os.getenv("POSTGRES_PASSWORD", "zero_secure_pass"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
    tesseract_path=os.getenv("TESSERACT_PATH", ""),
    log_level=os.getenv("LOG_LEVEL", "INFO").upper(),

    groq_chat_model=os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile"),
    groq_router_model=os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant"),
    groq_memory_model=os.getenv("GROQ_MEMORY_MODEL", "llama-3.1-8b-instant"),
    groq_research_model=os.getenv("GROQ_RESEARCH_MODEL", "llama-3.3-70b-versatile"),
    groq_code_model=os.getenv("GROQ_CODE_MODEL", "llama-3.3-70b-versatile"),
    groq_automation_model=os.getenv("GROQ_AUTOMATION_MODEL", "llama-3.1-8b-instant"),
    groq_screen_model=os.getenv("GROQ_SCREEN_MODEL", "llama-3.3-70b-versatile"),
    groq_transcribe_model=os.getenv("GROQ_TRANSCRIBE_MODEL", "whisper-large-v3"),
)
