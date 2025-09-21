# src/config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Low-level Google GenAI SDK (your snippet)
from google import genai
from google.genai.types import HttpOptions

# LangChain wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional news tool (we instantiate later)
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

load_dotenv()

def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _as_int(v: str | None, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

@dataclass
class Settings:
    google_api_key: str
    use_vertex: bool
    news_lookback_days: int
    news_top_k: int

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            use_vertex=_as_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), False),
            news_lookback_days=_as_int(os.getenv("NEWS_LOOKBACK_DAYS"), 7),
            news_top_k=_as_int(os.getenv("NEWS_TOP_K"), 5),
        )

def get_settings() -> Settings:
    s = Settings.load()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is empty. Add it to .env")
    if s.use_vertex:
        raise RuntimeError(
            "GOOGLE_GENAI_USE_VERTEXAI=True detected, but we’re using Google GenAI API now. "
            "Set it to False or we’ll wire Vertex later."
        )
    return s

def get_raw_genai_client() -> genai.Client:
    s = get_settings()
    return genai.Client(
        api_key=s.google_api_key,
        http_options=HttpOptions(api_version="v1"),
    )

def get_gemini_chat_model(model: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=s.google_api_key,
        temperature=0.2,
    )

def get_yahoo_news_tool() -> YahooFinanceNewsTool:
    return YahooFinanceNewsTool()

def sanity_summary() -> tuple[str, str, str]:
    s = get_settings()
    return (
        f"Google GenAI SDK: api_version=v1, key_present={bool(s.google_api_key)}",
        "LangChain LLM: ChatGoogleGenerativeAI(model='gemini-2.0-flash')",
        "News tool: YahooFinanceNewsTool()",
    )

__all__ = [
    "Settings",
    "get_settings",
    "get_raw_genai_client",
    "get_gemini_chat_model",
    "get_yahoo_news_tool",
    "sanity_summary",
]
