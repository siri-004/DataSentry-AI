import os
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve setting from environment variable, falling back to Streamlit secrets if running in cloud."""
    val = os.getenv(key)
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = str(st.secrets[key])
        except Exception:
            pass
    return val or default

class Config:
    """Application configuration and business rule defaults."""

    # LLM Provider & Keys
    GEMINI_API_KEY: Optional[str] = _get_setting("GEMINI_API_KEY") or _get_setting("GOOGLE_API_KEY")
    OPENAI_API_KEY: Optional[str] = _get_setting("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = _get_setting("OPENAI_BASE_URL", None)

    # Determine default provider
    LLM_PROVIDER: str = _get_setting("LLM_PROVIDER", "gemini" if (_get_setting("GEMINI_API_KEY") or _get_setting("GOOGLE_API_KEY")) else "openai").lower()
    
    # Model name default based on provider
    raw_model = _get_setting("LLM_MODEL", "")
    if not raw_model or "2.0-flash" in raw_model:
        default_model = "gemini-2.5-flash" if LLM_PROVIDER == "gemini" else "gpt-4o-mini"
    else:
        default_model = raw_model
    LLM_MODEL: str = default_model

    LLM_TEMPERATURE: float = float(_get_setting("LLM_TEMPERATURE", "0.1"))
    LLM_TIMEOUT_SECONDS: int = int(_get_setting("LLM_TIMEOUT_SECONDS", "30"))

    # File and Data Processing Limits
    MAX_FILE_SIZE_MB: int = int(_get_setting("MAX_FILE_SIZE_MB", "25"))
    MAX_ROWS_LIMIT: int = int(_get_setting("MAX_ROWS_LIMIT", "100000"))
    MAX_SAMPLE_RECORDS_IN_EVIDENCE: int = 5

    # Default Business Rules
    DEFAULT_NON_NEGATIVE_PATTERNS: List[str] = [
        "quantity", "qty", "price", "unit_price", "amount", "total", 
        "sales", "revenue", "cost", "discount", "tax", "fee", "rate", "age"
    ]

    DEFAULT_REQUIRED_ID_PATTERNS: List[str] = [
        "id", "customer_id", "client_id", "order_id", "transaction_id", 
        "user_id", "account_id", "invoice_id"
    ]

    # Outlier Settings
    DEFAULT_IQR_MULTIPLIER: float = 1.5

    @classmethod
    def reload_env(cls) -> None:
        """Reload .env variables dynamically, falling back to st.secrets."""
        load_dotenv(override=True)
        cls.GEMINI_API_KEY = _get_setting("GEMINI_API_KEY") or _get_setting("GOOGLE_API_KEY")
        cls.OPENAI_API_KEY = _get_setting("OPENAI_API_KEY")
        cls.OPENAI_BASE_URL = _get_setting("OPENAI_BASE_URL", None)
        cls.LLM_PROVIDER = _get_setting("LLM_PROVIDER", "gemini" if cls.GEMINI_API_KEY else "openai").lower()
        raw_model = _get_setting("LLM_MODEL", "")
        if not raw_model or "2.0-flash" in raw_model:
            default_model = "gemini-2.5-flash" if cls.LLM_PROVIDER == "gemini" else "gpt-4o-mini"
        else:
            default_model = raw_model
        cls.LLM_MODEL = default_model

    @classmethod
    def get_active_provider_info(cls) -> dict:
        """Returns provider status and masked info for UI display without leaking the key."""
        cls.reload_env()
        has_gemini = bool(cls.GEMINI_API_KEY and not any(p in cls.GEMINI_API_KEY.lower() for p in ["placeholder", "your-", "your_"]))
        has_openai = bool(cls.OPENAI_API_KEY and not any(p in cls.OPENAI_API_KEY.lower() for p in ["placeholder", "your-", "sk-..."]))

        source = "Streamlit Secrets / .env"
        if cls.LLM_PROVIDER == "gemini" and has_gemini:
            return {
                "active": True,
                "provider": "Google Gemini",
                "model": cls.LLM_MODEL,
                "source": source,
            }
        elif has_openai:
            return {
                "active": True,
                "provider": "OpenAI Compatible",
                "model": cls.LLM_MODEL,
                "source": source,
            }
        elif has_gemini:
            return {
                "active": True,
                "provider": "Google Gemini",
                "model": cls.LLM_MODEL,
                "source": source,
            }
        return {
            "active": False,
            "provider": "Deterministic Rule Engine Only",
            "model": "None (No API Key Configured)",
            "source": "Local Heuristics",
        }

    @classmethod
    def is_llm_configured(cls) -> bool:
        """Check if any LLM API key is non-empty and not a placeholder."""
        info = cls.get_active_provider_info()
        return info["active"]
