from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from creditlens_agentic.config import get_settings


def get_llm(temperature: float = 0.2) -> Optional[BaseChatModel]:
    """Return a chat model if configured, else None (offline / fallback mode)."""
    settings = get_settings()
    if not settings.llm_available:
        return None

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            temperature=temperature,
        )

    return None
