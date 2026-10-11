"""Shared LLM endpoint client factory."""

from .base import LLMError, LLMProvider
from .litellm_provider import LiteLLMProvider


def build_provider(llm_cfg) -> LLMProvider:
    """Construct the endpoint client used by every LLM stage."""
    return LiteLLMProvider(llm_cfg, model=llm_cfg.model)


__all__ = ["LLMError", "LLMProvider", "build_provider"]
