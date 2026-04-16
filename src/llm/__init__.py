"""LLM provider abstraction."""

from __future__ import annotations

from types import SimpleNamespace

from .base import LLMError, LLMProvider


def build_provider(llm_cfg: SimpleNamespace) -> LLMProvider:
    """Factory: instantiate the provider selected by `llm_cfg.provider`."""
    provider_name = llm_cfg.provider
    if provider_name == "claude_cli":
        from .claude_cli import ClaudeCLIProvider
        return ClaudeCLIProvider(llm_cfg.claude_cli)
    if provider_name == "anthropic_api":
        from .anthropic_api import AnthropicAPIProvider
        return AnthropicAPIProvider(llm_cfg.anthropic_api, model=llm_cfg.model)
    raise LLMError(f"unknown llm.provider: {provider_name!r}")


__all__ = ["LLMError", "LLMProvider", "build_provider"]
