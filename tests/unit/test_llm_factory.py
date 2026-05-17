"""Characterization tests for `src/llm/__init__.py` (the provider factory).

Module under test: `build_provider(llm_cfg)` — dispatches by `llm_cfg.provider`
to one of three known classes: ClaudeCLIProvider, AnthropicAPIProvider, or
raises LLMError for unknown names.

This factory is the load-bearing piece for the Phase 1 LiteLLM swap.
The `anthropic_api` branch is scheduled for deletion in Phase 1, and a
`litellm` branch will replace it — the tests below will need both updates
in the same commit.

Each provider class is mocked so this file is purely about routing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm import LLMError, build_provider


def test_build_provider_routes_claude_cli_to_claudecliprovider(mocker) -> None:
    """`provider: 'claude_cli'` constructs ClaudeCLIProvider with `cfg.claude_cli`."""
    fake_class = mocker.patch("src.llm.claude_cli.ClaudeCLIProvider")
    cfg = SimpleNamespace(
        provider="claude_cli",
        claude_cli=SimpleNamespace(timeout_seconds=900),
    )

    result = build_provider(cfg)

    fake_class.assert_called_once_with(cfg.claude_cli)
    assert result is fake_class.return_value


def test_build_provider_routes_anthropic_api_to_anthropicapiprovider(mocker) -> None:
    """`provider: 'anthropic_api'` constructs AnthropicAPIProvider with cfg.anthropic_api + model.

    This test must be DELETED in Phase 1 alongside `src/llm/anthropic_api.py`.
    """
    fake_class = mocker.patch("src.llm.anthropic_api.AnthropicAPIProvider")
    cfg = SimpleNamespace(
        provider="anthropic_api",
        model="claude-sonnet-4-5",
        anthropic_api=SimpleNamespace(api_key_env="ANTHROPIC_API_KEY"),
    )

    result = build_provider(cfg)

    fake_class.assert_called_once_with(cfg.anthropic_api, model="claude-sonnet-4-5")
    assert result is fake_class.return_value


def test_build_provider_routes_litellm_to_litellmprovider(mocker) -> None:
    """`provider: 'litellm'` constructs LiteLLMProvider with cfg.litellm + model."""
    fake_class = mocker.patch("src.llm.litellm_provider.LiteLLMProvider")
    cfg = SimpleNamespace(
        provider="litellm",
        model="anthropic/claude-sonnet-4-5",
        litellm=SimpleNamespace(api_base=None, timeout_seconds=900, num_retries=2),
    )

    result = build_provider(cfg)

    fake_class.assert_called_once_with(cfg.litellm, model="anthropic/claude-sonnet-4-5")
    assert result is fake_class.return_value


def test_build_provider_raises_llmerror_for_unknown_provider_name() -> None:
    """Unknown provider names raise LLMError with the name in the message."""
    cfg = SimpleNamespace(provider="some_unknown_provider")

    with pytest.raises(LLMError, match="some_unknown_provider"):
        build_provider(cfg)
