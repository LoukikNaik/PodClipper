"""TDD tests for `src/llm/litellm_provider.py` — LiteLLMProvider.

Module under test does NOT yet exist; this file is built up cycle by
cycle alongside it. Each test is the RED for one cycle in the Phase 1
plan (docs/TDD_PLAN.md). All real network calls are mocked at the
`litellm.completion` SDK boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm import LLMError


# --------------------------------------------------------------------------- #
# Cycle 1.1 — init stores model and exposes name
# --------------------------------------------------------------------------- #

def test_init_sets_name_attribute_to_litellm_and_stores_model() -> None:
    """LiteLLMProvider(cfg, model) → name='litellm', self.model=model."""
    from src.llm.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-sonnet-4-5")

    assert provider.name == "litellm"
    assert provider.model == "anthropic/claude-sonnet-4-5"


# --------------------------------------------------------------------------- #
# Cycle 1.2 — complete() returns content from litellm response
# --------------------------------------------------------------------------- #

def _make_litellm_response(mocker, content: str):
    """Build a stand-in for litellm.ModelResponse with one choice."""
    message = mocker.MagicMock(content=content)
    choice = mocker.MagicMock(message=message)
    return mocker.MagicMock(choices=[choice])


def test_complete_returns_message_content_from_litellm_response(mocker) -> None:
    """complete() returns response.choices[0].message.content unchanged."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "hello world")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    result = provider.complete("anything")

    assert result == "hello world"


# --------------------------------------------------------------------------- #
# Cycle 1.3 — forward self.model as model= kwarg
# --------------------------------------------------------------------------- #

def test_complete_passes_configured_model_string_to_litellm(mocker) -> None:
    """The model string passed to __init__ is forwarded as model= to litellm.completion."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(SimpleNamespace(), model="openai/gpt-5-mini")

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["model"] == "openai/gpt-5-mini"


# --------------------------------------------------------------------------- #
# Cycle 1.4 — wrap user prompt as messages=[{role:user,content:...}]
# --------------------------------------------------------------------------- #

def test_complete_wraps_user_prompt_in_user_role_message(mocker) -> None:
    """With no system_prompt, messages contains exactly one user-role entry."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    provider.complete("hello?")

    assert mock_completion.call_args.kwargs["messages"] == [
        {"role": "user", "content": "hello?"},
    ]


# --------------------------------------------------------------------------- #
# Cycle 1.5 — system prompt prepended as system-role message when given
# --------------------------------------------------------------------------- #

def test_complete_prepends_system_prompt_as_system_message_when_provided(
    mocker,
) -> None:
    """Non-empty system_prompt → messages=[{system}, {user}] in that order."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    provider.complete("the question", system_prompt="be brief")

    assert mock_completion.call_args.kwargs["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the question"},
    ]


# --------------------------------------------------------------------------- #
# Cycle 1.6 — max_tokens forwarded to litellm
# --------------------------------------------------------------------------- #

def test_complete_forwards_max_tokens_kwarg_to_litellm(mocker) -> None:
    """max_tokens passed to complete() is forwarded to litellm.completion."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    provider.complete("hi", max_tokens=1234)

    assert mock_completion.call_args.kwargs["max_tokens"] == 1234


# --------------------------------------------------------------------------- #
# Cycle 1.7 — wrap any litellm exception in LLMError
# --------------------------------------------------------------------------- #

def test_complete_raises_llmerror_when_litellm_call_raises(mocker) -> None:
    """Any exception from litellm.completion is wrapped in LLMError."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.side_effect = RuntimeError("upstream blew up")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    with pytest.raises(LLMError, match="litellm"):
        provider.complete("hi")


# --------------------------------------------------------------------------- #
# Cycle 1.8 — empty / whitespace-only content raises LLMError
# --------------------------------------------------------------------------- #

def test_complete_raises_llmerror_when_response_content_is_empty(mocker) -> None:
    """Whitespace-only content from litellm → LLMError (don't return empty string)."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "   \n  ")
    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-x")

    with pytest.raises(LLMError, match="empty"):
        provider.complete("hi")


# --------------------------------------------------------------------------- #
# Cycle 1.9 — api_base forwarded when set in cfg (for Ollama/vLLM/proxies)
# --------------------------------------------------------------------------- #

def test_complete_forwards_api_base_when_set_in_cfg(mocker) -> None:
    """cfg.api_base (when truthy) → api_base= kwarg on litellm.completion."""
    from src.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    cfg = SimpleNamespace(api_base="http://localhost:11434")
    provider = LiteLLMProvider(cfg, model="ollama/llama3")

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:11434"
