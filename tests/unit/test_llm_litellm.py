"""TDD tests for `src/llm/litellm_provider.py` — LiteLLMProvider.

Module under test does NOT yet exist; this file is built up cycle by
cycle alongside it. Each test is the RED for one cycle in the Phase 1
plan (docs/TDD_PLAN.md). All real network calls are mocked at the
`litellm.completion` SDK boundary.
"""

from __future__ import annotations

from types import SimpleNamespace


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
