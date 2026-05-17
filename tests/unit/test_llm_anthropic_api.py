"""Characterization tests for `src/llm/anthropic_api.py` — AnthropicAPIProvider.

Module under test: thin wrapper over `anthropic.Anthropic.messages.create`.

The entire file is SCHEDULED FOR DELETION in Phase 1 (LiteLLM speaks
Anthropic natively via `anthropic/<model>` strings). These tests will
also be deleted in that commit. They exist now so the swap is
demonstrably equivalent.

Behaviors locked:
  - `__init__` reads `cfg.api_key_env` (default ANTHROPIC_API_KEY), raises
    LLMError if the env var is unset.
  - `complete()` calls `messages.create` with model/max_tokens/system/user,
    extracts text blocks from the response, returns concatenated stripped
    text, raises LLMError on empty content or upstream exception.

All SDK calls are mocked. No network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm import LLMError
from src.llm.anthropic_api import AnthropicAPIProvider


def test_init_raises_llmerror_when_api_key_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key → LLMError mentioning the env var name."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY not set"):
        AnthropicAPIProvider(SimpleNamespace(), model="claude-x")


def test_init_uses_custom_api_key_env_var_when_configured(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """`cfg.api_key_env` overrides the default env var name — the value
    actually read from THAT env var (not ANTHROPIC_API_KEY) reaches the
    Anthropic client constructor."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MY_CUSTOM_KEY", "sk-test-123")
    mock_cls = mocker.patch("anthropic.Anthropic")

    AnthropicAPIProvider(
        SimpleNamespace(api_key_env="MY_CUSTOM_KEY"), model="claude-x",
    )

    mock_cls.assert_called_once_with(api_key="sk-test-123")


def test_init_passes_api_key_to_anthropic_client(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """The Anthropic client is constructed with the api_key from env."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-abc")
    mock_client_cls = mocker.patch("anthropic.Anthropic")

    AnthropicAPIProvider(SimpleNamespace(), model="claude-x")

    mock_client_cls.assert_called_once_with(api_key="sk-abc")


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> tuple[AnthropicAPIProvider, "mocker.MagicMock"]:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_client_cls = mocker.patch("anthropic.Anthropic")
    mock_client = mock_client_cls.return_value
    provider = AnthropicAPIProvider(SimpleNamespace(), model="claude-sonnet-4-5")
    return provider, mock_client


def _fake_message_with_text(mocker, text: str):
    """Build a fake SDK response: `message.content = [TextBlock(type='text', text=...)]`."""
    block = mocker.MagicMock()
    block.type = "text"
    block.text = text
    message = mocker.MagicMock()
    message.content = [block]
    return message


def test_complete_returns_text_from_first_text_block(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Happy path: single text block → its text (stripped)."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.return_value = _fake_message_with_text(mocker, "  hello  ")

    result = provider.complete("hi")

    assert result == "hello"


def test_complete_concatenates_multiple_text_blocks(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Multi-block responses are joined (concatenated) and stripped."""
    provider, client = _make_provider(monkeypatch, mocker)
    block1 = mocker.MagicMock(); block1.type = "text"; block1.text = "part1 "
    block2 = mocker.MagicMock(); block2.type = "text"; block2.text = "part2"
    msg = mocker.MagicMock(); msg.content = [block1, block2]
    client.messages.create.return_value = msg

    result = provider.complete("hi")

    assert result == "part1 part2"


def test_complete_ignores_non_text_blocks(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Tool-use blocks etc. are filtered — only `type == 'text'` is included."""
    provider, client = _make_provider(monkeypatch, mocker)
    text_block = mocker.MagicMock(); text_block.type = "text"; text_block.text = "keep me"
    tool_block = mocker.MagicMock(); tool_block.type = "tool_use"; tool_block.text = "skip me"
    msg = mocker.MagicMock(); msg.content = [tool_block, text_block]
    client.messages.create.return_value = msg

    result = provider.complete("hi")

    assert result == "keep me"


def test_complete_forwards_model_and_max_tokens_to_messages_create(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """The provider passes model and max_tokens through to the SDK."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.return_value = _fake_message_with_text(mocker, "ok")

    provider.complete("hi", max_tokens=500)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5"
    assert kwargs["max_tokens"] == 500


def test_complete_passes_system_prompt_when_provided(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Non-empty system_prompt → `system=<value>` in the SDK call."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.return_value = _fake_message_with_text(mocker, "ok")

    provider.complete("hi", system_prompt="be brief")

    assert client.messages.create.call_args.kwargs["system"] == "be brief"


def test_complete_passes_none_system_when_no_system_prompt(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Empty system_prompt → SDK receives `system=None` (omits system header)."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.return_value = _fake_message_with_text(mocker, "ok")

    provider.complete("hi")

    assert client.messages.create.call_args.kwargs["system"] is None


def test_complete_wraps_user_prompt_in_messages_list(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """User prompt becomes `messages=[{'role':'user','content':<prompt>}]`."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.return_value = _fake_message_with_text(mocker, "ok")

    provider.complete("ask")

    assert client.messages.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "ask"},
    ]


def test_complete_raises_llmerror_when_sdk_call_raises(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Any exception from messages.create is wrapped as LLMError."""
    provider, client = _make_provider(monkeypatch, mocker)
    client.messages.create.side_effect = RuntimeError("network died")

    with pytest.raises(LLMError, match="anthropic API call failed"):
        provider.complete("hi")


def test_complete_raises_llmerror_when_response_content_is_empty(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """`message.content == []` → LLMError 'empty content'."""
    provider, client = _make_provider(monkeypatch, mocker)
    msg = mocker.MagicMock(); msg.content = []
    client.messages.create.return_value = msg

    with pytest.raises(LLMError, match="empty content"):
        provider.complete("hi")


def test_complete_raises_llmerror_when_no_text_blocks_returned(
    monkeypatch: pytest.MonkeyPatch, mocker,
) -> None:
    """Only non-text blocks → no text after filtering → LLMError 'no text content'."""
    provider, client = _make_provider(monkeypatch, mocker)
    tool_block = mocker.MagicMock(); tool_block.type = "tool_use"
    msg = mocker.MagicMock(); msg.content = [tool_block]
    client.messages.create.return_value = msg

    with pytest.raises(LLMError, match="no text content"):
        provider.complete("hi")
