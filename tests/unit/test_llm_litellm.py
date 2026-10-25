"""TDD tests for `src/llm/litellm_provider.py` — LiteLLMProvider.

Module under test does NOT yet exist; this file is built up cycle by
cycle alongside it. Each test is the RED for one cycle in the Phase 1
plan (docs/TDD_PLAN.md). All real network calls are mocked at the
`litellm.completion` SDK boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podclipper.llm import LLMError


# --------------------------------------------------------------------------- #
# Cycle 1.1 — init stores model and exposes name
# --------------------------------------------------------------------------- #

def test_init_sets_name_attribute_to_litellm_and_stores_model() -> None:
    """LiteLLMProvider(cfg, model) → name='litellm', self.model=model."""
    from podclipper.llm.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(SimpleNamespace(), model="anthropic/claude-sonnet-4-5")

    assert provider.name == "litellm"
    assert provider.model == "openai/anthropic/claude-sonnet-4-5"


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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

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
    from podclipper.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    cfg = SimpleNamespace(endpoint="http://localhost:11434")
    provider = LiteLLMProvider(cfg, model="ollama/llama3")

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["api_base"] == "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Cycle 1.10 — timeout forwarded from cfg
# --------------------------------------------------------------------------- #

def test_complete_forwards_timeout_seconds_from_cfg(mocker) -> None:
    """cfg.timeout_seconds → timeout= kwarg on litellm.completion."""
    from podclipper.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(
        SimpleNamespace(timeout_seconds=42), model="anthropic/claude-x",
    )

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["timeout"] == 42


# --------------------------------------------------------------------------- #
# Cycle 1.11 — num_retries forwarded from cfg
# --------------------------------------------------------------------------- #

def test_complete_forwards_num_retries_from_cfg(mocker) -> None:
    """cfg.num_retries → num_retries= kwarg on litellm.completion."""
    from podclipper.llm.litellm_provider import LiteLLMProvider

    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(
        SimpleNamespace(num_retries=3), model="anthropic/claude-x",
    )

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["num_retries"] == 3


# --------------------------------------------------------------------------- #
# Cycle 1.16 — one generic API key for every configured endpoint
# --------------------------------------------------------------------------- #

def test_complete_forwards_single_generic_api_key(
    mocker, monkeypatch,
) -> None:
    """Every endpoint reads LLM_API_KEY without a key-variable config setting."""
    from podclipper.llm.litellm_provider import LiteLLMProvider

    monkeypatch.setenv("LLM_API_KEY", "generic-test-key")
    mock_completion = mocker.patch("litellm.completion")
    mock_completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(
        SimpleNamespace(),
        model="openai/anthropic/claude-sonnet-4.5",
    )

    provider.complete("hi")

    assert mock_completion.call_args.kwargs["api_key"] == "generic-test-key"


# --------------------------------------------------------------------------- #
# Cycle 1.17 — silence litellm's chatty internal logger
# --------------------------------------------------------------------------- #

def test_init_raises_litellm_internal_logger_to_warning() -> None:
    """LiteLLMProvider() init silences the noisy 'LiteLLM' logger.

    Critical: when our root logger is DEBUG (e.g. -v flag), litellm dumps
    full request payloads through its 'LiteLLM' logger. Combined with our
    Rich handler this can deadlock on long prompts (~30KB+ messages).
    Locking the level prevents that footgun.
    """
    import logging
    from podclipper.llm.litellm_provider import LiteLLMProvider

    # set it low to prove init actively raises it
    logging.getLogger("LiteLLM").setLevel(logging.DEBUG)

    LiteLLMProvider(SimpleNamespace(), model="anthropic/x")

    assert logging.getLogger("LiteLLM").level >= logging.WARNING


def test_plain_endpoint_model_routes_through_openai_protocol(mocker, monkeypatch):
    from podclipper.llm.litellm_provider import LiteLLMProvider
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    completion = mocker.patch("litellm.completion")
    completion.return_value = _make_litellm_response(mocker, "ok")
    provider = LiteLLMProvider(SimpleNamespace(endpoint="https://example.test/v1"), model="example")
    assert provider.complete("hi") == "ok"
    assert completion.call_args.kwargs["model"] == "openai/example"
    assert completion.call_args.kwargs["api_base"] == "https://example.test/v1"
    assert completion.call_args.kwargs["api_key"] == "test-key"


def test_provider_routing_forwarded_without_reasoning_override(mocker, tmp_path):
    from podclipper.config import load_config
    from podclipper.llm import build_provider
    completion = mocker.patch("litellm.completion")
    completion.return_value = _make_litellm_response(mocker, "Hi")
    config = tmp_path / "config.yaml"
    config.write_text("llm:\n  routing:\n    only: [deepinfra]\n    allow_fallbacks: false\n")
    provider = build_provider(load_config(config).llm)
    assert provider.complete("Hi") == "Hi"
    params = completion.call_args.kwargs
    assert params["extra_body"] == {"provider": {
        "only": ["deepinfra"], "allow_fallbacks": False, "require_parameters": True,
    }}
    assert "reasoning" not in params["extra_body"]
    assert "reasoning_effort" not in params


def test_strict_schema_reaches_openrouter_without_a_provider_allowlist(mocker):
    from podclipper.config import load_default_config
    from podclipper.llm import build_provider
    completion = mocker.patch('litellm.completion')
    completion.return_value = _make_litellm_response(mocker, '{"corrections": []}')
    cfg = load_default_config().llm
    cfg.endpoint = 'https://openrouter.ai/api/v1'
    schema = {'type': 'json_schema', 'json_schema': {
        'name': 'caption_corrections', 'strict': True,
        'schema': {'type': 'object', 'properties': {'corrections': {'type': 'array', 'items': {'type': 'string'}}},
                   'required': ['corrections'], 'additionalProperties': False},
    }}
    build_provider(cfg).complete('Correct captions', response_format=schema)
    params = completion.call_args.kwargs
    assert params['response_format'] == schema
    assert params['extra_body']['provider'] == {'allow_fallbacks': True, 'require_parameters': True}


def test_default_openrouter_routing_flags_do_not_leak_to_other_endpoints(mocker):
    from podclipper.config import load_default_config
    from podclipper.llm import build_provider
    completion = mocker.patch('litellm.completion')
    completion.return_value = _make_litellm_response(mocker, 'ok')
    build_provider(load_default_config().llm).complete('Hello')
    assert 'extra_body' not in completion.call_args.kwargs


def test_strict_openrouter_request_always_requires_parameter_support(mocker):
    from podclipper.config import load_default_config
    from podclipper.llm import build_provider
    completion = mocker.patch('litellm.completion')
    completion.return_value = _make_litellm_response(mocker, '{}')
    cfg = load_default_config().llm
    cfg.endpoint = 'https://openrouter.ai/api/v1'
    cfg.routing.require_parameters = False
    build_provider(cfg).complete('Hi', response_format={
        'type': 'json_schema', 'json_schema': {'name': 'test', 'strict': True, 'schema': {'type': 'object'}},
    })
    assert completion.call_args.kwargs['extra_body']['provider']['require_parameters'] is True
