"""Characterization tests for `src/llm/claude_cli.py` — ClaudeCLIProvider.

Module under test: shells out to the `claude` CLI binary via subprocess.
All tests mock `subprocess.run` and `shutil.which` so no real binary is
invoked and the suite passes on machines without Claude Code installed.

Behaviors locked:
  - `__init__` reads cfg overrides (binary, extra_args, timeout_seconds)
    and raises LLMError if the binary is not on PATH.
  - `complete()` builds the right subprocess command, combines system+user
    prompts (CLI has no --system flag across all versions), strips stdout,
    raises LLMError on timeout / non-zero exit / empty output.

DO NOT modify claude_cli.py to make a test pass — record surprises instead.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src.llm import LLMError
from src.llm.claude_cli import ClaudeCLIProvider


# --------------------------------------------------------------------------- #
# Init
# --------------------------------------------------------------------------- #

def test_init_uses_defaults_when_cfg_missing_attributes(mocker) -> None:
    """No cfg attributes → binary='claude', extra_args=[], timeout=900."""
    mocker.patch("src.llm.claude_cli.shutil.which", return_value="/usr/local/bin/claude")

    provider = ClaudeCLIProvider(SimpleNamespace())

    assert provider.binary == "claude"
    assert provider.extra_args == []
    assert provider.timeout == 900


def test_init_reads_cfg_overrides_for_binary_extra_args_and_timeout(mocker) -> None:
    """Custom cfg values override the defaults."""
    mocker.patch("src.llm.claude_cli.shutil.which", return_value="/opt/claude")
    cfg = SimpleNamespace(
        binary="/opt/claude",
        extra_args=["--verbose"],
        timeout_seconds=60,
    )

    provider = ClaudeCLIProvider(cfg)

    assert provider.binary == "/opt/claude"
    assert provider.extra_args == ["--verbose"]
    assert provider.timeout == 60


def test_init_raises_llmerror_when_claude_binary_not_on_path(mocker) -> None:
    """No binary at given path → LLMError with hint to install Claude Code."""
    mocker.patch("src.llm.claude_cli.shutil.which", return_value=None)

    with pytest.raises(LLMError, match="claude CLI binary"):
        ClaudeCLIProvider(SimpleNamespace())


# --------------------------------------------------------------------------- #
# complete()
# --------------------------------------------------------------------------- #

def _make_provider(mocker) -> ClaudeCLIProvider:
    mocker.patch("src.llm.claude_cli.shutil.which", return_value="/usr/local/bin/claude")
    return ClaudeCLIProvider(SimpleNamespace())


def test_complete_returns_stripped_stdout_on_success(mocker) -> None:
    """Stdout is returned with surrounding whitespace stripped."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="  hello world  \n", stderr="")

    result = provider.complete("question?")

    assert result == "hello world"


def test_complete_concatenates_system_and_user_prompts_with_double_newline(
    mocker,
) -> None:
    """When system_prompt given, stdin is `{system}\\n\\n{user}` (CLI has no --system flag)."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")

    provider.complete(user_prompt="ask", system_prompt="be brief")

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["input"] == "be brief\n\nask"


def test_complete_passes_only_user_prompt_when_no_system_prompt(mocker) -> None:
    """Empty system_prompt → stdin is just the user_prompt."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")

    provider.complete(user_prompt="just this")

    assert mock_run.call_args.kwargs["input"] == "just this"


def test_complete_builds_command_with_p_and_output_format_text(mocker) -> None:
    """Command = [binary, '-p', '--output-format', 'text', *extra_args]."""
    mocker.patch("src.llm.claude_cli.shutil.which", return_value="/x/claude")
    provider = ClaudeCLIProvider(SimpleNamespace(
        binary="/x/claude", extra_args=["--debug"],
    ))
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")

    provider.complete("hi")

    cmd = mock_run.call_args.args[0]
    assert cmd == ["/x/claude", "-p", "--output-format", "text", "--debug"]


def test_complete_passes_configured_timeout_to_subprocess(mocker) -> None:
    """timeout_seconds is forwarded to subprocess.run's timeout kwarg."""
    mocker.patch("src.llm.claude_cli.shutil.which", return_value="/x/claude")
    provider = ClaudeCLIProvider(SimpleNamespace(timeout_seconds=42))
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")

    provider.complete("hi")

    assert mock_run.call_args.kwargs["timeout"] == 42


def test_complete_raises_llmerror_on_subprocess_timeout(mocker) -> None:
    """TimeoutExpired → LLMError mentioning the configured timeout."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=900)

    with pytest.raises(LLMError, match="timed out after 900"):
        provider.complete("hi")


def test_complete_raises_llmerror_on_nonzero_exit_code(mocker) -> None:
    """CalledProcessError → LLMError including the last 500 chars of stderr."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=2, cmd="claude", stderr="bad things happened",
    )

    with pytest.raises(LLMError, match="claude CLI failed"):
        provider.complete("hi")


def test_complete_raises_llmerror_when_stdout_is_empty(mocker) -> None:
    """Empty stdout (even after timeout-free exit-0) → LLMError."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="   \n", stderr="")

    with pytest.raises(LLMError, match="empty output"):
        provider.complete("hi")


def test_complete_ignores_max_tokens_kwarg(mocker) -> None:
    """max_tokens is accepted but discarded (CLI manages its own context window).

    Verified by calling complete twice with very different max_tokens values
    and asserting the subprocess invocation is byte-identical (same argv,
    same stdin, same kwargs). Proves max_tokens influences NOTHING about
    the subprocess call — stronger than a substring check on argv.

    SURPRISE-aware: this is intentional behavior, not a bug. Lock so any
    future attempt to plumb max_tokens through (e.g. via a new --max-tokens
    CLI arg) breaks here loudly."""
    provider = _make_provider(mocker)
    mock_run = mocker.patch("src.llm.claude_cli.subprocess.run")
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")

    provider.complete("hi", max_tokens=100)
    call_with_100 = (mock_run.call_args.args, mock_run.call_args.kwargs)

    mock_run.reset_mock()
    mock_run.return_value = mocker.MagicMock(stdout="ok", stderr="")
    provider.complete("hi", max_tokens=5000)
    call_with_5000 = (mock_run.call_args.args, mock_run.call_args.kwargs)

    assert call_with_100 == call_with_5000
