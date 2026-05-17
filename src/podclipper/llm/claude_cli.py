"""Claude CLI provider — shells out to the `claude` binary."""

from __future__ import annotations

import logging
import shutil
import subprocess
from types import SimpleNamespace

from .base import LLMError

log = logging.getLogger("ave.llm.cli")


class ClaudeCLIProvider:
    name = "claude_cli"

    def __init__(self, cfg: SimpleNamespace):
        self.binary = getattr(cfg, "binary", "claude")
        self.extra_args = list(getattr(cfg, "extra_args", []) or [])
        self.timeout = int(getattr(cfg, "timeout_seconds", 900))
        if shutil.which(self.binary) is None:
            raise LLMError(
                f"claude CLI binary {self.binary!r} not found on PATH. "
                f"Install Claude Code or point claude_cli.binary at the executable."
            )

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        del max_tokens  # CLI has its own context limits
        # The CLI doesn't accept --system across all versions, so combine prompts.
        combined = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"

        cmd = [self.binary, "-p", "--output-format", "text", *self.extra_args]
        log.debug(f"claude-cli cmd: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                input=combined,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude CLI timed out after {self.timeout}s") from e
        except subprocess.CalledProcessError as e:
            raise LLMError(
                f"claude CLI failed (exit {e.returncode}): {e.stderr.strip()[-500:]}"
            ) from e

        text = result.stdout.strip()
        if not text:
            raise LLMError("claude CLI returned empty output")
        return text
