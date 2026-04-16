"""LLM provider abstraction.

A provider exposes a single `complete(user_prompt, system_prompt)` method.
Everything downstream (analyze.py, future vision-enabled passes) treats
providers identically — you pick one via config.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """Raised by any provider when a call fails unrecoverably."""


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt and return the raw assistant text.

        Callers (e.g. analyze.py) are responsible for parsing structured output
        out of the response.
        """
        ...
