"""LLM provider protocol — exposes a single `complete(...)` method."""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    pass


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt and return the raw assistant text."""
        ...
