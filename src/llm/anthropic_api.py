"""Anthropic API provider — uses the official `anthropic` SDK."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

from .base import LLMError

log = logging.getLogger("ave.llm.api")


class AnthropicAPIProvider:
    name = "anthropic_api"

    def __init__(self, cfg: SimpleNamespace, model: str):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMError(
                "anthropic SDK not installed — run `pip install anthropic`"
            ) from e

        api_key_env = getattr(cfg, "api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMError(
                f"env var {api_key_env} not set; cannot use anthropic_api provider"
            )

        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt or None,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:  # noqa: BLE001 — SDK raises many exception types
            raise LLMError(f"anthropic API call failed: {e}") from e

        if not message.content:
            raise LLMError("anthropic API returned empty content")

        # Concatenate text blocks (ignores any non-text tool-use blocks)
        parts = [b.text for b in message.content if getattr(b, "type", None) == "text"]
        text = "".join(parts).strip()
        if not text:
            raise LLMError("anthropic API returned no text content")
        return text
