"""Unified LLM provider via the litellm SDK."""

from __future__ import annotations

from types import SimpleNamespace


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, cfg: SimpleNamespace, model: str):
        self.model = model
