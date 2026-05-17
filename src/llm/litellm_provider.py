"""Unified LLM provider via the litellm SDK."""

from __future__ import annotations

from types import SimpleNamespace

from .base import LLMError


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, cfg: SimpleNamespace, model: str):
        self.model = model

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
    ) -> str:
        import litellm

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"litellm call failed: {e}") from e
        return response.choices[0].message.content
