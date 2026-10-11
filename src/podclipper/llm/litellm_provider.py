"""Unified LLM provider via the litellm SDK."""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

from .base import LLMError


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, cfg: SimpleNamespace, model: str):
        # litellm dumps full request payloads at INFO/DEBUG; combined with our
        # Rich handler it deadlocks on long prompts (~30KB+). Clamp the level.
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        self.model = model if model.startswith("openai/") else f"openai/{model}"
        self.api_base = getattr(cfg, "endpoint", None)
        self.timeout = getattr(cfg, "timeout_seconds", None)
        self.num_retries = getattr(cfg, "num_retries", None)
        self.retry_max_tokens = getattr(cfg, "retry_max_tokens", 32768)
        api_key_env = getattr(cfg, "api_key_env", None)
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        from ..config import ns_to_dict
        self.routing = ns_to_dict(getattr(cfg, "routing", {}))

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

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.num_retries is not None:
            kwargs["num_retries"] = self.num_retries
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.routing.get("only") or self.routing.get("order"):
            routing = {key: value for key, value in self.routing.items()
                       if key not in {"only", "order"} or value}
            kwargs["extra_body"] = {"provider": routing}

        for attempt in range(2):
            try:
                response = litellm.completion(**kwargs)
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"litellm call failed: {e}") from e
            if response.choices[0].finish_reason != "length":
                break
            retry_budget = min(kwargs["max_tokens"] * 2, self.retry_max_tokens)
            if attempt == 1 or retry_budget <= kwargs["max_tokens"]:
                raise LLMError(
                    f"LLM response was truncated at {kwargs['max_tokens']} tokens. "
                    "Increase llm.max_tokens and llm.retry_max_tokens in podclipper.yaml "
                    "if your endpoint supports a larger output budget."
                )
            logging.getLogger(__name__).warning(
                "LLM response was truncated; retrying once with %s tokens (may increase API cost)",
                retry_budget,
            )
            kwargs["max_tokens"] = retry_budget
        text = response.choices[0].message.content or ""
        if not text.strip():
            raise LLMError("litellm returned empty content")
        return text
