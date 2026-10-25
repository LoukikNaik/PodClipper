"""Unified LLM provider via the litellm SDK."""

from __future__ import annotations

import logging
import os
import base64
from types import SimpleNamespace
from urllib.parse import urlparse

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
        self.api_key = os.environ.get("LLM_API_KEY")
        from ..config import ns_to_dict
        self.routing = ns_to_dict(getattr(cfg, "routing", {}))

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        *,
        images: list[bytes] | None = None,
        response_format: dict | None = None,
    ) -> str:
        import litellm

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = user_prompt
        if images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(value).decode("ascii")}}
                for value in images)
        messages.append({"role": "user", "content": content})

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
        if response_format is not None:
            kwargs["response_format"] = response_format
        openrouter = bool(self.api_base) and urlparse(self.api_base).hostname == "openrouter.ai"
        strict_schema = (response_format is not None and response_format.get("type") == "json_schema"
                         and response_format.get("json_schema", {}).get("strict") is True)
        if (self.routing and (openrouter or self.routing.get("only") or self.routing.get("order"))) or (openrouter and strict_schema):
            routing = {key: value for key, value in self.routing.items()
                       if key not in {"only", "order"} or value}
            if openrouter and strict_schema:
                routing["require_parameters"] = True
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
