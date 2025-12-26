"""
OpenAI-compatible chat client with tool/function calling.

Works with:
- OpenAI (https://api.openai.com/v1)
- DeepSeek (https://api.deepseek.com) [OpenAI-compatible]
- OpenRouter (https://openrouter.ai/api/v1)
...and many other OpenAI-compatible gateways.

Environment variables (recommended):
- LLM_PROVIDER: openai | openrouter | deepseek | (anything else => openai defaults)
- LLM_API_KEY: API key for the chosen provider (or gateway)
- LLM_BASE_URL: override the API base URL (e.g. https://openrouter.ai/api/v1)
- OPENROUTER_API_KEY: fallback key if using OpenRouter and LLM_API_KEY is unset
- OPENROUTER_REFERER: optional, for OpenRouter app attribution (HTTP-Referer)
- OPENROUTER_TITLE: optional, for OpenRouter app attribution (X-Title)

Notes:
- OpenRouter's OpenAI-compatible endpoint is https://openrouter.ai/api/v1. Their attribution
  headers are optional. See OpenRouter docs for details.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


def _looks_like_openrouter(base_url: str) -> bool:
    return "openrouter.ai" in (base_url or "").lower()


def _json_loads_maybe(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


class LLMClient:
    """
    Minimal wrapper around the official OpenAI Python SDK that also works with OpenAI-compatible gateways.

    This harness uses Chat Completions + tool calling, which OpenRouter supports on its OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()

        # Resolve base_url
        base_url = base_url or os.environ.get("LLM_BASE_URL")
        if not base_url:
            if provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            elif provider == "deepseek":
                base_url = "https://api.deepseek.com"
            else:
                base_url = "https://api.openai.com/v1"

        # Resolve api_key (support provider-specific fallbacks)
        api_key = api_key or os.environ.get("LLM_API_KEY")
        if not api_key and _looks_like_openrouter(base_url):
            api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "Missing API key. Set LLM_API_KEY (or OPENROUTER_API_KEY if using OpenRouter)."
            )

        # Optional OpenRouter attribution headers
        headers: Dict[str, str] = {}
        if default_headers:
            headers.update(default_headers)

        if _looks_like_openrouter(base_url):
            referer = os.environ.get("OPENROUTER_REFERER")
            title = os.environ.get("OPENROUTER_TITLE")
            if referer:
                headers.setdefault("HTTP-Referer", referer)
            if title:
                headers.setdefault("X-Title", title)

        # Instantiate OpenAI SDK client with optional custom headers
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        max_output_tokens: int = 700,
        temperature: float = 0.2,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          - content: assistant text (may be "")
          - tool_calls: list of tool call dicts (may be [])
          - raw: full response object (dict)
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # OpenAI-compatible endpoints generally accept `max_tokens` for chat completions.
        kwargs["max_tokens"] = max_output_tokens

        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Allow passing routing/provider-specific params (useful for OpenRouter).
        # You can also set LLM_EXTRA_BODY_JSON='{"provider":{"require_parameters":true}}' etc.
        body_from_env = _json_loads_maybe(os.environ.get("LLM_EXTRA_BODY_JSON"))
        merged_extra: Dict[str, Any] = {}
        if body_from_env:
            merged_extra.update(body_from_env)
        if extra_body:
            merged_extra.update(extra_body)
        if merged_extra:
            kwargs["extra_body"] = merged_extra

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                )

        return {"content": msg.content or "", "tool_calls": tool_calls, "raw": resp.model_dump()}
