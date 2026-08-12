"""LLM provider runner — adapts different LLM APIs to a common interface."""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx


class LLMRunner:
    """Unified interface for calling LLMs across providers."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self.total_tokens = 0
        self.total_calls = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the response text."""
        provider = self._detect_provider()
        if provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider for model: {self.model}")

    def _detect_provider(self) -> str:
        model_lower = self.model.lower()
        if any(m in model_lower for m in ["gpt", "o1", "o3", "o4"]):
            return "openai"
        if any(m in model_lower for m in ["claude", "sonnet", "haiku", "opus"]):
            return "anthropic"
        # Default: try openai-compatible
        return "openai"

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set (pass --api-key or set env var)")

        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(api_key=api_key, **kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.kwargs.get("temperature", 0.7),
            max_tokens=self.kwargs.get("max_tokens", 2048),
        )
        self.total_tokens += response.usage.total_tokens if response.usage else 0
        self.total_calls += 1
        content = response.choices[0].message.content or ""
        # Reasoning models may put output in reasoning_content with empty content
        if not content and hasattr(response.choices[0].message, "reasoning_content"):
            reasoning = response.choices[0].message.reasoning_content or ""
            if reasoning:
                content = reasoning
        return content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.kwargs.get("max_tokens", 2048),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.kwargs.get("temperature", 0.7),
        )
        tokens = (response.usage.input_tokens + response.usage.output_tokens
                   if response.usage else 0)
        self.total_tokens += tokens
        self.total_calls += 1
        return response.content[0].text if response.content else ""


class MockLLMRunner:
    """Mock runner for testing — returns canned responses."""

    def __init__(self, responses: Optional[list[str]] = None):
        self.responses = responses or []
        self.call_count = 0
        self.total_tokens = 0
        self.total_calls = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = '{"action": "skip", "reasoning": "no response configured"}'
        self.call_count += 1
        self.total_calls += 1
        return resp
