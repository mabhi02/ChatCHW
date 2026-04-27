"""LLM client for the sequential pipeline -- raw API calls with caching.

Uses Anthropic for maker/repair (Sonnet with 1h prompt caching) and
OpenAI for red team (gpt-5.4-mini with auto-cache, 3x majority vote).

No rlm library dependency. This is the raw API layer.
"""

import asyncio
import json
import logging
import os
import time
from collections import Counter
from typing import Any

import anthropic
import openai

logger = logging.getLogger(__name__)


class SequentialLLMClient:
    """Manages LLM calls for the sequential pipeline with prompt caching."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        openai_key: str | None = None,
    ):
        self._anthropic_key = anthropic_key or os.environ.get("ANTHROPIC_KEY", "")
        self._openai_key = openai_key or os.environ.get("OPENAI_API_KEY", "")

        self._anthropic = anthropic.Anthropic(api_key=self._anthropic_key)
        self._openai = openai.OpenAI(api_key=self._openai_key)

        # Cumulative token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_calls = 0
        self.total_cost_usd = 0.0

    async def call_anthropic(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 16384,
        temperature: float = 0,
        cache_system: bool = True,
    ) -> dict:
        """Call Anthropic API with optional prompt caching on system block.

        Returns: {content, input_tokens, output_tokens, cached_tokens, model, duration_ms}
        """
        t0 = time.time()

        system_blocks: list[dict[str, Any]] = []
        if system_prompt:
            block: dict[str, Any] = {"type": "text", "text": system_prompt}
            if cache_system:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_blocks if system_blocks else anthropic.NOT_GIVEN,
                    messages=[{"role": "user", "content": user_message}],
                ),
            )
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            raise

        duration_ms = int((time.time() - t0) * 1000)
        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens
        self.total_calls += 1

        # Cost estimate (Sonnet pricing)
        cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
        if cached_tokens > 0:
            cost -= cached_tokens * 2.7 / 1_000_000  # 90% discount on cached
        if cache_write > 0:
            cost += cache_write * 3.75 / 1_000_000  # 1.25x write cost
        self.total_cost_usd += cost

        logger.info(
            f"Anthropic {model}: {input_tokens}in/{output_tokens}out "
            f"(cached={cached_tokens}, write={cache_write}) "
            f"${cost:.4f} {duration_ms}ms"
        )

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write,
            "model": model,
            "duration_ms": duration_ms,
            "cost_usd": cost,
        }

    async def call_openai(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "gpt-5.4-mini",
        max_completion_tokens: int = 8192,
        temperature: float = 0,
    ) -> dict:
        """Call OpenAI API. Uses max_completion_tokens (gpt-5.x requirement).

        Returns: {content, input_tokens, output_tokens, cached_tokens, model, duration_ms}
        """
        t0 = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                ),
            )
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise

        duration_ms = int((time.time() - t0) * 1000)
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = 0
        if hasattr(response.usage, "prompt_tokens_details") and response.usage.prompt_tokens_details:
            cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cached_tokens += cached_tokens
        self.total_calls += 1

        # Cost estimate (gpt-5.4-mini pricing)
        cost = (input_tokens * 0.75 / 1_000_000) + (output_tokens * 4.50 / 1_000_000)
        if cached_tokens > 0:
            cost -= cached_tokens * 0.5625 / 1_000_000  # 75% discount
        self.total_cost_usd += cost

        logger.info(
            f"OpenAI {model}: {input_tokens}in/{output_tokens}out "
            f"(cached={cached_tokens}) ${cost:.4f} {duration_ms}ms"
        )

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "model": model,
            "duration_ms": duration_ms,
            "cost_usd": cost,
        }

    async def call_openai_majority(
        self,
        system_prompt: str,
        user_message: str,
        n: int = 3,
        model: str = "gpt-5.4-mini",
        max_completion_tokens: int = 8192,
        temperature: float = 0,
    ) -> dict:
        """Call OpenAI N times and return majority-vote result.

        Used for red team (3x majority vote at temperature=0).
        Returns: {content, votes, input_tokens, output_tokens, cached_tokens, duration_ms}
        """
        t0 = time.time()
        tasks = [
            self.call_openai(system_prompt, user_message, model, max_completion_tokens, temperature)
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, dict)]
        if not successes:
            raise RuntimeError(f"All {n} majority vote calls failed")

        # For red team, the "vote" is pass/fail. Extract from JSON response.
        votes = []
        for r in successes:
            try:
                parsed = json.loads(r["content"])
                passed = parsed.get("passed", parsed.get("pass", None))
                if passed is not None:
                    votes.append(bool(passed))
            except (json.JSONDecodeError, KeyError):
                # If not parseable as JSON, treat the whole response as the vote
                content_lower = r["content"].lower()
                votes.append("pass" in content_lower and "fail" not in content_lower)

        majority = Counter(votes).most_common(1)[0][0] if votes else False
        duration_ms = int((time.time() - t0) * 1000)

        total_in = sum(r.get("input_tokens", 0) for r in successes)
        total_out = sum(r.get("output_tokens", 0) for r in successes)
        total_cached = sum(r.get("cached_tokens", 0) for r in successes)
        total_cost = sum(r.get("cost_usd", 0) for r in successes)

        return {
            "content": successes[0]["content"],  # first response as representative
            "passed": majority,
            "votes": votes,
            "n_calls": len(successes),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cached_tokens": total_cached,
            "model": model,
            "duration_ms": duration_ms,
            "cost_usd": total_cost,
        }

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }
