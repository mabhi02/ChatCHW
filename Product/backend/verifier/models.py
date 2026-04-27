"""Verifier model selection with Anthropic same-family fallback.

The spec prefers an independent GPT verifier. In the user's current
environment, GPT quota is unreliable, so the default runtime behavior
is:
  1. Attempt a cheap GPT probe -- if it succeeds, use GPT and tag
     `verifier_independence = "different-family"`.
  2. Otherwise, fall back to an Anthropic model from the opposite
     family (Opus -> Sonnet, Sonnet -> Opus) and tag
     `verifier_independence = "same-family-fallback"`.

Fallback is not a silent failure: every divergence list from a
same-family verifier has a trailing `same_family_fallback`
`severity: warn` divergence appended so reviewers can see they're
looking at weakened-independence verification.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


GPT_PRIMARY_MODEL = "gpt-5.4-mini"


def select_verifier_model(generator_model: str) -> tuple[str, str, str | None]:
    """Return (model_id, independence_label, openai_api_key_or_None).

    The third tuple element is the OpenAI API key the probe verified as
    billable, returned to the caller so the runner can pass it to the
    OpenAI client without going through `os.environ`. None when the
    fallback is in effect.
    """
    ok, key = _probe_gpt()
    if ok:
        return GPT_PRIMARY_MODEL, "different-family", key
    logger.warning("verifier: GPT unavailable; falling back to same-family Anthropic verifier")
    if generator_model.startswith("claude-opus"):
        return "claude-sonnet-4-6", "same-family-fallback", None
    if generator_model.startswith("claude-sonnet"):
        return "claude-opus-4-6", "same-family-fallback", None
    return "claude-sonnet-4-6", "same-family-fallback", None


def _probe_gpt() -> tuple[bool, str | None]:
    """Cheap synchronous probe.

    Returns (True, working_key) on success, (False, None) on missing key,
    HTTP error, or timeout. Does NOT write to os.environ -- the caller is
    responsible for using the returned key.
    """
    for env_name in ("OPENAI_API_KEY", "OPENAI_API_KEY_ALT"):
        key = os.environ.get(env_name, "")
        if not key:
            continue
        body = json.dumps({
            "model": GPT_PRIMARY_MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "max_completion_tokens": 4,
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as _:
                return True, key
        except urllib.error.HTTPError as exc:
            logger.info("verifier: GPT probe on %s returned HTTP %s", env_name, exc.code)
            continue
        except Exception as exc:
            logger.info("verifier: GPT probe on %s failed: %s", env_name, exc)
            continue
    return False, None
