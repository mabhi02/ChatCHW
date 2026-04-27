"""Per-artifact verifier.

Reads one artifact + its generator prompts + the source manual text,
asks an independent model whether the artifact represents the source
correctly, and returns a `VerificationBlock` with zero or more
`Divergence` records.

The verifier NEVER mutates the artifact. If GPT is unavailable the call
falls back to a same-family Anthropic model and a
`type: "same_family_fallback"` divergence is appended so reviewers see
the weaker-independence case explicitly.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.provenance.schema import Divergence, VerificationBlock
from backend.verifier.models import select_verifier_model

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = (
    "You are an independent verifier of a medical decision-logic extraction pipeline.\n"
    "You receive (a) excerpts of the source clinical guideline, (b) the artifact\n"
    "produced by another AI system, and (c) the prompts that produced it.\n\n"
    "Your job:\n"
    "1. Read the source.\n"
    "2. Read the artifact.\n"
    "3. Decide whether the artifact represents the source correctly.\n"
    "4. For each disagreement, emit a divergence object:\n"
    '   {"type": "...", "severity": "info|warn|error", "detail": "...", "evidence": {...}}.\n\n'
    "Severity guide:\n"
    '- "error": clinical correctness compromised (wrong threshold, missing danger sign).\n'
    '- "warn":  structural issue that could mislead reviewers (duplicate predicate,\n'
    "           missing units, ambiguous wording).\n"
    '- "info":  cosmetic / stylistic.\n\n'
    "OUTPUT FORMAT (NON-NEGOTIABLE):\n"
    "Respond with EXACTLY ONE JSON object and NOTHING ELSE. No prose before or after.\n"
    "No markdown code fences. No commentary. No 'Here is my analysis'.\n"
    "Your entire response must be parseable by json.loads on the first character.\n\n"
    "Required shape:\n"
    '{"agree": true|false, "divergences": [{"type": "...", "severity": "info|warn|error", "detail": "...", "evidence": {}}]}\n\n'
    "If you agree with the artifact, return:\n"
    '{"agree": true, "divergences": []}\n\n'
    "DO NOT edit the artifact. DO NOT propose corrections. Flag, do not fix."
)


def verify_artifact(
    artifact_path: Path,
    *,
    artifact_kind: str,
    artifact_content_sha: str,
    source_manual_text: str,
    generator_prompts: dict[str, Any],
    generator_model: str,
    max_source_chars: int = 50_000,
    max_artifact_chars: int = 30_000,
    max_prompt_chars: int = 10_000,
) -> VerificationBlock:
    """Run one verification pass and return a `VerificationBlock`."""
    try:
        artifact_data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        artifact_data = artifact_path.read_text(encoding="utf-8", errors="replace")

    model_id, independence, openai_key = select_verifier_model(generator_model)
    user_msg = (
        f"=== SOURCE MANUAL (first {max_source_chars} chars) ===\n"
        f"{source_manual_text[:max_source_chars]}\n\n"
        f"=== ARTIFACT ({artifact_kind}) ===\n"
        f"{json.dumps(artifact_data, indent=2, default=str)[:max_artifact_chars]}\n\n"
        f"=== PROMPTS THAT PRODUCED THIS ARTIFACT ===\n"
        f"{json.dumps(generator_prompts, indent=2, default=str)[:max_prompt_chars]}\n\n"
        "Verify."
    )

    try:
        response = _call_verifier(model_id, VERIFIER_SYSTEM_PROMPT, user_msg, openai_key=openai_key)
    except Exception as exc:
        logger.warning("verifier: call failed for %s with %s: %s", artifact_kind, model_id, exc)
        response = {
            "agree": False,
            "divergences": [{
                "type": "verifier_call_failed",
                "severity": "warn",
                "detail": f"Verifier call raised: {exc!r}",
                "evidence": {"model": model_id},
            }],
        }

    divergences: list[Divergence] = []
    for d in response.get("divergences", []) or []:
        try:
            divergences.append(Divergence(**d))
        except Exception:
            divergences.append(Divergence(
                type="malformed_divergence",
                severity="warn",
                detail=f"Verifier returned a malformed divergence: {d!r}",
            ))

    if independence == "same-family-fallback":
        divergences.append(Divergence(
            type="same_family_fallback",
            severity="warn",
            detail=(
                f"Verifier {model_id} is same model family as generator {generator_model}; "
                f"independence weakened."
            ),
            evidence={"verifier_model": model_id, "generator_model": generator_model},
        ))

    has_error = any(d.severity == "error" for d in divergences)
    agree = bool(response.get("agree", False)) and not has_error

    return VerificationBlock(
        artifact_kind=artifact_kind,
        artifact_content_sha256=artifact_content_sha,
        verifier_model=model_id,
        verifier_independence=independence,
        verifier_run_at=datetime.now(timezone.utc).isoformat(),
        agree=agree,
        divergences=divergences,
    )


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _call_verifier(
    model_id: str,
    system_prompt: str,
    user_msg: str,
    openai_key: str | None = None,
) -> dict:
    if model_id.startswith("gpt"):
        return _call_openai(model_id, system_prompt, user_msg, openai_key)
    return _call_anthropic(model_id, system_prompt, user_msg)


def _call_openai(model_id: str, system_prompt: str, user_msg: str,
                 openai_key: str | None = None) -> dict:
    # Prefer the key the probe in `select_verifier_model` confirmed as billable.
    api_key = openai_key or os.environ.get("OPENAI_API_KEY", "")
    from openai import OpenAI
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=2048,
    )
    content = resp.choices[0].message.content or "{}"
    parsed = _extract_json(content)
    if parsed is None:
        return {"agree": False, "divergences": [{
            "type": "malformed_verifier_response",
            "severity": "warn",
            "detail": "OpenAI verifier returned non-JSON content",
            "evidence": {"content_preview": content[:500]},
        }]}
    _track_verifier_usage(model_id, resp.usage.prompt_tokens, resp.usage.completion_tokens, 0, 0)
    return parsed


def _call_anthropic(model_id: str, system_prompt: str, user_msg: str) -> dict:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text if resp.content else "{}"
    parsed = _extract_json(text)
    if parsed is None:
        return {"agree": False, "divergences": [{
            "type": "malformed_verifier_response",
            "severity": "warn",
            "detail": "Anthropic verifier returned non-JSON content",
            "evidence": {"content_preview": text[:500]},
        }]}
    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
    _track_verifier_usage(
        model_id, resp.usage.input_tokens, resp.usage.output_tokens,
        cache_read, cache_write,
    )
    return parsed


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Parse a JSON object out of a possibly-prose-wrapped LLM response.

    Strategy:
      1. Try direct json.loads on the trimmed text.
      2. Strip leading/trailing markdown code fences (```json ... ```).
      3. Scan for the first balanced `{...}` and try to parse that.
    Returns None if no JSON object can be recovered.
    """
    if not text:
        return None
    s = text.strip()

    # Direct parse
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences anywhere in the response.
    # Match ```json ... ``` or ``` ... ``` (non-greedy).
    import re
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

    # Balanced-brace scan: find the first `{` and walk to its matching `}`,
    # respecting string literals and escapes. Skips prose prefixes/suffixes.
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    result = json.loads(candidate)
                    return result if isinstance(result, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Verifier cost tracking
# ---------------------------------------------------------------------------

def _track_verifier_usage(model_id: str, input_tokens: int, output_tokens: int,
                          cache_read: int, cache_write: int) -> None:
    """Feed verifier API usage into the global run-cost accumulator.

    Without this, the 7 verifier calls per gen8 run are billed to the user
    but invisible in `costEstimateUsd` / `callsOpus` / `callsSonnet`. The
    runner uses the official OpenAI/Anthropic SDKs directly, bypassing the
    rlm_runner monkey-patched accumulator, so we have to bridge it manually.
    """
    try:
        from backend.rlm_runner import accumulate_catcher_usage
        accumulate_catcher_usage(
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
    except Exception:
        # Cost tracking is best-effort -- never let it crash a verification.
        pass
