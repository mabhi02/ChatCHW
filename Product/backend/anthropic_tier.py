"""Anthropic API key tier pre-flight check.

Fires one tiny Haiku call with the user's BYOK key and reads the rate-limit
headers off the response. Returns a structured tier assessment the server
uses to gate session starts and the frontend uses to display a tier badge.

Why this exists:
    CHW Navigator's chunked catcher path bursts 21 parallel Haiku calls per
    completeness check on full WHO runs. Anthropic Tier 1 (50 RPM / 50K
    input TPM) cannot sustain this — the burst exceeds both the RPM and the
    input TPM ceilings, and the run silently dies in a cloud of 429 errors
    and garbled catcher output. Tier 2+ (1000 RPM / 450K input TPM) handles
    it with margin.

    Before this check, a Tier 1 user would:
      1. Paste their key
      2. Click "Start extraction"
      3. Watch the run burn for ~3 minutes
      4. See "Extraction failed" with no explanation of WHY
      5. Assume the tool is broken

    With this check, a Tier 1 user sees a clear "Your key is Tier 1. This
    tool requires Tier 2 or higher." message BEFORE starting, with direct
    instructions on how to upgrade. Fails fast, actionable.

How it works:
    Anthropic responses include these headers on every API call:
        anthropic-ratelimit-requests-limit
        anthropic-ratelimit-input-tokens-limit
        anthropic-ratelimit-output-tokens-limit

    We fire a single Haiku call with max_tokens=1 (the smallest valid
    output — costs ~$0.0001), read the headers via the SDK's
    with_raw_response helper, and classify the tier based on the limits.

Cost: ~$0.0001 per check. Essentially free.

Notes:
    - The check uses the same model family our catchers use (Haiku 4.5), so
      the returned limits are the ones that actually matter for chunked
      catcher performance.
    - Anthropic's tier ladder has shifted historically; the threshold
      constants below are conservative (Tier 2 starts at 400K input TPM
      even though the documented value is 450K, to give a small buffer for
      any account-specific variation).
    - We do NOT cache the tier result. Each start_session re-runs the check
      because the user may have upgraded their tier between runs, or may
      have pasted a different key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# Anthropic Haiku 4.5 is the model whose rate limits matter for chunked
# catchers. Using Haiku 4.5 for the probe means the returned headers reflect
# exactly the pool our catchers will share at runtime.
_PROBE_MODEL = "claude-haiku-4-5-20251001"

# Minimum tier thresholds. A key is considered "meets_minimum" if all three
# of these are satisfied. Values are slightly below the documented Tier 2
# limits to give a small buffer for account-specific variation.
_MIN_INPUT_TPM = 400_000   # Tier 2 documented: 450K
_MIN_OUTPUT_TPM = 80_000   # Tier 2 documented: 90K
_MIN_RPM = 500             # Tier 2 documented: 1000

# Tier classification boundaries (by input TPM, which is the most reliable
# proxy across Anthropic's documented tier ladders). These are ballpark —
# the authoritative source is the Claude Console UI.
_TIER_THRESHOLDS = [
    (2_000_000, 4),   # Tier 4: 2M+ input TPM
    (900_000, 3),     # Tier 3: 900K+
    (400_000, 2),     # Tier 2: 400K+
    (0, 1),           # Tier 1: anything below
]


@dataclass
class TierCheckResult:
    """Structured result of an Anthropic tier probe.

    Attributes:
        tier: Integer tier level (1-4). None if the probe failed entirely.
        requests_limit: RPM ceiling reported by Anthropic for the probed model.
        input_tpm: Input tokens per minute ceiling (excluding cache reads).
        output_tpm: Output tokens per minute ceiling.
        meets_minimum: True iff the key can run chunked-catcher extractions.
        message: Human-readable status message for the frontend banner.
        error: Error details if the probe failed (auth error, network, etc.).
    """
    tier: int | None
    requests_limit: int
    input_tpm: int
    output_tpm: int
    meets_minimum: bool
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "requestsLimit": self.requests_limit,
            "inputTpm": self.input_tpm,
            "outputTpm": self.output_tpm,
            "meetsMinimum": self.meets_minimum,
            "message": self.message,
            "error": self.error,
        }


def _classify_tier(input_tpm: int) -> int:
    for threshold, tier in _TIER_THRESHOLDS:
        if input_tpm >= threshold:
            return tier
    return 1


def _build_message(tier: int, meets_minimum: bool) -> str:
    """Build the user-facing status message based on tier.

    Kept in this module so the frontend doesn't have to duplicate tier-level
    copy — the backend owns the canonical messaging.
    """
    if not meets_minimum:
        return (
            f"Your Anthropic API key is Tier {tier}. CHW Navigator requires "
            f"Tier 2 or higher for full-guide extractions. Options:\n\n"
            f"  1. Upgrade your Anthropic account — auto-upgrades after $40 "
            f"cumulative spend + 7 days since your first purchase. Check "
            f"https://console.anthropic.com/settings/limits for your current "
            f"status.\n\n"
            f"  2. Use the WHO sample guide (6 pages) — chunking is not "
            f"triggered on small guides, so Tier 1 keys can still run sample "
            f"extractions.\n\n"
            f"Why: chunked catcher validation on full guides bursts ~21 "
            f"parallel Haiku calls per round, which exceeds the Tier 1 rate "
            f"limit of 50 requests per minute."
        )
    if tier == 2:
        return (
            "Your Anthropic key is Tier 2 — ready to run full-guide "
            "extractions. You have 450K input TPM and 90K output TPM, which "
            "supports up to 3 concurrent extractions on the deployed instance."
        )
    if tier == 3:
        return (
            "Your Anthropic key is Tier 3 — ready to run full-guide "
            "extractions with generous headroom. You have ~900K input TPM "
            "and ~180K output TPM, which supports 6+ concurrent extractions."
        )
    if tier == 4:
        return (
            "Your Anthropic key is Tier 4 — maximum auto-upgrade tier. "
            "You have ~2M input TPM and ~400K output TPM, effectively "
            "unlimited for CHW Navigator's workload."
        )
    return f"Your Anthropic key is Tier {tier} — ready to run."


async def check_anthropic_tier(api_key: str) -> TierCheckResult:
    """Probe the Anthropic API with a tiny call and classify the key's tier.

    Args:
        api_key: The user's BYOK Anthropic key.

    Returns:
        TierCheckResult with tier classification and a user-facing message.
        On probe failure (auth error, network issue, etc.), returns a result
        with tier=None, meets_minimum=False, and an `error` field describing
        what went wrong — the caller should surface this to the user as a
        key-validation error rather than as a tier issue.

    Cost: ~$0.0001 per call (one Haiku call, 1-output-token max).
    """
    if not api_key or not api_key.startswith("sk-ant-"):
        return TierCheckResult(
            tier=None,
            requests_limit=0,
            input_tpm=0,
            output_tpm=0,
            meets_minimum=False,
            message="API key is missing or malformed (must start with 'sk-ant-').",
            error="invalid_key_format",
        )

    client = AsyncAnthropic(api_key=api_key)
    try:
        # with_raw_response gives us access to the HTTP-level response
        # including headers, which we need to read the rate-limit values.
        # The .parse() method on the returned object would give us the
        # typed Message back, but we don't need the content — only the
        # headers matter for the tier check.
        raw_response = await client.messages.with_raw_response.create(
            model=_PROBE_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
    except Exception as exc:
        # Auth errors, network errors, API errors all land here. We map
        # them into a user-facing message without leaking internal details.
        exc_name = type(exc).__name__
        exc_str = str(exc).lower()
        if "authentication" in exc_str or "invalid" in exc_str or "401" in exc_str:
            return TierCheckResult(
                tier=None,
                requests_limit=0,
                input_tpm=0,
                output_tpm=0,
                meets_minimum=False,
                message=(
                    "Anthropic rejected this API key — it may be invalid, "
                    "revoked, or expired. Check your key at "
                    "https://console.anthropic.com/settings/keys and try again."
                ),
                error=f"auth_error:{exc_name}",
            )
        logger.warning("Tier check probe failed: %s: %s", exc_name, exc)
        return TierCheckResult(
            tier=None,
            requests_limit=0,
            input_tpm=0,
            output_tpm=0,
            meets_minimum=False,
            message=(
                "Could not reach the Anthropic API to verify your key. "
                "Please check your network connection and try again."
            ),
            error=f"probe_error:{exc_name}",
        )

    # Extract rate-limit headers. Anthropic uses lower-case header names in
    # the SDK's response object. Values are strings; we coerce to int with
    # a fallback to 0 on parse failure (means "unknown", treated as low tier).
    headers = raw_response.headers

    def _get_int(name: str) -> int:
        raw = headers.get(name, "0") or "0"
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 0

    requests_limit = _get_int("anthropic-ratelimit-requests-limit")
    input_tpm = _get_int("anthropic-ratelimit-input-tokens-limit")
    output_tpm = _get_int("anthropic-ratelimit-output-tokens-limit")

    tier = _classify_tier(input_tpm)
    meets_minimum = (
        input_tpm >= _MIN_INPUT_TPM
        and output_tpm >= _MIN_OUTPUT_TPM
        and requests_limit >= _MIN_RPM
    )

    logger.info(
        "Tier check: tier=%d rpm=%d input_tpm=%d output_tpm=%d meets_minimum=%s",
        tier, requests_limit, input_tpm, output_tpm, meets_minimum,
    )

    return TierCheckResult(
        tier=tier,
        requests_limit=requests_limit,
        input_tpm=input_tpm,
        output_tpm=output_tpm,
        meets_minimum=meets_minimum,
        message=_build_message(tier, meets_minimum),
        error=None,
    )
