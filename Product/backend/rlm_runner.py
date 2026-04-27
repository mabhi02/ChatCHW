"""RLM Runner -- configures and runs the rlms library for clinical extraction.

Wraps the rlms library with our custom system prompt, structural validators,
Z3 verifier, and the hybrid-plan per-phase artifact validators.

Exposes these functions in the REPL namespace:
  - validate(logic)        structural validators on a full clinical_logic dict
  - z3_check(logic)        Z3 exhaustiveness / reachability proofs
  - emit_artifact(name, v) auto-validating checkpoint between phases

`emit_artifact` is the hybrid plan's key contribution. The model MUST call it
between phases (see EXTRACTION_STRATEGY in backend/system_prompt.py). Each
call persists the artifact to disk + Neon AND synchronously runs the catcher
validators for that artifact type. If the validators return critical_issues,
the call's return value tells the model to fix and re-emit before proceeding.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from rlm import RLM
from rlm.logger import RLMLogger

from backend.db import create_intermediate_artifact
from backend.system_prompt import build_system_prompt, build_initial_user_message
from backend.validators import run_all_validators
from backend.validators.phases import (
    valid_artifact_names,
    validate_artifact,
    PhaseValidationResult,
    _ARTIFACT_PHASES,
)
from backend.z3_verifier import verify_clinical_logic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# rlm library monkey-patch: bounded format_iteration truncation
# ---------------------------------------------------------------------------
# The library's format_iteration (rlm.utils.parsing) defaults to preserving
# up to 20,000 characters of REPL output per code block in the message history
# passed to every subsequent LLM iteration.
#
# RAISED 2026-04-11: 20000 → 60000 chars. The 20K default was the single
# biggest silent-recall risk in the model's self-view across iterations.
# When the model did `print(json.dumps(context['sections'], indent=2))` to
# inspect sections past its initial metadata, anything past ~character 20000
# got silently dropped from the model's view on the NEXT iteration. The model
# thought it had read the output; the model never saw the tail. At 60K chars
# (~15K tokens) per code block, all realistic section-inspection prints fit.
#
# Why 60K and not larger: per-code-block stdout accumulates across iterations
# in the compaction=False message history. With 50 iterations × 60K per block
# worst case = 3M chars ≈ 750K tokens, still well under the 1M context limit
# and matching Sonnet 4.6 / Opus 4.6 headroom. Combined with the system prompt
# telling the model "inspect via slicing, don't print giant dicts" the real
# usage is far below this cap.
#
# The cap still LOGS a warning when truncation fires, so the gate run surfaces
# any case where the model is accidentally approaching the cap.
#
# The patch is idempotent: applying it twice is a no-op.
try:
    from rlm.utils import parsing as _rlm_parsing
    _original_format_iteration = _rlm_parsing.format_iteration

    # The cap can be overridden via env var for experiments (e.g., gate runs
    # that want to measure the effect of different caps on recall).
    import os as _os
    _FORMAT_ITERATION_CAP = int(_os.environ.get("RLM_STDOUT_CAP_CHARS", "60000"))

    def _format_iteration_bounded(iteration, max_character_length: int = _FORMAT_ITERATION_CAP):
        """Wrapped format_iteration that LOGS when truncation actually fires.

        Cap raised from library default 20K → 60K chars (2026-04-11) to
        give the model a meaningfully larger scratchpad view of its own
        past stdout across iterations. The wrapper instruments truncation
        events so we can detect if even the raised cap is biting.
        """
        result = _original_format_iteration(iteration, max_character_length=max_character_length)
        # Heuristic: if the result is at or near the cap, the underlying
        # call almost certainly truncated. Log a WARNING so the gate run
        # surfaces it. Threshold is 95% of cap to allow for the truncation
        # marker the library appends.
        if isinstance(result, str) and len(result) >= int(max_character_length * 0.95):
            logger.warning(
                "format_iteration cap likely triggered: result_len=%d, cap=%d. "
                "Model may be seeing TRUNCATED stdout. If this fires repeatedly, "
                "the system prompt's 'build incrementally' instruction is being ignored.",
                len(result), max_character_length,
            )
        return result

    # Only patch once — check via a sentinel attribute
    if not getattr(_rlm_parsing.format_iteration, "_chw_patched", False):
        _format_iteration_bounded._chw_patched = True  # type: ignore[attr-defined]
        _rlm_parsing.format_iteration = _format_iteration_bounded
        # Also patch the binding imported into rlm.core.rlm (already-imported name)
        try:
            from rlm.core import rlm as _rlm_core
            _rlm_core.format_iteration = _format_iteration_bounded
        except Exception as _exc:  # pragma: no cover - defensive
            logger.debug("format_iteration core patch skipped: %s", _exc)
        logger.info(
            "Patched rlm format_iteration: cap=%d chars (raised from library default 20K "
            "to give model more scratchpad recall across iterations) + truncation warning",
            _FORMAT_ITERATION_CAP,
        )
except ImportError as _exc:  # pragma: no cover - rlm library missing
    logger.warning("Could not patch rlm.utils.parsing.format_iteration: %s", _exc)


# ---------------------------------------------------------------------------
# rlm library monkey-patch: Anthropic prompt caching on the system prompt
# ---------------------------------------------------------------------------
# The rlm library's AnthropicClient passes the system prompt as a plain string
# to `messages.create`, which means the 17K-token system prompt is re-processed
# from scratch on every one of our ~50 iterations. Anthropic supports prompt
# caching via cache_control markers on individual text blocks; marking the
# system prompt as cached gives us:
#
#   Cost math for a 50-iter run, 17K-token system prompt on Sonnet
#   ($3/Mtok input):
#     Without cache: 50 × 17K × $3/Mtok = $2.55 in system-prompt input alone
#     With cache (first call writes at 1.25x, rest read at 0.1x):
#       1 × 17K × $3/Mtok × 1.25 + 49 × 17K × $3/Mtok × 0.1
#       = $0.064 + $0.250 = $0.31
#     Savings: ~$2.24 per run on the system prompt alone
#
#   Latency:
#     First iteration: ~25% slower prefill (cache write path)
#     Iterations 2-50: cached prefix is skipped during prefill, net ~15-30%
#       faster time-to-first-token on each iteration
#     Overall wall clock: faster on average since 49 of 50 iterations benefit
#
# The cache TTL is set to "1h" rather than the default 5-minute ephemeral.
# Reason: a single gate run executes 6 sequential extractions back-to-back
# (3 hybrid + 3 all-at-once), each ~12 minutes, total ~75 minutes. With 5m
# TTL the cache would expire and have to be re-written ~14 times across the
# gate. With 1h TTL the cache survives the whole gate (refreshed on use,
# not on creation, per Anthropic docs). The 1h cache write costs 2x base
# (vs 1.25x for 5m) but is paid ONCE per gate instead of per extraction,
# so net cost is lower across a multi-extraction gate run. Same content,
# same cache state across all 6 extractions = stationary baseline.
#
# The patch wraps AnthropicClient.completion and AnthropicClient.acompletion
# to transform the string `system` argument into the Anthropic list-of-blocks
# form with cache_control={"type": "ephemeral"}. Idempotent.
try:
    import httpx as _httpx
    from rlm.clients import anthropic as _rlm_anthropic
    import anthropic as _anthropic_sdk

    _original_init = _rlm_anthropic.AnthropicClient.__init__
    _original_completion = _rlm_anthropic.AnthropicClient.completion
    _original_acompletion = _rlm_anthropic.AnthropicClient.acompletion

    # Adaptive model routing: Opus 4.6 for complex reasoning phases,
    # Sonnet 4.6 for mechanical execution phases.
    #
    # Opus phases (multi-artifact reasoning, high error surfaces):
    #   - Iteration 0: system prompt parsing + extraction planning
    #   - After modules emit: integrative table (cross-module comorbidity)
    #   - After integrative emit: self-validation repair (holistic fixes)
    #   - After modules fail: module DMN repair (complex table logic)
    #
    # Sonnet phases (single-artifact, mechanical):
    #   - supply_list, variables, predicates, router, phrase_bank
    #   - Simple re-emits after catcher fixes
    _SONNET = "claude-sonnet-4-6"

    # Artifacts that trigger Opus upshift for the NEXT iteration.
    # When emit_artifact fires for one of these AND has critical issues,
    # the model needs Opus-level reasoning to fix them.
    _OPUS_TRIGGER_ARTIFACTS = {"modules", "integrative"}
    # Also upshift when the model enters the self-validation repair loop
    # (it just emitted phrase_bank and is about to assemble + validate).
    _OPUS_TRIGGER_FINAL = {"phrase_bank"}

    # Patch __init__ to use a 10-minute timeout so the SDK's non-streaming
    # guard doesn't block on high max_tokens values.
    # Shared mutable dict for cross-closure signaling between emit_artifact
    # (which knows what phase just ran) and _completion_with_cache (which
    # picks the model). Both run in the same worker thread so no lock needed.
    _model_router_signal: dict = {"use_opus_next": True}

    # Cumulative usage accumulator for live cost tracking. Updated on every
    # completion call. Read by on_iteration_complete to publish live snapshots
    # to the SSE stream. Reset at the start of each run.
    #
    # `calls_haiku` tracks catcher calls made by backend/validators/phases.py
    # (which does NOT go through our patched AnthropicClient). The catcher
    # code imports and writes to this dict directly via accumulate_catcher_usage().
    _run_usage: dict = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": 0.0,
        "calls_opus": 0,
        "calls_sonnet": 0,
        "calls_haiku": 0,
    }

    # Holder for the current run's guide JSON. Set at the start of
    # run_extraction() and cleared in the finally path. When populated,
    # _completion_with_cache / _acompletion_with_cache append the full guide
    # as a SECOND cached system block (1h TTL) alongside the system prompt,
    # so all Opus/Sonnet calls get direct guide access from cache (~$0.50/MTok
    # read) instead of sub-querying via llm_query() on every iteration.
    #
    # Size discipline: Opus 4.6 and Sonnet 4.6 are natively 1M context (no
    # beta header required). The system prompt is ~7K tokens, the WHO guide
    # is ~493K tokens, total well under 1M. Cache writes cost 2x for 1h TTL
    # and are paid once per run; every subsequent call reads at 0.1x.
    _current_guide_json: dict | None = None
    # Sentinel to log the guide caching only on the first call per run.
    _guide_cache_logged: dict = {"logged": False}

    # Gen 7 extension: cached content beyond the guide JSON. Set once per
    # Gen 7 run before rlm.completion() is invoked; cleared at run end.
    #   _current_gen7_guide_text: the reconstructed source guide text
    #     (concatenated chunk text) — attached as one cache block in the
    #     system prompt so the REPL's Module Maker sub-calls see the same
    #     full-text context that the labeler saw.
    #   _current_gen7_labels: the deduped label list (full label objects
    #     with span/id/type/quote_context/source_section_ids) — attached as
    #     a second cache block so every REPL turn + every llm_query_batched
    #     sub-call reads the whole label inventory at cache-read rates.
    _current_gen7_guide_text: str | None = None
    _current_gen7_labels: list | None = None
    _gen7_blocks_logged: dict = {"logged": False}
    # Rough safety cap (chars, not tokens): if serialized guide + system
    # prompt exceed this threshold, fall back to fewer guide repetitions
    # or skip the guide block entirely. Raised from 3.2M → 3.5M in gen 2.3
    # to accommodate 2× prompt repetition on full WHO (1.58M raw × 2 +
    # system prompt ≈ 3.2M chars, fits under 3.5M with margin).
    #
    # Context budget math at 3.5M chars:
    #   cached content: 3.5M chars ≈ 875K tokens
    #   + user msg accumulator: ~100K tokens (growing over iterations)
    #   + output max: 16K tokens
    #   = ~991K tokens total ≤ 1M Opus/Sonnet 4.6 native context
    # Tight but fits. Guides > 1.75M chars raw will fall back to 1×.
    _GUIDE_CACHE_MAX_CHARS = 3_500_000

    # Gen 2.3 guide prompt repetition count. Per Google 2512.14982, repeating
    # the cached guide 2× inside the system block gives attention-restructuring
    # benefits for multi-needle retrieval tasks without the cost/size overhead
    # of 3×. 3× would push full WHO over the context ceiling. 2× fits.
    # The _maybe_attach_guide_block function attempts this many repetitions,
    # falling back to 1× if the result would exceed _GUIDE_CACHE_MAX_CHARS.
    _GUIDE_REPETITION_N = int(_os.environ.get("GENERATOR_GUIDE_REPETITION", "2"))

    # Per-session TOC cache. Keys are SHA256 of guide content. Values are
    # the Opus-generated table-of-contents preamble (rich domain summary).
    # Populated lazily by _generate_guide_toc at session start and read
    # synchronously by _maybe_attach_guide_block on every main call.
    # Negative cache (empty string) is valid: on TOC generation failure
    # we store "" so we don't retry on every call.
    _toc_cache: dict[str, str] = {}

    # Cost per MTok by model family.
    # Format: (input, output, cache_read, cache_write_1h)
    #
    # IMPORTANT: cache_write_1h is 2.0x the input rate because we set
    # `ttl: "1h"` on cache_control. The 5-minute TTL rate would be 1.25x,
    # but we never use that. Using the wrong multiplier underestimates the
    # cache write cost on the first call of each run by ~60%.
    #
    # Rates from Anthropic platform docs (2026-04):
    #   - Opus 4.6:   $5 in / $25 out / $0.50 read / $10.00 write (2x)
    #   - Sonnet 4.6: $3 in / $15 out / $0.30 read / $6.00 write (2x)
    #   - Haiku 4.5:  $1 in / $5 out  / $0.10 read / $2.00 write (2x)
    _COST_TABLE = {
        "claude-opus-4-6": (5.0, 25.0, 0.5, 10.0),
        "claude-sonnet-4-6": (3.0, 15.0, 0.3, 6.0),
        "claude-haiku-4-5-20251001": (1.0, 5.0, 0.1, 2.0),
        "claude-haiku-4-5": (1.0, 5.0, 0.1, 2.0),
    }

    def _accumulate_run_usage(response, model: str) -> None:
        """Tally tokens and cost from an Anthropic response into _run_usage."""
        try:
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

            _run_usage["input_tokens"] += input_tokens
            _run_usage["output_tokens"] += output_tokens
            _run_usage["cached_tokens"] += cached_tokens
            _run_usage["cache_write_tokens"] += cache_write

            # Fallback rates default to Opus (safest overestimate if the
            # model name doesn't match the table, so we never silently
            # underbill the user).
            in_price, out_price, cache_read_price, cache_write_price = _COST_TABLE.get(
                model, (5.0, 25.0, 0.5, 10.0)
            )
            # Billed input = input_tokens minus cached (cached charged at lower rate)
            billed_input = max(0, input_tokens - cached_tokens - cache_write)
            call_cost = (
                (billed_input / 1_000_000) * in_price
                + (cached_tokens / 1_000_000) * cache_read_price
                + (cache_write / 1_000_000) * cache_write_price
                + (output_tokens / 1_000_000) * out_price
            )
            _run_usage["cost_usd"] += call_cost

            model_lower = (model or "").lower()
            if "opus" in model_lower:
                _run_usage["calls_opus"] += 1
            elif "haiku" in model_lower:
                _run_usage["calls_haiku"] += 1
            else:
                _run_usage["calls_sonnet"] += 1

            logger.info(
                "Usage accumulator: +%d in / +%d out / +%d cached / +$%.4f "
                "(run total: $%.2f, %d calls)",
                input_tokens, output_tokens, cached_tokens, call_cost,
                _run_usage["cost_usd"],
                _run_usage["calls_opus"] + _run_usage["calls_sonnet"],
            )
        except Exception as exc:
            logger.error("_accumulate_run_usage failed: %s", exc, exc_info=True)

    def _reset_run_usage() -> None:
        """Reset usage counters at the start of each extraction run."""
        for k in _run_usage:
            _run_usage[k] = 0 if isinstance(_run_usage[k], int) else 0.0
        # Reset the guide-cache logging sentinel so the next run re-logs
        # when the guide is first attached to a system block.
        _guide_cache_logged["logged"] = False

    def _snapshot_run_usage() -> dict:
        """Return a snapshot of the current cumulative usage."""
        return dict(_run_usage)

    def accumulate_catcher_usage(
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Public entry point for any code path that calls Anthropic directly
        (not through the RLM client) to record its usage in the same
        _run_usage accumulator the main extraction path uses. Without this,
        direct Anthropic calls -- catcher validators AND Gen 7 labeler calls --
        are invisible to the frontend Cost Tracker.

        Unlike `_accumulate_run_usage`, this version takes plain integers
        instead of an Anthropic response object, so callers don't need to
        import the rlm library's response shape.

        Fallback rates are Haiku ($1/$5) since the original caller was
        Haiku catchers. Model-specific overrides come from _COST_TABLE.
        Also increments the right model call counter (calls_opus /
        calls_sonnet / calls_haiku) so the UI Model Routing strip is
        accurate across all code paths.
        """
        _run_usage["input_tokens"] += input_tokens
        _run_usage["output_tokens"] += output_tokens
        _run_usage["cached_tokens"] += cached_tokens
        _run_usage["cache_write_tokens"] += cache_write_tokens

        in_price, out_price, cache_read_price, cache_write_price = _COST_TABLE.get(
            model, (1.0, 5.0, 0.1, 2.0)  # default to Haiku rates for catchers
        )
        billed_input = max(0, input_tokens - cached_tokens - cache_write_tokens)
        call_cost = (
            (billed_input / 1_000_000) * in_price
            + (cached_tokens / 1_000_000) * cache_read_price
            + (cache_write_tokens / 1_000_000) * cache_write_price
            + (output_tokens / 1_000_000) * out_price
        )
        _run_usage["cost_usd"] += call_cost

        # Increment the right model call counter so the UI routing strip
        # reflects direct-to-Anthropic calls (labeler, catchers) too.
        model_lower = (model or "").lower()
        if "opus" in model_lower:
            _run_usage["calls_opus"] += 1
        elif "haiku" in model_lower:
            _run_usage["calls_haiku"] += 1
        else:
            _run_usage["calls_sonnet"] += 1

    # -------------------------------------------------------------------
    # Fix B state: per-run catcher pass/fail history for required artifacts
    # -------------------------------------------------------------------
    # Keys are artifact names (supply_list, variables, predicates, modules,
    # router, integrative, phrase_bank). Values are the LATEST pass/fail
    # result (bool) from the most recent emit of that artifact in the
    # current run. Written by _make_emit_artifact_fn after the catcher
    # runs, read by session_manager._run_extraction_task as a soft gate
    # to downgrade the run status if the model called FINAL_VAR while
    # any required artifact never had a passing catcher emit.
    _run_artifact_status: dict[str, bool] = {}

    def _reset_run_artifact_status() -> None:
        """Reset the per-run artifact pass/fail tracker."""
        _run_artifact_status.clear()

    def _snapshot_run_artifact_status() -> dict:
        """Return a shallow copy of the current run's artifact pass/fail map."""
        return dict(_run_artifact_status)

    def _set_current_guide_json(guide: dict | None) -> None:
        """Install (or clear) the guide JSON for the current run.

        Called from run_extraction() at the start of a run to install the
        guide so _completion_with_cache can attach it as a second cached
        system block, and from the finally/except paths to clear it so a
        later run doesn't leak stale guide state into its cache.
        """
        global _current_guide_json  # noqa: PLW0603 — intentional module state
        _current_guide_json = guide

    def _set_gen7_cached_context(
        guide_text: str | None,
        deduped_labels: list | None,
    ) -> None:
        """Install (or clear) the Gen 7 cached context for the REPL phase.

        Gen 7 caches TWO blocks beyond the system prompt itself:
          - The reconstructed full guide text (concatenated chunk.text), so
            Module Maker sub-calls can ground their DMN in the source.
          - The deduped label inventory, so every REPL turn + every
            llm_query_batched sub-call reads the same clean label list.

        Both are attached as cache_control blocks in the system prompt by
        _maybe_attach_gen7_blocks(), which is called from the monkey-patched
        AnthropicClient._completion_with_cache.

        Call with (None, None) at the end of a run to clear the state so a
        later legacy-pipeline run doesn't inherit stale Gen 7 cache state.
        """
        global _current_gen7_guide_text, _current_gen7_labels  # noqa: PLW0603
        global _gen7_blocks_logged  # noqa: PLW0603
        _current_gen7_guide_text = guide_text
        _current_gen7_labels = deduped_labels
        _gen7_blocks_logged["logged"] = False

    def _maybe_attach_gen7_blocks(kwargs: dict) -> None:
        """Append Gen 7 guide-text and label-list blocks to the system prompt.

        Only fires when _set_gen7_cached_context has been called (i.e. on
        a Gen 7 run). Legacy extraction runs are unaffected because the
        state variables are None.

        Each block is attached with cache_control: ephemeral, ttl: 1h. The
        1h TTL fits the typical ~15-25 minute REPL compile phase with
        headroom; cache writes cost 2x of input on the first call but every
        subsequent call reads at 0.1x, which is essential when the Module
        Maker phase fires N llm_query_batched sub-calls.
        """
        if _current_gen7_guide_text is None and _current_gen7_labels is None:
            return
        if "system" not in kwargs or not isinstance(kwargs["system"], list):
            return

        blocks_added = []

        if _current_gen7_guide_text is not None:
            guide_block_text = (
                "FULL SOURCE GUIDE (reconstructed from micro-chunks, for grounding):\n\n"
                + _current_gen7_guide_text
            )
            kwargs["system"].append({
                "type": "text",
                "text": guide_block_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            })
            blocks_added.append(f"guide_text={len(guide_block_text)} chars")

        if _current_gen7_labels is not None:
            labels_block_text = (
                f"DEDUPED LABEL INVENTORY ({len(_current_gen7_labels)} unique (id, type) entries, "
                "verified by distillation pass):\n\n"
                + json.dumps(_current_gen7_labels, indent=1)
            )
            kwargs["system"].append({
                "type": "text",
                "text": labels_block_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            })
            blocks_added.append(f"labels={len(_current_gen7_labels)} entries")

        if blocks_added and not _gen7_blocks_logged["logged"]:
            _gen7_blocks_logged["logged"] = True
            logger.info(
                "Gen7 cached system blocks attached: %s",
                ", ".join(blocks_added),
            )

    async def _generate_guide_toc(guide_json: dict, api_key: str) -> str:
        """Generate a rich table-of-contents preamble via one Opus call.

        This is the generator-side analog of the per-chunk contextual
        headers we use in the catcher layer. A one-time ~500-800 word
        domain-dense summary of the guide's structure gives the generator
        a focus prior before it starts reading the full guide content.

        Model choice: Opus (not Haiku). The user explicitly wants rich
        description — Opus produces substantially more structured and
        clinically-precise summaries than Haiku on complex source material.
        Opus input at $5/MTok on full WHO (~490K tokens) = ~$2.45 per
        first-time generation. Cached across iterations of the same run
        and across re-runs within process lifetime.

        Result is stored in _toc_cache keyed by SHA256 of guide content.
        Subsequent lookups for the same guide return the cached string
        for free. Negative cache (empty string) is used on generation
        failure to avoid retry storms.

        Called from run_extraction() at session start, BEFORE the first
        main call dispatches. Must be async because it makes an API call.
        The _maybe_attach_guide_block function (sync) reads _toc_cache
        directly to pick up the generated TOC.
        """
        import hashlib
        try:
            content_hash = hashlib.sha256(
                json.dumps(guide_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError):
            logger.warning("TOC: could not hash guide content; skipping")
            return ""

        if content_hash in _toc_cache:
            cached = _toc_cache[content_hash]
            logger.info(
                "TOC cache HIT (hash=%s, %d chars)",
                content_hash[:8], len(cached),
            )
            return cached

        # Build the generation prompt. We send the full guide so Opus can
        # actually read its structure and produce a substantive summary.
        try:
            guide_text = json.dumps(guide_json)
        except (TypeError, ValueError) as exc:
            logger.warning("TOC: could not serialize guide: %s", exc)
            _toc_cache[content_hash] = ""
            return ""

        prompt = (
            "You will receive a clinical guide as structured JSON. Produce a "
            "dense, factual table-of-contents preamble (500-800 words) that "
            "tells a downstream reader what to expect when they read the full "
            "guide. This preamble will be prepended to the guide in a cached "
            "system block used by a clinical-logic extractor — its job is to "
            "give the extractor a focus prior about what to look for.\n"
            "\n"
            "Cover:\n"
            "  1. What clinical domain(s) does this guide cover? (pediatrics, "
            "adult medicine, TB, cardiology, emergency, etc.)\n"
            "  2. What major clinical topics or conditions does it address? "
            "List them by name.\n"
            "  3. For each topic, what kinds of content does it contain? "
            "(assessment questions, examination findings, threshold "
            "definitions, treatment instructions with dosing, referral "
            "criteria, dose-by-weight tables, danger-sign enumerations)\n"
            "  4. Notable structural features: decision algorithms, priority "
            "short-circuits, multi-condition handling, phase-based workflows.\n"
            "  5. Approximate item counts per category if deducible: number "
            "of distinct supplies, distinct variables, distinct predicates, "
            "distinct modules, distinct phrase types.\n"
            "\n"
            "Output requirements:\n"
            "  - 500-800 words of dense factual prose. No markdown headers. "
            "No bullet points in the final preamble. No emoji. No meta "
            "commentary like 'Here is a summary of...' or 'This guide is...'.\n"
            "  - Start directly with substantive content. First sentence "
            "should name the clinical domain and the target clinician role.\n"
            "  - Be concrete and specific. Name drugs, thresholds, sections, "
            "and categories. Avoid generic language like 'various conditions' "
            "or 'several topics'.\n"
            "  - Quote specific phrases from the guide when helpful.\n"
            "\n"
            "GUIDE:\n"
            + guide_text
        )

        try:
            client = _anthropic_sdk.AsyncAnthropic(
                api_key=api_key,
                timeout=_httpx.Timeout(600.0, connect=30.0),
            )
            response = await client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1500,  # ~500-800 words of dense prose
                temperature=0.0,  # deterministic, reproducible across runs
                messages=[{"role": "user", "content": prompt}],
            )
            toc_text = ""
            for block in (response.content or []):
                btext = getattr(block, "text", None)
                if btext:
                    toc_text += btext
            toc_text = toc_text.strip()

            # Record usage in the main accumulator so the cost tracker
            # sees this Opus call.
            try:
                _accumulate_run_usage(response, "claude-opus-4-6")
            except Exception:
                pass

            _toc_cache[content_hash] = toc_text
            logger.info(
                "TOC generated via Opus: %d chars (hash=%s)",
                len(toc_text), content_hash[:8],
            )
            return toc_text
        except Exception as exc:
            logger.warning(
                "TOC generation failed (non-fatal, run will continue without TOC preamble): %s",
                exc,
            )
            _toc_cache[content_hash] = ""  # negative cache to prevent retry
            return ""

    def _init_with_timeout(self, api_key, model_name=None, max_tokens=16384, **kwargs):
        _original_init(self, api_key=api_key, model_name=model_name, max_tokens=max_tokens, **kwargs)
        long_timeout = _httpx.Timeout(600.0, connect=30.0)
        self.client = _anthropic_sdk.Anthropic(api_key=api_key, timeout=long_timeout)
        self.async_client = _anthropic_sdk.AsyncAnthropic(api_key=api_key, timeout=long_timeout)
        self._call_count = 0
        self._opus_model = model_name  # remember the configured Opus model

    def _pick_model(self) -> str:
        """Choose Opus or Sonnet based on phase-aware routing signal.

        Signal is set by emit_artifact in the same worker thread via
        _model_router_signal dict. Opus fires for:
        - Iteration 0 (planning)
        - After modules/integrative fail (complex repair)
        - After phrase_bank emit (self-validation phase)
        """
        self._call_count = getattr(self, "_call_count", 0) + 1
        use_opus = _model_router_signal.get("use_opus_next", self._call_count <= 1)

        if use_opus:
            effective = getattr(self, "_opus_model", None) or "claude-opus-4-6"
            _model_router_signal["use_opus_next"] = False  # reset after use
        else:
            effective = _SONNET

        logger.info(
            "Model router: call #%d -> %s%s",
            self._call_count, effective,
            " (Opus upshift)" if use_opus else "",
        )
        return effective

    def _call_with_retry(client_call, kwargs, max_retries: int = 3):
        """Call Anthropic with retry-and-backoff on transient errors.

        Retries on APIConnectionError, APITimeoutError, InternalServerError,
        RateLimitError. Backoff: 2s, 5s, 15s. Raises after exhausting retries.
        """
        import time as _time
        backoffs = [2, 5, 15]
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return client_call(**kwargs)
            except (
                _anthropic_sdk.APIConnectionError,
                _anthropic_sdk.APITimeoutError,
                _anthropic_sdk.InternalServerError,
                _anthropic_sdk.RateLimitError,
            ) as exc:
                last_exc = exc
                if attempt >= max_retries:
                    logger.error(
                        "Anthropic call failed after %d retries: %s",
                        max_retries, exc,
                    )
                    raise
                delay = backoffs[attempt] if attempt < len(backoffs) else 15
                logger.warning(
                    "Anthropic transient error (attempt %d/%d): %s -- retrying in %ds",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                _time.sleep(delay)
        # unreachable but satisfies type checkers
        if last_exc:
            raise last_exc
        return None

    def _maybe_attach_guide_block(kwargs: dict, system_text: str) -> None:
        """If a guide JSON is set for the current run, append it as a second
        cached system block alongside the existing system prompt.

        Gen 2.3 layout (2026-04-11):
            [TOC preamble if cached] + N× repeated guide text
            where N = _GUIDE_REPETITION_N (default 2) if it fits under the
            cap, else 1, else skip the block entirely.

        The TOC preamble is a ~500-800 word Opus-generated summary of the
        guide's structure, generated once per unique guide per session via
        _generate_guide_toc() and cached in _toc_cache. It gives the
        generator a focus prior about what to expect before reading the
        full guide content.

        The N× repetition applies Google's prompt-repetition finding
        (2512.14982): duplicating the cached content lets causal-attention
        models simulate bidirectional attention over it, improving
        multi-needle retrieval by 15-30 percentage points on list-heavy
        tasks. Our main extraction phases (supply_list, variables,
        predicates, modules, phrase_bank) are all list-retrieval tasks.

        Size-guards against blowing the 1M context window: if the combined
        TOC + N× guide + system prompt would exceed _GUIDE_CACHE_MAX_CHARS,
        we try a smaller repetition count first, then fall back to
        skipping the block entirely.
        """
        if _current_guide_json is None:
            return
        try:
            guide_text_single = f"CLINICAL GUIDE (JSON):\n{json.dumps(_current_guide_json)}"
        except (TypeError, ValueError) as exc:
            logger.warning("Could not serialize guide for cache block: %s", exc)
            return

        # Look up cached TOC preamble (generated at session start by
        # _generate_guide_toc). Empty string means TOC generation failed
        # or wasn't attempted — fall through with no prefix.
        import hashlib
        try:
            content_hash = hashlib.sha256(
                json.dumps(_current_guide_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        except (TypeError, ValueError):
            content_hash = ""
        toc_text = _toc_cache.get(content_hash, "")
        toc_prefix = (
            f"GUIDE TABLE OF CONTENTS (Opus-generated preamble for focus prior):\n"
            f"{toc_text}\n\n"
            f"================================================================\n"
            f"END OF TOC — GUIDE CONTENT FOLLOWS\n"
            f"================================================================\n\n"
        ) if toc_text else ""

        single_block_size = len(toc_prefix) + len(guide_text_single)
        overhead = len(system_text or "")

        # Try N× first, fall back to (N-1)× if it doesn't fit, then 1×,
        # then skip entirely.
        target_n = max(1, _GUIDE_REPETITION_N)
        chosen_n = 0
        for attempt_n in range(target_n, 0, -1):
            # Account for the per-repetition delimiter overhead (small).
            delimiter_overhead = max(0, attempt_n - 1) * 100
            total = single_block_size + (attempt_n - 1) * len(guide_text_single) + delimiter_overhead + overhead
            if total <= _GUIDE_CACHE_MAX_CHARS:
                chosen_n = attempt_n
                break

        if chosen_n == 0:
            logger.warning(
                "Guide cache block SKIPPED: guide+system chars exceed cap "
                "even at 1× (single=%d, cap=%d). Falling back to no guide "
                "in cache.",
                single_block_size + overhead, _GUIDE_CACHE_MAX_CHARS,
            )
            return

        if chosen_n < target_n:
            logger.warning(
                "Guide too large for %d× prompt repetition; falling back "
                "to %d×. (guide=%d chars, cap=%d)",
                target_n, chosen_n, len(guide_text_single), _GUIDE_CACHE_MAX_CHARS,
            )

        # Build the cached block text: TOC prefix + N repetitions of the
        # guide separated by explicit delimiters. Anthropic's prompt cache
        # treats the whole block as one cache key.
        if chosen_n == 1:
            cached_text = toc_prefix + guide_text_single
        else:
            parts = [toc_prefix + guide_text_single]
            for i in range(1, chosen_n):
                parts.append(
                    f"\n\n--- REPEATED FOR ATTENTION (copy {i + 1} of {chosen_n}) ---\n\n"
                    f"{guide_text_single}"
                )
            cached_text = "".join(parts)

        kwargs["system"].append({
            "type": "text",
            "text": cached_text,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })
        if not _guide_cache_logged["logged"]:
            _guide_cache_logged["logged"] = True
            logger.info(
                "Guide cached in system block: %d chars (~%d tokens), "
                "repetition=%d×, TOC=%s",
                len(cached_text), len(cached_text) // 4, chosen_n,
                "yes" if toc_text else "no",
            )

    def _completion_with_cache(self, prompt, model=None):
        messages, system = self._prepare_messages(prompt)
        effective_model = self._pick_model()

        kwargs: dict = {
            "model": effective_model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            system_text = system if isinstance(system, str) else str(system)
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ]
            _maybe_attach_guide_block(kwargs, system_text)
            _maybe_attach_gen7_blocks(kwargs)
        response = _call_with_retry(self.client.messages.create, kwargs)
        self._track_cost(response, effective_model)
        _accumulate_run_usage(response, effective_model)
        return response.content[0].text

    async def _acompletion_with_cache(self, prompt, model=None):
        import asyncio as _asyncio
        messages, system = self._prepare_messages(prompt)
        effective_model = self._pick_model()

        kwargs: dict = {
            "model": effective_model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            system_text = system if isinstance(system, str) else str(system)
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ]
            _maybe_attach_guide_block(kwargs, system_text)
            _maybe_attach_gen7_blocks(kwargs)

        # Async retry loop (can't use _call_with_retry which uses sync time.sleep)
        backoffs = [2, 5, 15]
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                response = await self.async_client.messages.create(**kwargs)
                self._track_cost(response, effective_model)
                _accumulate_run_usage(response, effective_model)
                return response.content[0].text
            except (
                _anthropic_sdk.APIConnectionError,
                _anthropic_sdk.APITimeoutError,
                _anthropic_sdk.InternalServerError,
                _anthropic_sdk.RateLimitError,
            ) as exc:
                last_exc = exc
                if attempt >= 3:
                    logger.error("Anthropic async call failed after 3 retries: %s", exc)
                    raise
                delay = backoffs[attempt]
                logger.warning(
                    "Anthropic async transient error (attempt %d/4): %s -- retrying in %ds",
                    attempt + 1, exc, delay,
                )
                await _asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    # Only patch once (idempotent across imports)
    if not getattr(
        _rlm_anthropic.AnthropicClient.completion, "_chw_cache_patched", False
    ):
        _init_with_timeout._chw_cache_patched = True  # type: ignore[attr-defined]
        _completion_with_cache._chw_cache_patched = True  # type: ignore[attr-defined]
        _acompletion_with_cache._chw_cache_patched = True  # type: ignore[attr-defined]
        _rlm_anthropic.AnthropicClient.__init__ = _init_with_timeout  # type: ignore[assignment]
        _rlm_anthropic.AnthropicClient._pick_model = _pick_model  # type: ignore[attr-defined]
        _rlm_anthropic.AnthropicClient.completion = _completion_with_cache  # type: ignore[assignment]
        _rlm_anthropic.AnthropicClient.acompletion = _acompletion_with_cache  # type: ignore[assignment]
        logger.info(
            "Patched rlm AnthropicClient: cache_control + 600s timeout + Opus 4.6 support"
        )
except ImportError as _exc:  # pragma: no cover - rlm library missing
    logger.warning("Could not patch AnthropicClient for prompt caching: %s", _exc)


class ExtractionResult:
    """Result of an RLM extraction session.

    The seven intermediate artifact fields (supply_list, variables,
    predicates, modules, router, integrative, phrase_bank) are populated
    from the session_artifacts dict after rlm.completion() returns. They
    hold the LATEST version of each artifact the model emitted during the
    run. Previous versions (if the model re-emitted after fixing critical
    issues) are still persisted in Neon via create_intermediate_artifact.
    """

    def __init__(
        self,
        run_id: str,
        status: str,
        clinical_logic: dict | None,
        total_iterations: int,
        total_subcalls: int,
        validation_errors: int,
        cost_estimate_usd: float,
        trajectory: list[dict],
        supply_list: dict | None = None,
        variables: dict | None = None,
        predicates: dict | None = None,
        modules: dict | None = None,
        router: dict | None = None,
        integrative: dict | None = None,
        phrase_bank: dict | None = None,
    ):
        self.run_id = run_id
        self.status = status
        self.clinical_logic = clinical_logic
        self.total_iterations = total_iterations
        self.total_subcalls = total_subcalls
        self.validation_errors = validation_errors
        self.cost_estimate_usd = cost_estimate_usd
        self.trajectory = trajectory
        # Phase-level intermediate artifacts (hybrid plan)
        self.supply_list = supply_list
        self.variables = variables
        self.predicates = predicates
        self.modules = modules
        self.router = router
        self.integrative = integrative
        self.phrase_bank = phrase_bank


def _make_validate_fn(
    *,
    main_loop: asyncio.AbstractEventLoop,
    on_step: Any,
    step_counter: dict,
) -> callable:
    """Create the validate() function exposed in the REPL.

    Runs the structural validators AND pushes an SSE event with the result so
    the frontend's ValidationPanel can render the pass/fail and error list.
    """

    def validate(logic: Any) -> dict:
        # Defensive: the model occasionally passes the wrong variable here
        # (e.g. validate(clinical_logic_json_string) instead of the dict).
        # Return a structured error so the REPL sees a useful message
        # instead of a raw AttributeError crash that halts the run.
        if not isinstance(logic, dict):
            logger.warning(
                "validate() called with non-dict (%s) -- returning error "
                "response instead of crashing. First 200 chars: %s",
                type(logic).__name__,
                str(logic)[:200],
            )
            return {
                "passed": False,
                "error_count": 1,
                "errors": [{
                    "validator": "type_guard",
                    "message": (
                        f"validate() expects a dict, got {type(logic).__name__}. "
                        "Did you pass a stringified JSON instead of the parsed "
                        "Python dict? Try validate(clinical_logic) not "
                        "validate(json.dumps(clinical_logic))."
                    ),
                    "severity": "error",
                }],
            }
        result = run_all_validators(logic)

        # Push the result to the SSE stream via on_step
        if on_step is not None:
            step_counter["count"] += 1
            try:
                asyncio.run_coroutine_threadsafe(
                    on_step({
                        "step_number": step_counter["count"],
                        "step_type": "validate",
                        "validation_result": result,
                    }),
                    main_loop,
                )
            except Exception as exc:
                logger.debug("on_step for validate failed: %s", exc)

        return result

    return validate


def _make_z3_check_fn(
    *,
    main_loop: asyncio.AbstractEventLoop,
    on_step: Any,
    step_counter: dict,
) -> callable:
    """Create the z3_check() function exposed in the REPL.

    Runs the Z3 exhaustiveness/consistency proofs AND pushes an SSE event so
    the frontend's Z3Panel can render the per-check pass/fail.
    """

    def z3_check(logic: Any) -> dict:
        # Same defensive guard as validate() -- if the model passes a
        # non-dict, return a structured error rather than crashing on
        # logic.get("modules", []) inside compile_to_dmn_package.
        if not isinstance(logic, dict):
            logger.warning(
                "z3_check() called with non-dict (%s) -- returning error "
                "response instead of crashing. First 200 chars: %s",
                type(logic).__name__,
                str(logic)[:200],
            )
            return {
                "all_passed": False,
                "checks": [],
                "warnings": [
                    f"z3_check() expects a dict, got {type(logic).__name__}. "
                    "Pass the parsed clinical_logic dict, not a string or list."
                ],
            }
        result = verify_clinical_logic(logic)

        if on_step is not None:
            step_counter["count"] += 1
            try:
                asyncio.run_coroutine_threadsafe(
                    on_step({
                        "step_number": step_counter["count"],
                        "step_type": "z3",
                        "z3_result": result,
                    }),
                    main_loop,
                )
            except Exception as exc:
                logger.debug("on_step for z3_check failed: %s", exc)

        return result

    return z3_check


def _make_emit_artifact_fn(
    *,
    run_id: str,
    artifact_dir: Path,
    guide_json: dict,
    api_key: str,
    main_loop: asyncio.AbstractEventLoop,
    session_artifacts: dict,
    on_step: Any,
    step_counter: dict,
    model_router_signal: dict,
) -> callable:
    """Create the emit_artifact() function exposed in the REPL.

    This is the hybrid plan's per-phase checkpoint mechanism. The model calls
    it between phases with a name (supply_list, variables, predicates, modules,
    router, integrative, phrase_bank) and the artifact JSON. The function:

      1. Writes the artifact to {artifact_dir}/artifacts/{name}.json
      2. Routes to the right phase validator (auto-validate on emit)
      3. Persists artifact + validator result to Neon via create_intermediate_artifact
      4. Pushes a step event so the SSE stream shows the checkpoint
      5. Returns {passed, critical_issues, warnings, phase} to the model

    Fix C (re-emit intervention, 2026-04-12):
      When the catcher verdict is passed=False, the function ALSO prints a
      prominent action block to REPL stdout containing:
        - the structured critical_issues as a Python literal `_missing = [...]`
          the model can copy-paste,
        - enumerated preview of each critical (suggested_id, quote, section),
        - explicit NEXT ACTION steps telling the model to APPEND items to
          the existing artifact variable, not re-run Stage 1,
        - a programmatic access path via `emit_artifact.last_failed_structured`.
      This bypasses the "prose re-emit contract gets crowded out by REPL
      scratchpad" failure mode we observed in the gen 2.4 + A/B run —
      stdout is prominent in every next-iteration prompt and the model
      cannot miss executable code patterns the way it can miss system
      prompt instructions.

    The model MUST fix critical_issues and re-emit before proceeding. The
    returned dict is structured so the model can easily branch on `passed`.

    Because the REPL runs in a worker thread (rlm.completion is sync, we
    dispatch via asyncio.to_thread), this function is SYNCHRONOUS but bridges
    back to the main event loop for the async validator + Neon calls.
    """
    artifacts_subdir = artifact_dir / "artifacts"
    artifacts_subdir.mkdir(parents=True, exist_ok=True)

    # Fix C state — shared across emit_artifact calls. The inner function
    # mutates this dict and `emit_artifact.last_failed_structured` is a
    # reference to it, so the model can reach the structured criticals for
    # any failed artifact via `emit_artifact.last_failed_structured[name]`.
    _fixc_state: dict = {"last_failed": {}}

    def _extract_structured_criticals(validator_result: Any) -> list:
        """Pull the structured (dict-form) criticals out of a validator
        result's per-catcher outputs. Returns a flat list of dicts with the
        shape {catcher, suggested_id, description, guide_quote, section}.

        Non-structured criticals (plain strings from older catchers like
        completeness_router or comorbidity_coverage) are ignored here —
        they'll still appear in critical_issues as string form, but Fix C
        only injects the structured ones because those are the ones we
        can turn into a for-loop template.
        """
        out: list = []
        catcher_outputs = getattr(validator_result, "catcher_outputs", None) or {}
        for catcher_name, catcher_result in catcher_outputs.items():
            if not isinstance(catcher_result, dict):
                continue
            structured = catcher_result.get("_structured_criticals") or []
            for sc in structured:
                if not isinstance(sc, dict):
                    continue
                sid = sc.get("suggested_id")
                quote = sc.get("guide_quote")
                if not sid or not quote:
                    continue
                out.append({
                    "catcher": catcher_name,
                    "suggested_id": str(sid),
                    "description": str(sc.get("description", "")),
                    "guide_quote": str(quote),
                    "section": str(sc.get("section", "")),
                    "repair_instruction": str(sc.get("repair_instruction", "")),
                })
        return out

    # -----------------------------------------------------------------
    # Gen 3 in-REPL repair: synchronous Sonnet call for one entry
    # -----------------------------------------------------------------
    # Uses the synchronous anthropic.Anthropic client because emit_artifact
    # runs in the REPL worker thread (not the async main loop). Async
    # clients would require event-loop bridging.
    import anthropic as _anthropic_sdk

    _repair_client: _anthropic_sdk.Anthropic | None = None

    def _get_repair_client() -> _anthropic_sdk.Anthropic:
        nonlocal _repair_client
        if _repair_client is None:
            _repair_client = _anthropic_sdk.Anthropic(api_key=api_key)
        return _repair_client

    def _generate_one_entry(
        artifact_name: str,
        critical: dict,
        schema_example: dict,
        model: str = "claude-opus-4-6",
    ) -> dict | None:
        """Call Opus (Sonnet as fallback) to generate ONE artifact entry.

        Uses Opus by default for best schema understanding on complex
        entries (modules with nested rules, predicates with threshold
        expressions). Sonnet fallback on Opus failure.
        Synchronous. Returns the parsed dict or None on failure.
        """
        schema_str = json.dumps(schema_example, indent=2) if schema_example else "{}"
        prompt = (
            f"Generate exactly ONE JSON dict entry for the '{artifact_name}' artifact.\n\n"
            f"SCHEMA EXAMPLE (copy this field structure, fill in new values):\n"
            f"{schema_str}\n\n"
            f"ENTRY TO GENERATE:\n"
            f"  ID: {critical.get('suggested_id', 'unknown')}\n"
            f"  Guide quote: {critical.get('guide_quote', '')}\n"
            f"  Section: {critical.get('section', '')}\n"
            f"  Description: {critical.get('description', '')}\n"
            f"  Repair instruction: {critical.get('repair_instruction', '')}\n\n"
            f"Return ONLY the JSON dict. No markdown fences. No explanation.\n"
            f"The 'id' field must be '{critical.get('suggested_id', '')}'."
        )

        for attempt, attempt_model in enumerate([model, "claude-sonnet-4-6"]):
            try:
                client = _get_repair_client()
                response = client.messages.create(
                    model=attempt_model,
                    max_tokens=16384,  # max Anthropic allows; modules need 2000+ tokens
                    system="Return ONLY a valid JSON object. No markdown, no prose.",
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.content or not hasattr(response.content[0], "text"):
                    raise ValueError("Empty response from API")
                # Track cost of this repair call
                try:
                    ru = response.usage
                    r_in = getattr(ru, "input_tokens", 0) or 0
                    r_out = getattr(ru, "output_tokens", 0) or 0
                    r_cached = getattr(ru, "cache_read_input_tokens", 0) or 0
                    r_cw = getattr(ru, "cache_creation_input_tokens", 0) or 0
                    ip, op, crp, cwp = _COST_TABLE.get(attempt_model, (3.0, 15.0, 0.3, 6.0))
                    r_billed = max(0, r_in - r_cached - r_cw)
                    r_cost = (
                        (r_billed / 1_000_000) * ip
                        + (r_cached / 1_000_000) * crp
                        + (r_cw / 1_000_000) * cwp
                        + (r_out / 1_000_000) * op
                    )
                    _run_usage["input_tokens"] = _run_usage.get("input_tokens", 0) + r_in
                    _run_usage["output_tokens"] = _run_usage.get("output_tokens", 0) + r_out
                    _run_usage["cached_tokens"] = _run_usage.get("cached_tokens", 0) + r_cached
                    _run_usage["cache_write_tokens"] = _run_usage.get("cache_write_tokens", 0) + r_cw
                    _run_usage["cost_usd"] = _run_usage.get("cost_usd", 0.0) + r_cost
                    _run_usage["api_calls"] = _run_usage.get("api_calls", 0) + 1
                    # Increment model call counter for frontend display
                    ml = (attempt_model or "").lower()
                    if "opus" in ml:
                        _run_usage["calls_opus"] = _run_usage.get("calls_opus", 0) + 1
                    elif "haiku" in ml:
                        _run_usage["calls_haiku"] = _run_usage.get("calls_haiku", 0) + 1
                    else:
                        _run_usage["calls_sonnet"] = _run_usage.get("calls_sonnet", 0) + 1
                except Exception:
                    pass  # cost tracking failure is non-fatal
                text = response.content[0].text.strip()
                # Strip markdown fences if model included them
                import re as _re
                text = _re.sub(r"^```(?:json)?\s*", "", text)
                text = _re.sub(r"\s*```$", "", text)
                entry = json.loads(text)
                if not isinstance(entry, dict):
                    raise ValueError(f"Expected dict, got {type(entry).__name__}")
                if attempt == 1:
                    logger.info("Gen3 repair: Sonnet fallback succeeded for %s/%s",
                                artifact_name, critical.get("suggested_id", "?"))
                return entry
            except (json.JSONDecodeError, ValueError, KeyError,
                    IndexError, AttributeError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Gen3 repair: Opus malformed JSON for %s/%s, falling back to Sonnet: %s",
                        artifact_name, critical.get("suggested_id", "?"), exc,
                    )
                    continue
                else:
                    logger.error("Gen3 repair: Sonnet also failed for %s/%s: %s",
                                 artifact_name, critical.get("suggested_id", "?"), exc)
                    return None
            except Exception as exc:
                logger.warning("Gen3 repair: API error for %s/%s: %s",
                               artifact_name, critical.get("suggested_id", "?"), exc)
                return None
        return None

    def _patch_existing_entry(
        artifact_name: str,
        critical: dict,
        existing_entry: dict,
        schema_example: dict,
        model: str = "claude-opus-4-6",
    ) -> dict | None:
        """Call Opus to fix/patch an EXISTING artifact entry.

        Includes the current (broken) entry in the prompt so the model
        can see what exists and modify only what's broken. Returns the
        fixed entry dict or None on failure.
        """
        schema_str = json.dumps(schema_example, indent=2) if schema_example else "{}"
        existing_str = json.dumps(existing_entry, indent=2)
        entry_id = existing_entry.get("id", critical.get("suggested_id", ""))

        prompt = (
            f"Fix the EXISTING JSON dict entry in the '{artifact_name}' artifact.\n\n"
            f"SCHEMA EXAMPLE (reference for field structure):\n"
            f"{schema_str}\n\n"
            f"CURRENT (BROKEN) ENTRY TO FIX:\n"
            f"{existing_str}\n\n"
            f"WHAT'S BROKEN:\n"
            f"  Issue: {critical.get('description', '')}\n"
            f"  Guide quote: {critical.get('guide_quote', '')}\n"
            f"  Section: {critical.get('section', '')}\n"
            f"  Fix instruction: {critical.get('repair_instruction', '')}\n\n"
            f"Return ONLY the FIXED JSON dict with id='{entry_id}'. "
            f"Keep all unchanged fields exactly as-is. Modify ONLY what the "
            f"fix instruction says to change.\n"
            f"No markdown fences. No explanation."
        )

        for attempt, attempt_model in enumerate([model, "claude-sonnet-4-6"]):
            try:
                client = _get_repair_client()
                response = client.messages.create(
                    model=attempt_model,
                    max_tokens=16384,
                    system="Return ONLY a valid JSON object. No markdown, no prose.",
                    messages=[{"role": "user", "content": prompt}],
                )
                if not response.content or not hasattr(response.content[0], "text"):
                    raise ValueError("Empty response from API")
                # Cost tracking
                try:
                    ru = response.usage
                    r_in = getattr(ru, "input_tokens", 0) or 0
                    r_out = getattr(ru, "output_tokens", 0) or 0
                    r_cached = getattr(ru, "cache_read_input_tokens", 0) or 0
                    r_cw = getattr(ru, "cache_creation_input_tokens", 0) or 0
                    ip, op, crp, cwp = _COST_TABLE.get(attempt_model, (3.0, 15.0, 0.3, 6.0))
                    r_billed = max(0, r_in - r_cached - r_cw)
                    r_cost = (
                        (r_billed / 1_000_000) * ip
                        + (r_cached / 1_000_000) * crp
                        + (r_cw / 1_000_000) * cwp
                        + (r_out / 1_000_000) * op
                    )
                    _run_usage["input_tokens"] = _run_usage.get("input_tokens", 0) + r_in
                    _run_usage["output_tokens"] = _run_usage.get("output_tokens", 0) + r_out
                    _run_usage["cached_tokens"] = _run_usage.get("cached_tokens", 0) + r_cached
                    _run_usage["cache_write_tokens"] = _run_usage.get("cache_write_tokens", 0) + r_cw
                    _run_usage["cost_usd"] = _run_usage.get("cost_usd", 0.0) + r_cost
                    _run_usage["api_calls"] = _run_usage.get("api_calls", 0) + 1
                    ml = (attempt_model or "").lower()
                    if "opus" in ml:
                        _run_usage["calls_opus"] = _run_usage.get("calls_opus", 0) + 1
                    elif "haiku" in ml:
                        _run_usage["calls_haiku"] = _run_usage.get("calls_haiku", 0) + 1
                    else:
                        _run_usage["calls_sonnet"] = _run_usage.get("calls_sonnet", 0) + 1
                except Exception:
                    pass
                text = response.content[0].text.strip()
                import re as _re
                text = _re.sub(r"^```(?:json)?\s*", "", text)
                text = _re.sub(r"\s*```$", "", text)
                entry = json.loads(text)
                if not isinstance(entry, dict):
                    raise ValueError(f"Expected dict, got {type(entry).__name__}")
                # Preserve original id
                if entry.get("id") != entry_id and entry_id:
                    entry["id"] = entry_id
                if attempt == 1:
                    logger.info("Gen3 patch: Sonnet fallback succeeded for %s/%s",
                                artifact_name, critical.get("suggested_id", "?"))
                return entry
            except (json.JSONDecodeError, ValueError, KeyError,
                    IndexError, AttributeError) as exc:
                if attempt == 0:
                    logger.warning(
                        "Gen3 patch: Opus malformed JSON for %s/%s, falling back to Sonnet: %s",
                        artifact_name, critical.get("suggested_id", "?"), exc,
                    )
                    continue
                else:
                    logger.error("Gen3 patch: Sonnet also failed for %s/%s: %s",
                                 artifact_name, critical.get("suggested_id", "?"), exc)
                    return None
            except Exception as exc:
                logger.warning("Gen3 patch: API error for %s/%s: %s",
                               artifact_name, critical.get("suggested_id", "?"), exc)
                return None
        return None

    def _find_existing_entry(artifact_value: Any, entry_id: str) -> tuple:
        """Find an existing entry in the artifact by id.

        Returns (entry_dict, index_or_key) if found, (None, None) otherwise.
        For lists: searches by entry['id']. For dicts: direct key lookup.
        """
        if isinstance(artifact_value, list):
            for idx, item in enumerate(artifact_value):
                if isinstance(item, dict) and item.get("id") == entry_id:
                    return item, idx
                # Also check rule_id for modules (rules use rule_id, not id)
                if isinstance(item, dict) and item.get("rule_id") == entry_id:
                    return item, idx
        elif isinstance(artifact_value, dict):
            if entry_id in artifact_value:
                val = artifact_value[entry_id]
                return val if isinstance(val, dict) else None, entry_id
            # For modules: entries are dicts with module_id, rules have rule_id
            # Search nested: module.rules[].rule_id
            for mod_key, mod_val in artifact_value.items():
                if not isinstance(mod_val, dict):
                    continue
                for ridx, rule in enumerate(mod_val.get("rules", [])):
                    if isinstance(rule, dict) and rule.get("rule_id") == entry_id:
                        return rule, (mod_key, ridx)
        return None, None

    def _run_inline_repair(
        name: str,
        value: Any,
        result: Any,
        max_rounds: int = 3,
    ) -> tuple[Any, Any]:
        """In-REPL surgical repair loop. Runs synchronously in the worker thread.

        For each structured critical, calls Sonnet to generate ONE entry,
        system-inserts it into the artifact, then re-validates. Repeats
        until 0 criticals or max_rounds exhausted.

        Returns (updated_value, final_result).
        """
        # Get a schema example from the existing artifact
        schema_example = {}
        if isinstance(value, list) and value:
            schema_example = value[0] if isinstance(value[0], dict) else {}
        elif isinstance(value, dict):
            first_key = next(iter(value), None)
            if first_key and isinstance(value[first_key], dict):
                schema_example = value[first_key]

        for round_num in range(max_rounds):
            if result.passed:
                break

            structured = _extract_structured_criticals(result)
            if not structured:
                # No structured criticals to fix (string-only or empty)
                break

            logger.info(
                "Gen3 repair: %s round %d — %d structured criticals to fix",
                name, round_num + 1, len(structured),
            )

            repaired = 0
            for critical in structured:
                suggested_id = critical.get("suggested_id", f"{name}_repair")

                # Detect mode: does this entry already exist? (patch vs append)
                existing_entry, location = _find_existing_entry(value, suggested_id)
                is_patch = existing_entry is not None

                if is_patch:
                    logger.info("Gen3 repair: %s — PATCH mode for %s", name, suggested_id)
                    entry = _patch_existing_entry(name, critical, existing_entry, schema_example)
                else:
                    entry = _generate_one_entry(name, critical, schema_example)

                if entry is None:
                    continue

                # System inserts or replaces
                if is_patch and location is not None:
                    # Replace existing entry in-place
                    if isinstance(value, list) and isinstance(location, int):
                        value[location] = entry
                    elif isinstance(value, dict):
                        if isinstance(location, tuple):
                            # Nested: (module_key, rule_index) for module rules
                            mod_key, ridx = location
                            value[mod_key]["rules"][ridx] = entry
                        else:
                            value[location] = entry
                else:
                    # Append new entry
                    if isinstance(value, list):
                        value.append(entry)
                    elif isinstance(value, dict):
                        entry_id = entry.get("id", suggested_id)
                        value[entry_id] = entry
                repaired += 1

                # Push SSE event
                mode_str = "patch" if is_patch else "append"
                if on_step is not None:
                    step_counter["count"] += 1
                    try:
                        asyncio.run_coroutine_threadsafe(
                            on_step({
                                "step_number": step_counter["count"],
                                "step_type": "repair",
                                "prompt": f"repair({name}, {suggested_id}, {mode_str})",
                                "response": json.dumps({
                                    "entry_id": suggested_id,
                                    "round": round_num + 1,
                                    "mode": mode_str,
                                    "success": True,
                                }),
                            }),
                            main_loop,
                        )
                    except Exception:
                        pass

            logger.info("Gen3 repair: %s round %d — inserted %d/%d entries, re-validating",
                        name, round_num + 1, repaired, len(structured))

            # Persist repaired artifact to disk
            try:
                artifact_path = artifacts_subdir / f"{name}.json"
                artifact_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            except Exception:
                pass

            # Re-validate
            try:
                future = asyncio.run_coroutine_threadsafe(
                    validate_artifact(name, value, guide_json, api_key),
                    main_loop,
                )
                result = future.result(timeout=300)
            except Exception as exc:
                logger.warning("Gen3 repair: re-validation crashed for %s: %s", name, exc)
                break

            # Pin criticals on re-validation (Fix G logic inline).
            # Filter out NEW criticals whose suggested_id wasn't in the
            # prior round's set. If pinning drops all criticals to 0,
            # update the result to passed=True.
            if not result.passed:
                new_structured = _extract_structured_criticals(result)
                prior_sids = {
                    str(sc.get("suggested_id", "")).strip().lower()
                    for sc in structured if isinstance(sc, dict) and sc.get("suggested_id")
                }
                if prior_sids:
                    pinned_criticals = []
                    for c in result.critical_issues:
                        c_lower = c.lower()
                        if any(sid in c_lower for sid in prior_sids):
                            pinned_criticals.append(c)
                        elif not any(tag in c_lower for tag in (
                            "[completeness_", "[comorbidity_",
                        )):
                            pinned_criticals.append(c)

                    if len(pinned_criticals) < len(result.critical_issues):
                        logger.info(
                            "Gen3 repair: pinned %s from %d to %d criticals on re-validation",
                            name, len(result.critical_issues), len(pinned_criticals),
                        )
                        # Update the result object so emit_artifact sees the pinned count
                        result = PhaseValidationResult(
                            artifact_name=result.artifact_name,
                            phase=result.phase,
                            passed=len(pinned_criticals) == 0,
                            critical_issues=pinned_criticals,
                            warnings=result.warnings,
                            catcher_outputs=result.catcher_outputs,
                        )

        return value, result

    def emit_artifact(name: str, value: Any) -> dict:
        # Validate the name first so a typo gets a clear error
        if name not in valid_artifact_names():
            return {
                "passed": False,
                "critical_issues": [
                    f"Unknown artifact name {name!r}. "
                    f"Valid names: {valid_artifact_names()}"
                ],
                "warnings": [],
                "phase": -1,
            }

        # Guard: skip re-validation for artifacts that already passed.
        if name not in _fixc_state["last_failed"] and name in session_artifacts:
            print(f"[emit_artifact] {name!r} already PASSED. Skipping re-validation.")
            session_artifacts[name] = value
            return {
                "passed": True,
                "critical_issues": [],
                "warnings": [],
                "phase": _ARTIFACT_PHASES.get(name, -1),
            }

        # Defensive: require a dict or list.
        if not isinstance(value, (dict, list)):
            return {
                "passed": False,
                "critical_issues": [
                    f"Artifact {name!r} must be a dict or list, got {type(value).__name__}"
                ],
                "warnings": [],
                "phase": _ARTIFACT_PHASES.get(name, -1),
            }

        # Persist to disk
        try:
            artifact_path = artifacts_subdir / f"{name}.json"
            artifact_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write %s to disk: %s", name, exc)

        # Validate via catchers
        try:
            future = asyncio.run_coroutine_threadsafe(
                validate_artifact(name, value, guide_json, api_key),
                main_loop,
            )
            result = future.result(timeout=300)
        except Exception as exc:
            logger.exception("Validator for %s crashed: %s", name, exc)
            return {
                "passed": False,
                "critical_issues": [f"Validator crashed: {exc}"],
                "warnings": [],
                "phase": _ARTIFACT_PHASES.get(name, -1),
            }

        # Persist to Neon (best-effort)
        try:
            db_future = asyncio.run_coroutine_threadsafe(
                create_intermediate_artifact(
                    run_id=run_id,
                    artifact_name=name,
                    phase=result.phase,
                    artifact_json=value if isinstance(value, dict) else {"items": value},
                    validator_passed=result.passed,
                    critical_issues=result.critical_issues,
                    warnings=result.warnings,
                    catcher_outputs=result.catcher_outputs,
                ),
                main_loop,
            )
            db_future.result(timeout=30)
        except Exception as exc:
            logger.warning("Failed to persist artifact %s to Neon: %s", name, exc)

        # -----------------------------------------------------------
        # Gen 3: In-REPL surgical repair loop (2026-04-12)
        # -----------------------------------------------------------
        # If the artifact failed, run the repair loop BEFORE returning
        # to the model. The model sees a clean pass/fail after repair.
        # This enforces the DAG: the model can't move to the next
        # artifact until this one either passes or exhausts repair rounds.
        if not result.passed:
            structured = _extract_structured_criticals(result)
            if structured:
                logger.info(
                    "Gen3 repair: starting in-REPL repair for %s (%d structured criticals)",
                    name, len(structured),
                )
                value, result = _run_inline_repair(name, value, result)

        # Update session artifacts with the (possibly repaired) value
        session_artifacts[name] = value

        # Push SSE step event
        if on_step is not None:
            step_counter["count"] += 1
            try:
                asyncio.run_coroutine_threadsafe(
                    on_step({
                        "step_number": step_counter["count"],
                        "step_type": "artifact",
                        "prompt": f"emit_artifact({name!r})",
                        "response": json.dumps({
                            "passed": result.passed,
                            "critical_issues": result.critical_issues,
                            "warnings": result.warnings,
                            "phase": result.phase,
                        }),
                    }),
                    main_loop,
                )
            except Exception as exc:
                logger.debug("on_step for emit_artifact %s failed: %s", name, exc)

        # Model router upshift signals
        if name in _OPUS_TRIGGER_ARTIFACTS and not result.passed:
            model_router_signal["use_opus_next"] = True
            logger.info("Model router: upshift to Opus for %s repair (%d criticals)", name, len(result.critical_issues))
        elif name in _OPUS_TRIGGER_FINAL:
            model_router_signal["use_opus_next"] = True
            logger.info("Model router: upshift to Opus for self-validation phase")

        logger.info(
            "emit_artifact: name=%s phase=%d passed=%s critical=%d warnings=%d",
            name,
            result.phase,
            result.passed,
            len(result.critical_issues),
            len(result.warnings),
        )

        # Update catcher pass/fail tracker
        try:
            _run_artifact_status[name] = result.passed
        except NameError:
            pass

        # Update fixc_state
        if not result.passed:
            structured = _extract_structured_criticals(result)
            _fixc_state["last_failed"][name] = structured
        else:
            _fixc_state["last_failed"].pop(name, None)

        return {
            "passed": result.passed,
            "critical_issues": result.critical_issues,
            "warnings": result.warnings,
            "phase": result.phase,
        }

    # Expose the Fix C shared state as an attribute on the returned function
    # so the model can reach the structured criticals programmatically via
    # `emit_artifact.last_failed_structured[name]`. The attribute is bound to
    # the same dict that the closure mutates, so subsequent updates are
    # visible through the attribute automatically.
    emit_artifact.last_failed_structured = _fixc_state["last_failed"]

    return emit_artifact


    # Gen 3: standalone repair functions removed. The repair loop now runs
    # INSIDE emit_artifact (see _run_inline_repair above) so each artifact
    # is repaired before the model moves to the next phase in the DAG.



async def run_extraction(
    guide_json: dict,
    api_key: str,
    run_id: str,
    manual_name: str = "unknown",
    model_name: str = "claude-opus-4-6",
    subcall_model: str = "claude-sonnet-4-6",
    max_iterations: int = 50,
    environment: str = "local",
    log_dir: str | None = None,
    on_step: Any = None,
) -> ExtractionResult:
    """Run an RLM extraction session.

    Args:
        guide_json: The clinical manual as JSON dict.
        api_key: Anthropic API key (BYOK, from user).
        run_id: Unique identifier for this run.
        manual_name: Name of the manual (for logging).
        model_name: Root model (Claude Opus).
        subcall_model: Sub-call model (Claude Sonnet).
        max_iterations: Max REPL iterations before halt.
        environment: REPL environment ("local" or "docker").
        log_dir: Directory for trajectory JSONL logs.
        on_step: Async callback for each step (for SSE streaming).

    Returns:
        ExtractionResult with the validated clinical logic or halt status.
    """
    logger.info(
        "Starting extraction run=%s manual=%s model=%s",
        run_id,
        manual_name,
        model_name,
    )

    # Build system prompt
    system_prompt = build_system_prompt()
    initial_message = build_initial_user_message(guide_json)

    # Configure RLM logger
    rlm_logger = RLMLogger(log_dir=log_dir) if log_dir else RLMLogger()

    # Step counter for callbacks
    step_counter = {"count": 0, "subcalls": 0}

    # RLM callbacks are synchronous and will be called from a worker thread
    # once rlm.completion() is dispatched via asyncio.to_thread below.
    # From a worker thread, asyncio.get_event_loop() raises in Python 3.12,
    # and asyncio.get_running_loop() raises unconditionally.
    #
    # To bridge worker-thread callbacks back to the main event loop, we
    # capture the running loop HERE (before dispatching to the thread) and
    # use asyncio.run_coroutine_threadsafe(), which is the correct API for
    # scheduling coroutines from non-loop threads.
    main_loop = asyncio.get_running_loop()

    # Hybrid-plan artifact state. Populated by emit_artifact() calls from
    # the REPL. Holds the LATEST version of each named artifact. If the
    # model re-emits after fixing critical_issues, overrides the previous
    # value (Neon keeps the history via create_intermediate_artifact).
    session_artifacts: dict[str, Any] = {}

    def _on_iteration_complete(depth: int, iteration_num: int, duration: float) -> None:
        """Called by the rlm library after each REPL iteration.

        Signature: (depth, iteration_num, duration) per rlm.core.rlm line 77.
        """
        step_counter["count"] += 1
        # Snapshot live cumulative usage from the shared counter
        try:
            usage = _snapshot_run_usage()
        except NameError:
            usage = {}
        if on_step:
            try:
                asyncio.run_coroutine_threadsafe(
                    on_step({
                        "step_number": step_counter["count"],
                        "step_type": "exec",
                        "iteration": iteration_num,
                        "depth": depth,
                        "executionMs": int(duration * 1000),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cached_tokens": usage.get("cached_tokens", 0),
                        "cost_usd": usage.get("cost_usd", 0.0),
                        "calls_opus": usage.get("calls_opus", 0),
                        "calls_sonnet": usage.get("calls_sonnet", 0),
                    }),
                    main_loop,
                )
            except Exception as e:
                logger.debug("on_iteration_complete callback error: %s", e)

    def _on_subcall_complete(depth: int, model: str, duration: float, error: str | None = None) -> None:
        """Called by the rlm library after each sub-call completes.

        Signature: (depth, model, duration, error) per rlm.core.rlm line 812.
        """
        step_counter["subcalls"] += 1
        step_counter["count"] += 1
        if on_step:
            try:
                asyncio.run_coroutine_threadsafe(
                    on_step({
                        "step_number": step_counter["count"],
                        "step_type": "subcall",
                        "model": model,
                        "executionMs": int(duration * 1000),
                        "error": error,
                    }),
                    main_loop,
                )
            except Exception as e:
                logger.debug("on_subcall_complete callback error: %s", e)

    # Configure RLM instance using actual library API.
    # The emit_artifact tool requires the artifact directory, main loop,
    # and guide JSON to be captured in its closure.
    artifact_dir_path = Path(log_dir) if log_dir else Path(".")
    # Reset the model router signal for this run (Opus for iteration 0)
    _model_router_signal["use_opus_next"] = True

    # Reset the cumulative usage accumulator for this run
    try:
        _reset_run_usage()
    except NameError:
        pass  # patch failed to load

    # Reset the catcher gate's per-artifact pass/fail history (Fix B)
    try:
        _reset_run_artifact_status()
    except NameError:
        pass  # patch failed to load

    # Install the guide as the second cached system block for every Opus/
    # Sonnet call in this run (Fix A). The cleanup in the finally path below
    # clears this so the next run doesn't leak stale guide state.
    try:
        _set_current_guide_json(guide_json)
    except NameError:
        pass  # patch failed to load

    # Gen 2.3: generate the guide TOC preamble via one Opus call BEFORE the
    # first main extraction call dispatches. The TOC is stored in _toc_cache
    # keyed by guide content hash, and _maybe_attach_guide_block reads it
    # synchronously to prepend it to the cached guide block on every
    # subsequent Opus/Sonnet call. This is a one-time ~$2.45 call that
    # primes the generator with a rich focus prior about the guide's
    # structure. On re-runs of the same guide within the same process,
    # the TOC is cached for free.
    try:
        await _generate_guide_toc(guide_json, api_key)
    except NameError:
        pass  # patch failed to load
    except Exception as exc:
        logger.warning(
            "Guide TOC generation failed (non-fatal, run continues without TOC): %s",
            exc,
        )

    # Gen 4: generate or load frozen test suite for this guide.
    # The test suite replaces per-round LLM completeness catchers with
    # deterministic Python matching. Generated once per guide (cached by
    # content hash), then activated for all validate_artifact calls.
    try:
        from backend.validators.test_suite import generate_test_suite
        from backend.validators.phases import set_active_test_suite, clear_active_test_suite

        test_suite = await generate_test_suite(guide_json, api_key)
        set_active_test_suite(test_suite)
        logger.info("Gen4: test suite ready (%d items, hash=%s)",
                     len(test_suite.items), test_suite.guide_hash)
    except Exception as exc:
        logger.warning("Gen4: test suite generation failed (falling back to LLM catchers): %s", exc)
        try:
            clear_active_test_suite()
        except Exception:
            pass

    emit_artifact_fn = _make_emit_artifact_fn(
        run_id=run_id,
        artifact_dir=artifact_dir_path,
        guide_json=guide_json,
        api_key=api_key,
        main_loop=main_loop,
        session_artifacts=session_artifacts,
        on_step=on_step,
        step_counter=step_counter,
        model_router_signal=_model_router_signal,
    )

    # RLM efficiency configuration (audited from rlm library internals):
    #
    # compaction=False:
    #   DISABLED for the gate run. Compaction is an LLM-mediated
    #   nondeterministic step: when the conversation history exceeds the
    #   threshold, the rlm library calls an LLM to summarize older turns.
    #   The summary varies between runs (LLM stochasticity), and the trigger
    #   point varies between runs (small token-count differences shift the
    #   85% threshold by 1-3 iterations). Two identical inputs can therefore
    #   produce divergent message histories from compaction onward, which
    #   directly violates the identification constraint in the failure
    #   analysis. We accept the cost of longer message histories on long
    #   runs in exchange for clean reproducibility. The system prompt's
    #   "build incrementally, don't print huge dicts" instruction keeps
    #   per-iter stdout small enough that the 50-iter trajectory typically
    #   fits in ~150-180K tokens (under Sonnet's 200K limit). If a run
    #   exceeds the limit, it fails with a clear context-limit error rather
    #   than silently compacting and diverging.
    #
    # Parallel sub-calls via llm_query_batched (no explicit concurrency cap):
    #   The installed rlm library's LMHandler processes batched requests via
    #   asyncio.gather on client.acompletion — all prompts in a batch run
    #   concurrently at the async layer (no ThreadPoolExecutor, no cap).
    #   The model has to CALL llm_query_batched([p1, p2, ...]) instead of
    #   a sequential loop for this to activate. The system prompt has been
    #   updated to teach the batched variant as the preferred sub-call API.
    #
    # Safety valves (fail-closed under pathological behavior):
    #   max_budget=20.0    $20 hard ceiling per run
    #   max_errors=5       halt after 5 consecutive code-execution errors
    #   max_timeout=1500   25 minute wall-clock hard cap per run (raised
    #                       from 20 to give the no-compaction runs headroom)
    #   None of these affect happy-path cost/time; they only prevent runaway.
    #
    # Sub-call backend routing:
    #   subcall_model is passed as other_backends/other_backend_kwargs so
    #   llm_query() routes to the configured sub-call model. The gate
    #   harness defaults to Haiku for sub-calls. The argument for Haiku
    #   over Sonnet here is generalization, not cost: if the architecture
    #   is so brittle that the choice between two frontier checkpoints
    #   from the same family materially changes the DMN output, then the
    #   project's hypothesis (that LLMs-as-a-class can extract DMN) is
    #   already false and a model swap won't save it. The gate harness
    #   measures the DELTA between hybrid and all-at-once modes, both
    #   using the same sub-call model, so any sub-call model variance
    #   cancels out of the verdict. Sub-calls are also middle-of-chain
    #   work-doing calls whose output is checked by downstream phase
    #   validators, so the variance is bounded by the catcher gate that
    #   follows them.
    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": model_name,
            "api_key": api_key,
            "max_tokens": 16384,  # Stay under SDK non-streaming timeout guard
        },
        other_backends=["anthropic"],
        other_backend_kwargs=[{
            "model_name": subcall_model,
            "api_key": api_key,
        }],
        environment=environment,
        max_iterations=max_iterations,
        custom_system_prompt=system_prompt,
        custom_tools={
            "validate": _make_validate_fn(
                main_loop=main_loop,
                on_step=on_step,
                step_counter=step_counter,
            ),
            "z3_check": _make_z3_check_fn(
                main_loop=main_loop,
                on_step=on_step,
                step_counter=step_counter,
            ),
            "emit_artifact": emit_artifact_fn,
        },
        compaction=False,  # see comment block above — disabled for identification
        max_budget=50.0,   # raised for ralph-loop mode — let model iterate until clean
        max_errors=10,     # raised — more iterations means more chances for transient errors
        max_timeout=3600.0,  # 60 min — ralph loop needs headroom to iterate to convergence
        logger=rlm_logger,
        verbose=False,
        on_iteration_complete=_on_iteration_complete,
        on_subcall_complete=_on_subcall_complete,
    )

    start_time = time.time()
    trajectory: list[dict] = []

    try:
        # Run the RLM completion with the guide as context payload.
        # The guide is loaded as the `context` variable in the REPL.
        #
        # CRITICAL: rlm.completion() is a SYNCHRONOUS call that blocks for
        # 5-30 minutes while the model iterates. Running it directly inside
        # this async function would block the entire asyncio event loop,
        # freezing health checks, SSE streams, ingestion polling, and
        # cancellation requests for the duration of the extraction. We
        # dispatch it to a worker thread via asyncio.to_thread() so the loop
        # stays responsive. The sync callbacks (_on_iteration_complete,
        # _on_subcall_complete) fire from this worker thread and bridge back
        # to the main loop via the asyncio.run_coroutine_threadsafe() calls
        # configured above.
        result = await asyncio.to_thread(rlm.completion, initial_message)

        # -----------------------------------------------------------
        # Gen 3: Post-RLM surgical repair loop (2026-04-12)
        # Gen 3: repair now runs INSIDE emit_artifact (per-artifact, in the
        # REPL worker thread). By the time we reach this point, each artifact
        # has already been through its repair loop. No post-RLM repair needed.

        # Gen 3 Phase 6: assemble clinical_logic from session_artifacts
        # instead of relying on FINAL_VAR parsing. This is the primary
        # path; the FINAL_VAR parse below is the fallback.
        if all(name in session_artifacts for name in [
            "supply_list", "variables", "predicates", "modules",
            "router", "integrative", "phrase_bank",
        ]):
            clinical_logic = {
                name: session_artifacts[name]
                for name in [
                    "supply_list", "variables", "predicates", "modules",
                    "router", "integrative", "phrase_bank",
                ]
            }
            logger.info("Gen3: assembled clinical_logic from session_artifacts (7/7 present)")
            status = "passed"
        else:
            clinical_logic = None
            status = "passed"  # will be set below based on FINAL_VAR parse

        elapsed = time.time() - start_time

        # Fallback: Parse the FINAL_VAR response if Gen 3 assembly didn't produce clinical_logic
        if clinical_logic is None:
            response_text = getattr(result, "response", None)
            logger.info("FINAL_VAR fallback: response type=%s len=%s", type(response_text).__name__, len(str(response_text or "")))
            if response_text:
                if isinstance(response_text, dict):
                    clinical_logic = response_text
                elif isinstance(response_text, str):
                    try:
                        clinical_logic = json.loads(response_text)
                    except (json.JSONDecodeError, TypeError):
                        # ast.literal_eval is safe: only accepts Python literals
                        # (dict, list, str, int, float, bool, None). It does NOT
                        # execute arbitrary code. Used here because the model
                        # returns Python dict repr with single quotes and True/False.
                        import ast
                        try:
                            clinical_logic = ast.literal_eval(response_text)
                            logger.info("FINAL_VAR parsed via ast.literal_eval")
                        except (ValueError, SyntaxError, MemoryError) as exc:
                            clinical_logic = None
                            logger.warning("Could not parse FINAL_VAR: %s", exc)

                    # Fallback B: brace-balanced extraction
                    if clinical_logic is None:
                        try:
                            import re as _re
                            stripped = _re.sub(
                                r"^```(?:json|python)?\s*|\s*```$", "",
                                response_text.strip(), flags=_re.MULTILINE,
                            )
                            brace_start = stripped.find("{")
                            if brace_start >= 0:
                                depth = 0
                                for idx, ch in enumerate(stripped[brace_start:], start=brace_start):
                                    if ch == "{": depth += 1
                                    elif ch == "}":
                                        depth -= 1
                                        if depth == 0:
                                            candidate = stripped[brace_start : idx + 1]
                                            try:
                                                clinical_logic = json.loads(candidate)
                                            except (json.JSONDecodeError, TypeError):
                                                import ast as _ast
                                                clinical_logic = _ast.literal_eval(candidate)
                                            if clinical_logic is not None:
                                                logger.info("FINAL_VAR parsed via brace-balanced extraction")
                                            break
                        except Exception as exc:
                            logger.debug("Brace-balanced fallback failed: %s", exc)

        # Get usage stats from the result object or backend client
        usage_summary = getattr(result, "usage_summary", None)
        if not usage_summary:
            try:
                usage_summary = rlm.get_usage_summary()
            except Exception:
                pass

        # Validate final output. Guard against malformed parses: the JSON /
        # ast.literal_eval fallback path can return a string, list, or other
        # non-dict type if the model wrapped FINAL_VAR around the wrong
        # variable. run_all_validators expects a dict, so if we got anything
        # else, log a warning, null out clinical_logic, and mark the run as
        # failed instead of crashing.
        validation_errors = 0
        if clinical_logic and not isinstance(clinical_logic, dict):
            logger.warning(
                "FINAL_VAR parsed into a non-dict (%s) -- cannot validate. "
                "The model likely returned the wrong variable. Preview: %r",
                type(clinical_logic).__name__,
                str(clinical_logic)[:200],
            )
            clinical_logic = None
            status = "failed"
        if clinical_logic:
            val_result = run_all_validators(clinical_logic)
            validation_errors = val_result["error_count"]
            if on_step:
                await on_step({
                    "step_number": step_counter["count"] + 1,
                    "step_type": "validate",
                    "validation_result": val_result,
                })

        # Use the live cost accumulator which tracks per-model pricing + caching
        # for all calls (RLM main, catchers, Gen3 repairs).
        try:
            cost_estimate = _run_usage.get("cost_usd", 0.0)
        except NameError:
            cost_estimate = 0.0

        # Get iteration/subcall counts from metadata
        total_iterations = step_counter["count"]
        total_subcalls = step_counter["subcalls"]
        if result.metadata:
            total_iterations = result.metadata.get("total_iterations", total_iterations)
            total_subcalls = result.metadata.get("total_subcalls", total_subcalls)

        logger.info(
            "Extraction complete run=%s status=%s iterations=%s elapsed=%.1fs cost=$%.2f",
            run_id, status, total_iterations, elapsed, cost_estimate,
        )

        return ExtractionResult(
            run_id=run_id,
            status=status,
            clinical_logic=clinical_logic,
            total_iterations=total_iterations,
            total_subcalls=total_subcalls,
            validation_errors=validation_errors,
            cost_estimate_usd=cost_estimate,
            trajectory=trajectory,
            # Hybrid plan: latest version of each intermediate artifact
            supply_list=session_artifacts.get("supply_list"),
            variables=session_artifacts.get("variables"),
            predicates=session_artifacts.get("predicates"),
            modules=session_artifacts.get("modules"),
            router=session_artifacts.get("router"),
            integrative=session_artifacts.get("integrative"),
            phrase_bank=session_artifacts.get("phrase_bank"),
        )

    except Exception as e:
        logger.error("Extraction failed run=%s error=%s", run_id, str(e), exc_info=True)
        return ExtractionResult(
            run_id=run_id,
            status="failed",
            clinical_logic=None,
            total_iterations=step_counter["count"],
            total_subcalls=step_counter["subcalls"],
            validation_errors=0,
            cost_estimate_usd=0.0,
            trajectory=trajectory,
            # Preserve any partial artifacts the model managed to emit before failing
            supply_list=session_artifacts.get("supply_list"),
            variables=session_artifacts.get("variables"),
            predicates=session_artifacts.get("predicates"),
            modules=session_artifacts.get("modules"),
            router=session_artifacts.get("router"),
            integrative=session_artifacts.get("integrative"),
            phrase_bank=session_artifacts.get("phrase_bank"),
        )
    finally:
        # Clear the per-run guide JSON so a later run doesn't leak stale
        # guide state into its cached system block. (Fix A)
        try:
            _set_current_guide_json(None)
        except NameError:
            pass  # patch failed to load
        # Gen 4: clear the active test suite so subsequent runs don't
        # use a stale suite from a different guide.
        try:
            clear_active_test_suite()
        except NameError:
            pass
