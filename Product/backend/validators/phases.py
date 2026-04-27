"""Phase validators: catcher prompts invoked at each artifact emission.

Mapping of artifacts to catchers (Option C: JSON rewrites for intermediate
artifacts, original DMN-focused catchers for final DMN output). Each
artifact now has its own scoped completeness catcher (replaces the generic
completeness_json, which was causing false-positive criticals on
supply_list and variables because its coverage checklist described
module/phrase_bank content):

    supply_list    -> completeness_supply_list + provenance_json
    variables      -> completeness_variables + provenance_json
    predicates     -> completeness_predicates + clinical_review_json + provenance_json
    modules        -> completeness_modules + clinical_review_json + provenance_json + module_architecture_json
    router         -> completeness_router + module_architecture_json + consistency_json
    integrative    -> completeness_integrative + module_architecture_json + consistency_json + comorbidity_coverage
    phrase_bank    -> completeness_phrase_bank
    final_dmn      -> dmn_audit + boundary_condition (runs on XML post-conversion)

Each validator calls Anthropic Claude Haiku (cheap, fast, well-calibrated for
this kind of structural check) with the catcher prompt as the system message
and a user message containing the artifact JSON (and guide JSON where needed).

Validators are BLOCKING: the REPL cannot proceed to the next phase until the
current artifact's validators all pass, or the model explicitly fixes the
critical_issues and re-emits.

Validators are FROZEN per the Levine memo: one system prompt is the sole
experimental control surface. The catcher prompts are utility infrastructure
and must not vary between gate runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from anthropic import AsyncAnthropic

from backend.validators.deterministic import DETERMINISTIC_CATCHERS
from backend.validators.artifact_architecture import ARCHITECTURE_CHECKS

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "validators"

# Catcher model: Haiku for the LLM-based catchers (cheap, fast).
# The memo froze this choice to minimize experimental confounders on the gate.
_CATCHER_MODEL = os.environ.get("CATCHER_MODEL", "claude-haiku-4-5-20251001")
# Max output tokens per catcher call (gen 2.4 hotfix, 2026-04-12).
# Raised from 2000 → 6000 after the first gen 2.4 run observed systematic
# truncation (output=exactly 2000) on multiple catchers:
#   - boundary_condition   : 5/5 voters maxed on every round (DMN audit is
#                            structurally verbose, grows with module count)
#   - completeness_phrase_bank on chunk_large: 5/6 voters maxed
#   - clinical_review_json on modules artifact: 2-3/5 voters maxed per round
#   - dmn_audit: borderline, occasional max hits
# At 2000, the balanced-brace recovery in _extract_json absorbed the free-JSON
# truncations (landed on valid JSON boundaries) and strict-mode decoder
# landed tool_use.input on schema-valid partial dicts, but verdict content
# was silently clipped — criticals arrays would be cut mid-enumeration,
# missing real items. 6000 covers every observed truncation case with a
# comfortable margin; small-output catchers (router, integrative, supply_list
# verdicts at ~400 tokens) pay nothing extra because output is billed on
# actual tokens generated, not on max_tokens.
_CATCHER_MAX_TOKENS = int(os.environ.get("CATCHER_MAX_TOKENS", "6000"))
# LLM catchers run with temperature=0 + N-way majority vote to bound the
# residual stochasticity. N=3 is a sweet spot: triples cost but Anthropic
# Haiku at temp=0 produces 2-of-3 agreement on the same input the vast
# majority of the time.
# Catcher voter temperatures (generation 2, 2026-04-11 → expanded to 5 voters).
#
# See INFRA.md section 9 for the full math derivation. Short version:
# N calls at the same temperature are highly correlated and majority voting
# collapses to single-voter performance. N calls at varied temperatures are
# decorrelated by construction and actual variance reduction kicks in.
#
# Going from 3 → 5 voters reduces per-chunk error rate by an additional
# factor of ~2-3× (not the theoretical 6× from binomial math because voters
# aren't perfectly independent — they share the underlying Haiku distribution
# and only differ in sampling exploration). Combined with 3× prompt repetition
# and per-chunk artifact caching, this is what config D on the Pareto frontier
# looks like in practice.
#
# Chosen quintuple: (0.0, 0.1, 0.2, 0.3, 0.4)
#   0.0:  deterministic grounding baseline
#   0.1:  small variance, mostly same as voter 1 with occasional catches
#   0.2:  moderate exploration
#   0.3:  strong exploration, approaches the safe ceiling
#   0.4:  maximum safe exploration; past this Haiku's JSON structure starts
#         breaking (~0.5 is the knee where malformed JSON rates climb)
#
# The quote-or-drop rule in gen-2 catcher prompts filters out any
# temperature-induced hallucinations: a voter that invents an item without
# a verbatim guide quote fails the prompt's format requirement and the
# _extract_json regex fallback catches it.
#
# N_VOTES is derived from the length of _CATCHER_TEMPERATURES so changing
# the tuple automatically reconfigures the voting.
_CATCHER_TEMPERATURES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
_CATCHER_MAJORITY_N = len(_CATCHER_TEMPERATURES)

# Fix F (2026-04-12): minimum voter corroboration for a critical to survive.
#
# Prior to Fix F, criticals used UNION aggregation: any 1-of-5 voters
# finding a unique critical added it to the result, even if the other 4
# voters disagreed. This caused catcher instability across rounds: a voter
# at temp=0.4 on round 2 might enumerate a guide phrase it missed on round 1,
# flag it as missing, and that single-voter flag would block the artifact
# and appear as a "new" critical despite the artifact being bigger.
#
# Fix F introduces a per-critical vote count. After semantic dedupe, each
# unique critical (by dedupe key) must be independently flagged by at
# least _CRITICAL_VOTE_THRESHOLD voters to survive aggregation. This
# filters out single-voter noise while preserving items that multiple
# decorrelated voters agree are genuinely missing.
#
# Threshold = 2: "at least one other voter confirms." A single voter
# finding something alone is likely noise from temperature-driven
# enumeration variance. Two independent voters agreeing is corroboration.
# Higher thresholds (3, 4) would be too aggressive and drop genuine items.
_CRITICAL_VOTE_THRESHOLD = int(os.environ.get("CRITICAL_VOTE_THRESHOLD", "2"))

# Legacy alias kept for backwards-compat with any external code that
# imports _CATCHER_TEMPERATURE as a scalar. It points at the grounding
# voter's temperature (0.0) so a single-call consumer behaves conservatively.
_CATCHER_TEMPERATURE = _CATCHER_TEMPERATURES[0]

# ---------------------------------------------------------------------------
# Guide chunking config (recall fix, 2026-04-11 → generation 2)
# ---------------------------------------------------------------------------
# See INFRA.md section 9 for the full derivation. Summary:
#
# Chunk size 50K chars (~12.5K tokens) sits in Haiku 4.5's near-perfect
# attention zone (single-needle recall ≥99.5% up to ~50K tokens). The
# smaller window gives each catcher call a focused working set which
# empirically outperforms a single large call on multi-needle recall
# even though the larger call's model is technically smarter.
#
# For the WHO full guide (1.94M chars pretty), 50K chunks produce
# ~25-30 chunks depending on how sections pack. Each chunk gets 3 votes
# at diverse temperatures, so a completeness check fires ~75-90 Haiku
# calls total. With CATCHER_CONCURRENCY_CAP=3 (see below), those run
# in sequential batches of 3 → ~25-30 × ~5s ≈ 2-3 minutes per catcher
# round. The run accepts ~20 extra minutes per full-WHO extraction in
# exchange for the recall gain.
_CHUNK_MAX_CHARS = int(os.environ.get("CATCHER_CHUNK_MAX_CHARS", "50000"))

# Concurrency cap: max Haiku catcher calls in flight across the whole
# backend process. Lowered from 20 → 3 in gen 2 because:
#
# 1. With 50K chunks × 3 votes × 4 catchers × N re-emits, a single run can
#    fire ~100-300 catcher calls total. At cap=20, bursts would temporarily
#    exceed Tier 2 rate limits when all 3 concurrent REPLs fire together.
# 2. Dropping cap to 3 means at most 3 Haiku calls run at once process-wide.
#    A 75-call catcher round on one REPL takes ~125 seconds instead of ~8
#    seconds. Across 3 REPLs sharing the cap, they effectively serialize
#    during catcher rounds. Wall clock per full-WHO run grows from ~15 to
#    ~35 minutes.
# 3. This preserves MAX_CONCURRENT_SESSIONS=3 (three researchers can still
#    start runs at the same time) without any rate-limit risk. The
#    trade-off: wall clock roughly 3x for the catcher phases. Cost
#    unchanged (same call count).
#
# The user can override via env to 20 for a faster-but-risker dev loop,
# or to 1 for a maximally-safe gate baseline.
_CATCHER_CONCURRENCY_CAP = int(os.environ.get("CATCHER_CONCURRENCY_CAP", "3"))

# ---------------------------------------------------------------------------
# Prompt repetition (Google 2512.14982, Dec 2025)
# ---------------------------------------------------------------------------
# Repeating the chunk text N times in the cached system block lets every
# token of the first copy attend to its duplicates in subsequent copies,
# simulating bidirectional attention within the causal attention mechanism.
#
# The Google paper "Prompt Repetition Improves Non-Reasoning LLMs" showed
# this technique wins 47/70 head-to-head tests with zero losses across
# Claude 3 Haiku/Sonnet, GPT, Gemini, DeepSeek. Biggest single result:
# NameIndex task (find the Nth name from a list of 50) went from 21% →
# 97% on Gemini 2.0 Flash-Lite with a single repetition.
#
# Our catcher task is structurally identical to NameIndex: "find the items
# in this chunk that should be in the artifact but aren't". 3× repetition
# is the right choice because the paper showed "×3 often substantially
# outperforms" 2× for list-retrieval tasks specifically.
#
# Since the chunk lives in a cached system block, repetition costs 3× the
# cache write (paid once per unique chunk per 1h TTL) and 3× the cache
# read (~$0.0025 extra per subsequent call). Wall-clock impact: ~10-20%
# on first-write calls, ~0% on cached-read calls.
_PROMPT_REPETITION_N = int(os.environ.get("CATCHER_PROMPT_REPETITION", "3"))

# ---------------------------------------------------------------------------
# Artifact caching (per-chunk-round optimization)
# ---------------------------------------------------------------------------
# Within a single chunk-round, all N voters see the SAME artifact bytes.
# Without caching, each vote re-sends the full artifact as billed input.
# With 5 voters × 23 chunks × 12 rounds = 1380 redundant artifact sends
# per full-WHO run. At ~15K tokens per artifact, that's 20M wasted tokens.
#
# Fix: cache the artifact as a third system block with 5-minute TTL. Vote
# 1 writes the cache; votes 2-5 read it. 5m TTL is Anthropic's minimum
# (there is no 1m option) and is comfortably longer than the ~15-20s it
# takes for 5 voters at cap=3 to complete a chunk-round.
#
# Artifact size threshold (corrected 2026-04-11): Anthropic's prompt-caching
# docs specify 4096 tokens as the minimum cacheable block size on Haiku 4.5
# (not 2048 — that number applied to Haiku 3.x). Artifacts below ~16K chars
# (~4K tokens) will silently bypass the cache because the `cache_control`
# marker gets ignored. We use a 20K-char floor (~5K tokens) for a comfortable
# margin over the 4096-token minimum, so the cache always actually fires
# when the marker is set.
#
# Cache-key determinism: Anthropic caches the exact byte prefix, so the
# serialized artifact must be byte-identical across votes within the same
# round. We use `_coerce_json_safe` (from db.py) to normalize the artifact
# shape before serialization, then json.dumps with sort_keys=True for
# deterministic ordering.
_ARTIFACT_CACHE_MIN_CHARS = int(os.environ.get("CATCHER_ARTIFACT_CACHE_MIN_CHARS", "20000"))

# ---------------------------------------------------------------------------
# Strict catcher mode (gen 2.4, 2026-04-11)
# ---------------------------------------------------------------------------
# Anthropic's strict tool use (output_config / strict:true) lets us force the
# catcher's output into a schema-validated tool call. The model emits a
# tool_use block whose `input` is grammar-constrained to our schema, and we
# read the verdict directly from `response.content[0].input` without any
# JSON parsing, markdown-fence recovery, or balanced-brace scanning. This
# eliminates the entire class of Run #8-style parse failures where catcher
# output got wrapped in ```json ... ``` fences and _extract_json had to go
# through multiple fallback paths to recover.
#
# Why we dropped Citations API (gen 2.2 → gen 2.4):
#   Per Anthropic's structured-outputs docs, Citations is INCOMPATIBLE with
#   `output_config.format` / strict tool mode — enabling both returns a 400
#   error. We picked strict mode because:
#     1. Citations were already dormant on our JSON-output catchers (gen 2.2
#        notes in INFRA.md:893 confirm this) — they rarely attached because
#        Haiku's JSON output isn't natural prose.
#     2. The load-bearing quote-grounding mechanism has always been
#        programmatic substring verification (guide_quote ∈ chunk_text),
#        not Citations. That check still runs in gen 2.4 and doesn't need
#        Citations to work.
#     3. Strict mode eliminates an entire parse-failure class that Citations
#        couldn't help with.
#
# Scope: strict mode applies ONLY to guide-dependent (completeness_*)
# catchers. Non-completeness catchers (clinical_review, provenance,
# module_architecture, consistency, comorbidity) stay on the free-JSON path
# because their output shapes are simpler and their parse-failure rate is
# already near-zero. Mixing the two paths lets us minimize surface area in
# the gen 2.4 change.
_CATCHER_STRICT_MODE = os.environ.get("CATCHER_STRICT_MODE", "true").lower() in ("true", "1", "yes")

# Module-level tool schema used by all guide-dependent (completeness_*) catchers.
# MUST be a module-level constant (not constructed per-call) so the `tools`
# array is byte-identical across requests. If we rebuilt the dict per call,
# Python dict ordering variance would shift the tool-level cache key and
# invalidate the tools cache between calls, losing cache hits on the system
# blocks downstream.
#
# Schema rationale:
#   - `passed`, `enumeration`, `critical_issues`, `warnings` all REQUIRED.
#     No optional params (stays well under the 24-optional-param strict limit).
#   - No union types (stays under the 16 union-param strict limit).
#   - `critical_issues` is an array of 4-field dicts matching the gen 2.2
#     dict-critical shape (suggested_id, description, guide_quote, section).
#   - `enumeration` is an array of 2-field dicts (guide_quote, section).
#     The third field varies by catcher type (suggested_module for modules,
#     category for phrase_bank, etc.) and is advisory-only — no downstream
#     Python code reads `enumeration`. Simplifying to 2 fields means the
#     same schema works for all 5 dict-format catchers without losing any
#     functional behavior. Under strict mode `additionalProperties: false`
#     would forbid a third field anyway.
#   - `additionalProperties: False` on every nested object — mandatory for
#     Anthropic strict mode.
#
# There is ONE tool shared across all strict-mode catchers because the
# output shape is identical for all of them. Router and integrative
# completeness catchers use the simpler string-critical format and stay
# on the free-JSON path (see _STRICT_MODE_CATCHERS below).
STRICT_CATCHER_VERDICT_TOOL: dict[str, Any] = {
    "name": "catcher_verdict",
    "description": (
        "Return the recall-audit verdict for the given artifact. "
        "Populate `enumeration` with every target-type item found in the "
        "chunk (stage 1), and `critical_issues` with the items that are in "
        "the enumeration but missing from the artifact (stage 2). Set "
        "`passed` to true only when `critical_issues` is empty."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {
                "type": "boolean",
                "description": (
                    "True if the audit found zero critical recall failures. "
                    "Must be false if `critical_issues` is non-empty."
                ),
            },
            "enumeration": {
                "type": "array",
                "description": (
                    "Stage-1 output: every target-type item present in the "
                    "chunk, with its verbatim quote and section id. "
                    "Advisory only — used for observability, not compared "
                    "programmatically against the artifact."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "guide_quote": {"type": "string"},
                        "section": {"type": "string"},
                    },
                    "required": ["guide_quote", "section"],
                    "additionalProperties": False,
                },
            },
            "critical_issues": {
                "type": "array",
                "description": (
                    "Stage-2 output: enumeration entries that are NOT "
                    "represented in the artifact. Each critical must include "
                    "a verbatim quote from the guide — items without a quote "
                    "will be dropped by programmatic verification."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "suggested_id": {"type": "string"},
                        "description": {"type": "string"},
                        "guide_quote": {"type": "string"},
                        "section": {"type": "string"},
                        "repair_instruction": {
                            "type": "string",
                            "description": (
                                "A concrete, actionable instruction telling the "
                                "generator model HOW to add this item. Include "
                                "artifact-specific field values derived from the "
                                "guide quote (e.g. for predicates: the threshold "
                                "expression and source variable; for modules: "
                                "which input columns to use; for phrase_bank: "
                                "the category and literal text). The generator "
                                "will copy this instruction directly into its "
                                "repair code."
                            ),
                        },
                    },
                    "required": [
                        "suggested_id",
                        "description",
                        "guide_quote",
                        "section",
                        "repair_instruction",
                    ],
                    "additionalProperties": False,
                },
            },
            "warnings": {
                "type": "array",
                "description": (
                    "Non-critical observations that should not block the run."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["passed", "enumeration", "critical_issues", "warnings"],
        "additionalProperties": False,
    },
}

# The `tools` array passed on strict-mode catcher calls. Single-element list
# wrapping the shared verdict tool. Reference identity is stable because
# both the outer list and the inner dict are module-level constants.
_STRICT_CATCHER_TOOLS: list[dict[str, Any]] = [STRICT_CATCHER_VERDICT_TOOL]
_STRICT_CATCHER_TOOL_CHOICE: dict[str, Any] = {
    "type": "tool",
    "name": STRICT_CATCHER_VERDICT_TOOL["name"],
}

# ---------------------------------------------------------------------------
# Contextual chunk headers (gen 2.2, Anthropic Contextual Retrieval pattern)
# ---------------------------------------------------------------------------
# Before each chunk is sent to a catcher, we generate a ~100-token contextual
# header via one Haiku call that situates the chunk inside the full guide.
# Example (generic, Haiku fills this in per chunk): "This is chunk 7 of 23
# from the <section title> section of the source manual. It covers
# <topic A>, <procedure B>, and <treatment C>."
#
# The header is prepended to the chunk content before caching. This gives
# each catcher call a focus prior about what the chunk SHOULD contain,
# which sharpens the completeness audit per Anthropic's 35-49% RAG
# retrieval failure reduction benchmark (reduced further to 67% when
# combined with reranking, which we don't do).
#
# Headers are cached in-process by chunk content hash so we pay the
# generation cost once per unique chunk per process lifetime, not once
# per catcher call. ~$0.005 per chunk × 23 chunks = ~$0.12 one-time per
# unique guide per process.
_CATCHER_CONTEXTUAL_HEADERS = os.environ.get("CATCHER_CONTEXTUAL_HEADERS", "true").lower() in ("true", "1", "yes")
_chunk_header_cache: dict[str, str] = {}  # sha256 → generated header text

# Semaphore bounding concurrent Haiku calls across all chunked catcher runs
# in the process. Created lazily because asyncio locks must be created inside
# a running event loop.
_catcher_semaphore: Optional[asyncio.Semaphore] = None


def _get_catcher_semaphore() -> asyncio.Semaphore:
    """Lazy-create the process-global concurrency cap semaphore."""
    global _catcher_semaphore
    if _catcher_semaphore is None:
        _catcher_semaphore = asyncio.Semaphore(_CATCHER_CONCURRENCY_CAP)
    return _catcher_semaphore


# ---------------------------------------------------------------------------
# Contextual chunk headers (Anthropic Contextual Retrieval pattern)
# ---------------------------------------------------------------------------


def _chunk_hash(chunk: Any) -> str:
    """SHA-256 hash of a chunk's canonical JSON representation. Used as the
    cache key for contextual headers so we only generate a header once per
    unique chunk per process lifetime, regardless of how many catcher calls
    reference that chunk."""
    import hashlib
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


async def _get_or_build_chunk_header(chunk: Any, api_key: str) -> str:
    """Return a cached contextual header for `chunk`, generating it on first
    use via one Haiku call.

    The header is a ~100-token plain-English summary that situates the
    chunk within the full document: what sections it covers, what topics
    it describes, what clinical area it addresses. This gives each catcher
    call a focus prior so its attention is already pre-aimed at the right
    content before it starts the completeness audit.

    Caching: results are memoized by chunk content hash in an in-process
    dict. Two catcher calls on the same chunk (different voters, different
    re-emit rounds) reuse the same header and pay zero additional cost.

    Cost: ~$0.005 per first-ever catcher call on a new chunk; $0 thereafter.
    For a 23-chunk full-WHO run that's ~$0.12 one-time per process.

    Failure mode: if the header generation fails (network, auth, API
    error), the caller logs a debug message and proceeds with an empty
    header — gen 2.2 continues to work, just without the Contextual
    Retrieval focus prior.
    """
    h = _chunk_hash(chunk)
    cached = _chunk_header_cache.get(h)
    if cached is not None:
        return cached

    # Serialize the chunk; for the header we cap at ~12K chars (enough to
    # describe structure without wasting tokens on full content).
    chunk_preview_json = json.dumps(chunk, indent=2)[:12000]

    # Short, deterministic, one-shot: Haiku, temp 0, small max_tokens.
    client = AsyncAnthropic(api_key=api_key)
    prompt = (
        "You will receive one chunk of a larger clinical manual that has been "
        "split at section boundaries. Write a 2-3 sentence header that situates "
        "this chunk inside the broader guide: which sections it contains, what "
        "clinical topics it covers, and what type of content (procedures, "
        "thresholds, treatment instructions, etc.) a reader should expect. "
        "Do NOT quote the guide. Do NOT invent content. Do NOT use emojis. "
        "Keep it under 100 words.\n\n"
        "Output only the header text, no preamble.\n\n"
        f"CHUNK:\n{chunk_preview_json}"
    )
    try:
        response = await client.messages.create(
            model=_CATCHER_MODEL,
            max_tokens=200,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        header_text = ""
        for block in (response.content or []):
            btext = getattr(block, "text", None)
            if btext:
                header_text += btext
        header_text = header_text.strip()

        # Record usage against the catcher accumulator
        try:
            from backend.rlm_runner import accumulate_catcher_usage
            usage = response.usage
            accumulate_catcher_usage(
                model=_CATCHER_MODEL,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            )
        except (ImportError, AttributeError):
            pass

        # Wrap the header in a clear marker so the downstream catcher can
        # distinguish it from the raw chunk content
        framed = f"CHUNK CONTEXT:\n{header_text}"
        _chunk_header_cache[h] = framed
        logger.info(
            "Generated contextual header for chunk (hash=%s): %s",
            h[:8], header_text[:120],
        )
        return framed
    except Exception as exc:
        logger.debug("Header generation failed for chunk %s: %s", h[:8], exc)
        # Cache the empty string so we don't retry on every call
        _chunk_header_cache[h] = ""
        return ""


_CRITICAL_KEY_RE = __import__("re").compile(
    # Match the _flatten_critical output format:
    #   "MISSING <sid>: <desc>. GUIDE QUOTE: '<quote>'. SECTION: <section>."
    # with optional `[catcher_name] ` prefix and tolerant of minor whitespace.
    # Non-greedy on desc so we don't swallow GUIDE QUOTE; non-greedy on quote
    # so we don't run past the closing single quote when descriptions contain
    # apostrophes.
    r"(?:\[[^\]]+\]\s+)?"
    r"MISSING\s+(?P<sid>[^:]+?):\s*.*?"
    r"GUIDE QUOTE:\s*'(?P<quote>.*?)'\s*\.\s*"
    r"SECTION:\s*(?P<section>[^.]+?)\.?\s*$",
    __import__("re").DOTALL,
)


def _critical_dedupe_key(critical_str: str):
    """Produce a dedupe key from a flattened critical string.

    For structured criticals matching the `MISSING ... GUIDE QUOTE: ...
    SECTION: ...` format, returns a `(sid, quote, section)` tuple so that
    voters with identical structural triples but different descriptions
    collapse to a single entry on majority-vote aggregation.

    For non-matching strings (unstructured criticals from free-JSON
    catchers, API error messages, etc.), returns a whitespace-normalized
    lowercase string so identical error messages still dedupe.

    Used in `_call_catcher_majority` to prevent voter count from
    artificially inflating the critical_issues list on semantically
    identical items. See gen 2.4 hotfix notes (2026-04-12) — observed
    run had 9 variables criticals where 2 were structural duplicates
    from voters at temp 0.0 and temp 0.2.
    """
    s = critical_str.strip()
    m = _CRITICAL_KEY_RE.match(s)
    if m:
        return (
            m.group("sid").strip().lower(),
            m.group("quote").strip().lower(),
            m.group("section").strip().lower(),
        )
    return " ".join(s.lower().split())


@dataclass
class PhaseValidationResult:
    """Result of running all catchers for one artifact."""
    artifact_name: str
    phase: int
    passed: bool
    critical_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catcher_outputs: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "artifact_name": self.artifact_name,
            "phase": self.phase,
            "passed": self.passed,
            "critical_issues": self.critical_issues,
            "warnings": self.warnings,
            "catcher_outputs": self.catcher_outputs,
        }


# ---------------------------------------------------------------------------
# Catcher prompt loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _load_catcher(name: str) -> str:
    """Load a catcher prompt from disk. Cached per-process.

    Catcher files live in backend/prompts/validators/. Missing files raise
    at first call — fail-fast so a misconfigured install doesn't pretend to
    validate and silently pass everything.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise RuntimeError(
            f"Catcher prompt not found: {path}. "
            f"Expected one of: provenance_json, clinical_review_json, "
            f"consistency_json, completeness_json, module_architecture_json, "
            f"comorbidity_coverage, dmn_audit, boundary_condition, "
            f"completeness_supply_list, completeness_variables, "
            f"completeness_predicates, completeness_modules, "
            f"completeness_router, completeness_integrative, "
            f"completeness_phrase_bank."
        )
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Anthropic call helper
# ---------------------------------------------------------------------------


async def _call_catcher(
    catcher_name: str,
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
    extra_context: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dict:
    """Invoke one catcher on one artifact. Returns the parsed JSON response.

    Args:
        catcher_name: file stem under backend/prompts/validators/
        artifact: the intermediate JSON object being validated
        guide_json: the full source guide (only needed for catchers that cross-reference
                    source text, e.g. provenance_json and completeness_json)
        api_key: Anthropic API key (BYOK, passed from the session)
        extra_context: optional extra string appended before the artifact
        temperature: sampling temperature for this specific call. Defaults to
                    _CATCHER_TEMPERATURES[0] (0.0) when not supplied. When
                    called from _call_catcher_majority with the gen-2
                    diverse-temperature voting pattern, the caller passes
                    one of _CATCHER_TEMPERATURES per vote index so the
                    three votes decorrelate by construction.

    Returns:
        {"passed": bool, "critical_issues": [...], "warnings": [...]} on success.
        On any failure, returns a result with passed=False and one critical issue
        describing the failure — validators must fail CLOSED, not open.
    """
    if temperature is None:
        temperature = _CATCHER_TEMPERATURES[0]
    try:
        system_prompt = _load_catcher(catcher_name)
    except Exception as exc:
        return {
            "passed": False,
            "critical_issues": [f"Catcher {catcher_name} not loadable: {exc}"],
            "warnings": [],
        }

    # Build the system and user messages.
    #
    # Gen 2.4 cache layout (single path, no Citations branch):
    #   - system block 1: catcher prompt                  (cached 1h, ~700 tok)
    #   - system block 2: GUIDE chunk × 3 repetitions     (cached 1h, ~37K tok)
    #   - system block 3: ARTIFACT (if large enough)      (cached 5m, ~15K tok)
    #   - user message: chunk_context_hint                (NOT cached, <1K tok)
    #
    # This is the gen 2.1 layout brought back. Gen 2.2 briefly put the chunk
    # in a citations-enabled `document` block inside the user message; gen
    # 2.4 reverts that because (a) Citations is incompatible with strict
    # tool mode, and (b) Citations were mostly dormant on our JSON-output
    # catchers even when they did fire. Dropping Citations also restores
    # gen 2.1's 3× prompt repetition (which was blocked in gen 2.2 because
    # repeated chunk text would break Citations' character-index pointers).
    #
    # Three layers of caching with different TTLs:
    #
    # Block 1 (catcher prompt, 1h TTL):
    #   Shared across all calls for the same catcher. Writes once per
    #   catcher per run, reads for free on every subsequent call. Tiny
    #   — ~700 tokens — but caching is still mathematically worthwhile.
    #
    # Block 2 (guide chunk × 3 repetitions, 1h TTL):
    #   The prompt-repetition trick from Google 2512.14982. Including the
    #   chunk 3 times in the cached block lets every token attend to every
    #   other token across the three copies, simulating bidirectional
    #   attention within the causal mechanism. The paper shows this turns
    #   21% → 97% accuracy on list-retrieval tasks (NameIndex). Our
    #   completeness checks are structurally identical to NameIndex.
    #   The cached block contains:
    #     "GUIDE:\n<chunk_json>\n---\nGUIDE (repeat 1):\n<chunk_json>\n---\nGUIDE (repeat 2):\n<chunk_json>"
    #   Cache write/read costs scale 3×, but the 3× is applied inside the
    #   cache layer so it only matters on write. Reads after the first
    #   vote on each chunk are near-free against TPM.
    #
    # Block 3 (artifact, 5m TTL):
    #   Shared across all N voters on one chunk-round. Vote 1 of each
    #   chunk-round writes this cache; votes 2-N read it. 5m TTL is
    #   Anthropic's minimum (there is no 1m option) and is ~15× longer
    #   than the ~20s window of a 5-voter chunk-round at cap=3.
    #
    #   Only cached if the artifact is ≥ _ARTIFACT_CACHE_MIN_CHARS (default
    #   20K chars ≈ 5K tokens) — smaller artifacts are below Anthropic's
    #   4096-token minimum for Haiku 4.5 cache blocks and the cache_control
    #   marker would be silently ignored. For small artifacts we skip the
    #   third block and put the artifact in the user message instead.
    #
    # Why artifact caching is the prerequisite for 5 voters on Tier 2:
    #   With 5 voters and NO artifact caching, each chunk-round sends
    #   5 × artifact_tokens as billed input. Across 23 chunks × 12 rounds
    #   that's ~1.7M tokens/round — over Tier 2's 450K input TPM ceiling.
    #   With artifact caching, it collapses to ~15K billed tokens per
    #   chunk-round (only vote 1's write counts) — well under the ceiling.
    # Serialize the artifact with a deterministic key ordering so the
    # cache key is byte-identical across the N voters within a chunk-round.
    # sort_keys=True guarantees Python dict iteration order doesn't change
    # the cache key; _coerce_json_safe normalizes sets/tuples/numeric-keys
    # into plain JSON types before serialization.
    try:
        from backend.db import _coerce_json_safe
        artifact_normalized = _coerce_json_safe(artifact)
    except ImportError:
        artifact_normalized = artifact
    artifact_str = json.dumps(artifact_normalized, indent=2, sort_keys=True)
    if len(artifact_str) > 200_000:
        logger.warning(
            "Catcher %s: artifact truncated (%d -> 200000 chars). "
            "Consider artifact-level chunking for large extractions.",
            catcher_name, len(artifact_str),
        )
        artifact_str = artifact_str[:200_000] + "\n... [TRUNCATED — artifact exceeds 200K chars]"
    artifact_text = "ARTIFACT:\n" + artifact_str

    # Optionally fetch/generate the contextual header for this chunk.
    # Adds a ~100-token situational summary prepended to the chunk text.
    # No-op for the None-guide case (artifact-only catchers like clinical_review).
    chunk_context_header = ""
    if guide_json is not None and _CATCHER_CONTEXTUAL_HEADERS:
        try:
            chunk_context_header = await _get_or_build_chunk_header(
                guide_json, api_key
            )
        except Exception as exc:
            logger.debug("Chunk header generation failed (non-fatal): %s", exc)
            chunk_context_header = ""

    # ---- Gen 2.4 layout: cached system blocks, no Citations ----
    # Layout (for guide-dependent catchers):
    #   system[0] = catcher prompt           (cached 1h, ~700 tok)
    #   system[1] = GUIDE chunk × 3 repeats  (cached 1h, ~37K tok)
    #   system[2] = ARTIFACT                 (cached 5m, only if large enough)
    #   messages[0].user = chunk_context_hint (small, NOT cached)
    #
    # For artifact-only catchers (guide_json is None), system[1] and system[2]
    # may be omitted or collapsed depending on artifact size.
    #
    # This is the gen 2.1 layout resurrected; gen 2.2 briefly moved the
    # chunk into a citations-enabled document block in the user message, but
    # Citations are incompatible with strict tool mode (returns 400), and
    # Citations were already dormant on our JSON-output catchers per our own
    # gen 2.2 notes. Killing Citations reverts the chunk to a cached system
    # block, which also restores gen 2.1's 3× prompt repetition.
    system_blocks: list[dict] = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    if guide_json is not None:
        chunk_text = json.dumps(guide_json, indent=2)
        # Prepend contextual header if enabled
        if chunk_context_header:
            chunk_text = chunk_context_header + "\n\n" + chunk_text
        if _PROMPT_REPETITION_N > 1:
            repeated_parts = [f"GUIDE:\n{chunk_text}"]
            for i in range(1, _PROMPT_REPETITION_N):
                repeated_parts.append(
                    f"\n\n--- REPEATED FOR ATTENTION (copy {i + 1} of {_PROMPT_REPETITION_N}) ---\n\n"
                    f"GUIDE:\n{chunk_text}"
                )
            guide_block_text = "".join(repeated_parts)
        else:
            guide_block_text = "GUIDE:\n" + chunk_text
        system_blocks.append({
            "type": "text",
            "text": guide_block_text,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        })

    cache_artifact = len(artifact_text) >= _ARTIFACT_CACHE_MIN_CHARS
    if cache_artifact:
        system_blocks.append({
            "type": "text",
            "text": artifact_text,
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
        })

    user_parts: list[str] = []
    if extra_context:
        user_parts.append(extra_context)
    if not cache_artifact:
        user_parts.append(artifact_text)
    user_message_str = "\n\n".join(user_parts) if user_parts else "Audit the artifact provided in the system context."

    # Gen 2.4: strict mode only for catchers that emit the dict-critical
    # shape matching STRICT_CATCHER_VERDICT_TOOL's schema. This is the
    # 5-catcher subset of _GUIDE_DEPENDENT_CATCHERS — excludes router and
    # integrative (which use string criticals) and all non-completeness
    # catchers (clinical_review, provenance, module_architecture,
    # consistency, comorbidity). Everything else stays on the free-JSON
    # path because their output shapes don't match the strict schema.
    use_strict = _CATCHER_STRICT_MODE and catcher_name in _STRICT_MODE_CATCHERS

    try:
        client = AsyncAnthropic(api_key=api_key)
        # Retry loop: chunked catchers can burst 20+ parallel calls on large
        # guides. On Tier 1 rate limits (50 RPM) we'll hit RateLimitError; on
        # Tier 2+ we only hit transient 500s. Retry with exponential backoff
        # so a single 429 doesn't poison a whole catcher round. The main
        # extraction path has an equivalent retry loop in rlm_runner._call_with_retry.
        import anthropic as _anthropic_sdk
        backoffs = [2, 5, 15]
        user_message_payload: Any = user_message_str

        # Gen 2.4 strict mode: pass tools + forced tool_choice on guide-
        # dependent catchers. The grammar-constrained sampling guarantees the
        # response is a tool_use block whose `input` matches the schema, so
        # we never need _extract_json recovery on this path. The `tools`
        # array and `tool_choice` dict are module-level constants so the
        # tools-level cache key stays byte-identical across calls.
        create_kwargs: dict[str, Any] = {
            "model": _CATCHER_MODEL,
            "max_tokens": _CATCHER_MAX_TOKENS,
            "temperature": temperature,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_message_payload}],
        }
        if use_strict:
            create_kwargs["tools"] = _STRICT_CATCHER_TOOLS
            create_kwargs["tool_choice"] = _STRICT_CATCHER_TOOL_CHOICE

        response = None
        last_exc: Optional[Exception] = None
        for attempt in range(4):
            try:
                response = await client.messages.create(**create_kwargs)
                break
            except (
                _anthropic_sdk.APIConnectionError,
                _anthropic_sdk.APITimeoutError,
                _anthropic_sdk.InternalServerError,
                _anthropic_sdk.RateLimitError,
            ) as exc:
                last_exc = exc
                if attempt >= len(backoffs):
                    raise
                delay = backoffs[attempt]
                logger.info(
                    "Catcher %s transient error (attempt %d/4): %s — retrying in %ds",
                    catcher_name, attempt + 1, type(exc).__name__, delay,
                )
                await asyncio.sleep(delay)
            except _anthropic_sdk.BadRequestError as exc:
                # Gen 2.4 hotfix: Anthropic occasionally returns a 400
                # `invalid_request_error` with body "Not Found" on strict-mode
                # catcher calls that hit large cached prefixes. The first gen
                # 2.4 run observed this twice in ~150 catcher calls (on
                # completeness_modules chunk_large and completeness_phrase_bank
                # chunk_large), different request IDs, so it's a transient
                # server-side condition not a permanent config issue.
                #
                # We retry ONLY when the error body contains "Not Found" —
                # generic 400s (schema violations, bad model names, etc.) are
                # permanent errors and should fail fast so they get flagged
                # in catcher_outputs as criticals.
                body_str = str(getattr(exc, "body", None) or getattr(exc, "message", "") or exc)
                if "Not Found" not in body_str:
                    raise  # permanent 400, fail fast
                last_exc = exc
                if attempt >= len(backoffs):
                    raise
                delay = backoffs[attempt]
                logger.info(
                    "Catcher %s transient 400 Not Found (attempt %d/4): %s — retrying in %ds",
                    catcher_name, attempt + 1, body_str[:200], delay,
                )
                await asyncio.sleep(delay)
        if response is None:
            raise last_exc or RuntimeError(
                f"Catcher {catcher_name} failed after retries"
            )

        # Read the verdict from the response.
        #
        # Gen 2.4 strict path: the response contains a tool_use block whose
        # `input` attribute is the grammar-validated verdict dict. We read
        # it directly — no JSON parsing, no markdown-fence recovery. If the
        # model managed to not produce a tool_use block (which strict mode
        # shouldn't allow), we fail closed with a descriptive error.
        #
        # Free-JSON path (non-completeness catchers): walk text blocks to
        # reconstruct the raw model output, then _extract_json the result.
        parsed: Optional[dict] = None
        raw: str = ""
        if use_strict:
            for block in (response.content or []):
                if getattr(block, "type", None) == "tool_use":
                    tool_input = getattr(block, "input", None)
                    if isinstance(tool_input, dict):
                        parsed = tool_input
                        break
            if parsed is None:
                logger.warning(
                    "Catcher %s strict-mode call returned no tool_use block "
                    "(this should be impossible under strict mode — check "
                    "Anthropic model version + strict mode availability)",
                    catcher_name,
                )
        else:
            raw_parts: list[str] = []
            for block in (response.content or []):
                btext = getattr(block, "text", None)
                if btext:
                    raw_parts.append(btext)
            raw = "".join(raw_parts)

        # Record Haiku catcher usage in the global _run_usage accumulator
        # so the Cost Tracker on the frontend sees catcher calls. Without
        # this, every catcher LLM call is invisible (the accumulator only
        # hears from the main extraction path via _completion_with_cache).
        # Best-effort: if the rlm_runner patch isn't loaded, just skip.
        usage = response.usage
        _billed_input = getattr(usage, "input_tokens", 0) or 0
        _cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        _cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        _output = getattr(usage, "output_tokens", 0) or 0
        try:
            from backend.rlm_runner import accumulate_catcher_usage
            accumulate_catcher_usage(
                model=_CATCHER_MODEL,
                input_tokens=_billed_input,
                output_tokens=_output,
                cached_tokens=_cache_read,
                cache_write_tokens=_cache_write,
            )
        except (ImportError, AttributeError) as usage_exc:
            logger.debug("Catcher usage accumulator unavailable: %s", usage_exc)

        # Gen 2.4 cache verification log. Every catcher call logs its cache
        # token breakdown at INFO so we can eyeball whether caching is
        # actually firing as predicted. Expected pattern per chunk-round:
        #   - voter 1 (temp=0.0):   large cache_write, small cache_read
        #   - voters 2..5 (temp>0): zero cache_write, large cache_read
        # If voters 2..5 also show cache_write tokens, the cache-write race
        # protection in `_call_catcher_majority` is broken (all voters are
        # racing on the same cache key instead of serializing voter 1).
        # If voter 1 shows zero cache_write, the tools/system block layout
        # is below Haiku 4.5's 4096-token minimum.
        _mode_tag = "STRICT" if use_strict else "FREE_JSON"
        logger.info(
            "[gen2.4-cache] catcher=%s mode=%s temp=%.2f "
            "billed_input=%d cache_write=%d cache_read=%d output=%d",
            catcher_name, _mode_tag, temperature,
            _billed_input, _cache_write, _cache_read, _output,
        )
    except Exception as exc:
        logger.warning("Catcher %s API call failed: %s", catcher_name, exc)
        return {
            "passed": False,
            "critical_issues": [f"Catcher {catcher_name} API error: {exc}"],
            "warnings": [],
        }

    # On the free-JSON path, extract the JSON object from the response. The
    # catcher prompts all instruct "return ONLY the JSON object" but models
    # occasionally wrap it in markdown code fences or prose. Be defensive.
    # The strict path already populated `parsed` from the tool_use block, so
    # we only run _extract_json when strict mode is OFF or unavailable.
    if parsed is None:
        parsed = _extract_json(raw)
    if parsed is None:
        return {
            "passed": False,
            "critical_issues": [
                f"Catcher {catcher_name} returned non-parseable response: "
                f"{raw[:200] if raw else '<empty>'}"
            ],
            "warnings": [],
        }

    # Normalize the result shape. Fail closed if required keys are missing.
    passed = bool(parsed.get("passed", False))
    critical = parsed.get("critical_issues") or []
    warnings = parsed.get("warnings") or []
    if not isinstance(critical, list):
        critical = [critical]
    if not isinstance(warnings, list):
        warnings = [warnings]

    # Dict-typed criticals (introduced in gen 2.2 for Citations compatibility,
    # retained in gen 2.4 because they match the strict-tool schema). Flatten
    # to strings for downstream aggregation, which expects a list of strings.
    # Preserve the structured raw form in `_structured_criticals` for
    # observability.
    #
    # Gen 2.4 hotfix (2026-04-12): returns Optional[str] — empty/placeholder
    # dicts return None and are filtered at the call site. Before the fix,
    # an empty dict `{}` was stringified as
    #   "MISSING <unknown_id>: . GUIDE QUOTE: ''. SECTION: <unknown_section>."
    # which propagated into the catcher verdict as a fake critical. The first
    # gen 2.4 run failed `modules` entirely because clinical_review_json
    # emitted one empty-dict critical in the `critical_issues` array (likely
    # from a truncated or malformed partial response) and that placeholder
    # string failed the fail-closed aggregation.
    def _flatten_critical(item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item if item.strip() else None
        if isinstance(item, dict):
            sid = item.get("suggested_id")
            desc = item.get("description")
            quote = item.get("guide_quote")
            section = item.get("section")
            # Drop if ALL meaningful fields are missing or blank. A critical
            # without either a suggested_id or a description is not actionable;
            # no downstream consumer can use it to guide a re-emit, and the
            # fail-closed aggregation will bogusly fail the gate on it.
            has_content = bool(
                (sid and str(sid).strip())
                or (desc and str(desc).strip())
                or (quote and str(quote).strip())
            )
            if not has_content:
                return None
            sid_str = sid or "<unknown_id>"
            desc_str = desc or ""
            quote_str = quote or ""
            section_str = section or "<unknown_section>"
            return (
                f"MISSING {sid_str}: {desc_str}. "
                f"GUIDE QUOTE: '{quote_str}'. SECTION: {section_str}."
            )
        return str(item) if item is not None else None

    def _flatten_many(items: list[Any]) -> list[str]:
        """Apply _flatten_critical to a list and drop None (empty) entries."""
        flattened = [_flatten_critical(c) for c in items]
        return [c for c in flattened if c is not None]

    structured_criticals = [
        c for c in critical
        if isinstance(c, dict) and _flatten_critical(c) is not None
    ]
    structured_warnings = [
        w for w in warnings
        if isinstance(w, dict) and _flatten_critical(w) is not None
    ]

    # Programmatic quote verification (load-bearing in gen 2.4 — Citations
    # were dropped because they're incompatible with strict mode and were
    # already dormant on our JSON-output catchers).
    #
    # For each structured critical, check that its `guide_quote` field is a
    # real substring of the chunk. Quotes that can't be found are
    # HALLUCINATED and we drop them. This is the sole mechanism that
    # enforces the "quote or drop" rule in gen 2.4 — strict mode guarantees
    # the shape, this guarantees the grounding.
    if structured_criticals and guide_json is not None:
        # Build a lowercased haystack once. Use the raw chunk text
        # (not pretty-printed) because the model might match across
        # JSON whitespace, and the raw form preserves every character.
        chunk_haystack = json.dumps(guide_json).lower()
        verified_criticals = []
        dropped_count = 0
        for sc in structured_criticals:
            quote = (sc.get("guide_quote") or "").strip()
            if not quote:
                # No quote → not verifiable → drop (quote-or-drop rule)
                dropped_count += 1
                continue
            # Normalize whitespace in the needle to handle minor formatting
            # differences (extra spaces, newlines) between guide source and
            # model output.
            needle = " ".join(quote.lower().split())
            haystack = " ".join(chunk_haystack.split())
            if needle in haystack:
                verified_criticals.append(sc)
            else:
                dropped_count += 1
                logger.debug(
                    "Catcher %s: dropping ungrounded critical (quote not in chunk): %r",
                    catcher_name, quote[:100],
                )
        if dropped_count > 0:
            logger.info(
                "Catcher %s: verified %d/%d structured criticals (dropped %d ungrounded)",
                catcher_name,
                len(verified_criticals),
                len(structured_criticals),
                dropped_count,
            )
        structured_criticals = verified_criticals
        # Rebuild the flat critical list from the verified structured set.
        # Keep any originally-string criticals as-is (gen 2.1 backward compat).
        # Empty/blank strings and placeholder dicts are dropped by _flatten_many.
        critical = (
            _flatten_many([c for c in critical if isinstance(c, str)])
            + _flatten_many(verified_criticals)
        )
    else:
        critical = _flatten_many(critical)

    warnings = _flatten_many(warnings)

    result = {
        "passed": passed and len(critical) == 0,
        "critical_issues": [str(c) for c in critical],
        "warnings": [str(w) for w in warnings],
    }
    if structured_criticals:
        # Gen 2.2+: preserve the structured (dict) form for observability.
        # In gen 2.4 strict mode this is the raw model-emitted verdict after
        # programmatic quote verification has dropped ungrounded items.
        result["_structured_criticals"] = structured_criticals
    if structured_warnings:
        result["_structured_warnings"] = structured_warnings
    return result


async def _call_catcher_majority(
    catcher_name: str,
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
    extra_context: Optional[str] = None,
    n: int = _CATCHER_MAJORITY_N,
) -> dict:
    """Run an LLM catcher N times with DIVERSE temperatures and combine via majority vote.

    Generation 2 (2026-04-11): voters run at different temperatures from
    _CATCHER_TEMPERATURES instead of all at temp=0. This decorrelates the
    voters so the majority vote actually reduces variance. See INFRA.md
    section 9 for the math.

    Aggregation strategy (CONSERVATIVE - fail-closed):
        - `passed` = True only if (a) majority of runs voted passed AND
          (b) the union of critical_issues across all runs is empty.
          In other words, ANY run finding a critical issue blocks the
          phase, even if 2-of-3 said clean.
        - `critical_issues` = union of critical_issues across all N runs
          (deduplicated, preserving order of first appearance).
        - `warnings` = union of warnings across all N runs (deduplicated).
        - `_majority_vote` = diagnostic field showing the vote breakdown
          and the temperatures used for each voter.

    Voter ordering (gen-2 cache-aware dispatch):
        Voter 1 is dispatched ALONE first and awaited. After a brief yield
        (500ms) to let Anthropic's cache backend register the writes,
        voters 2..N are dispatched in parallel via asyncio.gather. This
        guarantees the (catcher_prompt, chunk, artifact) cache entries
        are warm before the parallel voters race for them.

        Without the serialized first voter, all N voters dispatch at once,
        all N hit the same cache key, and Anthropic's cache doesn't
        deduplicate across in-flight writes — all N would write instead of
        1 writing + N-1 reading. The 500ms one-time penalty per round
        avoids paying N cache writes per round, which adds up to ~$10/run
        in wasted cache-write cost on full WHO extractions.
    """
    # Pick N temperatures from the configured tuple. If N > len(tuple), cycle.
    temps = [
        _CATCHER_TEMPERATURES[i % len(_CATCHER_TEMPERATURES)]
        for i in range(n)
    ]

    # Voter 1 alone: writes the cache entries for this (catcher, guide, artifact).
    first_result = await _call_catcher(
        catcher_name, artifact, guide_json, api_key, extra_context,
        temperature=temps[0],
    )
    # Brief yield to let the cache registration land on Anthropic's side
    # before voters 2..N dispatch and hit the same key. 500ms is enough in
    # practice — Anthropic's cache write is typically <100ms.
    if n > 1:
        await asyncio.sleep(0.5)

    # Voters 2..N in parallel: all read from the warm cache written by voter 1.
    rest_coros = [
        _call_catcher(
            catcher_name, artifact, guide_json, api_key, extra_context,
            temperature=temps[i],
        )
        for i in range(1, n)
    ]
    rest_results = await asyncio.gather(*rest_coros, return_exceptions=False) if rest_coros else []

    results = [first_result] + list(rest_results)

    passed_votes = sum(1 for r in results if r.get("passed", False))

    # Gen 2.4 hotfix (2026-04-12): semantic dedupe by the structural key
    # (suggested_id, guide_quote, section) instead of full-string equality.
    # See _critical_dedupe_key docstring for the rationale.
    #
    # Fix F (2026-04-12): vote-counted aggregation. Instead of UNION (any
    # 1-of-N voters adds a critical), count how many voters independently
    # flagged each dedupe key. Only keep criticals with count >=
    # _CRITICAL_VOTE_THRESHOLD. This filters single-voter noise that caused
    # catcher instability across rounds (new criticals appearing on r2 that
    # weren't on r1). A voter's internal duplicates (same key twice in one
    # voter's output) count as 1 vote, not 2.
    critical_vote_counts: dict[str | tuple, int] = {}
    critical_first_form: dict[str | tuple, str] = {}
    for r in results:
        voter_keys_seen: set = set()
        for issue in r.get("critical_issues", []):
            issue_str = str(issue)
            key = _critical_dedupe_key(issue_str)
            if key not in voter_keys_seen:
                voter_keys_seen.add(key)
                critical_vote_counts[key] = critical_vote_counts.get(key, 0) + 1
                if key not in critical_first_form:
                    critical_first_form[key] = issue_str

    threshold = min(_CRITICAL_VOTE_THRESHOLD, n)  # can't require more votes than voters
    surviving_keys = {
        key for key, count in critical_vote_counts.items()
        if count >= threshold
    }
    all_critical: list[str] = [
        critical_first_form[key]
        for key in surviving_keys
    ]

    # Gen 3: preserve structured criticals for the in-REPL repair loop.
    # The vote-counted aggregation above discards the dict-form criticals
    # and only keeps flattened strings. The repair loop needs the original
    # dicts (with suggested_id, guide_quote, repair_instruction). Collect
    # the structured forms that correspond to surviving vote-threshold keys.
    surviving_structured: list[dict] = []
    for r in results:
        for sc in r.get("_structured_criticals", []):
            if not isinstance(sc, dict):
                continue
            sid = sc.get("suggested_id", "")
            quote = sc.get("guide_quote", "")
            if not sid:
                continue
            # Build the same dedupe key the vote-counter used
            sc_key = (
                str(sid).strip().lower(),
                str(quote).strip().lower(),
                str(sc.get("section", "")).strip().lower(),
            )
            if sc_key in surviving_keys and sc_key not in {
                (s.get("suggested_id", "").strip().lower(),
                 s.get("guide_quote", "").strip().lower(),
                 s.get("section", "").strip().lower())
                for s in surviving_structured
            }:
                surviving_structured.append(sc)

    # Warnings remain union-based (no vote threshold; they're non-blocking).
    seen_warning_keys: set = set()
    all_warnings: list[str] = []
    for r in results:
        for warning in r.get("warnings", []):
            warning_str = str(warning)
            key = _critical_dedupe_key(warning_str)
            if key not in seen_warning_keys:
                seen_warning_keys.add(key)
                all_warnings.append(warning_str)

    final_passed = len(all_critical) == 0

    result = {
        "passed": final_passed,
        "critical_issues": all_critical,
        "warnings": all_warnings,
        "_majority_vote": {
            "passed_votes": passed_votes,
            "total_runs": n,
            "consensus": passed_votes == n or passed_votes == 0,
            "temperatures": list(temps),
        },
    }
    if surviving_structured:
        result["_structured_criticals"] = surviving_structured
    return result


# ---------------------------------------------------------------------------
# Guide chunking (recall fix)
# ---------------------------------------------------------------------------


def chunk_guide_for_catcher(
    guide_json: dict,
    max_chunk_chars: int = _CHUNK_MAX_CHARS,
) -> list[dict]:
    """Split `guide_json` into valid JSON chunks at section boundaries.

    Each chunk is a self-contained dict with:
      - top-level metadata fields copied from guide_json (minus `sections`
        and `pages`, both of which are rewritten per-chunk)
      - `sections`: a subset of the parent's sections, kept in original order
      - `_chunk_info`: marker with chunk_index, total_chunks, sections_included,
        and instructions the catcher uses to scope its audit

    The split respects section boundaries so no chunk ever contains
    malformed JSON. If a single section exceeds `max_chunk_chars` (rare — it
    would mean a section is larger than ~37K tokens), it is placed in its
    own chunk regardless of size. The catcher will see that oversized chunk
    intact — the library splits by structure, not by a character quota that
    cuts mid-object.

    For guides smaller than `max_chunk_chars`, returns a single-element list
    containing (a lightly-wrapped copy of) the input guide unchanged, so
    `_run_catcher` can use the same code path for both small and large guides.

    Caller contract:
      - Pass the FULL guide_json. Do not pre-slice.
      - The returned list has length >= 1, always. Never empty.
      - Each chunk, serialized with indent=2, is roughly max_chunk_chars or
        less (except when a single section is oversized).
    """
    # Serialize once to measure. If the whole thing fits, short-circuit.
    pretty_full = json.dumps(guide_json, indent=2)
    if len(pretty_full) <= max_chunk_chars:
        return [{
            **guide_json,
            "_chunk_info": {
                "chunk_index": 0,
                "total_chunks": 1,
                "sections_included": (
                    list(guide_json.get("sections", {}).keys())
                    if isinstance(guide_json.get("sections"), dict)
                    else []
                ),
                "is_full_guide": True,
            },
        }]

    sections = guide_json.get("sections", {})
    if not isinstance(sections, dict) or not sections:
        # No sections to split on — return whole guide as one oversized chunk.
        # The catcher will still see it (modulo Haiku's context limit).
        logger.warning(
            "chunk_guide_for_catcher: guide has no `sections` dict to split "
            "on (pretty=%d chars). Returning single oversized chunk.",
            len(pretty_full),
        )
        return [{
            **guide_json,
            "_chunk_info": {
                "chunk_index": 0,
                "total_chunks": 1,
                "sections_included": [],
                "is_full_guide": True,
            },
        }]

    # Metadata to include in every chunk (everything except sections + pages).
    # `pages` is excluded because it's the raw per-page text used upstream
    # during ingestion — sections already carry the extracted content. Keeping
    # `pages` would double each chunk's size.
    base_metadata = {
        k: v for k, v in guide_json.items()
        if k not in ("sections", "pages")
    }
    # Rough overhead of the metadata wrapper around each chunk (measured once
    # so we can budget accurately for section content inside each chunk).
    overhead = len(json.dumps({**base_metadata, "sections": {}, "_chunk_info": {}}, indent=2))
    chunk_budget = max(10_000, max_chunk_chars - overhead - 2_000)  # -2K for chunk_info

    chunks: list[dict] = []
    current_sections: dict[str, Any] = {}
    current_size = 0

    for slug, section in sections.items():
        # Serialize this section alone to measure its contribution
        section_str = json.dumps({slug: section}, indent=2)
        section_size = len(section_str)

        if current_sections and (current_size + section_size) > chunk_budget:
            # Flush current chunk
            chunks.append({
                **base_metadata,
                "sections": current_sections,
                "_chunk_info": {
                    "chunk_index": len(chunks),
                    "total_chunks": -1,  # patched at end
                    "sections_included": list(current_sections.keys()),
                    "is_full_guide": False,
                },
            })
            current_sections = {}
            current_size = 0

        current_sections[slug] = section
        current_size += section_size

    if current_sections:
        chunks.append({
            **base_metadata,
            "sections": current_sections,
            "_chunk_info": {
                "chunk_index": len(chunks),
                "total_chunks": -1,
                "sections_included": list(current_sections.keys()),
                "is_full_guide": False,
            },
        })

    total = len(chunks)
    for c in chunks:
        c["_chunk_info"]["total_chunks"] = total

    logger.info(
        "Guide chunked for catcher: %d sections → %d chunks (avg %d chars/chunk, cap %d)",
        len(sections), total, len(pretty_full) // max(1, total), max_chunk_chars,
    )
    return chunks


def _build_chunk_context_hint(chunk: dict) -> str:
    """Build the extra_context prefix that tells the catcher which chunk this
    is and how to scope its audit.

    Without this hint, a chunked catcher would falsely flag artifact items
    that reference sections not in its current chunk (because the catcher
    can't see them). The hint tells the catcher: "only audit items whose
    citations are in THIS chunk's section list; items citing other sections
    are audited in other chunks."
    """
    info = chunk.get("_chunk_info", {}) or {}
    if info.get("is_full_guide", False):
        return ""  # No chunking → no hint needed

    idx = info.get("chunk_index", 0)
    total = info.get("total_chunks", 1)
    included = info.get("sections_included", []) or []
    # Keep the hint short (<1K) — too long and we eat into the chunk budget.
    included_str = ", ".join(included[:50])
    if len(included) > 50:
        included_str += f", ... (and {len(included) - 50} more)"

    return (
        f"CHUNK_CONTEXT: This audit is for chunk {idx + 1}/{total} of a "
        f"large guide that has been split at section boundaries.\n"
        f"Sections in THIS chunk: {included_str}\n"
        f"\n"
        f"SCOPING RULES (read before auditing):\n"
        f"1. COMPLETENESS: Only flag items as missing if they should come "
        f"from sections in THIS chunk. Items that belong to sections in "
        f"other chunks are audited separately — do not flag them as missing "
        f"here.\n"
        f"2. PROVENANCE: If an artifact item cites a section that is NOT in "
        f"this chunk's section list above, SKIP it — do not flag the citation "
        f"as invalid. That citation is validated in the chunk where its "
        f"section lives.\n"
        f"3. Focus ONLY on items whose section_id or source references "
        f"match sections in THIS chunk's list above.\n"
    )


async def _call_catcher_chunked_majority(
    catcher_name: str,
    artifact: Any,
    guide_chunks: list[dict],
    api_key: str,
    extra_context: Optional[str] = None,
    n: int = _CATCHER_MAJORITY_N,
) -> dict:
    """Run a catcher across multiple guide chunks with N-way majority vote
    per chunk, then aggregate across chunks fail-closed.

    Call count: len(guide_chunks) × n  (all in parallel via asyncio.gather,
    throttled by the process-global semaphore to respect Anthropic rate
    limits).

    Per-chunk aggregation: majority vote within the n calls on that chunk
    (same logic as `_call_catcher_majority`).

    Cross-chunk aggregation: conservative fail-closed.
      - passed = True only if EVERY chunk passed
      - critical_issues = union across all chunks (deduped)
      - warnings = union across all chunks (deduped)

    This design means a single missed item in chunk 7 of 13 will block
    the artifact, which is the correct behavior for recall: we want to
    catch everything, everywhere.
    """
    sem = _get_catcher_semaphore()

    # Pick N temperatures from the configured tuple (gen-2 diverse voting).
    # Each chunk's N votes use the same N temperatures, so voter 1 across
    # all chunks is the temp=0 grounding voter, voter 2 is temp=0.15, etc.
    temps = [
        _CATCHER_TEMPERATURES[i % len(_CATCHER_TEMPERATURES)]
        for i in range(n)
    ]

    async def one_call(chunk: dict, vote_idx: int) -> dict:
        async with sem:
            chunk_hint = _build_chunk_context_hint(chunk)
            merged_context = (
                f"{extra_context}\n\n{chunk_hint}"
                if extra_context and chunk_hint
                else (chunk_hint or extra_context)
            )
            return await _call_catcher(
                catcher_name, artifact, chunk, api_key, merged_context,
                temperature=temps[vote_idx],
            )

    # Per-chunk voter orchestration (gen-2 serialized-first pattern):
    #
    # For each chunk:
    #   1. Vote 1 (grounding voter, temp=0.0) is dispatched alone and awaited.
    #      This single call writes the (catcher_prompt, chunk, artifact)
    #      cache entries to Anthropic's cache backend.
    #   2. Brief 500ms yield to let the cache write register before
    #      parallel voters dispatch and race against the same cache key.
    #   3. Votes 2..N are dispatched in parallel via asyncio.gather and
    #      all read from the warm cache written by voter 1. This is where
    #      the artifact cache actually earns its keep — without the
    #      serialization, all N voters would simultaneously hit the same
    #      cache key and Anthropic's cache doesn't deduplicate across
    #      in-flight writes, so we'd pay N writes instead of 1 write +
    #      (N-1) reads.
    #
    # Across chunks, run_chunk_voters tasks run in parallel via outer
    # asyncio.gather. The semaphore caps total in-flight calls to 3
    # regardless of how many chunks are active, so the interleaving is:
    #   - chunks A, B, C fire vote 1 (3 slots in use)
    #   - A, B, C finish vote 1, each enters 500ms sleep (0 slots used)
    #   - chunks D, E, F can use the slots for their vote 1 during A/B/C's sleep
    #   - A, B, C wake from sleep and fire votes 2..N
    #   - etc.
    #
    # Wall-clock overhead from the 500ms sleep: ~overlapped with other
    # chunks' calls, so net added ≈ 0.5s × (ceil(num_chunks / cap))
    # across the whole run. For 23 chunks at cap=3: ~4s added per round.

    async def run_chunk_voters(chunk: dict) -> list[dict]:
        # Vote 1: grounding call, writes the cache entries
        first = await one_call(chunk, 0)
        # Brief yield for cache registration. Only wait if there ARE more
        # voters to fire — single-voter configs skip this.
        if n > 1:
            await asyncio.sleep(0.5)
        # Votes 2..N: parallel reads from the warm cache
        if n > 1:
            rest = await asyncio.gather(*[
                one_call(chunk, i) for i in range(1, n)
            ])
        else:
            rest = []
        return [first] + list(rest)

    # Run all chunks' voter sequences in parallel. The semaphore inside
    # one_call serializes the actual API calls to _CATCHER_CONCURRENCY_CAP
    # simultaneous calls — so even with 23 chunks × 5 voters = 115 tasks
    # dispatched via asyncio.gather, at most 3 Haiku calls are in flight.
    per_chunk_buckets: list[list[dict]] = await asyncio.gather(*[
        run_chunk_voters(chunk) for chunk in guide_chunks
    ])

    # Per-chunk vote-counted aggregation (Fix F, 2026-04-12).
    # Previously used UNION + string-equality dedupe. Now uses:
    #   1. Semantic dedupe via _critical_dedupe_key (same as unchunked)
    #   2. Vote counting: only keep criticals with >= threshold votes
    threshold = min(_CRITICAL_VOTE_THRESHOLD, n)
    per_chunk_results: list[dict] = []
    for bucket in per_chunk_buckets:
        passed_votes = sum(1 for r in bucket if r.get("passed", False))

        # Vote-counted critical aggregation (mirrors _call_catcher_majority)
        crit_votes: dict[str | tuple, int] = {}
        crit_first: dict[str | tuple, str] = {}
        for r in bucket:
            voter_keys: set = set()
            for issue in r.get("critical_issues", []):
                s = str(issue)
                key = _critical_dedupe_key(s)
                if key not in voter_keys:
                    voter_keys.add(key)
                    crit_votes[key] = crit_votes.get(key, 0) + 1
                    if key not in crit_first:
                        crit_first[key] = s
        chunk_surviving_keys = {k for k, c in crit_votes.items() if c >= threshold}
        chunk_crit = [crit_first[k] for k in chunk_surviving_keys]

        # Gen 3: preserve structured criticals that survived vote threshold
        chunk_structured: list[dict] = []
        for r in bucket:
            for sc in r.get("_structured_criticals", []):
                if not isinstance(sc, dict) or not sc.get("suggested_id"):
                    continue
                sc_key = (
                    str(sc.get("suggested_id", "")).strip().lower(),
                    str(sc.get("guide_quote", "")).strip().lower(),
                    str(sc.get("section", "")).strip().lower(),
                )
                if sc_key in chunk_surviving_keys and sc_key not in {
                    (s.get("suggested_id", "").strip().lower(),
                     s.get("guide_quote", "").strip().lower(),
                     s.get("section", "").strip().lower())
                    for s in chunk_structured
                }:
                    chunk_structured.append(sc)

        # Warnings: union with semantic dedupe (no vote threshold)
        seen_w: set = set()
        chunk_warn: list[str] = []
        for r in bucket:
            for w in r.get("warnings", []):
                s = str(w)
                key = _critical_dedupe_key(s)
                if key not in seen_w:
                    seen_w.add(key)
                    chunk_warn.append(s)

        chunk_result: dict = {
            "passed": len(chunk_crit) == 0,
            "critical_issues": chunk_crit,
            "warnings": chunk_warn,
            "_passed_votes": passed_votes,
        }
        if chunk_structured:
            chunk_result["_structured_criticals"] = chunk_structured
        per_chunk_results.append(chunk_result)

    # Cross-chunk aggregation (fail-closed, semantic dedupe)
    total_seen_c: set = set()
    total_crit: list[str] = []
    total_seen_w: set = set()
    total_warn: list[str] = []
    total_structured: list[dict] = []
    total_structured_keys: set = set()
    all_chunks_passed = True
    for r in per_chunk_results:
        if not r["passed"]:
            all_chunks_passed = False
        for c in r["critical_issues"]:
            key = _critical_dedupe_key(c)
            if key not in total_seen_c:
                total_seen_c.add(key)
                total_crit.append(c)
        for w in r["warnings"]:
            key = _critical_dedupe_key(w)
            if key not in total_seen_w:
                total_seen_w.add(key)
                total_warn.append(w)
        # Gen 3: merge structured criticals across chunks (deduped by sid+section)
        for sc in r.get("_structured_criticals", []):
            if not isinstance(sc, dict):
                continue
            sc_dedup = (
                str(sc.get("suggested_id", "")).strip().lower(),
                str(sc.get("section", "")).strip().lower(),
            )
            if sc_dedup not in total_structured_keys:
                total_structured_keys.add(sc_dedup)
                total_structured.append(sc)

    cross_result = {
        "passed": all_chunks_passed and len(total_crit) == 0,
        "critical_issues": total_crit,
        "warnings": total_warn,
        "_chunked_majority": {
            "num_chunks": len(guide_chunks),
            "votes_per_chunk": n,
            "total_calls": len(guide_chunks) * n,
            "chunks_passed": sum(1 for r in per_chunk_results if r["passed"]),
            "per_chunk_passed_votes": [r["_passed_votes"] for r in per_chunk_results],
        },
    }
    if total_structured:
        cross_result["_structured_criticals"] = total_structured
    return cross_result


# Catchers whose job depends on reading the source guide. These are the ones
# that get routed through chunked-majority-vote on large guides. Everything
# else (architecture checks, consistency checks, clinical review on the
# artifact alone) skips chunking entirely because the guide isn't needed.
_GUIDE_DEPENDENT_CATCHERS: frozenset[str] = frozenset({
    "completeness_supply_list",
    "completeness_variables",
    "completeness_predicates",
    "completeness_modules",
    "completeness_router",
    "completeness_integrative",
    "completeness_phrase_bank",
    "completeness_json",  # legacy name, still routed if it appears
})

# Catchers that emit gen-2.2-style dict criticals (4-field: suggested_id,
# description, guide_quote, section) plus an enumeration array, and are
# therefore eligible for gen 2.4 strict tool mode.
#
# NOTE: This is a STRICT SUBSET of _GUIDE_DEPENDENT_CATCHERS. Router and
# integrative catchers are guide-dependent (chunked dispatch) but use the
# simpler string-critical format that predates gen 2.2, so they stay on
# the free-JSON output path. Their prompts explicitly say "Input: ONE JSON
# blob: ARTIFACT" and audit structural properties of the artifact, not
# guide-quoted recall failures, so the strict schema's required
# `guide_quote` field wouldn't match their output shape.
#
# If a new completeness catcher is added, put it in both sets (or neither).
# If an existing string-critical catcher is rewritten to emit dict criticals,
# move it from _GUIDE_DEPENDENT_CATCHERS only → both sets.
_STRICT_MODE_CATCHERS: frozenset[str] = frozenset({
    # Completeness catchers (recall audit per artifact)
    "completeness_supply_list",
    "completeness_variables",
    "completeness_predicates",
    "completeness_modules",
    "completeness_router",
    "completeness_integrative",
    "completeness_phrase_bank",
    "completeness_json",  # legacy
    # Clinical / consistency / architecture catchers
    "clinical_review_json",
    "consistency_json",
    "module_architecture_json",
    "comorbidity_coverage",
    "provenance_json",
    # Final validation catchers
    "dmn_audit",
    "boundary_condition",
})


async def _run_catcher(
    catcher_name: str,
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
    extra_context: Optional[str] = None,
) -> dict:
    """Unified catcher dispatcher: routes to deterministic, LLM majority vote,
    or LLM chunked-majority vote depending on the catcher type and guide size.

    Routing:
      1. If `catcher_name` has a deterministic Python implementation in
         DETERMINISTIC_CATCHERS, run that (zero LLM calls).
      2. Elif `catcher_name` is guide-dependent AND `guide_json` is provided
         AND the guide is large enough to chunk, run chunked-majority-vote.
      3. Else run the standard 3-way majority vote on the full guide.

    The deterministic functions are sync (no async, no I/O) so we can call
    them directly without await. They return the same dict shape as the
    LLM caller, so the dispatch is transparent to the validate_* functions.
    """
    deterministic = DETERMINISTIC_CATCHERS.get(catcher_name)
    if deterministic is not None:
        try:
            result = deterministic(artifact, guide_json)
            # Tag the result so the catcher_outputs trace shows it ran
            # deterministically — useful for the gate harness verdict.
            result["_source"] = "deterministic"
            return result
        except Exception as exc:
            logger.exception(
                "Deterministic catcher %s crashed: %s", catcher_name, exc
            )
            return {
                "passed": False,
                "critical_issues": [
                    f"Deterministic catcher {catcher_name} crashed: {exc}"
                ],
                "warnings": [],
                "_source": "deterministic_error",
            }

    # LLM catcher. Decide whether to chunk.
    should_chunk = (
        guide_json is not None
        and catcher_name in _GUIDE_DEPENDENT_CATCHERS
    )
    if should_chunk:
        chunks = chunk_guide_for_catcher(guide_json)
        if len(chunks) > 1:
            result = await _call_catcher_chunked_majority(
                catcher_name, artifact, chunks, api_key, extra_context
            )
            result["_source"] = (
                f"llm_chunked_majority_{_CATCHER_MAJORITY_N}"
                f"_x{len(chunks)}_chunks"
            )
            return result
        # len(chunks) == 1 means the guide fits in one call — fall through
        # to the standard 3-vote path on the (lightly wrapped) single chunk.
        # Pass the single chunk so the catcher sees the _chunk_info marker
        # consistently, even though no context hint is added.
        result = await _call_catcher_majority(
            catcher_name, artifact, chunks[0], api_key, extra_context
        )
        result["_source"] = f"llm_majority_{_CATCHER_MAJORITY_N}_unchunked"
        return result

    # Standard 3-way majority vote (artifact-only catchers, or guide too
    # small to need chunking, or guide_json=None was passed by the caller).
    result = await _call_catcher_majority(
        catcher_name, artifact, guide_json, api_key, extra_context
    )
    result["_source"] = f"llm_majority_{_CATCHER_MAJORITY_N}"
    return result


def _extract_json(text: str) -> Optional[dict]:
    """Pull a JSON object out of a model response that may include markdown
    fences, code blocks, or trailing prose. Returns None if no valid JSON
    found.

    Handles these formats (in order of preference):
      1. Raw JSON: `{"passed": true, ...}`
      2. Markdown code fence: ```json\n{...}\n```
      3. Generic code fence: ```\n{...}\n```
      4. JSON embedded in prose: "Here is the result: {...}"
      5. Largest balanced-brace object anywhere in the text

    Strategy: try cleaning first (strip markdown fences + surrounding
    whitespace), then try direct parse, then fall back to brace scan.

    Hardened 2026-04-11 after Run #8 hit a malformed variables catcher
    response where the prior extractor returned None on valid JSON
    wrapped in ```json ... ``` fences because the brace scanner started
    before the opening brace.
    """
    if not text:
        return None

    # Step 1: Strip surrounding markdown code fences if present. The model
    # commonly returns ```json\n{...}\n``` or ```\n{...}\n``` wrappers.
    import re
    stripped = text.strip()
    # Match: optional ``` or ```json at start, content, optional ``` at end
    fence_match = re.match(
        r"^```(?:json|JSON|javascript|js)?\s*\n?(.*?)\n?```\s*$",
        stripped,
        re.DOTALL,
    )
    if fence_match:
        stripped = fence_match.group(1).strip()

    # Step 2: Direct parse on the cleaned text
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 3: Try to find the LARGEST balanced JSON object via brace scan.
    # The old implementation returned the first valid object found, which
    # meant if the model wrote "Some prose {foo: 1} more prose {actual: result}"
    # it returned {foo: 1}. We want the outermost/largest valid object.
    best: Optional[dict] = None
    best_size = 0
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(stripped)):
            c = stripped[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and len(candidate) > best_size:
                            best = parsed
                            best_size = len(candidate)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        start = stripped.find("{", start + 1)

    if best is not None:
        return best

    # Step 4: Last resort — try to extract just the passed/critical_issues
    # structure using regex, in case the model wrote broken JSON but the
    # key fields are recognizable. This is the variables-catcher-crash case.
    # We look for `"passed":\s*(true|false)` and optionally try to parse
    # critical_issues as a list.
    passed_match = re.search(r'"passed"\s*:\s*(true|false)', stripped)
    if passed_match:
        logger.warning(
            "_extract_json: falling back to regex recovery — catcher output "
            "was malformed JSON but contained a parseable 'passed' field"
        )
        return {
            "passed": passed_match.group(1) == "true",
            "critical_issues": [
                f"Catcher output was malformed JSON; recovered 'passed' "
                f"value via regex fallback. Raw preview: {stripped[:500]}"
            ],
            "warnings": [],
        }

    return None


# ---------------------------------------------------------------------------
# Aggregate a set of catcher results into one PhaseValidationResult
# ---------------------------------------------------------------------------


def _aggregate(
    artifact_name: str,
    phase: int,
    catcher_results: dict[str, dict],
) -> PhaseValidationResult:
    """Combine multiple catcher outputs into a single pass/fail result.

    Pass requires ALL catchers to pass. Critical issues and warnings are
    concatenated with the catcher name as a prefix so consumers can trace
    which catcher flagged what.
    """
    all_critical: list[str] = []
    all_warnings: list[str] = []
    all_passed = True
    for catcher_name, result in catcher_results.items():
        if not result.get("passed", False):
            all_passed = False
        for issue in result.get("critical_issues", []):
            all_critical.append(f"[{catcher_name}] {issue}")
        for warning in result.get("warnings", []):
            all_warnings.append(f"[{catcher_name}] {warning}")
    return PhaseValidationResult(
        artifact_name=artifact_name,
        phase=phase,
        passed=all_passed and len(all_critical) == 0,
        critical_issues=all_critical,
        warnings=all_warnings,
        catcher_outputs=catcher_results,
    )


# ---------------------------------------------------------------------------
# Per-artifact validators (generation 2 — 2026-04-11)
# ---------------------------------------------------------------------------
#
# Each validator runs THREE layers:
#
#   1. Recall (LLM, chunked majority vote) — finds items the guide describes
#      that are missing from the artifact. HARD criticals here block the run.
#      Prompts enforce a "quote or drop" rule so hallucinated requirements
#      cannot slip through.
#
#   2. Architecture (deterministic Python, soft warnings) — structural
#      sanity checks: required fields, prefix conventions, hit policies,
#      row ordering. These are NOT recall checks and produce WARNINGS by
#      default. A small whitelist of HARD rules (see artifact_architecture.py
#      HARD_ARCHITECTURE_RULES) still produces criticals that block runs —
#      for invariants where a violation would produce a clinically-unsafe
#      or broken artifact downstream.
#
#   3. Clinical / consistency (mixed LLM + deterministic) — domain checks
#      that look at the artifact's internal coherence (clinical_review_json,
#      consistency_json, etc.). Unchanged from generation 1.
#
# The key insight from Run #8: generation 1 mixed all three layers into
# LLM completeness catchers, which caused runs to fail on style/convention
# issues instead of recall. Generation 2 separates them so a run with
# 100% recall but minor style warnings is a PASS.


async def _run_arch(artifact_type: str, artifact: Any, **extra) -> dict:
    """Run the deterministic architecture check for an artifact.

    Returns a catcher-shaped dict. Never raises — if the check function
    itself crashes, returns a warning-only dict so the run keeps moving.
    """
    check_fn = ARCHITECTURE_CHECKS.get(artifact_type)
    if check_fn is None:
        return {"passed": True, "critical_issues": [], "warnings": [], "_source": "architecture_no_check"}
    try:
        result = check_fn(artifact, **extra) if extra else check_fn(artifact)
        result["_source"] = "architecture_deterministic"
        return result
    except Exception as exc:
        logger.exception("Architecture check for %s crashed: %s", artifact_type, exc)
        return {
            "passed": True,  # fail OPEN on check crashes — don't block runs over validator bugs
            "critical_issues": [],
            "warnings": [f"architecture check crashed: {exc}"],
            "_source": "architecture_error",
        }


async def validate_supply_list(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 2 supply_list artifact.

    Catchers (gen 2):
        completeness_supply_list -> LLM recall (chunked majority vote)
        provenance_json          -> deterministic Python
        arch_supply_list         -> deterministic Python (warnings only)
    """
    catchers = {
        "completeness_supply_list": await _run_catcher("completeness_supply_list", artifact, guide_json, api_key),
        "provenance_json":          await _run_catcher("provenance_json",          artifact, guide_json, api_key),
        "arch_supply_list":         await _run_arch("supply_list", artifact, guide_json=guide_json),
    }
    return _aggregate("supply_list", phase=2, catcher_results=catchers)


async def validate_variables(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 2 variables artifact.

    Catchers (gen 2):
        completeness_variables -> LLM recall (chunked majority vote)
        provenance_json        -> deterministic Python
        arch_variables         -> deterministic Python (warnings only)
    """
    catchers = {
        "completeness_variables": await _run_catcher("completeness_variables", artifact, guide_json, api_key),
        "provenance_json":        await _run_catcher("provenance_json",        artifact, guide_json, api_key),
        "arch_variables":         await _run_arch("variables", artifact, guide_json=guide_json),
    }
    return _aggregate("variables", phase=2, catcher_results=catchers)


async def validate_predicates(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 3 predicates artifact.

    Catchers (gen 2):
        completeness_predicates -> LLM recall (chunked majority vote)
                                   NOW RECEIVES guide_json (was None in gen 1,
                                   which meant predicates recall was broken)
        clinical_review_json    -> LLM (majority vote, artifact-only)
        provenance_json         -> deterministic Python
        arch_predicates         -> deterministic Python (warnings only;
                                   fail_safe conventions moved here from
                                   the gen-1 LLM completeness prompt)
    """
    catchers = {
        "completeness_predicates": await _run_catcher("completeness_predicates", artifact, guide_json, api_key),
        "clinical_review_json":    await _run_catcher("clinical_review_json",    artifact, None,       api_key),
        "provenance_json":         await _run_catcher("provenance_json",         artifact, guide_json, api_key),
        "arch_predicates":         await _run_arch("predicates", artifact, guide_json=guide_json),
    }
    return _aggregate("predicates", phase=3, catcher_results=catchers)


async def validate_modules(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 3 modules artifact.

    Catchers (gen 2):
        completeness_modules     -> LLM recall (chunked majority vote)
        clinical_review_json     -> LLM (majority vote, artifact-only)
        provenance_json          -> deterministic Python
        module_architecture_json -> deterministic Python (legacy shape checks)
        arch_modules             -> deterministic Python (warnings only)
    """
    catchers = {
        "completeness_modules":     await _run_catcher("completeness_modules",     artifact, guide_json, api_key),
        "clinical_review_json":     await _run_catcher("clinical_review_json",     artifact, None,        api_key),
        "provenance_json":          await _run_catcher("provenance_json",          artifact, guide_json, api_key),
        "module_architecture_json": await _run_catcher("module_architecture_json", artifact, None,        api_key),
        "arch_modules":             await _run_arch("modules", artifact, guide_json=guide_json),
    }
    return _aggregate("modules", phase=3, catcher_results=catchers)


async def validate_router(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
    modules: Optional[Any] = None,
) -> PhaseValidationResult:
    """Validator for the Phase 4 router artifact.

    Catchers (gen 2):
        completeness_router      -> LLM recall (narrow scope: dead modules only)
        module_architecture_json -> deterministic Python
        consistency_json         -> deterministic Python
        arch_router              -> deterministic Python (row 0 MUST be
                                    danger-sign, row 0 and hit_policy HARD;
                                    everything else warnings)
    """
    catchers = {
        "completeness_router":      await _run_catcher("completeness_router",      artifact, None, api_key),
        "module_architecture_json": await _run_catcher("module_architecture_json", artifact, None, api_key),
        "consistency_json":         await _run_catcher("consistency_json",         artifact, None, api_key),
        "arch_router":              await _run_arch("router", artifact, modules=modules),
    }
    return _aggregate("router", phase=4, catcher_results=catchers)


async def validate_integrative(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 4 integrative artifact.

    The integrative table is the comorbidity hot spot — a child with two or
    more simultaneous conditions depends on this table to produce a safe
    combined care plan. The comorbidity_coverage catcher specifically probes
    whether the integrative handles clinically meaningful multi-module
    combinations beyond what the pure architecture + consistency checks
    would catch, so it stays as an LLM catcher with majority vote.

    Catchers:
        completeness_integrative -> LLM (majority vote)
        module_architecture_json -> deterministic Python
        consistency_json         -> deterministic Python
        comorbidity_coverage     -> LLM (majority vote)
    """
    catchers = {
        "completeness_integrative": await _run_catcher("completeness_integrative", artifact, None, api_key),
        "module_architecture_json": await _run_catcher("module_architecture_json", artifact, None, api_key),
        "consistency_json":         await _run_catcher("consistency_json",         artifact, None, api_key),
        "comorbidity_coverage":     await _run_catcher("comorbidity_coverage",     artifact, None, api_key),
        "arch_integrative":         await _run_arch("integrative", artifact),
    }
    return _aggregate("integrative", phase=4, catcher_results=catchers)


async def validate_phrase_bank(
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the Phase 5 phrase_bank artifact.

    Catchers (gen 2):
        completeness_phrase_bank -> LLM recall (chunked majority vote)
        arch_phrase_bank         -> deterministic Python (warnings only)
    """
    catchers = {
        "completeness_phrase_bank": await _run_catcher("completeness_phrase_bank", artifact, guide_json, api_key),
        "arch_phrase_bank":         await _run_arch("phrase_bank", artifact),
    }
    return _aggregate("phrase_bank", phase=5, catcher_results=catchers)


async def validate_final_dmn(
    dmn_xml: str,
    api_key: str,
) -> PhaseValidationResult:
    """Validator for the final DMN XML (runs POST-conversion).

    Both catchers stay LLM-based because they reason over XML structure
    that is not trivially expressible as Python without a DMN XSD validator.
    They run with majority vote (3 calls each) for stability.

    Catchers:
        dmn_audit          -> LLM (majority vote)
        boundary_condition -> LLM (majority vote)
    """
    # The dmn_audit and boundary_condition catchers expect DMN XML as input.
    # We pass it as a string in the "ARTIFACT:" slot since the generic caller
    # expects JSON-serializable input; wrap in a dict for consistency.
    wrapper = {"dmn_xml": dmn_xml}
    catchers = {
        "dmn_audit":          await _run_catcher("dmn_audit",          wrapper, None, api_key),
        "boundary_condition": await _run_catcher("boundary_condition", wrapper, None, api_key),
    }
    return _aggregate("final_dmn", phase=6, catcher_results=catchers)


# ---------------------------------------------------------------------------
# Dispatch table — the emit_artifact tool uses this to route to the right
# validator based on the artifact name the model passes.
# ---------------------------------------------------------------------------


_VALIDATOR_DISPATCH: dict[str, Any] = {
    "supply_list": validate_supply_list,
    "variables": validate_variables,
    "predicates": validate_predicates,
    "modules": validate_modules,
    "router": validate_router,
    "integrative": validate_integrative,
    "phrase_bank": validate_phrase_bank,
}

_ARTIFACT_PHASES: dict[str, int] = {
    "supply_list": 2,
    "variables": 2,
    "predicates": 3,
    "modules": 3,
    "router": 4,
    "integrative": 4,
    "phrase_bank": 5,
}


def valid_artifact_names() -> list[str]:
    """Return the list of artifact names the model is allowed to emit."""
    return list(_VALIDATOR_DISPATCH.keys())


# Gen 4: module-level test suite slot. When set by run_extraction,
# validate_artifact uses deterministic test-suite matching for guide-
# dependent completeness checks instead of per-round LLM catchers.
_active_test_suite: Any = None  # Optional[TestSuite]

_TEST_SUITE_ARTIFACT_TYPES = frozenset({
    "supply_list", "variables", "predicates", "modules", "phrase_bank",
})


def set_active_test_suite(suite: Any) -> None:
    """Set the active test suite for this process. Called by run_extraction."""
    global _active_test_suite
    _active_test_suite = suite
    if suite:
        logger.info("Gen4: test suite activated (%d items)", len(suite.items))


def clear_active_test_suite() -> None:
    """Clear the active test suite."""
    global _active_test_suite
    _active_test_suite = None


async def validate_artifact(
    artifact_name: str,
    artifact: Any,
    guide_json: Optional[dict],
    api_key: str,
) -> PhaseValidationResult:
    """Dispatch to the right validator based on artifact_name.

    Gen 4: when a test suite is active AND the artifact type is guide-
    dependent (supply_list, variables, predicates, modules, phrase_bank),
    replace the LLM completeness catchers with deterministic test-suite
    matching. Non-guide catchers (clinical_review, architecture,
    consistency) still run as LLM calls.

    Raises ValueError on unknown artifact names.
    """
    if artifact_name not in _VALIDATOR_DISPATCH:
        raise ValueError(
            f"Unknown artifact name {artifact_name!r}. "
            f"Valid names: {valid_artifact_names()}"
        )

    # Gen 4: if test suite is active, use hybrid validation
    if _active_test_suite and artifact_name in _TEST_SUITE_ARTIFACT_TYPES:
        from backend.validators.test_suite import validate_against_test_suite

        # Deterministic test suite matching (replaces completeness_* catchers)
        test_result = validate_against_test_suite(artifact_name, artifact, _active_test_suite)

        # Still run non-guide catchers (clinical_review, architecture, etc.)
        # These audit internal consistency, not guide completeness
        arch_result = await _run_arch(artifact_name, artifact)

        # For predicates and modules, also run clinical_review
        non_guide_catchers = {"arch": arch_result, "test_suite": test_result}
        if artifact_name in ("predicates", "modules"):
            try:
                cr_result = await _run_catcher(
                    "clinical_review_json", artifact, None, api_key,
                )
                non_guide_catchers["clinical_review_json"] = cr_result
            except Exception as exc:
                logger.warning("clinical_review_json failed (non-fatal): %s", exc)

        return _aggregate(
            artifact_name,
            phase=_ARTIFACT_PHASES.get(artifact_name, -1),
            catcher_results=non_guide_catchers,
        )

    # Default: use the full LLM validator pipeline
    validator = _VALIDATOR_DISPATCH[artifact_name]
    return await validator(artifact, guide_json, api_key)
