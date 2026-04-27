"""Gen 7 Phase 1: Single Opus labeling per chunk with Anthropic prompt caching.

One Opus call per micro-chunk at temp=0. The system prompt (codebook +
labeling instructions) is marked with cache_control so Anthropic caches it
across all 26+ chunk calls. After the first call, subsequent calls pay the
cache-read rate ($0.50/M instead of $5/M for input tokens in the system block).

No GPT-4o-mini swarm. No disagreement resolution. No cross-model arbitration.
Opus labels, Opus compiles. k=1 at the model level.
"""

import asyncio
import json
import logging
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

OPUS_MODEL = "claude-opus-4-6"


def _build_distillation_system_prompt(codebook: str) -> str:
    """Call 2 system prompt: quality-control verification of Call 1's labels.

    Drops hallucinations (labels not grounded in source text) and canonicalizes
    IDs against the codebook. Does NOT introduce new labels -- only removes
    bad ones and corrects IDs.
    """
    return (
        "You are a clinical label quality-control verifier. You examine candidate "
        "labels produced by a first-pass labeler and verify each is grounded in "
        "the source text and has a codebook-compliant ID.\n\n"
        "YOUR JOB:\n"
        "1. For each candidate label, verify its 'span' appears in the source text "
        "(verbatim or near-verbatim). Drop labels whose span cannot be found.\n"
        "2. Verify each label's 'id' follows the codebook prefix convention for "
        "its 'type'. If an ID is wrong-prefixed, either correct it or drop it.\n"
        "3. Drop labels that look like hallucinations (generic placeholder names, "
        "concepts not actually in the text).\n"
        "4. Keep verified labels with their original 'quote_context' intact.\n"
        "5. DO NOT add new labels. Only verify/clean the candidates you receive.\n\n"
        f"CODEBOOK (for ID canonicalization):\n{codebook}\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences, no prose). Shape:\n"
        '{"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}\n'
        "\n"
        "Keep every field present in the input (span, id, type, subtype, quote_context). "
        "Drop the whole label entry if it fails verification; do not partially populate."
    )


def _parse_json_robust(text: str) -> Any:
    """Parse JSON with repair for common LLM output issues.

    Handles: trailing commas, truncated output (unclosed brackets/strings),
    and Python-style True/False/None.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Repair: replace Python literals
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")

    # Repair: remove trailing commas before } or ]
    import re
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair: truncated output -- find the last complete label entry
    # Look for the labels array and close it properly
    labels_match = re.search(r'"labels"\s*:\s*\[', repaired)
    if labels_match:
        start = labels_match.end()
        # Find all complete label objects (ending with })
        last_complete = start
        brace_depth = 0
        i = start
        while i < len(repaired):
            c = repaired[i]
            if c == '{':
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    last_complete = i + 1
            elif c == '"':
                # Skip string content
                i += 1
                while i < len(repaired) and repaired[i] != '"':
                    if repaired[i] == '\\':
                        i += 1  # skip escaped char
                    i += 1
            i += 1

        # Truncate at last complete label and close the structure
        if last_complete > start:
            truncated = repaired[:last_complete].rstrip().rstrip(',')
            truncated += ']}'
            try:
                result = json.loads(truncated)
                if isinstance(result, dict):
                    logger.info("Gen7 label: repaired truncated JSON, %d labels recovered",
                                len(result.get("labels", [])))
                    return result
            except json.JSONDecodeError:
                pass

    logger.warning("Gen7 label: JSON repair failed, returning text[:200]: %s", text[:200])
    return None


def _build_labeling_system_prompt(codebook: str) -> str:
    """Build the labeling system prompt. This is cached across all chunk calls."""
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and label EVERY clinical item "
        "with a structured annotation.\n\n"
        "For EVERY clinical item in the text, produce a label with:\n"
        "- span: the exact verbatim text from the chunk (copy-paste, no paraphrase)\n"
        "- id: a canonical ID using the codebook prefixes below. IDs must be DESCRIPTIVE "
        "and content-derived, never numbered. "
        "GOOD: q_has_cough, p_fast_breathing_2mo, supply_amoxicillin_250mg, equip_thermometer. "
        "BAD: q_1, p_2, supply_3.\n"
        "- type: one of the 7 artifact types: supply_list, variables, predicates, modules, "
        "phrase_bank, router, integrative\n"
        "- subtype: optional finer classification (e.g. 'equipment' vs 'consumable' for supply_list)\n"
        "- quote_context: the surrounding sentence for provenance\n\n"
        "ARTIFACT TYPE MAPPING:\n"
        "  supply_list: physical items the CHW must possess.\n"
        "    - 'consumable' (supply_ prefix): medications, test kits, disposables that deplete\n"
        "    - 'equipment' (equip_ prefix): durable tools reused across patients\n"
        "  variables: runtime inputs the CHW collects during a visit.\n"
        "    - q_ = self-reported symptoms/history from patient\n"
        "    - ex_ = clinician observation/examination finding\n"
        "    - v_ = quantitative measurement requiring equipment\n"
        "    - lab_ = point-of-care test result\n"
        "    - hx_ = baseline history/chronic conditions\n"
        "    - demo_ = demographics\n"
        "  predicates: boolean thresholds computed from variables.\n"
        "    - p_ = threshold flag (e.g. p_fast_breathing when breaths_per_min >= 50)\n"
        "  modules: clinical decision topics/conditions.\n"
        "    - mod_ = decision module (e.g. mod_pneumonia, mod_diarrhoea)\n"
        "  phrase_bank: things the CHW says or advises.\n"
        "    - m_ = message/phrase (e.g. m_advise_fluids, m_refer_urgently)\n"
        "    - adv_ = advice/counseling text\n"
        "    - tx_ = treatment instruction\n"
        "    - rx_ = medication dosing instruction\n"
        "    - ref_ = referral message\n"
        "  router: routing/triage decisions (danger sign short-circuits, priority ordering)\n"
        "  integrative: comorbidity rules, cross-module interactions\n\n"
        "LABEL EVERYTHING you find:\n"
        "- Medications with dosages\n"
        "- Equipment and supplies\n"
        "- Questions to ask the patient/caregiver\n"
        "- Examination steps (look, feel, listen)\n"
        "- Vital sign thresholds and cutoffs\n"
        "- Age-based cutoffs\n"
        "- Danger signs and referral criteria\n"
        "- Classification/diagnosis labels\n"
        "- Treatment instructions and dosing\n"
        "- Advice and counseling phrases\n"
        "- Follow-up scheduling\n"
        "- Referral criteria and destinations\n\n"
        f"NAMING CODEBOOK:\n{codebook}\n\n"
        "CRITICAL RULES:\n"
        "1. Label EVERY clinical item. Missing items is worse than having extras.\n"
        "2. IDs must be descriptive and follow prefix conventions exactly.\n"
        "3. Numeric variables MUST end with unit suffix: _days, _per_min, _c, _mm, _mg, _kg.\n"
        "4. Use lowercase_with_underscores only. No spaces, hyphens, camelCase.\n"
        "5. If a concept does not fit any existing prefix, use the closest match "
        "and note it in the subtype field.\n"
        "6. Prefer over-labeling to under-labeling. The downstream compiler handles dedup.\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences, no explanation). Shape:\n"
        '{"section_id": "...", "section_title": "...", "text": "(original text)", '
        '"labels": [{"span": "...", "id": "prefix_descriptive_name", "type": "...", '
        '"subtype": "...", "quote_context": "..."}]}'
    )


async def label_chunk(
    chunk: dict,
    codebook: str,
    anthropic_key: str,
    _client: anthropic.AsyncAnthropic | None = None,
) -> dict:
    """Label one micro-chunk with a single Opus call.

    Uses cache_control on the system prompt so the codebook + instructions
    are cached across calls (Anthropic 5-min TTL). Each chunk's user message
    is the only uncached portion.

    Returns the labeled chunk dict with 'labels' array + '_labeling_meta'.
    """
    chunk_idx = chunk.get("chunk_index", -1)
    difficulty = chunk.get("difficulty", "medium")

    client = _client or anthropic.AsyncAnthropic(api_key=anthropic_key)

    system_prompt = _build_labeling_system_prompt(codebook)

    # Clean text input with section header
    clean_input = (
        f"Section: {chunk['section_title']} ({chunk['section_id']})\n\n"
        f"{chunk['text']}"
    )

    try:
        response = await client.messages.create(
            model=OPUS_MODEL,
            max_tokens=16384,
            temperature=0.0,
            # System prompt with cache_control for cross-call caching
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            messages=[
                {"role": "user", "content": f"CHUNK TO LABEL:\n{clean_input}"},
            ],
        )

        text = response.content[0].text.strip()

        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        parsed = _parse_json_robust(text)
        if not isinstance(parsed, dict):
            parsed = {"section_id": chunk["section_id"], "labels": [], "_error": "not dict"}
        parsed.setdefault("labels", [])

        call1_labels = list(parsed.get("labels", []))
        call1_in_tokens = response.usage.input_tokens
        call1_out_tokens = response.usage.output_tokens
        call1_cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        call1_cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0


        # Feed Call 1 usage into the global _run_usage accumulator so it
        # flows through to the frontend Cost Tracker. The labeler calls
        # Anthropic directly (bypassing the RLM client), so the monkey-
        # patched _accumulate_run_usage doesn't fire on these calls.
        try:
            from backend.rlm_runner import accumulate_catcher_usage
            accumulate_catcher_usage(
                model=OPUS_MODEL,
                input_tokens=call1_in_tokens,
                output_tokens=call1_out_tokens,
                cached_tokens=call1_cache_read,
                cache_write_tokens=call1_cache_creation,
            )
        except Exception:
            pass

        # ---- Call 2: distillation / quality-control pass ----
        # Verifies each Call 1 label is grounded in the source text and the ID
        # matches the codebook. Drops hallucinations, canonicalizes IDs.
        # Runs only if Call 1 actually returned labels (skip empty chunks).
        call2_in_tokens = 0
        call2_out_tokens = 0
        distilled_labels = call1_labels  # fallback to Call 1 on Call 2 failure

        if call1_labels:
            try:
                distill_system = _build_distillation_system_prompt(codebook)
                distill_user = (
                    f"SOURCE TEXT:\n{clean_input}\n\n"
                    f"CANDIDATE LABELS FROM CALL 1 ({len(call1_labels)} labels):\n"
                    f"{json.dumps(call1_labels, indent=1)}\n\n"
                    f"Verify each candidate against the source text and codebook. "
                    f"Drop hallucinations. Canonicalize IDs. Return the verified set."
                )
                call2_response = await client.messages.create(
                    model=OPUS_MODEL,
                    max_tokens=16384,
                    temperature=0.0,
                    system=[
                        {
                            "type": "text",
                            "text": distill_system,
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                    messages=[{"role": "user", "content": distill_user}],
                )

                call2_in_tokens = call2_response.usage.input_tokens
                call2_out_tokens = call2_response.usage.output_tokens
                call2_cache_read = getattr(call2_response.usage, "cache_read_input_tokens", 0) or 0
                call2_cache_creation = getattr(call2_response.usage, "cache_creation_input_tokens", 0) or 0

                # Feed Call 2 usage into the global _run_usage accumulator
                try:
                    from backend.rlm_runner import accumulate_catcher_usage
                    accumulate_catcher_usage(
                        model=OPUS_MODEL,
                        input_tokens=call2_in_tokens,
                        output_tokens=call2_out_tokens,
                        cached_tokens=call2_cache_read,
                        cache_write_tokens=call2_cache_creation,
                    )
                except Exception:
                    pass

                call2_text = call2_response.content[0].text.strip()
                call2_text = re.sub(r"^```(?:json)?\s*", "", call2_text)
                call2_text = re.sub(r"\s*```$", "", call2_text)
                call2_parsed = _parse_json_robust(call2_text)

                if isinstance(call2_parsed, dict) and isinstance(call2_parsed.get("labels"), list):
                    distilled_labels = call2_parsed["labels"]
                # else: silently fall back to Call 1 labels (already set above)

            except Exception as distill_exc:
                logger.warning(
                    "Gen7 label: chunk %d distillation (Call 2) failed, keeping Call 1 labels: %s",
                    chunk_idx, distill_exc,
                )

        parsed["labels"] = distilled_labels

        # Combined usage for cost tracking (Call 1 + Call 2)
        input_tokens = call1_in_tokens + call2_in_tokens
        output_tokens = call1_out_tokens + call2_out_tokens

        # Stage 1A red flag: if Opus's response body is shorter than the chunk
        # we fed it, it almost certainly wrote a shortcut (like echoing
        # "(original text as provided)") instead of a full structured response.
        # We compare against the chunk text directly (not the wrapped prompt).
        call1_response_chars = len(text or "")
        call2_response_chars = len(locals().get("call2_text", "") or "")
        input_chars = len(chunk.get("text", "") or "")
        call1_diff_chars = call1_response_chars - input_chars  # positive = added content
        call1_underproduced = call1_response_chars < input_chars

        meta = {
            "model": OPUS_MODEL,
            "difficulty": difficulty,
            "input_chars_to_opus": len(clean_input),
            "input_chars_chunk_only": input_chars,
            "label_count": len(distilled_labels),
            "stage1a": {
                "input_tokens": call1_in_tokens,
                "output_tokens": call1_out_tokens,
                "response_chars": call1_response_chars,
                "diff_chars": call1_diff_chars,
                "underproduced": call1_underproduced,
                "labels_added": len(call1_labels),
                "cache_read_tokens": call1_cache_read,
                "cache_creation_tokens": call1_cache_creation,
            },
            "stage1b": {
                "input_tokens": call2_in_tokens,
                "output_tokens": call2_out_tokens,
                "response_chars": call2_response_chars,
                "labels_after": len(distilled_labels),
                "labels_dropped": len(call1_labels) - len(distilled_labels),
            },
            # Flat aliases kept for the existing cost-tracker code path.
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "call1_input_tokens": call1_in_tokens,
            "call1_output_tokens": call1_out_tokens,
            "call1_response_chars": call1_response_chars,
            "call1_label_count": len(call1_labels),
            "call2_input_tokens": call2_in_tokens,
            "call2_output_tokens": call2_out_tokens,
            "call2_response_chars": call2_response_chars,
            "call2_label_count": len(distilled_labels),
            "cache_read_tokens": call1_cache_read,
            "cache_creation_tokens": call1_cache_creation,
        }

        parsed["_labeling_meta"] = meta
        parsed["chunk_index"] = chunk_idx
        parsed["difficulty"] = difficulty

        dropped = len(call1_labels) - len(distilled_labels)
        logger.info(
            "Gen7 label: chunk %d (%s) -- Call1=%d labels, Call2=%d labels (dropped %d), "
            "%d in/%d out tokens",
            chunk_idx, difficulty, len(call1_labels), len(distilled_labels), dropped,
            input_tokens, output_tokens,
        )
        return parsed

    except Exception as exc:
        logger.warning("Gen7 label: chunk %d failed: %s", chunk_idx, exc)
        return {
            "section_id": chunk.get("section_id", ""),
            "section_title": chunk.get("section_title", ""),
            "text": chunk.get("text", ""),
            "labels": [],
            "chunk_index": chunk_idx,
            "difficulty": difficulty,
            "_labeling_meta": {"error": str(exc), "model": OPUS_MODEL},
        }


# Tier 2 Anthropic rate limit: ~80K output tokens per minute for Opus.
# Each chunk produces ~10K output tokens. Dynamic batch size = floor(80K / 10K) = 8.
# Conservative: use 6 per batch to leave headroom for variance.
_OUTPUT_TPM_LIMIT = 80_000
_EST_OUTPUT_PER_CHUNK = 10_000
_MAX_BATCH_SIZE = max(1, _OUTPUT_TPM_LIMIT // _EST_OUTPUT_PER_CHUNK - 2)  # 6
_BATCH_COOLDOWN_SEC = 62  # wait 62s between batches to reset the TPM window


def _compute_batch_size(n_chunks: int) -> int:
    """Dynamic batch size: if fewer chunks than the limit, run them all at once."""
    if n_chunks <= _MAX_BATCH_SIZE:
        return n_chunks
    return _MAX_BATCH_SIZE


def deduplicate_labels(labeled_chunks: list[dict]) -> list[dict]:
    """Algorithmically dedupe labels across chunks on exact (id, type) string match.

    Pure-code dedup. No semantic matching, no approximations.
    - `supply_amoxicillin_250mg` (supply_list) == same ID in another chunk: dedup.
    - `supply_amoxicillin_250mg` vs `supply_amoxicillin_250mg_dispersible`:
      different strings, kept as separate entries. The REPL handles semantic
      merging later using reading comprehension.

    Keeps first occurrence's quote_context. Collects all source_section_ids
    (one per chunk where this (id, type) appeared) into a list.

    Returns a flat list of deduped label dicts, sorted by (type, id) for
    deterministic output.
    """
    seen: dict[tuple[str, str], dict] = {}
    for chunk in labeled_chunks:
        section_id = chunk.get("section_id", "")
        chunk_idx = chunk.get("chunk_index", -1)
        for label in chunk.get("labels", []):
            if not isinstance(label, dict):
                continue
            lid = str(label.get("id", "")).strip()
            ltype = str(label.get("type", "")).strip()
            if not lid or not ltype:
                continue
            key = (lid, ltype)
            if key not in seen:
                # First occurrence: keep full label + start source tracking
                entry = dict(label)
                entry["source_section_ids"] = [section_id] if section_id else []
                entry["source_chunk_indices"] = [chunk_idx] if chunk_idx >= 0 else []
                seen[key] = entry
            else:
                # Additional occurrence: just append to source tracking
                existing = seen[key]
                if section_id and section_id not in existing["source_section_ids"]:
                    existing["source_section_ids"].append(section_id)
                if chunk_idx >= 0 and chunk_idx not in existing["source_chunk_indices"]:
                    existing["source_chunk_indices"].append(chunk_idx)

    # Sort for deterministic output (same dedup run -> same list order)
    deduped = sorted(seen.values(), key=lambda d: (d.get("type", ""), d.get("id", "")))
    logger.info(
        "Gen7 labeler: dedup collapsed %d raw labels to %d unique (id, type) entries",
        sum(len(c.get("labels", [])) for c in labeled_chunks),
        len(deduped),
    )
    return deduped


async def label_all_chunks(
    chunks: list[dict],
    codebook: str,
    anthropic_key: str,
    on_chunk_labeled: Any = None,
) -> list[dict]:
    """Label all micro-chunks with dynamic batching and cache priming.

    Strategy:
      1. First chunk runs alone to prime the prompt cache (cache creation).
      2. Remaining chunks run in batches of _MAX_BATCH_SIZE with 62s cooldown
         between batches to respect Tier 2 output TPM limits.
      3. If total chunks <= batch size, everything runs in one parallel burst.

    Returns list of labeled chunk dicts sorted by chunk_index.
    """
    logger.info("Gen7 labeler: %d chunks to label with %s", len(chunks), OPUS_MODEL)

    client = anthropic.AsyncAnthropic(api_key=anthropic_key)
    batch_size = _compute_batch_size(len(chunks))

    results: list[Any] = [None] * len(chunks)

    total_batches = 1 if len(chunks) <= batch_size else (1 + ((len(chunks) - 1 + batch_size - 1) // batch_size))
    labeled_so_far = 0

    async def _notify(chunk_result: Any, chunk_idx: int, batch_num: int) -> None:
        nonlocal labeled_so_far
        if isinstance(chunk_result, dict):
            labeled_so_far += 1
        if on_chunk_labeled:
            meta = chunk_result.get("_labeling_meta", {}) if isinstance(chunk_result, dict) else {}
            await on_chunk_labeled({
                "event": "chunk_labeled",
                "chunk_index": chunk_idx,
                "batch": batch_num,
                "total_batches": total_batches,
                "labeled_so_far": labeled_so_far,
                "total_chunks": len(chunks),
                "label_count": meta.get("label_count", 0),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "error": meta.get("error"),
            })

    if len(chunks) <= batch_size:
        # Small guide: all chunks in one burst
        logger.info("Gen7 labeler: small guide, all %d chunks in parallel (1 batch)", len(chunks))
        if on_chunk_labeled:
            await on_chunk_labeled({
                "event": "batch_start", "batch": 1, "total_batches": 1,
                "chunk_count": len(chunks), "parallel": True,
            })
        batch_results = await asyncio.gather(*[
            label_chunk(chunk, codebook, anthropic_key, _client=client)
            for chunk in chunks
        ], return_exceptions=True)
        for i, r in enumerate(batch_results):
            results[i] = r
            await _notify(r, i, 1)
    else:
        # Large guide: prime cache with first chunk, then batch the rest
        logger.info("Gen7 labeler: priming cache with chunk 0, then batches of %d", batch_size)
        if on_chunk_labeled:
            await on_chunk_labeled({
                "event": "cache_prime_start", "chunk_index": 0,
                "total_chunks": len(chunks), "batch_size": batch_size,
            })
        results[0] = await label_chunk(chunks[0], codebook, anthropic_key, _client=client)
        await _notify(results[0], 0, 0)

        remaining = list(enumerate(chunks[1:], start=1))
        batch_num = 0
        while remaining:
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]
            batch_num += 1

            logger.info(
                "Gen7 labeler: batch %d -- chunks %s (%d remaining after this)",
                batch_num,
                [idx for idx, _ in batch],
                len(remaining),
            )
            if on_chunk_labeled:
                await on_chunk_labeled({
                    "event": "batch_start", "batch": batch_num, "total_batches": total_batches - 1,
                    "chunk_count": len(batch), "parallel": True,
                    "chunk_indices": [idx for idx, _ in batch],
                })

            batch_results = await asyncio.gather(*[
                label_chunk(chunk, codebook, anthropic_key, _client=client)
                for _, chunk in batch
            ], return_exceptions=True)

            for (idx, _), r in zip(batch, batch_results):
                results[idx] = r
                await _notify(r, idx, batch_num)

            # Cooldown between batches to reset TPM window
            if remaining:
                logger.info("Gen7 labeler: cooldown %ds before next batch", _BATCH_COOLDOWN_SEC)
                if on_chunk_labeled:
                    await on_chunk_labeled({
                        "event": "batch_cooldown", "batch": batch_num,
                        "cooldown_sec": _BATCH_COOLDOWN_SEC,
                        "remaining_chunks": len(remaining),
                    })
                await asyncio.sleep(_BATCH_COOLDOWN_SEC)

    # Aggregate results
    labeled = []
    errors = 0
    total_labels = 0
    total_cache_read = 0
    total_cache_creation = 0

    for i, r in enumerate(results):
        if isinstance(r, dict):
            labeled.append(r)
            meta = r.get("_labeling_meta", {})
            total_labels += meta.get("label_count", len(r.get("labels", [])))
            total_cache_read += meta.get("cache_read_tokens", 0)
            total_cache_creation += meta.get("cache_creation_tokens", 0)
        else:
            errors += 1
            logger.warning("Gen7 label: chunk %d exception: %s", i, r)
            labeled.append({
                **chunks[i],
                "labels": [],
                "_labeling_meta": {"error": str(r), "model": OPUS_MODEL},
            })

    labeled.sort(key=lambda c: c.get("chunk_index", 0))

    cache_pct = (
        total_cache_read / (total_cache_read + total_cache_creation) * 100
        if (total_cache_read + total_cache_creation) > 0 else 0
    )
    logger.info(
        "Gen7 labeler: done -- %d labeled, %d errors, %d total labels, "
        "cache hit rate: %.0f%% (%d read / %d creation tokens)",
        len(labeled), errors, total_labels,
        cache_pct, total_cache_read, total_cache_creation,
    )
    return labeled
