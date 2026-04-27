"""gen8 labeler: Opus mono (1 Opus call per chunk + Call 2 QC).

Adapted from `backend/gen7/labeler.py`. The only gen8-specific change is
that `_labeling_meta` carries `manual_page_start` / `manual_page_end` so
Stage 3 can ground `provenance: "WHO 2012 page N"` citations correctly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

OPUS_MODEL = "claude-opus-4-6"


def _build_labeling_system_prompt(codebook: str) -> str:
    """Stage 1 system prompt (identical to gen7 except for the header)."""
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and label EVERY clinical item "
        "with a structured annotation.\n\n"
        "For EVERY clinical item in the text, produce a label with:\n"
        "- span: the exact verbatim text from the chunk (copy-paste, no paraphrase)\n"
        "- id: a canonical ID using the codebook prefixes below. IDs must be DESCRIPTIVE "
        "and content-derived, never numbered.\n"
        "- type: one of the 7 artifact types: supply_list, variables, predicates, modules, "
        "phrase_bank, router, integrative\n"
        "- subtype: optional finer classification\n"
        "- quote_context: the surrounding sentence for provenance\n\n"
        "ARTIFACT TYPE MAPPING:\n"
        "  supply_list: physical items the CHW must possess.\n"
        "    - 'consumable' (supply_ prefix): medications, test kits, disposables\n"
        "    - 'equipment' (equip_ prefix): durable tools\n"
        "  variables: runtime inputs the CHW collects.\n"
        "    - q_ self-reported; ex_ observed; v_ measured; lab_ test result;\n"
        "      hx_ history; demo_ demographics\n"
        "  predicates: boolean thresholds computed from variables. Prefix p_.\n"
        "  modules: clinical decision topics. Prefix mod_.\n"
        "  phrase_bank: things the CHW says (m_, adv_, tx_, rx_, ref_).\n"
        "  router: routing/triage decisions.\n"
        "  integrative: cross-module interactions.\n\n"
        f"NAMING CODEBOOK:\n{codebook}\n\n"
        "CRITICAL RULES:\n"
        "1. Label EVERY clinical item. Missing items is worse than having extras.\n"
        "2. IDs descriptive, following prefix conventions.\n"
        "3. Numeric variables MUST end with a unit suffix.\n"
        "4. Lowercase_with_underscores only.\n"
        "5. Prefer over-labeling to under-labeling; dedup happens downstream.\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences). Shape:\n"
        '{"section_id": "...", "section_title": "...", "text": "(original text)", '
        '"labels": [{"span": "...", "id": "prefix_descriptive_name", "type": "...", '
        '"subtype": "...", "quote_context": "..."}]}'
    )


def _build_distillation_system_prompt(codebook: str) -> str:
    """Stage 2 system prompt (verification QC)."""
    return (
        "You are a clinical label quality-control verifier. You examine candidate "
        "labels produced by a first-pass labeler and verify each is grounded in "
        "the source text and has a codebook-compliant ID.\n\n"
        "YOUR JOB:\n"
        "1. For each candidate label, verify its 'span' appears in the source text. "
        "Drop labels whose span cannot be found.\n"
        "2. Verify each label's 'id' follows the codebook prefix convention for its "
        "'type'. If wrong-prefixed, correct or drop.\n"
        "3. Drop hallucinations (generic placeholders, concepts not in the text).\n"
        "4. Keep verified labels with their original 'quote_context'.\n"
        "5. DO NOT add new labels.\n\n"
        f"CODEBOOK (for ID canonicalization):\n{codebook}\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences). Shape:\n"
        '{"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", '
        '"quote_context": "..."}]}'
    )


def _parse_json_robust(text: str) -> Any:
    """JSON parse with the same repairs as the gen7 labeler."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    labels_match = re.search(r'"labels"\s*:\s*\[', repaired)
    if labels_match:
        start = labels_match.end()
        last_complete = start
        brace_depth = 0
        i = start
        while i < len(repaired):
            c = repaired[i]
            if c == "{":
                brace_depth += 1
            elif c == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    last_complete = i + 1
            elif c == '"':
                i += 1
                while i < len(repaired) and repaired[i] != '"':
                    if repaired[i] == "\\":
                        i += 1
                    i += 1
            i += 1
        if last_complete > start:
            truncated = repaired[:last_complete].rstrip().rstrip(",")
            truncated += "]}"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass
    logger.warning("gen8 label: JSON repair failed, returning text[:200]: %s", text[:200])
    return None


async def label_chunk(
    chunk: dict,
    codebook: str,
    anthropic_key: str,
    _client: "anthropic.AsyncAnthropic | None" = None,
) -> dict:
    """Label one chunk with Opus (Call 1 + Call 2 QC).

    Threads `manual_page_start`/`manual_page_end` into `_labeling_meta` so
    Stage 3 can cite pages in predicate provenance.
    """
    chunk_idx = chunk.get("chunk_index", -1)
    difficulty = chunk.get("difficulty", "medium")
    page_start = chunk.get("manual_page_start")
    page_end = chunk.get("manual_page_end")

    client = _client or anthropic.AsyncAnthropic(api_key=anthropic_key)

    system_prompt = _build_labeling_system_prompt(codebook)
    page_hint = ""
    if page_start is not None:
        page_hint = f" (manual pages {page_start}-{page_end})" if page_end != page_start else f" (manual page {page_start})"
    clean_input = f"Section: {chunk['section_title']} ({chunk['section_id']}){page_hint}\n\n{chunk['text']}"

    try:
        response = await client.messages.create(
            model=OPUS_MODEL,
            max_tokens=16384,
            temperature=0.0,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            messages=[{"role": "user", "content": f"CHUNK TO LABEL:\n{clean_input}"}],
        )

        text = response.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        parsed = _parse_json_robust(text)
        if not isinstance(parsed, dict):
            parsed = {"section_id": chunk["section_id"], "labels": [], "_error": "not dict"}
        parsed.setdefault("labels", [])

        call1_labels = list(parsed.get("labels", []))
        call1_in = response.usage.input_tokens
        call1_out = response.usage.output_tokens
        call1_cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        call1_cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0

        try:
            from backend.rlm_runner import accumulate_catcher_usage
            accumulate_catcher_usage(
                model=OPUS_MODEL,
                input_tokens=call1_in,
                output_tokens=call1_out,
                cached_tokens=call1_cache_read,
                cache_write_tokens=call1_cache_creation,
            )
        except Exception:
            pass

        # Call 2: distillation / QC
        call2_in = 0
        call2_out = 0
        call2_cache_read = 0
        call2_cache_creation = 0
        distilled_labels = call1_labels
        call2_text = ""

        if call1_labels:
            try:
                distill_system = _build_distillation_system_prompt(codebook)
                distill_user = (
                    f"SOURCE TEXT:\n{clean_input}\n\n"
                    f"CANDIDATE LABELS FROM CALL 1 ({len(call1_labels)} labels):\n"
                    f"{json.dumps(call1_labels, indent=1)}\n\n"
                    f"Verify each candidate. Drop hallucinations. Canonicalize IDs."
                )
                call2_response = await client.messages.create(
                    model=OPUS_MODEL,
                    max_tokens=16384,
                    temperature=0.0,
                    system=[{
                        "type": "text",
                        "text": distill_system,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }],
                    messages=[{"role": "user", "content": distill_user}],
                )
                call2_in = call2_response.usage.input_tokens
                call2_out = call2_response.usage.output_tokens
                call2_cache_read = getattr(call2_response.usage, "cache_read_input_tokens", 0) or 0
                call2_cache_creation = getattr(call2_response.usage, "cache_creation_input_tokens", 0) or 0

                try:
                    from backend.rlm_runner import accumulate_catcher_usage
                    accumulate_catcher_usage(
                        model=OPUS_MODEL,
                        input_tokens=call2_in,
                        output_tokens=call2_out,
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
            except Exception as exc:
                logger.warning("gen8 label: chunk %d Call 2 failed; keeping Call 1: %s", chunk_idx, exc)

        parsed["labels"] = distilled_labels
        input_tokens = call1_in + call2_in
        output_tokens = call1_out + call2_out
        call1_response_chars = len(text or "")
        call2_response_chars = len(call2_text or "")
        input_chars = len(chunk.get("text", "") or "")

        meta = {
            "pipeline": "gen8",
            "model": OPUS_MODEL,
            "method": "opus-mono",
            "difficulty": difficulty,
            "manual_page_start": page_start,
            "manual_page_end": page_end,
            "input_chars_to_opus": len(clean_input),
            "input_chars_chunk_only": input_chars,
            "label_count": len(distilled_labels),
            "stage1a": {
                "input_tokens": call1_in, "output_tokens": call1_out,
                "response_chars": call1_response_chars,
                "diff_chars": call1_response_chars - input_chars,
                "underproduced": call1_response_chars < input_chars,
                "labels_added": len(call1_labels),
                "cache_read_tokens": call1_cache_read,
                "cache_creation_tokens": call1_cache_creation,
            },
            "stage1b": {
                "input_tokens": call2_in, "output_tokens": call2_out,
                "response_chars": call2_response_chars,
                "labels_after": len(distilled_labels),
                "labels_dropped": len(call1_labels) - len(distilled_labels),
            },
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": call1_cache_read + call2_cache_read,
            "cache_creation_tokens": call1_cache_creation + call2_cache_creation,
        }

        parsed["_labeling_meta"] = meta
        parsed["chunk_index"] = chunk_idx
        parsed["difficulty"] = difficulty
        parsed["manual_page_start"] = page_start
        parsed["manual_page_end"] = page_end

        dropped = len(call1_labels) - len(distilled_labels)
        logger.info(
            "gen8 label: chunk %d (%s, p.%s-%s) -- Call1=%d Call2=%d (dropped %d), %d in/%d out",
            chunk_idx, difficulty, page_start, page_end,
            len(call1_labels), len(distilled_labels), dropped, input_tokens, output_tokens,
        )
        return parsed

    except Exception as exc:
        logger.warning("gen8 label: chunk %d failed: %s", chunk_idx, exc)
        return {
            "section_id": chunk.get("section_id", ""),
            "section_title": chunk.get("section_title", ""),
            "text": chunk.get("text", ""),
            "labels": [],
            "chunk_index": chunk_idx,
            "difficulty": difficulty,
            "manual_page_start": page_start,
            "manual_page_end": page_end,
            "_labeling_meta": {
                "pipeline": "gen8",
                "error": str(exc),
                "model": OPUS_MODEL,
                "manual_page_start": page_start,
                "manual_page_end": page_end,
            },
        }


_OUTPUT_TPM_LIMIT = 80_000
_EST_OUTPUT_PER_CHUNK = 10_000
_MAX_BATCH_SIZE = max(1, _OUTPUT_TPM_LIMIT // _EST_OUTPUT_PER_CHUNK - 2)
_BATCH_COOLDOWN_SEC = 62


def _compute_batch_size(n: int) -> int:
    return n if n <= _MAX_BATCH_SIZE else _MAX_BATCH_SIZE


def deduplicate_labels(labeled_chunks: list[dict]) -> list[dict]:
    """Exact-string dedup on (id, type). Keeps first occurrence's context."""
    seen: dict[tuple[str, str], dict] = {}
    for chunk in labeled_chunks:
        section_id = chunk.get("section_id", "")
        chunk_idx = chunk.get("chunk_index", -1)
        page_start = chunk.get("manual_page_start")
        for label in chunk.get("labels", []):
            if not isinstance(label, dict):
                continue
            lid = str(label.get("id", "")).strip()
            ltype = str(label.get("type", "")).strip()
            if not lid or not ltype:
                continue
            key = (lid, ltype)
            if key not in seen:
                entry = dict(label)
                entry["source_section_ids"] = [section_id] if section_id else []
                entry["source_chunk_indices"] = [chunk_idx] if chunk_idx >= 0 else []
                entry["source_pages"] = [page_start] if page_start is not None else []
                seen[key] = entry
            else:
                existing = seen[key]
                if section_id and section_id not in existing["source_section_ids"]:
                    existing["source_section_ids"].append(section_id)
                if chunk_idx >= 0 and chunk_idx not in existing["source_chunk_indices"]:
                    existing["source_chunk_indices"].append(chunk_idx)
                if page_start is not None and page_start not in existing["source_pages"]:
                    existing["source_pages"].append(page_start)
    return sorted(seen.values(), key=lambda d: (d.get("type", ""), d.get("id", "")))


async def label_all_chunks(
    chunks: list[dict],
    codebook: str,
    anthropic_key: str,
    on_chunk_labeled: Any = None,
) -> list[dict]:
    """Parallel per-chunk labeling with cache-priming and TPM-aware batching."""
    logger.info("gen8 labeler: %d chunks to label with %s", len(chunks), OPUS_MODEL)

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
        logger.info("gen8 labeler: all %d chunks in parallel", len(chunks))
        if on_chunk_labeled:
            await on_chunk_labeled({"event": "batch_start", "batch": 1, "total_batches": 1,
                                    "chunk_count": len(chunks), "parallel": True})
        batch_results = await asyncio.gather(*[
            label_chunk(c, codebook, anthropic_key, _client=client) for c in chunks
        ], return_exceptions=True)
        for i, r in enumerate(batch_results):
            results[i] = r
            await _notify(r, i, 1)
    else:
        if on_chunk_labeled:
            await on_chunk_labeled({"event": "cache_prime_start", "chunk_index": 0,
                                    "total_chunks": len(chunks), "batch_size": batch_size})
        results[0] = await label_chunk(chunks[0], codebook, anthropic_key, _client=client)
        await _notify(results[0], 0, 0)

        remaining = list(enumerate(chunks[1:], start=1))
        batch_num = 0
        while remaining:
            batch = remaining[:batch_size]
            remaining = remaining[batch_size:]
            batch_num += 1
            if on_chunk_labeled:
                await on_chunk_labeled({
                    "event": "batch_start", "batch": batch_num,
                    "total_batches": total_batches - 1,
                    "chunk_count": len(batch), "parallel": True,
                    "chunk_indices": [idx for idx, _ in batch],
                })
            batch_results = await asyncio.gather(*[
                label_chunk(c, codebook, anthropic_key, _client=client) for _, c in batch
            ], return_exceptions=True)
            for (idx, _), r in zip(batch, batch_results):
                results[idx] = r
                await _notify(r, idx, batch_num)
            if remaining:
                if on_chunk_labeled:
                    await on_chunk_labeled({"event": "batch_cooldown", "batch": batch_num,
                                            "cooldown_sec": _BATCH_COOLDOWN_SEC,
                                            "remaining_chunks": len(remaining)})
                await asyncio.sleep(_BATCH_COOLDOWN_SEC)

    labeled = []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            labeled.append(r)
        else:
            labeled.append({
                **chunks[i],
                "labels": [],
                "_labeling_meta": {"pipeline": "gen8", "error": str(r), "model": OPUS_MODEL},
            })
    labeled.sort(key=lambda c: c.get("chunk_index", 0))
    return labeled
