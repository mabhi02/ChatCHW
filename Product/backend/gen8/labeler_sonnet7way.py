"""gen8.5 labeler: 7 parallel narrow-prompt Sonnet calls per chunk.

Each pass asks Sonnet to label exactly one artifact type on the chunk
(supply_list, variables, predicates, modules, phrase_bank, router,
integrative). A deterministic reconciler then merges, dedups, and
preserves genuinely cross-type spans (rare but real: a phrase that is
also a supply reference).

Output shape mirrors gen7/gen8 labeler contract: a per-chunk dict with
`labels: [...]`, `_labeling_meta`, and the page-range fields gen8 needs
for Stage 3 provenance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from anthropic import AsyncAnthropic

from backend.gen8.concurrency import throttled_gather

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"

TYPE_NAMES: tuple[str, ...] = (
    "supply_list", "variables", "predicates", "modules",
    "phrase_bank", "router", "integrative",
)


def _pass_prompt(type_name: str, codebook: str) -> str:
    """Narrow-prompt per-pass system prompt.

    The type is hard-pinned: Sonnet MUST set every label's `type` field to
    this exact string. Cross-type hallucinations are filtered later by the
    reconciler but the prompt keeps them rare by only asking about one
    artifact type at a time.
    """
    return (
        f"You are a clinical document labeler. You label ONLY `{type_name}` items.\n"
        f"Every label you produce MUST have type = {type_name!r} (exact string).\n"
        "If there are no items of this type in the chunk, return an empty labels array.\n\n"
        "Every label has:\n"
        "- span: exact verbatim text from the chunk\n"
        f"- id: prefix-compliant ID for {type_name}\n"
        f"- type: {type_name!r}\n"
        "- subtype: optional\n"
        "- quote_context: the surrounding sentence\n\n"
        f"CODEBOOK (prefix conventions):\n{codebook}\n\n"
        "OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences). Shape:\n"
        '{"labels": [...]}\n'
    )


def _parse_json_robust(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


async def _run_pass(
    client: AsyncAnthropic,
    chunk: dict,
    type_name: str,
    codebook: str,
) -> tuple[str, list[dict], dict]:
    """Run one narrow-type Sonnet pass over one chunk."""
    page_start = chunk.get("manual_page_start")
    page_end = chunk.get("manual_page_end")
    page_hint = ""
    if page_start is not None:
        page_hint = (
            f" (manual pages {page_start}-{page_end})"
            if page_end != page_start else f" (manual page {page_start})"
        )
    clean_input = (
        f"Section: {chunk['section_title']} ({chunk['section_id']}){page_hint}\n\n"
        f"{chunk['text']}"
    )

    system = _pass_prompt(type_name, codebook)
    try:
        resp = await client.messages.create(
            model=SONNET_MODEL,
            max_tokens=8192,
            temperature=0.0,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }],
            messages=[{"role": "user", "content": f"CHUNK TO LABEL:\n{clean_input}"}],
        )
        text = resp.content[0].text.strip() if resp.content else "{}"
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = _parse_json_robust(text) or {}
        labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        # Feed into the global run-usage accumulator
        try:
            from backend.rlm_runner import accumulate_catcher_usage
            accumulate_catcher_usage(
                model=SONNET_MODEL,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_tokens=usage["cache_read_tokens"],
                cache_write_tokens=usage["cache_creation_tokens"],
            )
        except Exception:
            pass
        return type_name, labels, usage
    except Exception as exc:
        logger.warning("gen8.5 pass %s on chunk %s failed: %s",
                       type_name, chunk.get("chunk_index"), exc)
        return type_name, [], {"error": str(exc), "input_tokens": 0, "output_tokens": 0,
                               "cache_read_tokens": 0, "cache_creation_tokens": 0}


def _reconcile(
    raw_by_pass: dict[str, list[dict]],
    chunk_text: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic reconciliation of the 7 pass outputs for one chunk.

    Returns (final, dropped, cross_type):
      - final: accepted labels, one entry per (id, type) key
      - dropped: labels rejected because their span does not appear in chunk_text
      - cross_type: (span) pairs that were emitted by multiple passes with
        different types -- preserved separately so reviewers can see them.
    """
    # Dedup is by (normalized_span, type) so that two passes emitting the same
    # span with different ids collapse to one canonical entry (shorter id wins).
    # Keying by (id, type) instead would make the shorter-id tiebreaker
    # unreachable, since identical ids already collide on the id alone.
    final_by_key: dict[tuple[str, str], dict] = {}
    dropped: list[dict] = []
    raw_span_to_types: dict[str, set[str]] = {}

    for expected_type, labels in raw_by_pass.items():
        for lab in labels:
            if not isinstance(lab, dict):
                continue
            # Cross-type detection runs on raw labels (before grounding +
            # type-match filtering) so reviewers see every span that >1 pass
            # claimed, not just the ones that survived.
            raw_span = str(lab.get("span") or "").strip()
            raw_ltype = str(lab.get("type") or expected_type).strip()
            if raw_span and raw_ltype:
                raw_span_to_types.setdefault(raw_span.lower(), set()).add(raw_ltype)

            # Enforce the narrow-pass promise: only keep the type we asked for
            if raw_ltype != expected_type:
                continue
            lid = str(lab.get("id") or "").strip()
            span = raw_span
            if not lid or not span:
                continue
            # Span-grep grounding: span must appear in chunk text (case-sensitive
            # then case-insensitive fallback)
            grounded = span in chunk_text or span.lower() in chunk_text.lower()
            if not grounded:
                d = dict(lab)
                d["_drop_reason"] = "span_not_in_chunk"
                dropped.append(d)
                continue

            key = (span.lower(), expected_type)
            if key in final_by_key:
                existing = final_by_key[key]
                # Prefer shorter id as canonical; ties go to first seen.
                if len(lid) < len(str(existing.get("id", ""))):
                    d = dict(existing)
                    d["_drop_reason"] = f"superseded_by_shorter_id_{lid!r}"
                    dropped.append(d)
                    final_by_key[key] = lab
                else:
                    d = dict(lab)
                    d["_drop_reason"] = f"shorter_id_already_kept_{existing.get('id')!r}"
                    dropped.append(d)
            else:
                final_by_key[key] = lab

    cross_type: list[dict] = [
        {"span": span, "types": sorted(types)}
        for span, types in raw_span_to_types.items()
        if len(types) > 1
    ]
    final = sorted(final_by_key.values(), key=lambda d: (d.get("type", ""), d.get("id", "")))
    return final, dropped, cross_type


async def label_one_chunk(
    chunk: dict,
    codebook: str,
    client: AsyncAnthropic,
    pass_concurrency: int = 7,
) -> dict:
    """Run the 7 narrow-type passes on one chunk and reconcile."""
    pass_coros = [_run_pass(client, chunk, t, codebook) for t in TYPE_NAMES]
    results = await throttled_gather(pass_coros, max_concurrent=pass_concurrency)

    raw_by_pass: dict[str, list[dict]] = {}
    usage_by_pass: dict[str, dict] = {}
    raw_total = 0
    for type_name, labels, usage in results:
        raw_by_pass[type_name] = labels
        usage_by_pass[type_name] = usage
        raw_total += len(labels)

    final, dropped, cross_type = _reconcile(raw_by_pass, chunk.get("text", "") or "")

    total_in = sum(u.get("input_tokens", 0) for u in usage_by_pass.values())
    total_out = sum(u.get("output_tokens", 0) for u in usage_by_pass.values())

    return {
        **{k: v for k, v in chunk.items() if k not in ("blocks",)},
        "labels": final,
        "_labeling_meta": {
            "pipeline": "gen8.5",
            "model": SONNET_MODEL,
            "method": "7-way-sonnet",
            "passes": list(TYPE_NAMES),
            "raw_count": raw_total,
            "final_count": len(final),
            "dropped_count": len(dropped),
            "cross_type_count": len(cross_type),
            "manual_page_start": chunk.get("manual_page_start"),
            "manual_page_end": chunk.get("manual_page_end"),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "per_pass_usage": usage_by_pass,
            "label_count": len(final),
        },
        "_dropped_labels": dropped,
        "_cross_type_spans": cross_type,
        "blocks": chunk.get("blocks", []),
    }


async def label_all_chunks(
    chunks: list[dict],
    codebook: str,
    anthropic_key: str,
    on_chunk_labeled: Any = None,
    chunk_concurrency: int = 3,
    pass_concurrency: int = 7,
) -> list[dict]:
    """Run the 7-way pipeline across every chunk."""
    client = AsyncAnthropic(api_key=anthropic_key)
    logger.info("gen8.5 labeler: %d chunks x 7 passes (chunk_concurrency=%d, pass_concurrency=%d)",
                len(chunks), chunk_concurrency, pass_concurrency)

    results: list[dict | None] = [None] * len(chunks)
    labeled_so_far = 0

    async def _wrapped(idx: int, chunk: dict) -> None:
        nonlocal labeled_so_far
        out = await label_one_chunk(chunk, codebook, client, pass_concurrency=pass_concurrency)
        results[idx] = out
        labeled_so_far += 1
        if on_chunk_labeled:
            meta = out.get("_labeling_meta", {})
            await on_chunk_labeled({
                "event": "chunk_labeled",
                "chunk_index": idx,
                "batch": 1,
                "total_batches": 1,
                "labeled_so_far": labeled_so_far,
                "total_chunks": len(chunks),
                "label_count": meta.get("label_count", 0),
                "input_tokens": meta.get("input_tokens", 0),
                "output_tokens": meta.get("output_tokens", 0),
                "error": None,
            })

    if on_chunk_labeled:
        await on_chunk_labeled({
            "event": "batch_start", "batch": 1, "total_batches": 1,
            "chunk_count": len(chunks), "parallel": True,
        })

    await throttled_gather(
        [_wrapped(i, c) for i, c in enumerate(chunks)],
        max_concurrent=chunk_concurrency,
    )

    # Replace None entries (shouldn't happen; defensive)
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                **chunks[i],
                "labels": [],
                "_labeling_meta": {"pipeline": "gen8.5", "model": SONNET_MODEL, "error": "no_result"},
            }
    return [r for r in results if r is not None]


def deduplicate_labels(labeled_chunks: list[dict]) -> list[dict]:
    """Exact-string dedup across chunks (same contract as gen7/gen8 labeler)."""
    seen: dict[tuple[str, str], dict] = {}
    for chunk in labeled_chunks:
        section_id = chunk.get("section_id", "")
        chunk_idx = chunk.get("chunk_index", -1)
        page_start = chunk.get("manual_page_start")
        for label in chunk.get("labels", []) or []:
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
