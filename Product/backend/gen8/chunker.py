"""gen8 Phase 0: sequential micro-chunker with per-chunk page ranges.

Adapted from `backend/gen7/chunker.py`. The only functional addition is
that every emitted chunk carries `manual_page_start` and `manual_page_end`
derived from the block `.page` field threaded by the ingestion layer.
Tier 0 of gen8 depends on these page numbers so Stage 3 predicates can
cite "WHO 2012 page N" instead of the chunker's section slug.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TARGET_TOKENS = 2000
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def micro_chunk_guide(guide_json: dict) -> list[dict]:
    """Split guide into sequential ~2K-token chunks, threading page numbers."""
    sections = guide_json.get("sections", {})

    # Flatten blocks into a sequential stream while keeping the block dict so
    # we can later read `.page` off it.
    stream: list[tuple[str, str, str, dict]] = []
    for sec_id, sec_data in sections.items():
        title = sec_data.get("title", sec_id)
        blocks = sec_data.get("blocks", [])
        if not blocks:
            raw = (sec_data.get("raw_text", "") or "").strip()
            if raw:
                stream.append((sec_id, title, raw, {"text": raw, "page": sec_data.get("page_start")}))
        else:
            for block in blocks:
                bt = (block.get("text", "") or "").strip()
                if bt:
                    stream.append((sec_id, title, bt, block))

    if not stream:
        return []

    chunks: list[dict] = []
    chunk_index = 0
    current_text = ""
    current_blocks: list[dict] = []
    current_sections: set[tuple[str, str]] = set()
    current_tokens = 0

    for sec_id, sec_title, block_text, block_obj in stream:
        block_tokens = _estimate_tokens(block_text)
        if (sec_id, sec_title) not in current_sections and current_text:
            current_text += f"\n[{sec_title}]\n"
        current_sections.add((sec_id, sec_title))
        current_text += block_text + "\n"
        current_blocks.append(block_obj)
        current_tokens += block_tokens

        if current_tokens >= TARGET_TOKENS:
            chunks.append(_make_chunk(chunk_index, current_sections, current_text, current_blocks))
            chunk_index += 1
            current_text = ""
            current_blocks = []
            current_sections = set()
            current_tokens = 0

    if current_text.strip():
        chunks.append(_make_chunk(chunk_index, current_sections, current_text, current_blocks))

    # Merge trailing short chunk into previous one
    merged: list[dict] = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk["text"])
        if tokens < TARGET_TOKENS and merged:
            prev = merged[-1]
            prev["text"] = prev["text"] + "\n\n" + chunk["text"]
            prev["blocks"].extend(chunk["blocks"])
            prev["sections_in_chunk"] = prev.get("sections_in_chunk", 1) + chunk.get("sections_in_chunk", 1)
            if "+" not in prev["section_id"]:
                prev["section_id"] = prev["section_id"] + "+" + chunk["section_id"]
            # Extend page range to cover the merged chunk
            if chunk.get("manual_page_start") is not None:
                prev_start = prev.get("manual_page_start")
                prev["manual_page_start"] = (
                    chunk["manual_page_start"] if prev_start is None
                    else min(prev_start, chunk["manual_page_start"])
                )
            if chunk.get("manual_page_end") is not None:
                prev_end = prev.get("manual_page_end")
                prev["manual_page_end"] = (
                    chunk["manual_page_end"] if prev_end is None
                    else max(prev_end, chunk["manual_page_end"])
                )
        else:
            merged.append(chunk)

    chunks = merged
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    # Difficulty classification (reuses gen7 infrastructure; safe to call)
    from backend.validators.test_suite import classify_chunk_difficulty
    for chunk in chunks:
        fake = {"sections": {chunk["section_id"]: {
            "title": chunk["section_title"], "raw_text": chunk["text"],
            "blocks": chunk["blocks"],
        }}}
        diff = classify_chunk_difficulty(fake)
        chunk["difficulty"] = diff["difficulty"]
        chunk["difficulty_score"] = diff["score"]

    sizes = [_estimate_tokens(c["text"]) for c in chunks]
    pages_known = sum(1 for c in chunks if c.get("manual_page_start") is not None)
    logger.info(
        "gen8 chunker: %d blocks -> %d chunks (min=%d avg=%d max=%d tokens; %d/%d chunks have page ranges)",
        len(stream), len(chunks),
        min(sizes) if sizes else 0,
        sum(sizes) // len(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
        pages_known, len(chunks),
    )

    return chunks


def _make_chunk(index: int, sections: set, text: str, blocks: list[dict]) -> dict:
    """Build a chunk dict, computing `manual_page_start/end` from block pages."""
    sec_list = list(sections)
    if len(sec_list) == 1:
        sec_id, sec_title = sec_list[0]
    else:
        sec_id = "+".join(s[0] for s in sec_list[:3])
        sec_title = " | ".join(s[1] for s in sec_list[:3])

    pages = [b.get("page") for b in blocks if isinstance(b, dict) and b.get("page") is not None]
    manual_page_start = min(pages) if pages else None
    manual_page_end = max(pages) if pages else None

    return {
        "chunk_index": index,
        "section_id": sec_id,
        "section_title": sec_title,
        "text": text.strip(),
        "blocks": blocks,
        "sections_in_chunk": len(sec_list),
        "manual_page_start": manual_page_start,
        "manual_page_end": manual_page_end,
    }
