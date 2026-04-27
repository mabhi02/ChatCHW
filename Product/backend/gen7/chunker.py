"""Gen 7 Phase 0: Sequential chunking to ~2K+ tokens.

Walk through the guide JSON sequentially, accumulate blocks until hitting
2K tokens, then cut at the FIRST clean block boundary after 2K. This
maximizes the cached portion of each labeling call (only the ~2K chunk
text varies; the ~1.7K system prompt is cached).

Minimum chunk size is 2K tokens. No small trailing chunks; they get merged
into the previous chunk.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

TARGET_TOKENS = 2000  # walk to 2K, then cut at next clean boundary
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def micro_chunk_guide(guide_json: dict) -> list[dict]:
    """Split guide into sequential chunks of ~2K+ tokens.

    Algorithm: walk blocks sequentially, accumulate text. Once accumulated
    tokens >= TARGET_TOKENS, flush at the current block boundary (first
    clean spot after 2K). Oversized single blocks become their own chunk.
    Trailing chunks smaller than TARGET_TOKENS merge into the previous one.
    """
    sections = guide_json.get("sections", {})

    # Flatten all blocks into a sequential stream with section context
    stream = []  # list of (section_id, section_title, block_text, block_obj)
    for sec_id, sec_data in sections.items():
        title = sec_data.get("title", sec_id)
        blocks = sec_data.get("blocks", [])
        if not blocks:
            raw = (sec_data.get("raw_text", "") or "").strip()
            if raw:
                stream.append((sec_id, title, raw, {"text": raw}))
        else:
            for block in blocks:
                bt = (block.get("text", "") or "").strip()
                if bt:
                    stream.append((sec_id, title, bt, block))

    if not stream:
        return []

    # Walk the stream, accumulate until >= TARGET_TOKENS then cut
    chunks = []
    chunk_index = 0
    current_text = ""
    current_blocks = []
    current_sections = set()
    current_tokens = 0

    for sec_id, sec_title, block_text, block_obj in stream:
        block_tokens = _estimate_tokens(block_text)

        # Add section header if entering a new section
        if (sec_id, sec_title) not in current_sections and current_text:
            current_text += f"\n[{sec_title}]\n"

        current_sections.add((sec_id, sec_title))
        current_text += block_text + "\n"
        current_blocks.append(block_obj)
        current_tokens += block_tokens

        # Once we've passed TARGET_TOKENS, cut at this block boundary
        if current_tokens >= TARGET_TOKENS:
            chunks.append(_make_chunk(
                chunk_index, current_sections, current_text, current_blocks
            ))
            chunk_index += 1
            current_text = ""
            current_blocks = []
            current_sections = set()
            current_tokens = 0

    # Flush final buffer
    if current_text.strip():
        chunks.append(_make_chunk(
            chunk_index, current_sections, current_text, current_blocks
        ))

    # Enforce minimum: merge any trailing small chunk into the previous one
    merged_chunks = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk["text"])
        if tokens < TARGET_TOKENS and merged_chunks:
            prev = merged_chunks[-1]
            prev["text"] = prev["text"] + "\n\n" + chunk["text"]
            prev["blocks"].extend(chunk["blocks"])
            prev["sections_in_chunk"] = prev.get("sections_in_chunk", 1) + chunk.get("sections_in_chunk", 1)
            if "+" not in prev["section_id"]:
                prev["section_id"] = prev["section_id"] + "+" + chunk["section_id"]
        else:
            merged_chunks.append(chunk)

    # Re-index after merge
    chunks = merged_chunks
    for i, c in enumerate(chunks):
        c["chunk_index"] = i

    # Compute difficulty for each chunk
    from backend.validators.test_suite import classify_chunk_difficulty
    for chunk in chunks:
        fake = {"sections": {chunk["section_id"]: {
            "title": chunk["section_title"], "raw_text": chunk["text"],
            "blocks": chunk["blocks"],
        }}}
        diff = classify_chunk_difficulty(fake)
        chunk["difficulty"] = diff["difficulty"]
        chunk["difficulty_score"] = diff["score"]

    # Stats
    sizes = [_estimate_tokens(c["text"]) for c in chunks]
    logger.info(
        "Gen7 chunker: %d blocks -> %d chunks (min=%d avg=%d max=%d tokens)",
        len(stream), len(chunks),
        min(sizes) if sizes else 0,
        sum(sizes) // len(sizes) if sizes else 0,
        max(sizes) if sizes else 0,
    )

    return chunks


def _make_chunk(index: int, sections: set, text: str, blocks: list) -> dict:
    """Build a chunk dict from accumulated data."""
    sec_list = list(sections)
    if len(sec_list) == 1:
        sec_id, sec_title = sec_list[0]
    else:
        sec_id = "+".join(s[0] for s in sec_list[:3])
        sec_title = " | ".join(s[1] for s in sec_list[:3])

    return {
        "chunk_index": index,
        "section_id": sec_id,
        "section_title": sec_title,
        "text": text.strip(),
        "blocks": blocks,
        "sections_in_chunk": len(sec_list),
    }
