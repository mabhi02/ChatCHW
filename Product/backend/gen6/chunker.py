"""Gen 6 Phase 0: Sequential chunking to ~1-1.5K tokens.

Simple algorithm: walk through the guide JSON sequentially, accumulate
blocks until hitting ~1.2K tokens, cut at the nearest clean boundary
(end of block), start next chunk. Every chunk gets the section header
for context.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

TARGET_TOKENS = 1200  # target per chunk
MAX_TOKENS = 1500     # hard cap
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def micro_chunk_guide(guide_json: dict) -> list[dict]:
    """Split guide into sequential chunks of ~1-1.5K tokens.

    Walks through all sections and blocks in order. Accumulates text
    until hitting TARGET_TOKENS, then cuts at the block boundary.
    Each chunk carries its section header(s) for context.
    """
    sections = guide_json.get("sections", {})

    # Flatten all blocks into a sequential stream with section context
    stream = []  # list of (section_id, section_title, block_text, block_obj)
    for sec_id, sec_data in sections.items():
        title = sec_data.get("title", sec_id)
        blocks = sec_data.get("blocks", [])
        if not blocks:
            # Section has raw_text but no blocks — treat as one block
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

    # Walk the stream, fill chunks to target size
    chunks = []
    chunk_index = 0
    current_text = ""
    current_blocks = []
    current_sections = set()
    current_tokens = 0

    for sec_id, sec_title, block_text, block_obj in stream:
        block_tokens = _estimate_tokens(block_text)

        # If this single block exceeds MAX, it becomes its own chunk
        if block_tokens > MAX_TOKENS:
            # Flush current buffer first
            if current_text.strip():
                chunks.append(_make_chunk(
                    chunk_index, current_sections, current_text, current_blocks
                ))
                chunk_index += 1
                current_text = ""
                current_blocks = []
                current_sections = set()
                current_tokens = 0

            # Oversized block as its own chunk
            header = f"[Section: {sec_title}]\n"
            chunks.append(_make_chunk(
                chunk_index, {(sec_id, sec_title)}, header + block_text, [block_obj]
            ))
            chunk_index += 1
            continue

        # Would adding this block exceed target?
        if current_tokens + block_tokens > TARGET_TOKENS and current_text.strip():
            # Flush
            chunks.append(_make_chunk(
                chunk_index, current_sections, current_text, current_blocks
            ))
            chunk_index += 1
            current_text = ""
            current_blocks = []
            current_sections = set()
            current_tokens = 0

        # Add section header if entering a new section
        if (sec_id, sec_title) not in current_sections and current_text:
            current_text += f"\n[{sec_title}]\n"

        current_sections.add((sec_id, sec_title))
        current_text += block_text + "\n"
        current_blocks.append(block_obj)
        current_tokens += block_tokens

    # Flush final buffer
    if current_text.strip():
        chunks.append(_make_chunk(
            chunk_index, current_sections, current_text, current_blocks
        ))

    # Enforce minimum: merge any trailing small chunk into the previous one
    MIN_TOKENS = 800
    merged_chunks = []
    for chunk in chunks:
        tokens = _estimate_tokens(chunk["text"])
        if tokens < MIN_TOKENS and merged_chunks:
            # Merge into previous chunk
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
        "Gen6 chunker: %d blocks -> %d chunks (min=%d avg=%d max=%d tokens)",
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
