"""Gen 5 Step 2: Cross-chunk deduplication.

Pure Python. No LLM calls. Merges items from all chunks into unique sets.
"""

import logging
import re

logger = logging.getLogger(__name__)


def _normalize_quote(quote: str) -> str:
    """Normalize a quote for dedup comparison."""
    return " ".join(quote.lower().split())[:80]


def _normalize_id(item_id: str) -> str:
    """Normalize an ID for dedup."""
    return re.sub(r"[^a-z0-9_]", "_", item_id.lower().strip())


def dedup_items(raw_items: list[dict]) -> list[dict]:
    """Deduplicate items by normalized quote + normalized ID.

    Keeps the first occurrence. Tracks which chunks contributed.
    """
    seen_quotes: set[str] = set()
    seen_ids: set[str] = set()
    unique: list[dict] = []

    for item in raw_items:
        quote = item.get("quote", "")
        item_id = item.get("id", "")

        norm_quote = _normalize_quote(quote)
        norm_id = _normalize_id(item_id)

        # Skip if we've seen this exact quote or this exact ID
        if norm_quote and norm_quote in seen_quotes:
            # Update provenance: add chunk to existing item
            for u in unique:
                if _normalize_quote(u.get("quote", "")) == norm_quote:
                    chunks = u.get("_chunks", [u.get("_chunk", -1)])
                    new_chunk = item.get("_chunk", -1)
                    if new_chunk not in chunks:
                        chunks.append(new_chunk)
                    u["_chunks"] = chunks
                    break
            continue

        if norm_id and norm_id in seen_ids:
            for u in unique:
                if _normalize_id(u.get("id", "")) == norm_id:
                    chunks = u.get("_chunks", [u.get("_chunk", -1)])
                    new_chunk = item.get("_chunk", -1)
                    if new_chunk not in chunks:
                        chunks.append(new_chunk)
                    u["_chunks"] = chunks
                    break
            continue

        # New unique item
        if norm_quote:
            seen_quotes.add(norm_quote)
        if norm_id:
            seen_ids.add(norm_id)

        item["_chunks"] = [item.get("_chunk", -1)]
        unique.append(item)

    return unique


def dedup_all(raw_items: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Dedup all artifact types. Returns {type: [unique_items]}."""
    result = {}
    for artifact_type, items in raw_items.items():
        unique = dedup_items(items)
        logger.info(
            "Gen5 dedup: %s %d -> %d unique (%.0f%% reduction)",
            artifact_type, len(items), len(unique),
            (1 - len(unique) / len(items)) * 100 if items else 0,
        )
        result[artifact_type] = unique
    return result
