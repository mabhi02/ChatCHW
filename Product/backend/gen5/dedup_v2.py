"""Gen 5 dedup v2: Dual-path deduplication with Opus arbitration.

Branch A: Python deterministic dedup -> Opus verification
Branch B: Sonnet batched dedup (items grouped by ~10K token windows)
Final: Opus merges both branches into the definitive dedup list.
"""

import asyncio
import json
import logging
import math
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Branch A: Deterministic Python dedup
# ---------------------------------------------------------------------------

def _normalize_quote(quote: str) -> str:
    return " ".join(quote.lower().split())[:80]

def _normalize_id(item_id: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", item_id.lower().strip())

def python_dedup(raw_items: list[dict]) -> list[dict]:
    """Deterministic dedup by normalized quote + ID."""
    seen_quotes: set[str] = set()
    seen_ids: set[str] = set()
    unique: list[dict] = []

    for item in raw_items:
        quote = item.get("quote", "")
        item_id = item.get("id", "")
        norm_quote = _normalize_quote(quote)
        norm_id = _normalize_id(item_id)

        if norm_quote and norm_quote in seen_quotes:
            # Track provenance
            for u in unique:
                if _normalize_quote(u.get("quote", "")) == norm_quote:
                    chunks = u.setdefault("_chunks", [])
                    c = item.get("_chunk", -1)
                    if c not in chunks:
                        chunks.append(c)
                    break
            continue

        if norm_id and norm_id in seen_ids:
            for u in unique:
                if _normalize_id(u.get("id", "")) == norm_id:
                    chunks = u.setdefault("_chunks", [])
                    c = item.get("_chunk", -1)
                    if c not in chunks:
                        chunks.append(c)
                    break
            continue

        if norm_quote:
            seen_quotes.add(norm_quote)
        if norm_id:
            seen_ids.add(norm_id)
        item.setdefault("_chunks", [item.get("_chunk", -1)])
        unique.append(item)

    return unique


# ---------------------------------------------------------------------------
# Branch A verification: Opus checks deterministic dedup
# ---------------------------------------------------------------------------

async def opus_verify_dedup(
    deduped: list[dict],
    artifact_type: str,
    api_key: str,
) -> dict:
    """Opus reviews the deterministic dedup for errors.

    Returns {verified: list[dict], issues: list[str]}
    """
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)

    items_json = json.dumps(
        [{"id": i.get("id", "?"), "quote": i.get("quote", "")[:120], "chunks": i.get("_chunks", [])}
         for i in deduped[:80]],
        indent=1,
    )

    prompt = (
        f"Review this deduplicated {artifact_type} list for errors.\n\n"
        f"ITEMS ({len(deduped)}):\n{items_json}\n\n"
        f"Check for:\n"
        f"1. FALSE MERGES: items that look like duplicates but are actually different clinical concepts\n"
        f"2. MISSED MERGES: items that should be the same but have different IDs/quotes\n"
        f"3. NAMING ISSUES: IDs that don't follow conventions\n\n"
        f"Return JSON: {{\"issues\": [\"description of each issue\"], "
        f"\"merge_suggestions\": [{{\"keep\": \"id_to_keep\", \"remove\": \"id_to_remove\", \"reason\": \"why\"}}], "
        f"\"split_suggestions\": [{{\"id\": \"id_to_split\", \"reason\": \"why these are different\"}}]}}"
    )

    try:
        r = await client.messages.create(
            model="claude-opus-4-6", max_tokens=4000, temperature=0.0,
            system="Review dedup quality. Return ONLY JSON. No markdown.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
        issues = result.get("issues", [])
        merges = result.get("merge_suggestions", [])
        splits = result.get("split_suggestions", [])

        logger.info("Gen5 dedup verify: %s — %d issues, %d merge suggestions, %d splits",
                     artifact_type, len(issues), len(merges), len(splits))

        # Apply merge suggestions
        remove_ids = {m["remove"].lower() for m in merges if "remove" in m}
        verified = [i for i in deduped if _normalize_id(i.get("id", "")) not in remove_ids]

        return {
            "verified": verified,
            "issues": issues,
            "merges_applied": len(remove_ids),
            "splits_flagged": len(splits),
        }
    except Exception as exc:
        logger.warning("Gen5 dedup verify failed for %s: %s", artifact_type, exc)
        return {"verified": deduped, "issues": [], "merges_applied": 0, "splits_flagged": 0}


# ---------------------------------------------------------------------------
# Branch B: Sonnet batched dedup
# ---------------------------------------------------------------------------

def _estimate_tokens(items: list[dict]) -> int:
    """Rough token estimate for a list of items."""
    return len(json.dumps(items)) // 4


def _batch_by_token_limit(items: list[dict], max_tokens: int = 10000) -> list[list[dict]]:
    """Split items into batches that fit under the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0

    for item in items:
        item_tokens = _estimate_tokens([item])
        if current_tokens + item_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(item)
        current_tokens += item_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


async def sonnet_dedup_batch(
    items: list[dict],
    batch_idx: int,
    artifact_type: str,
    api_key: str,
) -> list[dict]:
    """Sonnet deduplicates one batch of items."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)

    items_json = json.dumps(
        [{"id": i.get("id", "?"), "quote": i.get("quote", "")[:150],
          "chunk": i.get("_chunk", -1)}
         for i in items],
        indent=1,
    )

    prompt = (
        f"Deduplicate this list of {artifact_type} items.\n\n"
        f"ITEMS ({len(items)}):\n{items_json}\n\n"
        f"Group items that refer to the SAME clinical concept, even if they have "
        f"different IDs or slightly different quotes. Keep the BEST version of each "
        f"(most complete quote, most descriptive ID).\n\n"
        f"Return a JSON array of unique items. Each item: "
        f'{{\"id\": \"best_id\", \"quote\": \"best_quote\", \"chunks\": [list of chunk indices]}}\n'
        f"Return ONLY the JSON array. No markdown."
    )

    try:
        r = await client.messages.create(
            model="claude-sonnet-4-6", max_tokens=8000, temperature=0.0,
            system="Deduplicate clinical items. Return ONLY a JSON array.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
        if isinstance(result, list):
            logger.info("Gen5 Sonnet dedup batch %d: %d -> %d items",
                        batch_idx, len(items), len(result))
            return [i for i in result if isinstance(i, dict)]
        return items  # fallback
    except Exception as exc:
        logger.warning("Gen5 Sonnet dedup batch %d failed: %s", batch_idx, exc)
        return items  # fallback to raw


async def sonnet_dedup_all(
    raw_items: list[dict],
    artifact_type: str,
    api_key: str,
) -> list[dict]:
    """Run Sonnet dedup across all items in batched windows."""
    batches = _batch_by_token_limit(raw_items, max_tokens=10000)
    logger.info("Gen5 Sonnet dedup: %s — %d items in %d batches",
                artifact_type, len(raw_items), len(batches))

    # Run all batches in parallel
    coros = [
        sonnet_dedup_batch(batch, i, artifact_type, api_key)
        for i, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*coros, return_exceptions=True)

    # Flatten all batch results
    all_deduped = []
    for i, result in enumerate(batch_results):
        if isinstance(result, list):
            all_deduped.extend(result)
        else:
            logger.warning("Sonnet batch %d failed: %s", i, result)
            all_deduped.extend(batches[i])  # fallback

    # Cross-batch dedup (Sonnet deduped within batches, need to dedup across)
    final = python_dedup(all_deduped)
    logger.info("Gen5 Sonnet dedup final: %s — %d after cross-batch dedup", artifact_type, len(final))
    return final


# ---------------------------------------------------------------------------
# Final: Opus arbitration between Branch A and Branch B
# ---------------------------------------------------------------------------

async def opus_arbitrate(
    branch_a: list[dict],
    branch_b: list[dict],
    artifact_type: str,
    api_key: str,
) -> list[dict]:
    """Opus creates the FINAL dedup list from both branches."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Summarize both branches
    a_items = json.dumps(
        [{"id": i.get("id", "?"), "quote": i.get("quote", "")[:100]}
         for i in branch_a[:60]],
        indent=1,
    )
    b_items = json.dumps(
        [{"id": i.get("id", "?"), "quote": i.get("quote", "")[:100]}
         for i in branch_b[:60]],
        indent=1,
    )

    prompt = (
        f"Two deduplication passes produced different lists for {artifact_type}.\n\n"
        f"BRANCH A (deterministic + Opus verified, {len(branch_a)} items):\n{a_items}\n\n"
        f"BRANCH B (Sonnet semantic dedup, {len(branch_b)} items):\n{b_items}\n\n"
        f"Create the FINAL definitive list by:\n"
        f"1. Including every item that appears in EITHER branch (union)\n"
        f"2. Merging items that are the same concept with different names\n"
        f"3. Keeping the most descriptive ID and most complete quote for each\n"
        f"4. Removing true duplicates\n\n"
        f"Return a JSON array of the final unique items. Each: "
        f'{{\"id\": \"best_id\", \"quote\": \"best_quote\", \"source\": \"A\"|\"B\"|\"both\"}}\n'
        f"Return ONLY the JSON array."
    )

    try:
        r = await client.messages.create(
            model="claude-opus-4-6", max_tokens=16384, temperature=0.0,
            system="Create the definitive deduplicated list. Return ONLY a JSON array.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
        if isinstance(result, list):
            logger.info("Gen5 Opus arbitration: %s — A=%d B=%d -> final=%d",
                        artifact_type, len(branch_a), len(branch_b), len(result))
            return [i for i in result if isinstance(i, dict)]
        return branch_a  # fallback
    except Exception as exc:
        logger.error("Gen5 Opus arbitration failed for %s: %s", artifact_type, exc)
        return branch_a  # fallback


# ---------------------------------------------------------------------------
# Main: dual-path dedup pipeline
# ---------------------------------------------------------------------------

async def dual_dedup(
    raw_items: dict[str, list[dict]],
    anthropic_key: str,
) -> dict[str, list[dict]]:
    """Run dual-path dedup for all artifact types.

    Branch A: Python dedup -> Opus verify
    Branch B: Sonnet batched dedup
    Final: Opus arbitrates between A and B

    Returns {artifact_type: [final_deduped_items]}
    """
    result = {}

    for artifact_type, items in raw_items.items():
        if not items:
            result[artifact_type] = []
            continue

        logger.info("Gen5 dual dedup: %s — %d raw items", artifact_type, len(items))

        # Branch A: Python + Opus verify (parallel with Branch B)
        python_result = python_dedup(items)

        branch_a_coro = opus_verify_dedup(python_result, artifact_type, anthropic_key)
        branch_b_coro = sonnet_dedup_all(items, artifact_type, anthropic_key)

        # Run both branches in parallel
        a_result, b_result = await asyncio.gather(branch_a_coro, branch_b_coro)

        branch_a = a_result["verified"] if isinstance(a_result, dict) else python_result
        branch_b = b_result if isinstance(b_result, list) else python_result

        logger.info("Gen5 dual dedup: %s — A=%d B=%d, running Opus arbitration",
                    artifact_type, len(branch_a), len(branch_b))

        # Final: Opus arbitrates
        final = await opus_arbitrate(branch_a, branch_b, artifact_type, anthropic_key)
        result[artifact_type] = final

        logger.info("Gen5 dual dedup: %s — FINAL=%d items", artifact_type, len(final))

    return result
