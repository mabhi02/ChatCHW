"""Gen 5 main pipeline: extract -> dedup -> compile -> verify.

Single entry point that runs the full extraction.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from backend.gen5.extractor import extract_all
from backend.gen5.dedup import dedup_all
from backend.gen5.dedup_v2 import dual_dedup
from backend.gen5.compiler import compile_all
from backend.validators.test_suite import classify_chunk_difficulty

logger = logging.getLogger(__name__)


async def run_gen5_extraction(
    guide_json: dict,
    anthropic_key: str,
    openai_key: Optional[str] = None,
    naming_codebook: str = "",
    output_dir: Optional[Path] = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the full Gen 5 extraction pipeline.

    Phase 1: Multi-model parallel extraction (all chunks x all types)
    Step 2: Cross-chunk deduplication (pure Python)
    Phase 2: DAG-ordered schema compilation (Opus calls)

    Returns {
        "artifacts": {type: compiled_artifact},
        "raw_counts": {type: int},
        "deduped_counts": {type: int},
        "elapsed": float,
        "status": "passed" | "failed",
    }
    """
    start = time.time()

    # Pre-step: Compute chunk difficulty
    from backend.validators.phases import chunk_guide_for_catcher
    chunks = chunk_guide_for_catcher(guide_json)
    chunk_difficulty = []
    for i, chunk in enumerate(chunks):
        diff = classify_chunk_difficulty(chunk)
        diff["chunk_index"] = i
        secs = list(chunk.get("sections", {}).keys())
        diff["sections"] = secs[:5]
        chunk_difficulty.append(diff)
        logger.info("Gen5: chunk %d difficulty=%s score=%.3f passes=%d",
                    i, diff["difficulty"], diff["score"], diff["recommended_passes"])

    # Phase 1: Extract
    logger.info("Gen5: Phase 1 -- multi-model parallel extraction")
    if on_progress:
        await on_progress({"phase": "extract", "status": "starting"})

    raw_items = await extract_all(guide_json, anthropic_key, openai_key, on_progress)

    raw_counts = {at: len(items) for at, items in raw_items.items()}
    total_raw = sum(raw_counts.values())
    logger.info("Gen5: Phase 1 complete -- %d total raw items", total_raw)

    # Step 2: Dual-path dedup (Python + Sonnet -> Opus arbitration)
    logger.info("Gen5: Step 2 -- dual-path deduplication")
    deduped = await dual_dedup(raw_items, anthropic_key)

    deduped_counts = {at: len(items) for at, items in deduped.items()}
    total_deduped = sum(deduped_counts.values())
    logger.info("Gen5: Step 2 complete -- %d -> %d unique items (%.0f%% reduction)",
                total_raw, total_deduped,
                (1 - total_deduped / total_raw) * 100 if total_raw else 0)

    # Phase 2: Compile (DAG-ordered)
    logger.info("Gen5: Phase 2 — DAG-ordered schema compilation")
    if on_progress:
        await on_progress({"phase": "compile", "status": "starting"})

    artifacts = await compile_all(deduped, anthropic_key, naming_codebook, on_progress)

    elapsed = time.time() - start

    # Save artifacts if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for at, artifact in artifacts.items():
            path = output_dir / f"{at}.json"
            path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
            logger.info("Gen5: saved %s to %s", at, path)

        # Save chunk difficulty artifact
        diff_path = output_dir / "chunk_difficulty.json"
        diff_path.write_text(json.dumps({
            "total_chunks": len(chunk_difficulty),
            "distribution": {
                level: sum(1 for d in chunk_difficulty if d["difficulty"] == level)
                for level in ["trivial", "easy", "medium", "hard", "extreme"]
            },
            "chunks": chunk_difficulty,
        }, indent=2), encoding="utf-8")
        logger.info("Gen5: saved chunk difficulty to %s", diff_path)

        # Save the frozen test suite (deduped items)
        test_suite_path = output_dir / "test_suite.json"
        test_suite_path.write_text(json.dumps({
            "raw_counts": raw_counts,
            "deduped_counts": deduped_counts,
            "items": {at: items for at, items in deduped.items()},
        }, indent=2, default=str), encoding="utf-8")
        logger.info("Gen5: saved test suite to %s", test_suite_path)

    # Assemble clinical_logic (same shape as FINAL_VAR output)
    clinical_logic = {at: artifacts.get(at) for at in [
        "supply_list", "variables", "predicates", "modules",
        "router", "integrative", "phrase_bank",
    ]}

    status = "passed" if all(artifacts.get(at) for at in clinical_logic) else "failed"

    logger.info(
        "Gen5: complete in %.0fs — status=%s raw=%d deduped=%d artifacts=%d",
        elapsed, status, total_raw, total_deduped, len(artifacts),
    )

    return {
        "clinical_logic": clinical_logic,
        "artifacts": artifacts,
        "chunk_difficulty": chunk_difficulty,
        "raw_counts": raw_counts,
        "deduped_counts": deduped_counts,
        "elapsed": round(elapsed, 1),
        "status": status,
    }
