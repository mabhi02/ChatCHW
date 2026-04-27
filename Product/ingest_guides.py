"""Ingest both test and full WHO PDFs, cache in Neon.

Run: python ingest_guides.py
Requires: .env with UNSTRUCTURED_API_KEY, OPENAI_API_KEY, NEON_DB
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from backend.ingestion.config import load_ingestion_config
from backend.ingestion.pipeline import IngestionPipeline
from backend.db import get_db, disconnect_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def ingest_pdf(pipeline: IngestionPipeline, pdf_path: str, label: str) -> dict:
    """Ingest a single PDF and return the source guide record."""
    path = Path(pdf_path)
    if not path.exists():
        logger.error(f"PDF not found: {path}")
        return {}

    pdf_bytes = path.read_bytes()
    filename = path.name
    logger.info(f"[{label}] Starting ingestion: {filename} ({len(pdf_bytes):,} bytes)")

    t0 = time.time()
    job_id = f"cli-ingest-{label}-{int(t0)}"

    result = await pipeline.run(
        job_id=job_id,
        pdf_bytes=pdf_bytes,
        filename=filename,
        manual_name=filename.replace(".pdf", ""),
        check_dupes=True,
    )

    elapsed = time.time() - t0
    guide_id = result.get("guide_id", "unknown")
    status = result.get("status", "unknown")
    cached = result.get("cached", False)

    logger.info(
        f"[{label}] Done in {elapsed:.1f}s | status={status} | "
        f"guide_id={guide_id} | cached={cached}"
    )

    if result.get("manifest"):
        m = result["manifest"]
        logger.info(
            f"[{label}] Manifest: pages={m.get('total_pages', '?')}, "
            f"tables={m.get('vision_table_successes', '?')}/{m.get('total_tables', '?')}, "
            f"images={m.get('vision_image_successes', '?')}/{m.get('total_images', '?')}, "
            f"critical={m.get('critical_count', '?')}, warnings={m.get('warning_count', '?')}"
        )

    return result


async def main():
    config = load_ingestion_config()
    pipeline = IngestionPipeline(config)

    # Connect to Neon
    await get_db()

    results = {}

    # 1. Test PDF (6 pages)
    test_path = "WHO_CHW_guide_2012_test.pdf"
    if Path(test_path).exists():
        results["test"] = await ingest_pdf(pipeline, test_path, "test-6pg")
    else:
        logger.warning(f"Test PDF not found at {test_path}")

    # 2. Full WHO PDF (141 pages)
    who_path = "WHO CHW guide 2012.pdf"
    if Path(who_path).exists():
        results["who"] = await ingest_pdf(pipeline, who_path, "who-141pg")
    else:
        logger.warning(f"WHO PDF not found at {who_path}")

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label}: guide_id={r.get('guide_id', 'FAILED')} | cached={r.get('cached', '?')}")
    print("=" * 60)

    await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())
