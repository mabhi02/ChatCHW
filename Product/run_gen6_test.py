"""Standalone Gen 6 test on the 6-page WHO test guide."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gen6_test")

from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

if not ANTHROPIC_KEY or not OPENAI_KEY:
    print("ERROR: Need both ANTHROPIC_KEY and OPENAI_API_KEY in .env")
    sys.exit(1)

guide_json = json.loads(
    Path("sample_data/who_chw_2012_test_decomposed.json").read_text(encoding="utf-8")
)

try:
    from backend.system_prompt import NAMING_CODEBOOK
except ImportError:
    NAMING_CODEBOOK = ""

OUTPUT_DIR = Path("backend/output/gen6_test_run")


async def progress(event):
    logger.info("Progress: %s", json.dumps(event, default=str))


async def main():
    from backend.gen6.pipeline import run_gen6_extraction

    logger.info("Starting Gen 6 test")
    result = await run_gen6_extraction(
        guide_json=guide_json,
        anthropic_key=ANTHROPIC_KEY,
        openai_key=OPENAI_KEY,
        naming_codebook=NAMING_CODEBOOK,
        output_dir=OUTPUT_DIR,
        on_progress=progress,
    )

    logger.info("=" * 60)
    logger.info("GEN 6 COMPLETE")
    logger.info("=" * 60)
    logger.info("Status: %s", result["status"])
    logger.info("Stats: %s", result["stats"])

    if result["clinical_logic"]:
        for at, artifact in result["clinical_logic"].items():
            if isinstance(artifact, list):
                logger.info("  %s: %d entries", at, len(artifact))
            elif isinstance(artifact, dict):
                logger.info("  %s: %d keys", at, len(artifact))

    logger.info("Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(main())
