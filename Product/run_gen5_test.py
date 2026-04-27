"""Standalone Gen 5 test run on the 6-page WHO test guide.

Runs the full pipeline: extract -> dual dedup -> compile -> save.
Downloads: 7 artifact JSONs + chunk_difficulty.json + test_suite.json
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gen5_test")

# Load env
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

if not ANTHROPIC_KEY:
    print("ERROR: Set ANTHROPIC_KEY in .env")
    sys.exit(1)

logger.info("Keys: Anthropic=%s OpenAI=%s",
            "YES" if ANTHROPIC_KEY else "NO",
            "YES" if OPENAI_KEY else "NO")

# Load the test guide
GUIDE_PATH = Path("sample_data/who_chw_2012_test_decomposed.json")
if not GUIDE_PATH.exists():
    print(f"ERROR: Guide not found at {GUIDE_PATH}")
    sys.exit(1)

guide_json = json.loads(GUIDE_PATH.read_text(encoding="utf-8"))
logger.info("Loaded guide: %d sections, %d pages",
            len(guide_json.get("sections", {})),
            len(guide_json.get("pages", {})))

# Output directory
OUTPUT_DIR = Path("backend/output/gen5_test_run")

# Load naming codebook from system_prompt
try:
    from backend.system_prompt import NAMING_CODEBOOK
except ImportError:
    NAMING_CODEBOOK = ""


async def progress_callback(event):
    logger.info("Progress: %s", json.dumps(event, default=str))


async def main():
    from backend.gen5.pipeline import run_gen5_extraction

    logger.info("Starting Gen 5 test run on %s", GUIDE_PATH.name)

    result = await run_gen5_extraction(
        guide_json=guide_json,
        anthropic_key=ANTHROPIC_KEY,
        openai_key=OPENAI_KEY if OPENAI_KEY else None,
        naming_codebook=NAMING_CODEBOOK,
        output_dir=OUTPUT_DIR,
        on_progress=progress_callback,
    )

    logger.info("=" * 60)
    logger.info("GEN 5 TEST RUN COMPLETE")
    logger.info("=" * 60)
    logger.info("Status: %s", result["status"])
    logger.info("Elapsed: %.1fs", result["elapsed"])
    logger.info("Raw item counts: %s", result["raw_counts"])
    logger.info("Deduped counts: %s", result["deduped_counts"])
    logger.info("Chunk difficulty: %s",
                {d["difficulty"]: 1 for d in result.get("chunk_difficulty", [])})

    # Summary per artifact
    for at, artifact in result["artifacts"].items():
        if isinstance(artifact, list):
            logger.info("  %s: %d entries", at, len(artifact))
        elif isinstance(artifact, dict):
            logger.info("  %s: %d keys", at, len(artifact))
        else:
            logger.info("  %s: %s", at, type(artifact).__name__)

    logger.info("\nOutputs saved to: %s", OUTPUT_DIR)
    logger.info("Files:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            size = f.stat().st_size
            logger.info("  %s (%d bytes)", f.name, size)


if __name__ == "__main__":
    asyncio.run(main())
