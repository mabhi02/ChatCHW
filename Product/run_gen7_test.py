"""Standalone Gen 7 test on the 6-page WHO test guide.

Gen 7 = Opus-only pipeline. Single Anthropic key, no OpenAI.
Phase 0: Micro-chunk -> Phase 1: Opus label -> Phase 2: Opus REPL compile.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# rlm package lives at rlm/rlm/, add rlm/ to path so 'from rlm import RLM' works
sys.path.insert(0, str(Path(__file__).parent / "rlm"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gen7_test")

from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_KEY", "")

if not ANTHROPIC_KEY:
    print("ERROR: Need ANTHROPIC_KEY in .env")
    print("Gen 7 is Anthropic-only. No OpenAI key needed.")
    sys.exit(1)

guide_json = json.loads(
    Path("sample_data/who_chw_2012_test_decomposed.json").read_text(encoding="utf-8")
)

try:
    from backend.system_prompt import NAMING_CODEBOOK
except ImportError:
    NAMING_CODEBOOK = ""

OUTPUT_DIR = Path("backend/output/gen7_test_run")


async def progress(event):
    logger.info("Progress: %s", json.dumps(event, default=str))


async def main():
    from backend.gen7.pipeline import run_gen7_extraction

    logger.info("=" * 60)
    logger.info("GEN 7 TEST -- Opus-only pipeline")
    logger.info("=" * 60)
    logger.info("Guide: who_chw_2012_test_decomposed.json (6-page test)")
    logger.info("Model: claude-opus-4-6 (labeling + REPL)")
    logger.info("Provider: Anthropic only")
    logger.info("=" * 60)

    result = await run_gen7_extraction(
        guide_json=guide_json,
        anthropic_key=ANTHROPIC_KEY,
        naming_codebook=NAMING_CODEBOOK,
        output_dir=OUTPUT_DIR,
        on_progress=progress,
    )

    logger.info("=" * 60)
    logger.info("GEN 7 COMPLETE")
    logger.info("=" * 60)
    logger.info("Status: %s", result["status"])
    logger.info("Stats:")
    for k, v in result["stats"].items():
        logger.info("  %s: %s", k, v)

    if result["clinical_logic"]:
        logger.info("Artifacts:")
        for at, artifact in result["clinical_logic"].items():
            if isinstance(artifact, list):
                logger.info("  %s: %d entries", at, len(artifact))
            elif isinstance(artifact, dict):
                if "rows" in artifact:
                    logger.info("  %s: %d rows", at, len(artifact.get("rows", [])))
                elif "rules" in artifact:
                    logger.info("  %s: %d rules", at, len(artifact.get("rules", [])))
                else:
                    logger.info("  %s: %d keys", at, len(artifact))
    else:
        logger.warning("No clinical_logic produced!")

    logger.info("Outputs: %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
