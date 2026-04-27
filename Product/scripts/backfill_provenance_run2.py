"""One-shot backfill: emit provenance sidecars for run_8970cf65.

Walks every JSON under `backend/output/run_8970cf65/`, computes SHAs,
writes `.provenance.json` siblings, then writes a run-level `manifest.json`
rolled up into a container hash. The source-manual SHA is pulled from
Neon (`SourceGuide`) so the manifest is anchored in the real PDF bytes.

Usage (from repo root, with backend/.venv active):
    python -m scripts.backfill_provenance_run2

Idempotent: re-running overwrites sidecars with fresh content + the same
SHAs. The manifest gets rewritten each run; container_sha256 is stable
modulo the artifact bytes (and any timestamp drift in the sidecars, which
are not part of the container hash).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_DIR = REPO_ROOT / "backend" / "output" / "run_8970cf65"
GEN7_GIT_SHA = "9000e86"
TARGET_SHA_PREFIX = "1d2708940727"  # WHO 2012 guide per user


async def _fetch_source_manual():
    """Pull the real SHA for the WHO 2012 guide from Neon."""
    from prisma import Prisma
    db = Prisma()
    await db.connect()
    try:
        guides = await db.sourceguide.find_many(order={"ingestedAt": "desc"}, take=30)
        target = None
        for g in guides:
            if g.sha256 and g.sha256.startswith(TARGET_SHA_PREFIX):
                target = g
                break
        if target is None:
            # Fallback: pick the most recent guide whose manualName contains WHO / 2012
            for g in guides:
                name = (g.manualName or g.filename or "").lower()
                if "who" in name or "2012" in name:
                    target = g
                    break
        return target
    finally:
        await db.disconnect()


def backfill(source_manual) -> dict:
    from backend.provenance.schema import SourceManual
    from backend.provenance.sidecar import write_with_provenance, compute_sha256
    from backend.provenance.container import write_manifest

    sm = SourceManual(
        name=source_manual.manualName or source_manual.filename or "Caring for the sick child in the community",
        publisher="WHO, Geneva, 2012",
        pdf_sha256=source_manual.sha256,
        page_count=source_manual.pageCount,
        neon_guide_id=source_manual.id,
    )

    artifact_shas: dict[str, str] = {}
    for path in sorted(RUN_DIR.rglob("*.json")):
        name = path.name
        if name.endswith(".provenance.json") or name.endswith(".verification.json"):
            continue
        if name == "manifest.json":
            continue
        try:
            data = json.loads(path.read_bytes())
        except json.JSONDecodeError:
            continue
        sha = write_with_provenance(
            path, data,
            artifact_kind=path.stem,
            run_id=RUN_DIR.name,
            pipeline="gen7",
            labeler="opus",
            pipeline_git_sha=GEN7_GIT_SHA,
            model="claude-opus-4-6",
            source_manual=sm,
            parents=[],
        )
        key = str(path.relative_to(RUN_DIR)).replace("\\", "/")
        artifact_shas[key] = sha

    # Also hash non-JSON deliverable artifacts (dmn, xlsx, png, md, csv) so the
    # manifest covers the full reviewer-facing set.
    for pattern in ("*.dmn", "*.xlsx", "*.png", "*.md", "*.csv", "*.txt"):
        for path in sorted(RUN_DIR.rglob(pattern)):
            if path.name == "manifest.json":
                continue
            sha = compute_sha256(path.read_bytes())
            key = str(path.relative_to(RUN_DIR)).replace("\\", "/")
            artifact_shas[key] = sha

    container_sha = write_manifest(
        RUN_DIR,
        artifact_shas=artifact_shas,
        source_pdf_sha=sm.pdf_sha256,
        git_sha=GEN7_GIT_SHA,
        verifier_model="none",
        pipeline="gen7",
        labeler="opus",
        run_id=RUN_DIR.name,
    )
    return {
        "artifact_count": len(artifact_shas),
        "container_sha": container_sha,
        "source_pdf_sha": sm.pdf_sha256,
        "source_manual": sm.name,
    }


async def main() -> int:
    load_dotenv()
    if not RUN_DIR.exists():
        print(f"run dir missing: {RUN_DIR}")
        return 2
    source = await _fetch_source_manual()
    if source is None:
        print("Could not find a matching SourceGuide in Neon; aborting.")
        return 3
    result = backfill(source)
    print(f"Backfilled {result['artifact_count']} artifacts in {RUN_DIR}")
    print(f"  source PDF sha256: {result['source_pdf_sha']}")
    print(f"  container sha256:  {result['container_sha']}")
    print(f"  source manual:     {result['source_manual']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
