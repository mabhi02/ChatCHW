"""Run-level container hash.

`compute_container_hash` rolls every artifact SHA + the source PDF SHA +
the pipeline git SHA + the verifier model identifier into a single
canonical hash. If two runs produce byte-identical artifact sets against
the same source PDF using the same git SHA, they get the same container
hash -- that's the identification claim the research design depends on.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def get_git_sha(repo_root: Path | None = None) -> str:
    """Return the current HEAD short SHA, or `"unknown"` if not in a repo."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_container_hash(
    artifact_shas: dict[str, str],
    source_pdf_sha: str,
    git_sha: str,
    verifier_model: str,
) -> str:
    """Deterministic SHA-256 rollup.

    `artifact_shas` is sorted alphabetically before hashing so the container
    hash is stable regardless of the order in which artifacts were emitted.
    """
    payload = {
        "artifacts": dict(sorted(artifact_shas.items())),
        "source_pdf_sha256": source_pdf_sha,
        "pipeline_git_sha": git_sha,
        "verifier_model": verifier_model,
    }
    canonical = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_manifest(
    output_dir: Path,
    *,
    artifact_shas: dict[str, str],
    source_pdf_sha: str,
    git_sha: str,
    verifier_model: str,
    pipeline: str,
    labeler: str,
    run_id: str,
) -> str:
    """Write `output_dir/manifest.json` and return the container SHA."""
    container_sha = compute_container_hash(
        artifact_shas=artifact_shas,
        source_pdf_sha=source_pdf_sha,
        git_sha=git_sha,
        verifier_model=verifier_model,
    )
    manifest = {
        "container_sha256": container_sha,
        "run_id": run_id,
        "pipeline": pipeline,
        "labeler": labeler,
        "pipeline_git_sha": git_sha,
        "source_pdf_sha256": source_pdf_sha,
        "verifier_model": verifier_model,
        "artifact_shas": dict(sorted(artifact_shas.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return container_sha
