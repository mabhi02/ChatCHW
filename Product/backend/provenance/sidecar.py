"""Artifact write with provenance sidecar.

`write_with_provenance(path, data, ...)` replaces `path.write_text(json.dumps(data))`.
It canonically serializes `data`, writes it to `path`, computes a SHA-256 over
the serialized bytes, then writes a sibling `<path>.provenance.json` file
containing a `ProvenanceBlock`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.provenance.schema import ProvenanceBlock, ParentRef, SourceManual


def serialize_artifact(data: Any) -> bytes:
    """Canonical JSON serialization used for hashing AND on-disk content.

    `sort_keys=True` is critical: the container hash must be stable across
    runs regardless of insertion order.
    """
    return json.dumps(data, indent=2, sort_keys=True, default=str).encode("utf-8")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_with_provenance(
    path: Path,
    data: Any,
    *,
    artifact_kind: str,
    run_id: str,
    pipeline: str,
    labeler: str,
    pipeline_git_sha: str,
    model: str,
    source_manual: SourceManual,
    parents: list[ParentRef] | None = None,
) -> str:
    """Write `data` as canonical JSON to `path` and emit a provenance sidecar.

    Returns the content SHA-256 so the caller can collect them into the
    container hash without re-reading each file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_artifact(data)
    sha = compute_sha256(serialized)
    path.write_bytes(serialized)

    prov = ProvenanceBlock(
        artifact_kind=artifact_kind,
        content_sha256=sha,
        run_id=run_id,
        pipeline=pipeline,
        labeler=labeler,
        created_at=datetime.now(timezone.utc).isoformat(),
        pipeline_git_sha=pipeline_git_sha,
        model=model,
        source_manual=source_manual,
        parents=parents or [],
    )
    sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
    sidecar_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    return sha


def write_binary_with_provenance(
    path: Path,
    data: bytes,
    *,
    artifact_kind: str,
    run_id: str,
    pipeline: str,
    labeler: str,
    pipeline_git_sha: str,
    model: str,
    source_manual: SourceManual,
    parents: list[ParentRef] | None = None,
) -> str:
    """Same as `write_with_provenance` but for binary artifacts (xlsx, png, dmn)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = compute_sha256(data)
    path.write_bytes(data)

    prov = ProvenanceBlock(
        artifact_kind=artifact_kind,
        content_sha256=sha,
        run_id=run_id,
        pipeline=pipeline,
        labeler=labeler,
        created_at=datetime.now(timezone.utc).isoformat(),
        pipeline_git_sha=pipeline_git_sha,
        model=model,
        source_manual=source_manual,
        parents=parents or [],
    )
    sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
    sidecar_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    return sha


def verify_sidecar(path: Path) -> bool:
    """Return True iff the artifact's bytes match the SHA recorded in its sidecar."""
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar.exists() or not path.exists():
        return False
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    return compute_sha256(path.read_bytes()) == prov.get("content_sha256")
