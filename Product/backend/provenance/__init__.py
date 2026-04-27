"""Provenance + verification sidecar emission.

Each artifact write goes through `write_with_provenance` which:
 1. canonically serializes the artifact
 2. writes it to disk
 3. computes content_sha256
 4. writes a `.provenance.json` sidecar alongside

The run-level `manifest.json` is a container hash over every sidecar SHA,
the source PDF SHA, the pipeline git SHA, and the verifier model id.
"""

from backend.provenance.sidecar import (
    write_with_provenance,
    compute_sha256,
    serialize_artifact,
    verify_sidecar,
)
from backend.provenance.container import (
    compute_container_hash,
    write_manifest,
    get_git_sha,
)
from backend.provenance.schema import (
    ProvenanceBlock,
    VerificationBlock,
    SourceManual,
    ParentRef,
    Divergence,
)

__all__ = [
    "write_with_provenance",
    "compute_sha256",
    "serialize_artifact",
    "verify_sidecar",
    "compute_container_hash",
    "write_manifest",
    "get_git_sha",
    "ProvenanceBlock",
    "VerificationBlock",
    "SourceManual",
    "ParentRef",
    "Divergence",
]
