"""Pydantic schemas for provenance + verification sidecars.

Two sidecar files are emitted next to every artifact:
  * `<artifact>.provenance.json`  -> ProvenanceBlock (who/when/what/hash)
  * `<artifact>.verification.json` -> VerificationBlock (agree/divergences)

Keeping them separate lets Tier 3 verification run as a post-process
that never mutates the original provenance record.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class ParentRef(BaseModel):
    """Identifies an upstream artifact this one was derived from."""
    kind: str
    content_sha256: str


class SourceManual(BaseModel):
    """The source PDF/guide this run was extracted from."""
    name: str
    publisher: str
    pdf_sha256: str
    page_count: int
    neon_guide_id: str | None = None
    # Publication year (4-digit). Threaded into the Stage 3 prompt so
    # predicate `provenance` citations cite the actual year instead of
    # whatever year Opus guesses from context.
    publication_year: str | None = None


class ProvenanceBlock(BaseModel):
    artifact_kind: str
    content_sha256: str
    run_id: str
    pipeline: str                         # "gen7" | "gen8" | "gen8.5"
    labeler: str                          # "opus" | "sonnet7way"
    created_at: str                       # ISO 8601 UTC
    pipeline_git_sha: str
    model: str                            # generator model identifier
    source_manual: SourceManual
    parents: list[ParentRef] = Field(default_factory=list)
    schema_version: int = 1
    status: Literal["unverified", "verified", "verification_failed"] = "unverified"


class Divergence(BaseModel):
    type: str
    severity: Literal["info", "warn", "error"] = "warn"
    detail: str
    evidence: dict = Field(default_factory=dict)


class VerificationBlock(BaseModel):
    artifact_kind: str
    artifact_content_sha256: str
    verifier_model: str
    verifier_independence: Literal["different-family", "same-family-fallback"]
    verifier_run_at: str
    agree: bool
    divergences: list[Divergence] = Field(default_factory=list)
    schema_version: int = 1
