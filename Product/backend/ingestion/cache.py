"""Prisma-backed cache and storage for ingested guides.

Uses the SourceGuide table. Multiple rows may share the same sha256 — the
dupe-check slider decides whether the pipeline reuses an existing row or
creates a new one. This module doesn't know about that decision; the
orchestrator (pipeline.py) is in charge of calling find_latest_by_hash or not.
"""

import hashlib
import json
import logging
from typing import Any

from prisma import Json
from backend.db import get_db

logger = logging.getLogger(__name__)


def compute_content_hash(pdf_bytes: bytes) -> str:
    """SHA-256 hex of the PDF bytes. Content-addressable dedupe key."""
    return hashlib.sha256(pdf_bytes).hexdigest()


def _row_to_dict(row: Any) -> dict:
    """Normalize a Prisma SourceGuide row into a plain dict.

    Prisma Client Python returns pydantic models; model_dump() gives us a
    JSON-safe dict. We fall back to dict() or vars() if the shape ever changes.
    """
    if row is None:
        return {}
    if hasattr(row, "model_dump"):
        try:
            return row.model_dump()
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(row, "dict") and callable(row.dict):
        try:
            return row.dict()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(row, dict):
        return row
    return {k: v for k, v in vars(row).items() if not k.startswith("_")}


async def find_latest_by_hash(sha256: str) -> dict | None:
    """Return the most recent source_guides row matching this hash, or None.

    Returns a dict representation with fields: id, sha256, filename, pageCount,
    pdfBytes, guideJson, ingestionMeta, ingestedAt.

    Uses the (sha256, ingestedAt DESC) composite index so this is a single
    indexed lookup even with many duplicates of the same PDF stored.
    """
    db = await get_db()
    row = await db.sourceguide.find_first(
        where={"sha256": sha256},
        order={"ingestedAt": "desc"},
    )
    if row is None:
        logger.debug("cache miss for sha256=%s", sha256[:12])
        return None
    result = _row_to_dict(row)
    logger.info(
        "cache hit for sha256=%s -> guide_id=%s",
        sha256[:12],
        result.get("id"),
    )
    return result


async def find_by_id(guide_id: str) -> dict | None:
    """Load a specific source_guides row by id.

    Used by the session manager to resolve sourceGuideId -> guide_json when
    starting an extraction run.
    """
    db = await get_db()
    row = await db.sourceguide.find_unique(where={"id": guide_id})
    if row is None:
        logger.debug("no source_guides row for id=%s", guide_id)
        return None
    return _row_to_dict(row)


def _sanitize_json_keys(obj):
    """Recursively sanitize JSON keys so Prisma's GraphQL layer doesn't choke.

    Prisma Client Python serializes JSONB values through a GraphQL query where
    dict keys become GraphQL field names. GraphQL identifiers must match
    /[_A-Za-z][_0-9A-Za-z]*/ -- they cannot start with a digit. Section slugs
    like "1_introduction" from the assembler break this rule.

    Fix: prefix keys that start with a digit AND contain non-digit chars with
    "s_" ("1_introduction" -> "s_1_introduction"). Pure numeric keys like "1",
    "42", "141" (page numbers) are left as-is because Prisma handles those fine
    in JSON values (the issue is only with object-key positions in the GraphQL
    query, and pure numeric page keys are inside a nested JSON object that
    Prisma treats as opaque JSONB).

    Actually, Prisma's GraphQL serializer chokes on ALL keys starting with
    digits, even pure numeric ones. We must prefix them all. But to avoid
    breaking downstream code that does int(key) on page numbers, we use a
    different prefix: "p_" for pure numeric keys and "s_" for alphanumeric.
    This way page number iteration can strip "p_" and parse the rest.

    Simplest approach: just convert to Json() wrapper which bypasses GraphQL
    serialization entirely. The _sanitize_json_keys is no longer needed because
    prisma.Json wraps the value as raw JSON, not as a GraphQL literal.
    """
    # With prisma.Json(), the value is sent as raw JSON, bypassing the
    # GraphQL key-name restrictions entirely. No sanitization needed.
    return obj


async def store_guide(
    sha256: str,
    filename: str,
    manual_name: str | None,
    page_count: int,
    pdf_bytes_len: int,
    guide_json: dict,
    ingestion_meta: dict,
) -> str:
    """Insert a new source_guides row. Returns the new row's id.

    Always inserts a NEW row, never upserts. Multiple rows with the same
    sha256 are allowed (for dupe-off consistency testing). The orchestrator
    decides whether to reuse an existing row via find_latest_by_hash.
    """
    db = await get_db()
    # Sanitize JSON keys: Prisma's GraphQL layer rejects keys starting with digits
    safe_guide_json = _sanitize_json_keys(guide_json)
    safe_ingestion_meta = _sanitize_json_keys(ingestion_meta)
    # Wrap in prisma.Json to ensure proper JSONB serialization
    row = await db.sourceguide.create(
        data={
            "sha256": sha256,
            "filename": filename,
            "manualName": manual_name,
            "pageCount": page_count,
            "pdfBytes": pdf_bytes_len,
            "guideJson": Json(safe_guide_json),
            "ingestionMeta": Json(safe_ingestion_meta),
        }
    )
    new_id = getattr(row, "id", None) or _row_to_dict(row).get("id", "")
    # Summarize ingestion_meta for the log line without dumping the whole dict.
    meta_summary = {
        k: ingestion_meta.get(k)
        for k in ("hierarchy_quality", "section_count", "title_count")
        if k in ingestion_meta
    }
    logger.info(
        "stored source_guide id=%s sha256=%s pages=%d bytes=%d meta=%s",
        new_id,
        sha256[:12],
        page_count,
        pdf_bytes_len,
        meta_summary,
    )
    return new_id
