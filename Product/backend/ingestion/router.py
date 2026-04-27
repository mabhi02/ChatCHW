"""FastAPI router for PDF ingestion endpoints.

Mounted by backend/server.py under /api/ingest. Background task pattern:
POST returns immediately with a jobId, the pipeline runs in-process as an
asyncio task, progress is streamed to Redis, and the frontend polls
GET /api/ingest/{job_id} every second until status=done.
"""

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from backend.ingestion.config import load_ingestion_config
from backend.ingestion.pipeline import IngestionPipeline, update_job_state
from backend.ingestion.cache import find_by_id
from backend.redis_client import redis_get

logger = logging.getLogger(__name__)


# Module-level singletons. Config is loaded once at import time so missing env
# vars surface at startup instead of on the first request.
_config = load_ingestion_config()
_pipeline = IngestionPipeline(_config)

# Strong references to in-flight background tasks. Without this set, Python's
# garbage collector is free to collect a Task whose only reference was the
# `asyncio.create_task(...)` return value we previously discarded. Collected
# tasks silently stop running mid-execution. Holding a strong reference here
# keeps them alive until they complete; the done_callback below removes them.
_background_tasks: set[asyncio.Task] = set()


def _extract_api_key(request: Request) -> str:
    """Extract Anthropic API key from Authorization header.

    Note: ingestion doesn't USE this key, but we still require it to limit
    access to authenticated users (same pattern as the rest of the app).
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")


router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


class IngestResponse(BaseModel):
    jobId: str
    status: str  # "queued" | "running" | "done"


# Stream-read chunk size. 256 KB strikes a good balance: large enough that
# we don't spin through millions of tiny awaits on a 50 MB PDF, small enough
# that we can fail-fast on oversize without buffering much.
_UPLOAD_CHUNK_SIZE = 256 * 1024


async def _read_upload_with_size_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read an UploadFile into memory, aborting as soon as we exceed max_bytes.

    UploadFile.read() with no argument buffers the entire upload before we
    can check size, which is how we OOM on adversarial uploads. Reading in
    chunks lets us bail the moment we hit the limit.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds max size of {max_bytes} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("", response_model=IngestResponse)
async def start_ingest(
    request: Request,
    file: UploadFile = File(...),
    manual_name: str | None = Form(default=None),
    check_dupes: bool = Query(default=True, description="If True, return cached guide for same PDF hash"),
) -> IngestResponse:
    """Start a PDF ingestion job. Returns a jobId the frontend polls."""
    _extract_api_key(request)  # auth gate

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    # Streaming read with early-abort on size. Replaces the naive
    # `await file.read()` which would buffer the entire upload before any
    # size check could fire — meaning a 500 MB adversarial upload OOMs the
    # process before we even get to validate it.
    pdf_bytes = await _read_upload_with_size_limit(file, _config.max_pdf_bytes)
    if len(pdf_bytes) == 0:
        raise HTTPException(400, "Uploaded PDF is empty")

    job_id = str(uuid.uuid4())
    # Write initial queued state to Redis so the poll endpoint has something immediately
    await update_job_state(job_id, status="queued", stage="queued", progress=0.0)

    # Kick off the pipeline as a background task; KEEP the reference in a
    # module-level set so Python's GC doesn't collect the task mid-execution.
    # The done_callback removes the task after completion.
    task = asyncio.create_task(
        _run_ingest_task(pdf_bytes, file.filename, manual_name, job_id, check_dupes),
        name=f"ingest-{job_id}",
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return IngestResponse(jobId=job_id, status="queued")


async def _run_ingest_task(
    pdf_bytes: bytes,
    filename: str,
    manual_name: str | None,
    job_id: str,
    check_dupes: bool,
) -> None:
    """Background task wrapper. Catches unhandled exceptions and writes them to job state."""
    try:
        await _pipeline.run(pdf_bytes, filename, manual_name, job_id, check_dupes)
    except Exception as e:
        logger.error(f"Ingestion job {job_id} failed: {e}", exc_info=True)
        await update_job_state(
            job_id, status="failed", stage="failed", progress=0.0, error=str(e)
        )


@router.get("/{job_id}")
async def get_ingest_status(request: Request, job_id: str) -> dict:
    """Poll endpoint for ingestion progress. Frontend calls this every 1s."""
    _extract_api_key(request)
    state_json = await redis_get(f"ingest:job:{job_id}")
    if state_json is None:
        raise HTTPException(404, "Job not found or expired")
    return json.loads(state_json)


@router.get("/guides/{guide_id}")
async def get_guide(request: Request, guide_id: str) -> dict:
    """Load a stored source guide by id.

    Used by the frontend to preview the guide metadata, or by the session
    start endpoint to resolve sourceGuideId -> guide_json. Surfaces the full
    IngestionManifest so consumers can decide whether to start an RLM session
    or block on critical quality issues.
    """
    _extract_api_key(request)
    guide = await find_by_id(guide_id)
    if guide is None:
        raise HTTPException(404, "Source guide not found")
    ingested_at = guide["ingestedAt"]
    ingestion_meta = guide["guideJson"].get("metadata", {}).get("ingestion_meta", {}) or {}
    manifest = ingestion_meta.get("manifest", {}) or {}
    return {
        "id": guide["id"],
        "filename": guide["filename"],
        "manualName": guide.get("manualName"),
        "pageCount": guide["pageCount"],
        "ingestedAt": ingested_at.isoformat() if hasattr(ingested_at, "isoformat") else ingested_at,
        "sectionCount": len(guide["guideJson"].get("sections", {})),
        "hierarchyQuality": ingestion_meta.get("hierarchy_quality", "unknown"),
        "manifest": manifest,
        "criticalCount": manifest.get("critical_count", 0),
        "warningCount": manifest.get("warning_count", 0),
        "infoCount": manifest.get("info_count", 0),
    }
