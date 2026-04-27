"""FastAPI server for CHW Navigator.

Endpoints:
- POST /api/session/start -- Upload guide JSON, start extraction
- GET  /api/session/{id}/stream -- SSE stream of REPL steps
- GET  /api/session/{id}/status -- Poll status
- GET  /api/session/{id}/artifacts -- List artifacts
- GET  /api/session/{id}/artifacts/{type} -- Download specific artifact
- POST /api/session/{id}/cancel -- Cancel running session
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load .env BEFORE any backend imports that read os.environ at module scope
# (e.g., backend.ingestion.config raises at import if UNSTRUCTURED_API_KEY
# is missing). uvicorn invoked directly does not auto-load .env.
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.anthropic_tier import check_anthropic_tier
from backend.db import get_db, disconnect_db
from backend.ingestion.router import router as ingest_router
from backend.logging_config import setup_logging
from backend.session_manager import (
    create_session,
    get_session_events,
    get_session_status,
    get_session_artifacts,
    cancel_session,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup and teardown."""
    setup_logging()
    logger.info("CHW Navigator backend starting")
    import os
    if os.environ.get("NEON_DB"):
        try:
            await get_db()
            logger.info("Connected to Neon Postgres")
        except Exception as e:
            logger.warning("Could not connect to database: %s (running without DB)", e)
    else:
        logger.warning("NEON_DB not set; running without database (Neon logging disabled)")
    yield
    try:
        await disconnect_db()
    except Exception:
        pass
    logger.info("CHW Navigator backend shutting down")


# ---------------------------------------------------------------------------
# Multipart size limit override
# ---------------------------------------------------------------------------
# Starlette's default max_part_size is 1 MB. Our ingestion endpoint accepts
# PDFs up to 50 MB, but FastAPI's UploadFile parser calls `request.form()`
# with the default limit BEFORE our endpoint code runs, so the request is
# rejected with a generic 413 long before our in-endpoint size check fires.
#
# We patch Starlette's Request.form at import time to bump the default limit
# to INGEST_MAX_PDF_BYTES (50 MB by default, configurable via env). The actual
# per-endpoint size validation still happens in `_read_upload_with_size_limit`
# inside the router, using the same constant — the monkey-patch just raises
# the floor so our validation code gets a chance to run.
import starlette.requests as _starlette_requests

_MAX_MULTIPART_BYTES = int(os.environ.get("INGEST_MAX_PDF_BYTES", str(50 * 1024 * 1024)))
_original_form = _starlette_requests.Request.form

async def _form_with_higher_limit(
    self,
    *,
    max_files: int | float = 1000,
    max_fields: int | float = 1000,
    max_part_size: int = _MAX_MULTIPART_BYTES,
):
    return await _original_form(
        self,
        max_files=max_files,
        max_fields=max_fields,
        max_part_size=max_part_size,
    )

_starlette_requests.Request.form = _form_with_higher_limit  # type: ignore[assignment]


app = FastAPI(
    title="CHW Navigator API",
    description="Clinical decision logic extractor using Recursive Language Models",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose Content-Disposition so JS fetch-and-blob downloads can read the
    # server-suggested filename. Without this, browsers block the header
    # from reaching frontend code.
    expose_headers=["Content-Disposition", "Content-Length"],
)

app.include_router(ingest_router)


@app.get("/api/test-download/{fmt}")
async def test_download_endpoint(fmt: str):
    """Test endpoint for download smoke test page. No auth required.
    Returns a small file with Content-Disposition so the frontend test page
    can verify which download mechanism correctly picks up the filename.
    """
    from fastapi.responses import Response
    content = f"This is a test download. Format: {fmt}\nTimestamp: ok\n".encode()
    filename_map = {
        "json": "test-artifact.json",
        "csv": "test-artifact.csv",
        "zip": "test-bundle.zip",
        "txt": "hello-world.txt",
    }
    filename = filename_map.get(fmt, f"unknown-{fmt}.bin")
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _extract_api_key(request: Request) -> str:
    """Extract API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")


# ---------------------
# Request/Response models
# ---------------------


class TierCheckRequest(BaseModel):
    apiKey: str


class TierCheckResponse(BaseModel):
    tier: int | None
    requestsLimit: int
    inputTpm: int
    outputTpm: int
    meetsMinimum: bool
    message: str
    error: str | None = None


class StartSessionRequest(BaseModel):
    guide_json: dict | None = None
    sourceGuideId: str | None = None
    manual_name: str | None = None
    # Override flag: if True, start the session even if the IngestionManifest
    # has critical issues (e.g., hierarchy_fallback). Default False so consumers
    # have to opt-in explicitly. Ignored when guide_json is provided directly
    # (legacy path has no manifest).
    forceIngestionOverride: bool = False
    # If True, skip the tier pre-flight check on session start. Only useful
    # for local dev or bypassing the check temporarily -- production should
    # always leave this False so Tier 1 keys get rejected with a clear error.
    skipTierCheck: bool = False
    # Extraction pipeline to use:
    #   "gen7"    = Opus mono (frozen @ SHA 9000e86)
    #   "gen8"    = Opus mono + Tier 0-3 (predicate hygiene, provenance,
    #               traffic cops, stockout coverage, verifier)
    #   "gen8.5"  = 7-way Sonnet labeler + everything in gen8
    #   "legacy"  = original rlm_runner.py (hybrid Opus/Sonnet REPL)
    # Defaults to "gen7".
    pipeline: str = "gen7"


class StartSessionResponse(BaseModel):
    sessionId: str
    status: str


class SessionStatusResponse(BaseModel):
    sessionId: str
    status: str
    pipeline: str = "legacy"
    totalIterations: int
    totalSubcalls: int
    validationErrors: int
    costEstimateUsd: float
    inputTokensTotal: int = 0
    outputTokensTotal: int = 0
    cachedTokensTotal: int = 0
    callsOpus: int = 0
    callsSonnet: int = 0
    callsHaiku: int = 0
    cacheWriteTokensTotal: int = 0
    startedAt: str | None = None
    errorMessage: str | None = None


# ---------------------
# Endpoints
# ---------------------


@app.post("/api/anthropic/tier-check", response_model=TierCheckResponse)
async def anthropic_tier_check(body: TierCheckRequest):
    """Probe the user's Anthropic API key and return its tier classification.

    Called by the frontend after the user enters their API key, so we can
    show a green "Tier X — ready to run" badge (or a red "upgrade required"
    banner) before they attempt to start an extraction. Also called as a
    pre-flight check inside /api/session/start (see below) as a safety net
    against users bypassing the frontend badge.

    Cost: ~$0.0001 per call (one Haiku call, 1-output-token max).

    Never returns an HTTP error — auth failures, tier-too-low, and network
    errors are all returned as structured JSON with meetsMinimum=false so
    the frontend can display the appropriate message without catch/rethrow.
    """
    result = await check_anthropic_tier(body.apiKey)
    return TierCheckResponse(**result.to_dict())


@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(request: Request, body: StartSessionRequest):
    """Start a new extraction session.

    Accepts either an inline `guide_json` (legacy) or a `sourceGuideId`
    pointing at a previously ingested guide. Exactly one must be provided.
    """
    api_key = _extract_api_key(request)

    # Tier pre-flight: reject Tier 1 keys with a clear error BEFORE we create
    # a session record or spin up an extraction task. Without this, a Tier 1
    # user would see their run fail with garbled "rate_limit_error" messages
    # after ~3 minutes of wasted compute. Fails fast, actionable, ~$0.0001.
    if not body.skipTierCheck:
        tier_result = await check_anthropic_tier(api_key)
        if not tier_result.meets_minimum:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "insufficient_tier",
                    "tier": tier_result.tier,
                    "requestsLimit": tier_result.requests_limit,
                    "inputTpm": tier_result.input_tpm,
                    "outputTpm": tier_result.output_tpm,
                    "message": tier_result.message,
                    "probeError": tier_result.error,
                },
            )

    if (body.guide_json is None) == (body.sourceGuideId is None):
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of 'guide_json' or 'sourceGuideId'",
        )

    if body.sourceGuideId is not None:
        from backend.ingestion.cache import find_by_id
        source = await find_by_id(body.sourceGuideId)
        if source is None:
            raise HTTPException(status_code=404, detail="Source guide not found")
        guide_json = source["guideJson"]
        manual_name = body.manual_name or source.get("manualName")
        source_guide_id = source["id"]

        # Quality gate: refuse to start the session if the ingestion manifest
        # has critical issues, unless the caller explicitly overrides. This
        # protects the user from running a 5-10 minute extraction on a guide
        # whose section hierarchy completely failed to detect.
        ingestion_meta = (
            (guide_json.get("metadata", {}) or {}).get("ingestion_meta", {}) or {}
        )
        manifest = ingestion_meta.get("manifest", {}) or {}
        critical_count = int(manifest.get("critical_count", 0) or 0)
        if critical_count > 0 and not body.forceIngestionOverride:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "ingestion_quality_critical",
                    "criticalCount": critical_count,
                    "warningCount": int(manifest.get("warning_count", 0) or 0),
                    "hierarchyQuality": ingestion_meta.get("hierarchy_quality", "unknown"),
                    "message": (
                        f"Source guide has {critical_count} critical ingestion issue(s). "
                        f"Re-ingest the PDF or pass forceIngestionOverride=true to proceed anyway."
                    ),
                    "flaggedItems": manifest.get("flagged_items", [])[:10],
                },
            )
    else:
        guide_json = body.guide_json
        manual_name = body.manual_name
        source_guide_id = None

    # Validate pipeline choice
    pipeline = body.pipeline
    if pipeline not in ("gen7", "gen8", "gen8.5", "legacy"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid pipeline '{pipeline}'. "
                f"Must be one of: 'gen7', 'gen8', 'gen8.5', 'legacy'."
            ),
        )

    session_id = await create_session(
        api_key=api_key,
        guide_json=guide_json,
        manual_name=manual_name,
        source_guide_id=source_guide_id,
        pipeline=pipeline,
    )
    return StartSessionResponse(sessionId=session_id, status="running")


@app.get("/api/session/{session_id}/stream")
async def stream_session(session_id: str, token: str = Query(default="")):
    """SSE stream of REPL events.

    The API key is passed as a query param since EventSource doesn't support headers.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing token parameter")

    status = await get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        async for event in get_session_events(session_id):
            yield f"data: {json.dumps(event)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/session/{session_id}/status", response_model=SessionStatusResponse)
async def session_status(request: Request, session_id: str):
    """Get current session status."""
    _extract_api_key(request)
    status = await get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatusResponse(
        sessionId=session_id,
        status=status.get("status", "unknown"),
        pipeline=status.get("pipeline", "legacy"),
        totalIterations=status.get("total_iterations", 0),
        totalSubcalls=status.get("total_subcalls", 0),
        validationErrors=status.get("validation_errors", 0),
        costEstimateUsd=status.get("cost_estimate_usd", 0.0),
        inputTokensTotal=status.get("input_tokens_total", 0),
        outputTokensTotal=status.get("output_tokens_total", 0),
        cachedTokensTotal=status.get("cached_tokens_total", 0),
        callsOpus=status.get("calls_opus", 0),
        callsSonnet=status.get("calls_sonnet", 0),
        callsHaiku=status.get("calls_haiku", 0),
        cacheWriteTokensTotal=status.get("cache_write_tokens_total", 0),
        startedAt=status.get("started_at"),
        errorMessage=status.get("error_message"),
    )


@app.get("/api/session/{session_id}/journal")
async def get_journal(request: Request, session_id: str):
    """Get the live extraction journal (scratchpad.md content)."""
    _extract_api_key(request)
    from backend.session_manager import _active_sessions
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    journal = _active_sessions[session_id].get("journal")
    if not journal:
        return {"markdown": "", "stats": {}}
    return {"markdown": journal.get_journal_text(), "stats": journal._get_stats()}


@app.get("/api/session/{session_id}/artifacts")
async def list_artifacts(
    request: Request,
    session_id: str,
    token: str = Query(default=""),
):
    """List available artifacts for download.

    Auth: accepts Authorization header OR ?token query param.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") and not token:
        raise HTTPException(status_code=401, detail="Missing auth (header or ?token)")
    artifacts = await get_session_artifacts(session_id)
    return artifacts


@app.get("/api/session/{session_id}/artifacts/{artifact_type}")
async def download_artifact(
    request: Request,
    session_id: str,
    artifact_type: str,
    token: str = Query(default=""),
):
    """Download a specific artifact.

    Auth: accepts either `Authorization: Bearer <key>` header OR `?token=<key>`
    query param. The query param path is needed for `<a href>` download links
    which cannot send custom headers.
    """
    # Try header first, fall back to query param
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") and not token:
        raise HTTPException(status_code=401, detail="Missing auth (header or ?token)")

    artifacts = await get_session_artifacts(session_id)
    artifact = next((a for a in artifacts if a["type"] == artifact_type), None)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Map type to filename
    status = await get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    run_id = status.get("run_id", "")
    # Prefer the relativePath field (introduced with hybrid-plan intermediates
    # under artifacts/ subdirectory); fall back to filename for backwards compat
    # with any legacy artifact entries.
    relative = artifact.get("relativePath") or artifact["filename"]
    file_path = OUTPUT_DIR / run_id / relative

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(
        path=str(file_path),
        filename=artifact["filename"],
        media_type="application/octet-stream",
    )


@app.get("/api/session/{session_id}/artifacts.zip")
async def download_all_artifacts_zip(
    request: Request,
    session_id: str,
    token: str = Query(default=""),
):
    """Bundle all session artifacts into a single ZIP download.

    Auth: accepts either Authorization header OR ?token query param.
    """
    import io
    import zipfile

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") and not token:
        raise HTTPException(status_code=401, detail="Missing auth (header or ?token)")

    artifacts = await get_session_artifacts(session_id)
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts found for this session")

    status = await get_session_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")

    run_id = status.get("run_id", "")

    buffer = io.BytesIO()
    added: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in artifacts:
            relative = artifact.get("relativePath") or artifact["filename"]
            file_path = OUTPUT_DIR / run_id / relative
            if file_path.exists():
                zf.write(file_path, arcname=relative)
                added.add(relative.replace("\\", "/"))

        # Recursively walk the entire run directory to pick up everything
        # the registered-artifacts list doesn't know about: README.md,
        # stage3_repl.json, system_prompt.md, gen8 manifest.json,
        # divergence_worklist.json, all *.provenance.json + *.verification.json
        # sidecars, both per-module subdirectories, and any future additions.
        # This is broader than the previous targeted-extras loop and keeps the
        # zip self-contained as the artifact set evolves.
        run_dir = OUTPUT_DIR / run_id
        if run_dir.exists():
            for p in sorted(run_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(run_dir)).replace("\\", "/")
                if rel in added:
                    continue
                zf.write(p, arcname=rel)
                added.add(rel)

    buffer.seek(0)
    from fastapi.responses import Response
    return Response(
        content=buffer.read(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{session_id[:8]}-artifacts.zip"',
        },
    )


@app.post("/api/session/{session_id}/cancel")
async def cancel(request: Request, session_id: str):
    """Cancel a running extraction session."""
    _extract_api_key(request)
    success = await cancel_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already completed")
    return {"status": "cancelled"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "chw-navigator"}


# ---------------------------------------------------------------------------
# Arena A/B Test endpoints
# ---------------------------------------------------------------------------


class StartArenaRequest(BaseModel):
    sourceGuideId: str
    nRuns: int = 1  # 1 or 3
    manualName: str | None = None


class StartArenaResponse(BaseModel):
    arenaId: str
    status: str


def _extract_dual_keys(request: Request) -> tuple[str, str]:
    """Extract both Anthropic and OpenAI keys from request headers.

    Anthropic key: Authorization header (Bearer sk-ant-...)
    OpenAI key: X-OpenAI-Key header (sk-proj-...)
    """
    anthropic_key = ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        anthropic_key = auth[7:]
    if not anthropic_key:
        raise HTTPException(status_code=401, detail="Missing Anthropic API key in Authorization header")

    openai_key = request.headers.get("X-OpenAI-Key", "")
    if not openai_key:
        raise HTTPException(status_code=401, detail="Missing OpenAI API key in X-OpenAI-Key header")

    return anthropic_key, openai_key


@app.post("/api/arena/start", response_model=StartArenaResponse)
async def start_arena_endpoint(request: Request, body: StartArenaRequest):
    """Start an arena A/B comparison. Launches N REPL + N Sequential runs."""
    anthropic_key, openai_key = _extract_dual_keys(request)

    from backend.ingestion.cache import find_by_id
    source = await find_by_id(body.sourceGuideId)
    if source is None:
        raise HTTPException(status_code=404, detail="Source guide not found")

    guide_json = source["guideJson"]
    manual_name = body.manualName or source.get("manualName", "Unknown")

    from backend.arena import start_arena
    session = await start_arena(
        guide_json=guide_json,
        source_guide_id=body.sourceGuideId,
        manual_name=manual_name,
        n_runs=body.nRuns,
        anthropic_key=anthropic_key,
        openai_key=openai_key,
    )

    return StartArenaResponse(arenaId=session.arena_id, status="running")


@app.get("/api/arena/{arena_id}/stream")
async def stream_arena(arena_id: str):
    """SSE stream of arena events (both REPL and Sequential progress)."""
    from backend.arena import get_arena
    session = get_arena(arena_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Arena session not found")

    async def event_generator():
        async for event in session.stream_events():
            yield f"data: {json.dumps(event, default=str)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/arena/{arena_id}/status")
async def arena_status(request: Request, arena_id: str):
    """Get current arena status and results."""
    from backend.arena import get_arena
    session = get_arena(arena_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Arena session not found")

    return {
        "arenaId": session.arena_id,
        "status": session.status,
        "nRuns": session.n_runs,
        "manualName": session.manual_name,
        "replResults": session.repl_results,
        "sequentialResults": session.seq_results,
    }


@app.get("/api/arena/{arena_id}/artifacts/{pipeline}/{run_number}/{artifact_name}")
async def download_arena_artifact(
    arena_id: str,
    pipeline: str,
    run_number: int,
    artifact_name: str,
):
    """Download a specific artifact from an arena run."""
    from backend.session_manager import OUTPUT_DIR
    artifact_path = OUTPUT_DIR / arena_id / f"{pipeline}-{run_number}" / f"{artifact_name}.json"
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(
        path=str(artifact_path),
        filename=f"{artifact_name}.json",
        media_type="application/json",
    )


@app.get("/api/guides")
async def list_guides(request: Request):
    """List all ingested guides available for arena runs."""
    _extract_api_key(request)
    db = await get_db()
    guides = await db.sourceguide.find_many(
        order={"ingestedAt": "desc"},
        take=50,
    )
    return [
        {
            "id": g.id,
            "filename": g.filename,
            "manualName": g.manualName,
            "pageCount": g.pageCount,
            "pdfBytes": g.pdfBytes,
            "ingestedAt": g.ingestedAt.isoformat() if g.ingestedAt else None,
        }
        for g in guides
    ]
