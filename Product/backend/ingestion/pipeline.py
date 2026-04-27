"""End-to-end ingestion orchestrator.

Coordinates: hash -> cache lookup -> Unstructured parse (parallelized via
split_pdf_page) -> vision enrichment (parallelized via asyncio.gather) ->
assembly -> Neon storage. Emits progress to Redis throughout so the frontend
can poll job state without long-polling SSE.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from backend.ingestion.assembler import assemble_guide
from backend.ingestion.cache import (
    compute_content_hash,
    find_latest_by_hash,
    store_guide,
)
from backend.ingestion.config import IngestionConfig
from backend.ingestion.rendering import render_page_region  # noqa: F401  (re-exported for vision callers)
from backend.ingestion.schema import (
    IngestionFlaggedItem,
    IngestionJobState,
    IngestionManifest,
    RLMGuide,
    RLMGuideMetadata,
)
from backend.ingestion.unstructured_client import UnstructuredClient
from backend.ingestion.vision import VisionPipeline
from backend.ingestion.warmup import compute_probe_concurrency, warmup_burst
from backend.redis_client import redis_delete, redis_get, redis_setex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job state helper (Redis-backed, best-effort)
# ---------------------------------------------------------------------------


async def update_job_state(
    job_id: str,
    status: str,
    stage: str,
    progress: float,
    note: str = "",
    guide_id: str | None = None,
    content_hash: str | None = None,
    error: str | None = None,
) -> None:
    """Write the current ingestion job state to Redis.

    Key: f"ingest:job:{job_id}", TTL 2 hours. Failures to write are logged
    but never raised — the pipeline must keep running even if Redis is down.

    started_at is preserved across state transitions: on the first write for
    a given job_id, we initialize it to now(); on subsequent writes we read
    the existing state from Redis and keep whatever started_at was there.
    This makes the dict parseable as IngestionJobState (which requires both
    started_at and updated_at) and gives consumers an accurate start time.
    """
    now = time.time()
    started_at: float = now
    try:
        existing_raw = await redis_get(f"ingest:job:{job_id}")
        if existing_raw:
            existing = json.loads(existing_raw)
            prior_start = existing.get("started_at")
            if isinstance(prior_start, (int, float)) and prior_start > 0:
                started_at = float(prior_start)
    except Exception as e:
        logger.debug(f"Failed to read prior job state {job_id}: {e}")

    state = {
        "job_id": job_id,
        "status": status,
        "stage": stage,
        "progress": progress,
        "note": note,
        "guide_id": guide_id,
        "content_hash": content_hash,
        "error": error,
        "started_at": started_at,
        "updated_at": now,
    }
    try:
        await redis_setex(f"ingest:job:{job_id}", 7200, json.dumps(state))
    except Exception as e:
        logger.warning(f"Failed to update ingest job state {job_id}: {e}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """End-to-end ingestion orchestrator. One instance per process."""

    def __init__(self, config: IngestionConfig):
        self._config = config
        self._unstructured = UnstructuredClient(
            api_key=config.unstructured_api_key,
            api_url=config.unstructured_api_url,
        )
        self._vision = VisionPipeline(config)
        # Process-local single-flight dedup. The Redis lock at Stage 1 covers
        # cross-instance races (when we eventually scale to multiple Render
        # instances) but the current Upstash REST wrapper doesn't expose NX
        # semantics, so within a single process two simultaneous uploads of
        # the same PDF would still race past the cache check. This dict gives
        # us a deterministic in-process barrier: the second caller waits on
        # the first caller's asyncio.Lock and then re-checks the cache, which
        # by that point will be populated.
        self._inflight_locks: dict[str, asyncio.Lock] = {}
        self._inflight_locks_guard = asyncio.Lock()

    async def _acquire_inflight_lock(self, content_hash: str) -> asyncio.Lock:
        """Get-or-create an asyncio.Lock for this content hash. The guard lock
        ensures the dict mutation itself is atomic so two callers can't both
        create separate locks for the same hash."""
        async with self._inflight_locks_guard:
            lock = self._inflight_locks.get(content_hash)
            if lock is None:
                lock = asyncio.Lock()
                self._inflight_locks[content_hash] = lock
            return lock

    async def _release_inflight_lock(self, content_hash: str) -> None:
        """Drop the asyncio.Lock from the dict once the pipeline is done.
        Safe to call even if the entry was already evicted."""
        async with self._inflight_locks_guard:
            self._inflight_locks.pop(content_hash, None)

    def _build_manifest(
        self,
        pages,
        tables_in,
        images_in,
        table_refinements,
        image_descriptions,
        flowchart_structures,
        guide,
        expected_total_pages: int,
    ) -> IngestionManifest:
        """Compute the IngestionManifest from pipeline results.

        Issue rules:
          - Each Table element with no entry in table_refinements -> warning
            (vision pipeline failed for it; the raw text and HTML are still
            available so the RLM can try anyway).
          - Each Image element with no entry in image_descriptions -> warning.
          - Each is_flowchart image with no flowchart_structures entry -> warning.
          - hierarchy_quality == "fallback" -> critical (RLM can still use the
            pages dict, but section navigation is degraded).
          - hierarchy_quality == "noisy" -> warning.
          - Page count mismatch (pypdfium2 estimate vs Unstructured pages) -> warning.
          - Empty page (zero non-whitespace text) -> info.
        """
        manifest = IngestionManifest()
        manifest.total_pages = len(pages)
        manifest.total_elements = sum(len(p.elements) for p in pages)

        # Vision: tables
        manifest.total_tables = len(tables_in)
        manifest.vision_table_attempts = len(tables_in)
        manifest.vision_table_successes = sum(
            1 for (_, eid) in tables_in if eid in table_refinements
        )
        for page_number, eid in tables_in:
            if eid not in table_refinements:
                manifest.add(
                    IngestionFlaggedItem(
                        page_number=page_number,
                        element_id=eid,
                        issue_type="vision_table_failed",
                        severity="warning",
                        message=(
                            f"Vision table refinement failed for element {eid} "
                            f"on page {page_number}; raw HTML still available"
                        ),
                    )
                )

        # Vision: images
        manifest.total_images = len(images_in)
        manifest.vision_image_attempts = len(images_in)
        manifest.vision_image_successes = sum(
            1 for (_, eid) in images_in if eid in image_descriptions
        )
        for page_number, eid in images_in:
            if eid not in image_descriptions:
                manifest.add(
                    IngestionFlaggedItem(
                        page_number=page_number,
                        element_id=eid,
                        issue_type="vision_image_failed",
                        severity="warning",
                        message=(
                            f"Vision image description failed for element {eid} "
                            f"on page {page_number}"
                        ),
                    )
                )

        # Vision: flowcharts (subset of images flagged is_flowchart)
        flowchart_attempts = [
            (page_number, eid)
            for (page_number, eid) in images_in
            if eid in image_descriptions
            and image_descriptions[eid].is_flowchart
        ]
        manifest.total_flowcharts = len(flowchart_attempts)
        manifest.vision_flowchart_attempts = len(flowchart_attempts)
        manifest.vision_flowchart_successes = sum(
            1 for (_, eid) in flowchart_attempts if eid in flowchart_structures
        )
        for page_number, eid in flowchart_attempts:
            if eid not in flowchart_structures:
                manifest.add(
                    IngestionFlaggedItem(
                        page_number=page_number,
                        element_id=eid,
                        issue_type="vision_flowchart_failed",
                        severity="warning",
                        message=(
                            f"Vision flowchart parsing failed for element {eid} "
                            f"on page {page_number}; falling back to image description"
                        ),
                    )
                )

        # Hierarchy quality (assembler-derived)
        meta = guide.metadata.ingestion_meta or {}
        hierarchy_quality = str(meta.get("hierarchy_quality", "unknown"))
        manifest.hierarchy_quality = hierarchy_quality
        manifest.section_count = len(guide.sections)

        if hierarchy_quality == "fallback":
            manifest.add(
                IngestionFlaggedItem(
                    issue_type="hierarchy_fallback",
                    severity="critical",
                    message=(
                        "No section hierarchy detected; assembler used per-page "
                        "fallback sectioning. RLM section navigation will be degraded; "
                        "use guide['pages'] instead of guide['sections']"
                    ),
                    context={"section_count": len(guide.sections)},
                )
            )
        elif hierarchy_quality == "noisy":
            manifest.add(
                IngestionFlaggedItem(
                    issue_type="hierarchy_noisy",
                    severity="warning",
                    message=(
                        f"Section detection produced {len(guide.sections)} sections "
                        f"for {manifest.total_pages} pages; likely false-positive titles"
                    ),
                    context={"section_count": len(guide.sections)},
                )
            )

        # Page count mismatch (Unstructured vs pypdfium2 estimate)
        if expected_total_pages > 0 and manifest.total_pages != expected_total_pages:
            manifest.add(
                IngestionFlaggedItem(
                    issue_type="page_count_mismatch",
                    severity="warning",
                    message=(
                        f"Page count mismatch: pypdfium2 saw {expected_total_pages}, "
                        f"Unstructured returned elements for {manifest.total_pages}"
                    ),
                    context={
                        "pypdfium2": expected_total_pages,
                        "unstructured": manifest.total_pages,
                    },
                )
            )

        # Empty pages (zero non-whitespace text)
        for page in pages:
            if not page.raw_text.strip():
                manifest.add(
                    IngestionFlaggedItem(
                        page_number=page.page_number,
                        issue_type="empty_page",
                        severity="info",
                        message=f"Page {page.page_number} has no extracted text",
                    )
                )

        return manifest

    async def run(
        self,
        pdf_bytes: bytes,
        filename: str,
        manual_name: str | None,
        job_id: str,
        check_dupes: bool = True,
    ) -> dict:
        """Run the full ingestion pipeline.

        Returns: dict with keys {guide_id, content_hash, cached, guide_json}.

        Raises on fatal errors (no Unstructured response, DB failure, etc).
        Non-fatal errors (one vision call failing) are logged and produce a
        partial guide.
        """
        content_hash: str = ""
        lock_key: str | None = None
        warmup_task: asyncio.Task | None = None
        inflight_lock: asyncio.Lock | None = None
        inflight_acquired: bool = False

        try:
            # -------- Stage 0: hash + (parallel) warmup kickoff + cache lookup --------
            content_hash = compute_content_hash(pdf_bytes)

            # Process-local single-flight: if another upload of the SAME bytes
            # is already in flight in this process, wait for it. The wait
            # naturally releases when the first caller's pipeline finishes,
            # and our cache check below will then return the row it just wrote.
            # When check_dupes=False (consistency testing) we deliberately skip
            # this so multiple fresh runs can race.
            if check_dupes:
                inflight_lock = await self._acquire_inflight_lock(content_hash)
                await inflight_lock.acquire()
                inflight_acquired = True

            # Count pages locally (pypdfium2, <100ms for typical PDFs) so we can
            # size the warmup burst to match the expected parallelism of the real
            # parse. Fail open if page count detection fails — warmup still runs
            # with a conservative concurrency.
            total_pages_estimate = 0
            try:
                import pypdfium2 as pdfium
                _doc = pdfium.PdfDocument(pdf_bytes)
                try:
                    total_pages_estimate = len(_doc)
                finally:
                    _doc.close()
            except Exception as exc:
                logger.warning(
                    "Page count estimation failed (warmup will use max concurrency): %s",
                    exc,
                )
                total_pages_estimate = 200  # conservative max

            probe_concurrency = compute_probe_concurrency(total_pages_estimate)

            # Kick off the warmup burst in parallel with the cache check.
            # On cache hit we cancel it; on miss we await it before firing the
            # real parse so workers are warm when split_pdf_page lands.
            warmup_task = asyncio.create_task(
                warmup_burst(self._unstructured, probe_concurrency)
            )

            logger.info(
                "ingestion start: job=%s file=%s bytes=%d hash=%s pages=%d probes=%d check_dupes=%s",
                job_id,
                filename,
                len(pdf_bytes),
                content_hash[:12],
                total_pages_estimate,
                probe_concurrency,
                check_dupes,
            )
            await update_job_state(
                job_id,
                status="running",
                stage="cache_check",
                progress=0.02,
                note=f"checking cache for {total_pages_estimate}-page PDF",
                content_hash=content_hash,
            )

            if check_dupes:
                cached = await find_latest_by_hash(content_hash)
                if cached is not None:
                    # Cache hit — cancel the in-flight warmup, we don't need it.
                    if warmup_task and not warmup_task.done():
                        warmup_task.cancel()
                        try:
                            await warmup_task
                        except (asyncio.CancelledError, Exception):
                            pass
                    guide_id = cached.get("id", "")
                    guide_json = cached.get("guideJson") or cached.get("guide_json") or {}
                    await update_job_state(
                        job_id,
                        status="done",
                        stage="cache_hit",
                        progress=1.0,
                        note=f"reused existing guide {guide_id}",
                        guide_id=guide_id,
                        content_hash=content_hash,
                    )
                    logger.info(
                        "ingestion cache hit: job=%s guide_id=%s",
                        job_id,
                        guide_id,
                    )
                    # Release the in-flight lock so any waiting callers can proceed
                    # (they'll also hit this same cached row).
                    if inflight_acquired and inflight_lock is not None:
                        inflight_lock.release()
                        inflight_acquired = False
                        await self._release_inflight_lock(content_hash)
                    return {
                        "guide_id": guide_id,
                        "content_hash": content_hash,
                        "cached": True,
                        "guide_json": guide_json,
                    }

            # -------- Stage 1: acquire lock (best-effort) --------
            if check_dupes:
                lock_key = f"ingest:lock:{content_hash}"
                await update_job_state(
                    job_id,
                    status="running",
                    stage="acquiring_lock",
                    progress=0.05,
                    note="acquiring ingestion lock",
                    content_hash=content_hash,
                )
                # Upstash REST wrapper doesn't expose NX semantics; a plain
                # SETEX is good enough for v1. If a concurrent run overwrites,
                # both will still produce valid rows and later cache lookups
                # will resolve correctly via (sha256, ingested_at DESC).
                try:
                    await redis_setex(lock_key, 600, job_id)
                except Exception as e:
                    logger.warning(
                        "Failed to acquire ingest lock %s: %s", lock_key, e
                    )

            # -------- Stage 1b: wait for warmup burst to finish --------
            # The burst was kicked off in Stage 0 in parallel with the cache
            # lookup. Waiting here means the real parse fires against a warm
            # worker pool, dropping cold hi_res wall clock from ~10 min to ~90s
            # for a 200-page manual. Failures in warmup are non-fatal — we
            # still run the parse, just potentially cold.
            if warmup_task is not None:
                await update_job_state(
                    job_id,
                    status="running",
                    stage="warming_pool",
                    progress=0.07,
                    note=f"warming {probe_concurrency} Unstructured workers",
                    content_hash=content_hash,
                )
                try:
                    await warmup_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.warning(
                        "Warmup burst raised (non-fatal, continuing cold): %s",
                        exc,
                    )

            # -------- Stage 2: Unstructured parse --------
            await update_job_state(
                job_id,
                status="running",
                stage="parsing_with_unstructured",
                progress=0.10,
                note="sending PDF to Unstructured hi_res",
                content_hash=content_hash,
            )
            pages = await self._unstructured.partition_pdf(pdf_bytes, filename)

            table_count = sum(
                1 for p in pages for e in p.elements if e.element_type == "Table"
            )
            image_count = sum(
                1 for p in pages for e in p.elements if e.element_type == "Image"
            )
            logger.info(
                "parsed %d pages (%d tables, %d images) for job=%s",
                len(pages),
                table_count,
                image_count,
                job_id,
            )

            await update_job_state(
                job_id,
                status="running",
                stage="parsing_with_unstructured",
                progress=0.35,
                note=f"{len(pages)} pages parsed, {table_count} tables, {image_count} images",
                content_hash=content_hash,
            )

            # -------- Stage 3: Vision enrichment (parallel) --------
            await update_job_state(
                job_id,
                status="running",
                stage="vision_table_refinement",
                progress=0.35,
                note="starting vision enrichment",
                content_hash=content_hash,
            )

            async def progress_cb(stage: str, frac: float, note: str) -> None:
                # Vision reports its own 0..1 progress; we linearly map that
                # into the 0.35..0.80 slice reserved for vision work.
                scaled = 0.35 + 0.45 * max(0.0, min(1.0, frac))
                await update_job_state(
                    job_id,
                    status="running",
                    stage=stage,
                    progress=scaled,
                    note=note,
                    content_hash=content_hash,
                )

            # Snapshot the input element lists so we can compute vision failures
            # AFTER the call returns. We don't want vision.py to know about the
            # manifest; the orchestrator owns quality reporting.
            tables_in: list[tuple[int, str]] = [
                (p.page_number, e.element_id)
                for p in pages
                for e in p.elements
                if e.element_type == "Table"
            ]
            images_in: list[tuple[int, str]] = [
                (p.page_number, e.element_id)
                for p in pages
                for e in p.elements
                if e.element_type == "Image"
            ]

            (
                table_refinements,
                image_descriptions,
                flowchart_structures,
            ) = await self._vision.enrich_pages(
                pdf_bytes,
                pages,
                progress_callback=progress_cb,
            )

            # -------- Stage 4: assemble --------
            await update_job_state(
                job_id,
                status="running",
                stage="assembling",
                progress=0.85,
                note="assembling RLM guide",
                content_hash=content_hash,
            )

            total_pages = len(pages)
            metadata = RLMGuideMetadata(
                title=manual_name or filename.rsplit(".", 1)[0].replace("_", " "),
                source_filename=filename,
                content_hash=content_hash,
                total_pages=total_pages,
                ingested_at=datetime.now(timezone.utc).isoformat(),
                ingestion_model="unstructured-hi_res",
                ingestion_version="1.0",
                ingestion_meta={},
            )

            guide = assemble_guide(
                pages,
                table_refinements,
                image_descriptions,
                flowchart_structures,
                metadata,
            )

            await update_job_state(
                job_id,
                status="running",
                stage="assembling",
                progress=0.90,
                note=f"{len(guide.sections)} sections",
                content_hash=content_hash,
            )

            # -------- Stage 4b: build the IngestionManifest --------
            # Quality report covering vision failures, hierarchy quality, and
            # any structural issues the assembler flagged. Stored in
            # ingestion_meta["manifest"] alongside the existing
            # hierarchy_quality / section_count fields so consumers can gate
            # on critical_count == 0 before starting an RLM session.
            manifest = self._build_manifest(
                pages=pages,
                tables_in=tables_in,
                images_in=images_in,
                table_refinements=table_refinements,
                image_descriptions=image_descriptions,
                flowchart_structures=flowchart_structures,
                guide=guide,
                expected_total_pages=total_pages_estimate,
            )
            # Inject the manifest into ingestion_meta so it ships with guide_json
            # AND lives at the top of source_guides.ingestion_meta for fast lookup.
            guide.metadata.ingestion_meta["manifest"] = manifest.model_dump()
            logger.info(
                "ingestion manifest: critical=%d warning=%d info=%d (vision tables %d/%d, images %d/%d, flowcharts %d/%d)",
                manifest.critical_count,
                manifest.warning_count,
                manifest.info_count,
                manifest.vision_table_successes,
                manifest.vision_table_attempts,
                manifest.vision_image_successes,
                manifest.vision_image_attempts,
                manifest.vision_flowchart_successes,
                manifest.vision_flowchart_attempts,
            )

            # -------- Stage 5: store in Neon --------
            await update_job_state(
                job_id,
                status="running",
                stage="writing_to_neon",
                progress=0.92,
                note="persisting to source_guides",
                content_hash=content_hash,
            )

            guide_json_dict = guide.model_dump()
            guide_id = await store_guide(
                sha256=content_hash,
                filename=filename,
                manual_name=manual_name,
                page_count=total_pages,
                pdf_bytes_len=len(pdf_bytes),
                guide_json=guide_json_dict,
                ingestion_meta=guide.metadata.ingestion_meta,
            )

            await update_job_state(
                job_id,
                status="done",
                stage="done",
                progress=1.0,
                note=f"guide stored as {guide_id}",
                guide_id=guide_id,
                content_hash=content_hash,
            )

            # -------- Stage 6: release lock and return --------
            if lock_key is not None:
                try:
                    await redis_delete(lock_key)
                except Exception as e:
                    logger.warning(
                        "Failed to release ingest lock %s: %s", lock_key, e
                    )

            # Release the in-process single-flight lock now that the row exists
            # in source_guides; any waiting callers will hit the cache check
            # and return the freshly-written guide.
            if inflight_acquired and inflight_lock is not None:
                inflight_lock.release()
                inflight_acquired = False
                await self._release_inflight_lock(content_hash)

            logger.info(
                "ingestion done: job=%s guide_id=%s hash=%s",
                job_id,
                guide_id,
                content_hash[:12],
            )
            return {
                "guide_id": guide_id,
                "content_hash": content_hash,
                "cached": False,
                "guide_json": guide_json_dict,
            }

        except Exception as exc:
            logger.exception(
                "ingestion failed: job=%s file=%s error=%s",
                job_id,
                filename,
                exc,
            )
            # Clean up any in-flight warmup task so we don't leak a background
            # coroutine on failure.
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                try:
                    await warmup_task
                except (asyncio.CancelledError, Exception):
                    pass
            await update_job_state(
                job_id,
                status="failed",
                stage="failed",
                progress=0.0,
                note="ingestion failed",
                content_hash=content_hash or None,
                error=str(exc),
            )
            if lock_key is not None:
                try:
                    await redis_delete(lock_key)
                except Exception as release_err:
                    logger.warning(
                        "Failed to release ingest lock %s after error: %s",
                        lock_key,
                        release_err,
                    )
            # Release the single-flight lock on failure too — waiting callers
            # should be unblocked so they can retry.
            if inflight_acquired and inflight_lock is not None:
                inflight_lock.release()
                inflight_acquired = False
                if content_hash:
                    await self._release_inflight_lock(content_hash)
            raise
