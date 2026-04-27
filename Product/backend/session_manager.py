"""Session Manager -- BYOK session lifecycle.

Manages extraction session lifecycle:
- Create: generate session ID, store state in Redis (Upstash REST) + in-memory
- Run: launch RLM session in background, stream steps via SSE
- Artifacts: after completion, run converters, serve outputs
- Cleanup: clear session state

The API key is held in-memory and optionally in Redis (TTL 2 hours).
Never persisted to disk or Neon.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import HTTPException

from backend.db import create_extraction_run, update_extraction_run, log_repl_step
from backend.journal import JournalWriter
from backend.redis_client import redis_get, redis_setex, redis_delete
from backend.rlm_runner import run_extraction
from backend.converters import convert_to_dmn, convert_to_mermaid, convert_to_csv
from backend.converters.json_to_xlsx import convert_to_xlsx

# Gen 7 pipeline import (lazy to avoid import failures when gen7 deps missing)
_gen7_import_error: str | None = None
try:
    from backend.gen7.pipeline import run_gen7_extraction
except ImportError as _exc:
    _gen7_import_error = str(_exc)
    run_gen7_extraction = None  # type: ignore[assignment]

# Gen 8 / 8.5 pipeline import (lazy; shared module, labeler is a string arg)
_gen8_import_error: str | None = None
try:
    from backend.gen8.pipeline import run as run_gen8_extraction
except ImportError as _exc:
    _gen8_import_error = str(_exc)
    run_gen8_extraction = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Active sessions: in-memory is the primary store.
# Redis is a secondary store for cross-process visibility and TTL-based expiry.
_active_sessions: dict[str, dict] = {}

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SESSION_TTL = 7200  # 2 hours

# Concurrent extraction cap. Each extraction holds ~150-200 MB peak (journal,
# REPL state, Anthropic response buffers). Render Standard has 2 GB RAM and we
# need headroom for FastAPI + Prisma + ingestion. 2 concurrent extractions fit
# comfortably. Configurable via env var for larger plans.
MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "2"))

# Event ring buffer cap per session. A 30-minute, 50-iteration extraction with
# full Anthropic responses can produce 50-100 MB of events. We retain the last
# N in-memory for SSE catch-up + the frontend event log, and persist ALL events
# to Neon via log_repl_step(), so dropping old events from memory is lossless
# from the consumer's perspective — they can always replay from the DB.
MAX_EVENTS_IN_MEMORY = int(os.environ.get("MAX_EVENTS_IN_MEMORY", "500"))


def _count_running_sessions() -> int:
    """Return the number of sessions currently in `running` status."""
    return sum(
        1
        for s in _active_sessions.values()
        if s.get("session_data", {}).get("status") == "running"
    )


async def create_session(
    api_key: str,
    guide_json: dict,
    manual_name: str | None = None,
    source_guide_id: str | None = None,
    pipeline: str = "gen7",
) -> str:
    """Create a new extraction session. Returns the session_id.

    If `source_guide_id` is provided, it's linked to the ExtractionRun so the
    run can be traced back to the ingested SourceGuide row.

    `pipeline` selects the extraction engine:
      - "gen7" (default): Opus-only pipeline (micro-chunk, label, REPL compile)
      - "legacy": Original rlm_runner.py path (hybrid Opus/Sonnet REPL)

    Enforces the MAX_CONCURRENT_SESSIONS cap. If the cap is reached, raises
    HTTPException(429) -- Render Standard's 2 GB memory budget can't hold more
    than ~2 extractions at once safely.
    """
    running = _count_running_sessions()
    if running >= MAX_CONCURRENT_SESSIONS:
        logger.warning(
            "Session creation rejected: %d/%d extractions already running",
            running,
            MAX_CONCURRENT_SESSIONS,
        )
        # User-facing message is deliberately trust-based — this is an
        # internal research tool shared across the research group. The
        # cap is a shared resource; users coordinate out-of-band about who
        # is running what. The message should sound like "your colleagues
        # are busy", not "you exceeded a quota".
        raise HTTPException(
            status_code=429,
            detail={
                "error": "too_many_concurrent_sessions",
                "running": running,
                "limit": MAX_CONCURRENT_SESSIONS,
                "message": (
                    f"The tool is currently busy — {running} of {MAX_CONCURRENT_SESSIONS} "
                    f"extraction slots are in use by other members of the research group. "
                    f"A full extraction takes about 15-25 minutes. Please wait a few "
                    f"minutes and try again, or check with your colleagues about "
                    f"coordinating your runs."
                ),
            },
        )

    session_id = str(uuid.uuid4())
    run_id = f"run_{session_id[:8]}"

    session_data = {
        "session_id": session_id,
        "run_id": run_id,
        "status": "running",
        "pipeline": pipeline,
        "manual_name": manual_name or "unknown",
        "total_iterations": 0,
        "total_subcalls": 0,
        "validation_errors": 0,
        "cost_estimate_usd": 0.0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "cached_tokens_total": 0,
        "calls_opus": 0,
        "calls_sonnet": 0,
        "calls_haiku": 0,
        "cache_write_tokens_total": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Store in-memory (primary)
    _active_sessions[session_id] = {
        "run_id": run_id,
        "task": None,
        "events": [],
        "journal": None,
        "session_data": session_data,
        "api_key": api_key,  # held in memory only
    }

    # Store in Redis (secondary, best-effort)
    await redis_setex(
        f"session:{session_id}",
        SESSION_TTL,
        json.dumps(session_data),
    )

    # Create Neon extraction run record (best-effort)
    try:
        await create_extraction_run(run_id, manual_name, source_guide_id=source_guide_id)
    except Exception as e:
        logger.warning("Could not create Neon run record: %s", e)

    # Launch extraction in background: route to gen7 / gen8 / gen8.5 / legacy
    if pipeline == "gen7":
        if run_gen7_extraction is None:
            raise HTTPException(
                status_code=500,
                detail=f"Gen 7 pipeline unavailable: {_gen7_import_error}",
            )
        task = asyncio.create_task(
            _run_gen7_extraction_task(session_id, run_id, api_key, guide_json, manual_name)
        )
    elif pipeline in ("gen8", "gen8.5"):
        if run_gen8_extraction is None:
            raise HTTPException(
                status_code=500,
                detail=f"Gen 8 pipeline unavailable: {_gen8_import_error}",
            )
        labeler = "sonnet7way" if pipeline == "gen8.5" else "opus"
        task = asyncio.create_task(
            _run_gen8_extraction_task(session_id, run_id, api_key, guide_json,
                                      manual_name, labeler=labeler,
                                      pipeline_name=pipeline,
                                      source_guide_id=source_guide_id)
        )
    else:
        task = asyncio.create_task(
            _run_extraction_task(session_id, run_id, api_key, guide_json, manual_name)
        )
    _active_sessions[session_id]["task"] = task

    logger.info("Session created: %s (run: %s, pipeline: %s)", session_id, run_id, pipeline)
    return session_id


def _merge_final_usage_snapshot(payload: dict) -> None:
    """Merge the latest _run_usage snapshot into an update payload.

    The live-merge in `get_session_status` only fires while status=="running",
    so when the task reaches a terminal state the final cost/token numbers
    need to be captured explicitly. This helper is called from both the
    success and failure paths of `_run_extraction_task` right before
    `_update_session_data`, so the frontend sees the final cost/tokens
    instead of stale/zero values.
    """
    try:
        from backend.rlm_runner import _run_usage
        payload.setdefault("cost_estimate_usd", _run_usage.get("cost_usd", 0.0))
        # Always overwrite with the accumulator's values if they're non-zero
        if _run_usage.get("cost_usd", 0) > payload.get("cost_estimate_usd", 0):
            payload["cost_estimate_usd"] = _run_usage["cost_usd"]
        payload["input_tokens_total"] = _run_usage.get("input_tokens", 0)
        payload["output_tokens_total"] = _run_usage.get("output_tokens", 0)
        payload["cached_tokens_total"] = _run_usage.get("cached_tokens", 0)
        payload["cache_write_tokens_total"] = _run_usage.get("cache_write_tokens", 0)
        payload["calls_opus"] = _run_usage.get("calls_opus", 0)
        payload["calls_sonnet"] = _run_usage.get("calls_sonnet", 0)
        payload["calls_haiku"] = _run_usage.get("calls_haiku", 0)
    except ImportError:
        logger.debug("Final usage snapshot unavailable (rlm patch not loaded)")


def _assemble_from_intermediates(result) -> dict | None:
    """Assemble a clinical_logic dict from an ExtractionResult's per-phase
    intermediate artifacts when the final FINAL_VAR output failed to parse.

    The ExtractionResult carries seven phase artifacts (supply_list, variables,
    predicates, modules, router, integrative, phrase_bank) that were captured
    from `emit_artifact` calls during the run. When the model's FINAL_VAR
    result is None (parse failure) or doesn't match a dict shape, we can
    still produce a usable clinical_logic by stitching those pieces together.
    The converters' `_normalize_for_converters` will handle any dict-vs-list
    shape mismatches downstream.
    """
    names = ("supply_list", "variables", "predicates", "modules",
             "router", "integrative", "phrase_bank")
    assembled: dict = {}
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            assembled[name] = value
    if not assembled:
        return None
    logger.info(
        "Assembled clinical_logic from %d intermediate artifacts: %s",
        len(assembled),
        list(assembled.keys()),
    )
    return assembled


async def _run_extraction_task(
    session_id: str,
    run_id: str,
    api_key: str,
    guide_json: dict,
    manual_name: str | None,
) -> None:
    """Background task that runs the RLM extraction."""
    try:
        # Create the live journal (scratchpad.md)
        artifact_dir = OUTPUT_DIR / run_id
        journal = JournalWriter(artifact_dir, manual_name or "Unknown Guide")
        _active_sessions[session_id]["journal"] = journal

        # Step callback: logs to Neon + writes journal + pushes SSE events
        async def on_step(step_data: dict) -> None:
            step_number = step_data.get("step_number", 0)
            step_type = step_data.get("step_type", "exec")

            # Log to Neon (best-effort)
            try:
                await log_repl_step(
                    run_id=run_id,
                    step_number=step_number,
                    step_type=step_type,
                    code=step_data.get("code"),
                    stdout=step_data.get("stdout"),
                    stderr=step_data.get("stderr"),
                    prompt=step_data.get("prompt"),
                    response=step_data.get("response"),
                    execution_ms=step_data.get("execution_ms"),
                    input_tokens=step_data.get("input_tokens"),
                    output_tokens=step_data.get("output_tokens"),
                )
            except Exception as e:
                logger.debug("Neon log_repl_step failed: %s", e)

            # Increment live counters so the status endpoint reflects progress.
            # NOTE: `step_number` is a monotonic counter across ALL event types
            # (iteration completions, sub-calls, artifact emits). It is NOT the
            # REPL iteration count. A run with 4 iterations and 7 artifact emits
            # per iteration reports step_number≈32, which misleads the frontend's
            # sub-calls derivation (`llm_calls - iterations`). We store
            # step_number separately as `total_events` for SSE ordering and
            # only increment `total_iterations` when an actual exec step fires.
            if session_id in _active_sessions:
                sd = _active_sessions[session_id]["session_data"]
                sd["total_events"] = step_number
                if step_type == "exec":
                    sd["total_iterations"] = sd.get("total_iterations", 0) + 1
                elif step_type == "subcall":
                    sd["total_subcalls"] = sd.get("total_subcalls", 0) + 1
                # Pull live cumulative usage from the rlm_runner accumulator.
                # Step_data carries a snapshot from the last LLM call; we
                # overlay the global accumulator afterwards so catcher-side
                # Haiku calls (which the RLM step callback doesn't know about)
                # still show up in the live counters.
                if step_data.get("cost_usd") is not None:
                    sd["cost_estimate_usd"] = step_data.get("cost_usd", 0.0)
                    sd["input_tokens_total"] = step_data.get("input_tokens", 0)
                    sd["output_tokens_total"] = step_data.get("output_tokens", 0)
                    sd["cached_tokens_total"] = step_data.get("cached_tokens", 0)
                    sd["calls_opus"] = step_data.get("calls_opus", 0)
                    sd["calls_sonnet"] = step_data.get("calls_sonnet", 0)
                try:
                    from backend.rlm_runner import _run_usage as _live_usage
                    sd["calls_haiku"] = _live_usage.get("calls_haiku", 0)
                    sd["cache_write_tokens_total"] = _live_usage.get("cache_write_tokens", 0)
                    # If the accumulator's total cost is ahead of step_data's
                    # (e.g. catcher calls fired between steps), prefer it.
                    if _live_usage.get("cost_usd", 0) > sd.get("cost_estimate_usd", 0):
                        sd["cost_estimate_usd"] = _live_usage["cost_usd"]
                except ImportError:
                    pass

            # Push raw REPL event to SSE
            _push_event(session_id, {
                "type": step_type,
                "stepNumber": step_number,
                "code": step_data.get("code"),
                "stdout": step_data.get("stdout"),
                "stderr": step_data.get("stderr"),
                "prompt": step_data.get("prompt"),
                "response": step_data.get("response"),
                "validationResult": step_data.get("validation_result"),
                "z3Result": step_data.get("z3_result"),
                "tokens": {
                    "input": step_data.get("input_tokens", 0),
                    "output": step_data.get("output_tokens", 0),
                } if step_data.get("input_tokens") else None,
                "executionMs": step_data.get("executionMs") or step_data.get("execution_ms"),
            })

            # Write journal entry and push as separate SSE event
            journal_entry = journal.record_step(
                step_number=step_number,
                step_type=step_type,
                code=step_data.get("code"),
                stdout=step_data.get("stdout"),
                stderr=step_data.get("stderr"),
                prompt=step_data.get("prompt"),
                response=step_data.get("response"),
                validation_result=step_data.get("validation_result"),
                z3_result=step_data.get("z3_result"),
                iteration=step_data.get("iteration"),
                executionMs=step_data.get("executionMs"),
            )
            if journal_entry:
                _push_event(session_id, journal_entry)

        result = await run_extraction(
            guide_json=guide_json,
            api_key=api_key,
            run_id=run_id,
            manual_name=manual_name or "unknown",
            log_dir=str(artifact_dir),
            on_step=on_step,
        )

        # Fix B: soft catcher gate. The model can bypass emit_artifact's
        # critical_issues and call FINAL_VAR() anyway, producing a "passed"
        # run with unfixed criticals. Require each of the 7 required
        # artifacts to have had a PASSING catcher on its most recent emit.
        # If any required artifact never passed, downgrade the run to
        # "failed" and attach an error message. Runs BEFORE _generate_artifacts
        # so failed runs don't produce final DMN/XLSX outputs.
        try:
            from backend.rlm_runner import _snapshot_run_artifact_status
            required = [
                "supply_list",
                "variables",
                "predicates",
                "modules",
                "router",
                "integrative",
                "phrase_bank",
            ]
            status_history = _snapshot_run_artifact_status()
            failed_artifacts = [a for a in required if not status_history.get(a, False)]
            if result.status == "passed" and failed_artifacts:
                logger.warning(
                    "Extraction status downgraded: model called FINAL_VAR but %d artifacts never passed catchers: %s",
                    len(failed_artifacts), failed_artifacts,
                )
                result.status = "failed"
                _update_session_data(session_id, {
                    "error_message": (
                        f"Catcher gate failed: {', '.join(failed_artifacts)} "
                        f"never passed validation"
                    ),
                })
        except ImportError:
            logger.debug("Catcher gate snapshot unavailable (rlm patch not loaded)")

        # Update Neon (best-effort)
        try:
            await update_extraction_run(
                run_id=run_id,
                status=result.status.upper(),
                total_iterations=result.total_iterations,
                total_subcalls=result.total_subcalls,
                validation_errors=result.validation_errors,
                final_json=result.clinical_logic,
                cost_estimate_usd=result.cost_estimate_usd,
            )
        except Exception as e:
            logger.warning("Could not update Neon run record: %s", e)

        # Final artifact rendering. Run this for BOTH passed and failed
        # terminal states as long as we have enough intermediate data to
        # assemble a clinical_logic dict. The converters already tolerate
        # partial shapes, and users want the final DMN/XLSX/Mermaid even
        # when the catcher gate downgraded the run or FINAL_VAR failed.
        final_logic = result.clinical_logic or _assemble_from_intermediates(result)
        if final_logic:
            try:
                await _generate_artifacts(session_id, run_id, final_logic, manual_name)
            except Exception as exc:
                logger.warning("Final artifact generation failed (non-fatal): %s", exc)

        # Push completion event
        _push_event(session_id, {
            "type": "status",
            "stepNumber": -1,
            "status": result.status,
        })

        # Update in-memory + Redis. Snapshot the usage accumulator one last
        # time so the final cost/tokens are captured on the status endpoint
        # even though the live-merge guard in get_session_status only fires
        # while the run is "running".
        update_payload = {
            "status": result.status,
            "total_iterations": result.total_iterations,
            "total_subcalls": result.total_subcalls,
            "validation_errors": result.validation_errors,
            "cost_estimate_usd": result.cost_estimate_usd,
        }
        _merge_final_usage_snapshot(update_payload)
        # Surface error message for failed runs so the frontend can display it
        result_error = getattr(result, "error", None)
        if result_error:
            update_payload["error_message"] = result_error
        _update_session_data(session_id, update_payload)

    except Exception as e:
        error_msg = str(e)
        logger.error("Extraction task failed: session=%s error=%s", session_id, error_msg)
        try:
            await update_extraction_run(run_id=run_id, status="FAILED")
        except Exception:
            pass
        _push_event(session_id, {
            "type": "error",
            "stepNumber": -1,
            "stderr": error_msg,
        })
        failure_payload = {
            "status": "failed",
            "error_message": error_msg,
        }
        _merge_final_usage_snapshot(failure_payload)
        _update_session_data(session_id, failure_payload)


async def _run_gen7_extraction_task(
    session_id: str,
    run_id: str,
    api_key: str,
    guide_json: dict,
    manual_name: str | None,
) -> None:
    """Background task that runs the Gen 7 (Opus-only) extraction pipeline.

    Maps Gen 7's on_progress callback into the same SSE event format that the
    frontend expects, reusing _push_event and the session_data counters so the
    Cost Tracker, progress bar, and SSE stream all work unchanged.
    """
    try:
        artifact_dir = OUTPUT_DIR / run_id
        journal = JournalWriter(artifact_dir, manual_name or "Unknown Guide")
        _active_sessions[session_id]["journal"] = journal

        # Step counter for SSE event numbering
        step_counter = {"count": 0}

        async def on_progress(event: dict) -> None:
            """Translate Gen 7 progress events into the frontend SSE format.

            Gen 7 emits these event types:
              Phase events:  {"phase": "chunk"|"label"|"compile", "status": "starting"|"done", ...}
              Label events:  {"phase": "label", "event": "batch_start"|"chunk_labeled"|"batch_cooldown"|"cache_prime_start", ...}
              REPL events:   {"phase": "compile", "event": "repl_iteration", "iteration": N, "code": "...", "stdout": "..."}
            """
            step_counter["count"] += 1
            phase = event.get("phase", "unknown")
            status = event.get("status", "")
            evt_type = event.get("event", "")

            # --- Labeling granular events ---
            if evt_type == "cache_prime_start":
                _push_event(session_id, {
                    "type": "status", "stepNumber": step_counter["count"],
                    "stdout": f"Priming cache with chunk 0 of {event.get('total_chunks', '?')} "
                              f"(batches of {event.get('batch_size', '?')} after this)",
                })
                journal_entry = journal.record_step(step_counter["count"], "status",
                    stdout=f"## Labeling: cache priming (chunk 0 of {event.get('total_chunks', '?')})")
                if journal_entry:
                    _push_event(session_id, journal_entry)
                return

            if evt_type == "batch_start":
                batch = event.get("batch", 0)
                total = event.get("total_batches", "?")
                n = event.get("chunk_count", 0)
                indices = event.get("chunk_indices", [])
                _push_event(session_id, {
                    "type": "status", "stepNumber": step_counter["count"],
                    "stdout": f"Labeling batch {batch}/{total}: {n} chunks in parallel {indices}",
                })
                journal_entry = journal.record_step(step_counter["count"], "status",
                    stdout=f"## Labeling batch {batch}/{total}: {n} chunks parallel")
                if journal_entry:
                    _push_event(session_id, journal_entry)
                return

            if evt_type == "chunk_labeled":
                ci = event.get("chunk_index", "?")
                lc = event.get("label_count", 0)
                so_far = event.get("labeled_so_far", 0)
                total_c = event.get("total_chunks", "?")
                in_tok = event.get("input_tokens", 0)
                out_tok = event.get("output_tokens", 0)
                err = event.get("error")
                label_text = f"chunk {ci}: {lc} labels" if not err else f"chunk {ci}: ERROR {err}"
                _push_event(session_id, {
                    "type": "exec", "stepNumber": step_counter["count"],
                    "stdout": f"Labeled {label_text} ({so_far}/{total_c} done, {in_tok} in / {out_tok} out tokens)",
                })
                journal_entry = journal.record_step(step_counter["count"], "status",
                    stdout=f"Labeled {label_text} ({so_far}/{total_c} done)")
                if journal_entry:
                    _push_event(session_id, journal_entry)
                # Cost, tokens, and call counts now flow from the authoritative
                # rlm_runner._run_usage accumulator (labeler calls feed it via
                # accumulate_catcher_usage). The get_session_status() live-merge
                # will pull the correct numbers on every status poll. Do NOT
                # manually increment here -- that would double-count.
                return

            if evt_type == "batch_cooldown":
                secs = event.get("cooldown_sec", 62)
                remaining = event.get("remaining_chunks", 0)
                _push_event(session_id, {
                    "type": "status", "stepNumber": step_counter["count"],
                    "stdout": f"Rate limit cooldown: {secs}s ({remaining} chunks remaining)",
                })
                return

            # --- REPL iteration events ---
            if evt_type == "repl_iteration":
                iteration = event.get("iteration", 0)
                duration = event.get("duration_sec", 0)
                code = event.get("code", "")
                stdout = event.get("stdout", "")
                iter_in_tok = event.get("input_tokens", 0) or 0
                iter_out_tok = event.get("output_tokens", 0) or 0
                _push_event(session_id, {
                    "type": "exec", "stepNumber": step_counter["count"],
                    "code": code, "stdout": stdout,
                    "executionMs": int(duration * 1000),
                })
                journal_entry = journal.record_step(
                    step_counter["count"], "exec",
                    code=code, stdout=stdout[:2000])
                if journal_entry:
                    _push_event(session_id, journal_entry)
                # Keep total_iterations in sync here (not in _run_usage);
                # cost, tokens, and call counts come from _run_usage via the
                # live-merge in get_session_status() -- don't double-count.
                if session_id in _active_sessions:
                    sd = _active_sessions[session_id]["session_data"]
                    sd["total_iterations"] = iteration
                return

            # --- Phase-level events (starting/done) ---
            sse_event: dict = {
                "type": "status",
                "stepNumber": step_counter["count"],
                "phase": phase,
                "phaseStatus": status,
            }

            if phase == "chunk" and status == "done":
                sse_event["chunkCount"] = event.get("count", 0)
                sse_event["stdout"] = f"Created {event.get('count', 0)} micro-chunks (~2K tokens each)"
            elif phase == "label" and status == "starting":
                sse_event["stdout"] = "Starting Opus labeling phase (codebook annotation)"
            elif phase == "label" and status == "done":
                sse_event["totalLabels"] = event.get("total_labels", 0)
                sse_event["labelErrors"] = event.get("errors", 0)
                sse_event["stdout"] = (
                    f"Labeling complete: {event.get('total_labels', 0)} labels, "
                    f"{event.get('errors', 0)} errors"
                )
            elif phase == "compile" and status == "starting":
                sse_event["stdout"] = "Starting Opus REPL compilation (DAG order: 7 artifacts)"
            elif phase == "compile" and status == "done":
                sse_event["elapsed"] = event.get("elapsed", 0)
                sse_event["stdout"] = (
                    f"Compilation complete in {event.get('elapsed', 0)}s "
                    f"({event.get('total_iterations', '?')} REPL iterations)"
                )

            _push_event(session_id, sse_event)

            journal_text = f"## {phase.title()} - {status}\n"
            if phase == "chunk" and status == "done":
                journal_text += f"Created {event.get('count', 0)} micro-chunks\n"
            elif phase == "label" and status == "done":
                journal_text += (
                    f"Labeled {event.get('total_labels', 0)} spans "
                    f"({event.get('errors', 0)} errors)\n"
                )
            elif phase == "compile" and status == "done":
                journal_text += f"Compilation complete in {event.get('elapsed', 0)}s\n"

            journal_entry = journal.record_step(
                step_number=step_counter["count"],
                step_type="status",
                stdout=journal_text,
            )
            if journal_entry:
                _push_event(session_id, journal_entry)

        # Load the naming codebook for labeling
        try:
            from backend.system_prompt import NAMING_CODEBOOK
        except ImportError:
            NAMING_CODEBOOK = ""

        # Run Gen 7 pipeline. Pass manual_name as a hint so the bundle README
        # shows the source manual; the pipeline previously only read it from
        # guide_json.metadata.manual_name, which is empty for the cached Neon
        # ingestion shape.
        gen7_result = await run_gen7_extraction(
            guide_json=guide_json,
            anthropic_key=api_key,
            naming_codebook=NAMING_CODEBOOK,
            output_dir=artifact_dir,
            on_progress=on_progress,
            manual_name_hint=manual_name,
        )

        clinical_logic = gen7_result.get("clinical_logic")
        stats = gen7_result.get("stats", {})
        status = gen7_result.get("status", "failed")

        # Update session data with Gen 7 cost/token stats
        # total_cost_usd from pipeline.py already includes both label + REPL cost
        cost_usd = stats.get("total_cost_usd", 0.0)
        # Total tokens = labeling + REPL. Previously this only counted label
        # tokens, under-reporting usage in the UI by the REPL share.
        input_tokens = stats.get("label_input_tokens", 0) + stats.get("repl_input_tokens", 0)
        output_tokens = stats.get("label_output_tokens", 0) + stats.get("repl_output_tokens", 0)
        repl_iterations = stats.get("repl_iterations", 0)

        if session_id in _active_sessions:
            sd = _active_sessions[session_id]["session_data"]
            sd["cost_estimate_usd"] = cost_usd
            sd["input_tokens_total"] = input_tokens
            sd["output_tokens_total"] = output_tokens
            sd["calls_opus"] = (
                stats.get("total_chunks", 0)  # one Opus call per chunk (labeling)
                + repl_iterations             # Opus REPL iterations
            )
            sd["total_iterations"] = repl_iterations

        # Update Neon extraction run (best-effort)
        try:
            await update_extraction_run(
                run_id=run_id,
                status=status.upper(),
                total_iterations=repl_iterations,
                total_subcalls=stats.get("total_chunks", 0),
                validation_errors=stats.get("label_errors", 0),
                final_json=clinical_logic,
                cost_estimate_usd=cost_usd,
            )
        except Exception as e:
            logger.warning("Could not update Neon run record (gen7): %s", e)

        # Generate downloadable artifacts (DMN, XLSX, Mermaid, CSV)
        # Gen 7's _save_outputs already writes these inside the pipeline, but
        # we also run the session_manager's _generate_artifacts to get the
        # full artifact bundle (intermediate .validator.json sidecars, DMN
        # validation, flowchart.png rendering, etc.)
        if clinical_logic and isinstance(clinical_logic, dict):
            try:
                await _generate_artifacts(session_id, run_id, clinical_logic, manual_name)
            except Exception as exc:
                logger.warning("Gen7 artifact generation failed (non-fatal): %s", exc)

        # Bug fix: update session status BEFORE pushing the final SSE event.
        # The frontend polls /status every 3s and shows artifacts only when
        # status === "passed". If we push the final event first, the SSE
        # stream ends and the frontend may disconnect before the next poll
        # picks up the updated status.
        _update_session_data(session_id, {
            "status": status,
            "total_iterations": repl_iterations,
            "total_subcalls": stats.get("total_chunks", 0),
            "validation_errors": stats.get("label_errors", 0),
            "cost_estimate_usd": cost_usd,
            "input_tokens_total": input_tokens,
            "output_tokens_total": output_tokens,
        })

        # Push completion event (after status is already persisted)
        _push_event(session_id, {
            "type": "final",
            "stepNumber": -1,
            "status": status,
            "costEstimateUsd": cost_usd,
            "inputTokensTotal": input_tokens,
            "outputTokensTotal": output_tokens,
            "stats": stats,
        })

    except Exception as e:
        error_msg = str(e)
        logger.error("Gen7 extraction task failed: session=%s error=%s", session_id, error_msg)
        try:
            await update_extraction_run(run_id=run_id, status="FAILED")
        except Exception:
            pass
        _push_event(session_id, {
            "type": "error",
            "stepNumber": -1,
            "stderr": error_msg,
        })
        _update_session_data(session_id, {
            "status": "failed",
            "error_message": error_msg,
        })


async def _run_gen8_extraction_task(
    session_id: str,
    run_id: str,
    api_key: str,
    guide_json: dict,
    manual_name: str | None,
    labeler: str = "opus",
    pipeline_name: str = "gen8",
    source_guide_id: str | None = None,
) -> None:
    """Run the gen8 / gen8.5 pipeline and forward progress to SSE.

    Mirrors `_run_gen7_extraction_task`. Uses the same on_progress event
    shapes so the frontend progress tracker needs no changes.

    `source_guide_id` is looked up in Neon to populate the manifest's
    real PDF SHA + page count + publisher year. Without this, the
    container hash is anchored on the string "unknown" instead of the
    bytes of the actual source PDF.
    """
    # Pre-fetch SourceGuide metadata so the pipeline can populate the
    # manifest with the real PDF SHA. Best-effort: if Neon is unavailable
    # the pipeline falls back to whatever guide_json.metadata contains.
    source_guide_meta: dict | None = None
    if source_guide_id:
        try:
            from backend.ingestion.cache import find_by_id
            sg = await find_by_id(source_guide_id)
            if sg:
                source_guide_meta = {
                    "sha256": sg.get("sha256"),
                    "manualName": sg.get("manualName") or sg.get("filename"),
                    "pageCount": sg.get("pageCount"),
                    "neon_guide_id": sg.get("id"),
                }
        except Exception as exc:
            logger.warning("gen8: SourceGuide lookup failed: %s", exc)

    try:
        artifact_dir = OUTPUT_DIR / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        journal = JournalWriter(artifact_dir, manual_name or "Unknown Guide")
        _active_sessions[session_id]["journal"] = journal

        step_counter = {"count": 0}

        async def on_progress(event: dict) -> None:
            step_counter["count"] += 1
            phase = event.get("phase", "unknown")
            status = event.get("status", "")
            evt_type = event.get("event", "")

            if evt_type == "batch_start":
                n = event.get("chunk_count", 0)
                indices = event.get("chunk_indices", [])
                _push_event(session_id, {
                    "type": "status", "stepNumber": step_counter["count"],
                    "stdout": f"{pipeline_name}: labeling {n} chunks in parallel {indices}",
                })
                return

            if evt_type == "chunk_labeled":
                ci = event.get("chunk_index", "?")
                lc = event.get("label_count", 0)
                so_far = event.get("labeled_so_far", 0)
                total_c = event.get("total_chunks", "?")
                in_tok = event.get("input_tokens", 0)
                out_tok = event.get("output_tokens", 0)
                err = event.get("error")
                label_text = f"chunk {ci}: {lc} labels" if not err else f"chunk {ci}: ERROR {err}"
                _push_event(session_id, {
                    "type": "exec", "stepNumber": step_counter["count"],
                    "stdout": f"Labeled {label_text} ({so_far}/{total_c} done, "
                              f"{in_tok} in / {out_tok} out tokens)",
                })
                return

            # Phase-level events
            sse_event: dict = {
                "type": "status",
                "stepNumber": step_counter["count"],
                "phase": phase,
                "phaseStatus": status,
            }
            if phase == "chunk" and status == "done":
                sse_event["chunkCount"] = event.get("count", 0)
                sse_event["stdout"] = f"Created {event.get('count', 0)} micro-chunks (~2K tokens each)"
            elif phase == "label" and status == "starting":
                model = event.get("model", "")
                sse_event["stdout"] = f"Starting {pipeline_name} labeling ({model})"
            elif phase == "label" and status == "done":
                sse_event["totalLabels"] = event.get("total_labels", 0)
                sse_event["stdout"] = f"Labeling complete: {event.get('total_labels', 0)} labels"
            elif phase == "compile" and status == "starting":
                sse_event["stdout"] = "Starting Opus REPL compilation (gen8 rules active)"
            elif phase == "compile" and status == "done":
                sse_event["elapsed"] = event.get("elapsed", 0)
                sse_event["stdout"] = f"Compilation complete in {event.get('elapsed', 0)}s"
            _push_event(session_id, sse_event)

        try:
            from backend.system_prompt import NAMING_CODEBOOK
        except ImportError:
            NAMING_CODEBOOK = ""

        gen8_result = await run_gen8_extraction(
            guide_json=guide_json,
            anthropic_key=api_key,
            naming_codebook=NAMING_CODEBOOK,
            output_dir=artifact_dir,
            on_progress=on_progress,
            manual_name_hint=manual_name,
            labeler=labeler,
            source_guide_meta=source_guide_meta,
        )

        clinical_logic = gen8_result.get("clinical_logic")
        stats = gen8_result.get("stats", {})
        status = gen8_result.get("status", "failed")
        container_sha = gen8_result.get("container_sha", "")
        verification_summary = gen8_result.get("verification_summary", {})

        # Run the legacy artifact generator so the frontend's artifact list
        # picks up the gen7-style validator sidecars + flowchart.png + DMN
        # validator output. gen8/pipeline.py already writes the modern split
        # artifacts (form.xlsx, form_per_module/, flowchart_index.png, etc.);
        # _generate_artifacts adds the items the frontend hardcoded paths for.
        if clinical_logic and isinstance(clinical_logic, dict):
            try:
                await _generate_artifacts(session_id, run_id, clinical_logic, manual_name)
            except Exception as exc:
                logger.warning("%s artifact generation failed (non-fatal): %s",
                               pipeline_name, exc)

        cost_usd = stats.get("total_cost_usd", 0.0)
        input_tokens = stats.get("total_input_tokens", 0)
        output_tokens = stats.get("total_output_tokens", 0)

        if session_id in _active_sessions:
            sd = _active_sessions[session_id]["session_data"]
            sd["cost_estimate_usd"] = cost_usd
            sd["input_tokens_total"] = input_tokens
            sd["output_tokens_total"] = output_tokens

        try:
            await update_extraction_run(
                run_id=run_id,
                status=status.upper(),
                total_iterations=0,
                total_subcalls=stats.get("total_chunks", 0),
                validation_errors=stats.get("label_errors", 0),
                final_json=clinical_logic,
                cost_estimate_usd=cost_usd,
            )
        except Exception as e:
            logger.warning("Could not update Neon run record (%s): %s", pipeline_name, e)

        _update_session_data(session_id, {
            "status": status,
            "total_subcalls": stats.get("total_chunks", 0),
            "validation_errors": stats.get("label_errors", 0),
            "cost_estimate_usd": cost_usd,
            "input_tokens_total": input_tokens,
            "output_tokens_total": output_tokens,
        })

        _push_event(session_id, {
            "type": "final",
            "stepNumber": -1,
            "status": status,
            "costEstimateUsd": cost_usd,
            "inputTokensTotal": input_tokens,
            "outputTokensTotal": output_tokens,
            "containerSha": container_sha,
            "verification": verification_summary,
            "stats": stats,
        })

    except Exception as e:
        error_msg = str(e)
        logger.error("%s extraction task failed: session=%s error=%s",
                     pipeline_name, session_id, error_msg)
        try:
            await update_extraction_run(run_id=run_id, status="FAILED")
        except Exception:
            pass
        _push_event(session_id, {
            "type": "error", "stepNumber": -1, "stderr": error_msg,
        })
        _update_session_data(session_id, {
            "status": "failed", "error_message": error_msg,
        })


async def _generate_artifacts(
    session_id: str,
    run_id: str,
    clinical_logic: dict,
    manual_name: str | None,
) -> None:
    """Generate all output artifacts from the validated clinical logic.

    The four converters (DMN, Mermaid, CSV, XLSX) are sync CPU work, so we
    run them concurrently via asyncio.to_thread + asyncio.gather. XLSX writes
    to disk from inside the converter; the rest return strings/dicts we write
    here after the gather completes.

    Two extra artifacts are saved alongside the converters:
      - clinical_logic.json: the raw RLM output, source of truth for everything else
      - system_prompt.md: the full assembled system prompt that drove the extraction,
        useful for reproducibility and prompt iteration
    """
    artifact_dir = OUTPUT_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Raw clinical logic JSON (source of truth). All artifact writes below
    # MUST use encoding="utf-8" — on Windows, Path.write_text defaults to
    # cp1252, which fails to encode any non-ASCII characters the downstream
    # converters may emit. Keep the explicit encoding on every write even
    # though the mermaid converter is now ASCII-only (gen 2.4 cleanup) —
    # the cost is zero and this protects against regressions.
    json_path = artifact_dir / "clinical_logic.json"
    json_path.write_text(json.dumps(clinical_logic, indent=2), encoding="utf-8")

    # System prompt (assembled at extraction time, deterministic but version-tied
    # to the current code). Saved as Markdown so it renders nicely in the download.
    # Gen 7's bundle writer ships the multi-stage sendable version (Stage 1
    # labeling + Stage 2 distillation + Stage 3 REPL + naming codebook) and
    # writes it BEFORE this point. Detect that and don't clobber it with the
    # legacy single-prompt format. Marker: the sendable version contains
    # "## Stage 1: Chunk labeling" because that's the H2 the bundle uses.
    try:
        prompt_path = artifact_dir / "system_prompt.md"
        if prompt_path.exists() and "## Stage 1: Chunk labeling" in prompt_path.read_text(encoding="utf-8"):
            logger.info("system_prompt.md already contains the gen7 sendable bundle; not overwriting")
        else:
            from backend.system_prompt import build_system_prompt
            system_prompt_text = build_system_prompt()
            prompt_path.write_text(
                f"# CHW Navigator System Prompt\n\n"
                f"_Run ID: {run_id}_\n"
                f"_Manual: {manual_name or 'unknown'}_\n\n"
                f"---\n\n"
                f"{system_prompt_text}\n",
                encoding="utf-8",
            )
    except Exception as e:
        logger.warning("Failed to write system_prompt.md: %s", e)

    # Intermediate artifacts from the hybrid-plan phase checkpoints.
    # The emit_artifact() tool already wrote these to {artifact_dir}/artifacts/
    # during the run; this section pulls the LATEST version of each from Neon
    # (in case the model re-emitted after fixing critical_issues) and ensures
    # all seven files exist in the bundle.
    try:
        from backend.db import get_latest_intermediate_artifacts
        latest = await get_latest_intermediate_artifacts(run_id)
        intermediates_dir = artifact_dir / "artifacts"
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        for name in ("supply_list", "variables", "predicates", "modules",
                     "router", "integrative", "phrase_bank"):
            row = latest.get(name)
            if not row:
                continue
            artifact_json = row.get("artifactJson") or row.get("artifact_json") or {}
            (intermediates_dir / f"{name}.json").write_text(
                json.dumps(artifact_json, indent=2), encoding="utf-8"
            )
            # Also write a sidecar with the validator result for auditability
            critical = row.get("criticalIssues") or row.get("critical_issues") or []
            warnings = row.get("warnings") or []
            passed = row.get("validatorPassed") or row.get("validator_passed") or False
            (intermediates_dir / f"{name}.validator.json").write_text(
                json.dumps({
                    "artifact_name": name,
                    "phase": row.get("phase"),
                    "passed": passed,
                    "critical_issues": critical,
                    "warnings": warnings,
                }, indent=2),
                encoding="utf-8",
            )
    except Exception as e:
        logger.warning("Failed to bundle intermediate artifacts: %s", e)

    # Normalize clinical_logic shape for the converters. The model often
    # returns list-typed artifacts as dicts keyed by their id (e.g.,
    # `predicates: {"p_fever": {...}, "p_cough": {...}}` instead of
    # `predicates: [{predicate_id: "p_fever", ...}, ...]`). The converters
    # expect lists. Normalize here so both shapes work.
    normalized = _normalize_for_converters(clinical_logic)

    xlsx_path = str(artifact_dir / "form.xlsx")
    # Run converters individually so a single converter failure doesn't
    # destroy the whole download bundle. Each runs in its own try/except
    # so other converters still produce their outputs.
    converter_tasks = {
        "dmn": asyncio.to_thread(convert_to_dmn, normalized),
        "mermaid": asyncio.to_thread(convert_to_mermaid, normalized),
        "csv": asyncio.to_thread(convert_to_csv, normalized),
        "xlsx": asyncio.to_thread(convert_to_xlsx, normalized, xlsx_path, manual_name or "CHW Navigator"),
    }
    results = await asyncio.gather(*converter_tasks.values(), return_exceptions=True)
    dmn_xml = results[0] if not isinstance(results[0], Exception) else None
    mermaid = results[1] if not isinstance(results[1], Exception) else None
    csvs = results[2] if not isinstance(results[2], Exception) else None
    # Log any converter failures so the user sees what's missing
    for name, result in zip(converter_tasks.keys(), results):
        if isinstance(result, Exception):
            logger.warning(
                "Converter %s failed: %s -- skipping that artifact",
                name, result,
            )

    if dmn_xml:
        (artifact_dir / "clinical_logic.dmn").write_text(dmn_xml, encoding="utf-8")
    if mermaid:
        (artifact_dir / "flowchart.md").write_text(mermaid, encoding="utf-8")
        # Also render the mermaid to a PNG so users get a ready-to-view image
        # in the ZIP bundle without needing to paste the .md into an online
        # renderer. Uses mermaid.ink first, kroki.io second, and falls back
        # to a placeholder PNG if both are blocked/unreachable. Runs in a
        # worker thread so the sync HTTP calls don't block the event loop.
        try:
            from backend.converters.mermaid_to_png import write_flowchart_png
            png_path = artifact_dir / "flowchart.png"
            render_status = await asyncio.to_thread(
                write_flowchart_png, mermaid, png_path
            )
            logger.info(
                "flowchart.png written via %s (%d bytes)",
                render_status,
                png_path.stat().st_size,
            )
        except Exception as exc:
            logger.warning("flowchart.png generation failed: %s", exc)
    if csvs:
        (artifact_dir / "predicates.csv").write_text(csvs.get("predicates", ""), encoding="utf-8")
        (artifact_dir / "phrases.csv").write_text(csvs.get("phrases", ""), encoding="utf-8")

    # Final DMN validation via the frozen DMN-focused catchers. Only runs if
    # the DMN converter succeeded -- otherwise there's nothing to validate.
    if dmn_xml:
        try:
            from backend.validators.phases import validate_final_dmn
            api_key_for_catchers = _active_sessions.get(session_id, {}).get("api_key")
            if api_key_for_catchers:
                final_dmn_result = await validate_final_dmn(dmn_xml, api_key_for_catchers)
                (artifact_dir / "final_dmn.validator.json").write_text(
                    json.dumps(final_dmn_result.to_dict(), indent=2), encoding="utf-8"
                )
        except Exception as e:
            logger.warning("Final DMN validation failed (non-fatal): %s", e)

    logger.info("Artifacts generated for session %s at %s", session_id, artifact_dir)


def _normalize_for_converters(clinical_logic) -> dict:
    """Normalize clinical_logic shape so the converters can consume it.

    The converters expect list-of-dicts for predicates, modules, supply_list,
    variables, phrase_bank. The model frequently returns these as dicts keyed
    by id (because it's building them incrementally in Python). Convert to
    the list shape by injecting the key as the id field if the dict values
    don't already have the id.

    Defensive: if `clinical_logic` is not a dict (e.g. FINAL_VAR returned a
    string that got past the type guard, or the caller passed an
    _assemble_from_intermediates result that was somehow empty), return an
    empty dict so the converters no-op instead of crashing on `dict(...)`.
    """
    if not isinstance(clinical_logic, dict):
        logger.warning(
            "_normalize_for_converters received non-dict (%s) -- returning {}",
            type(clinical_logic).__name__,
        )
        return {}
    normalized = dict(clinical_logic)  # shallow copy

    id_field_by_name = {
        "predicates": "predicate_id",
        "modules": "module_id",
        "supply_list": "id",
        "variables": "id",
        "phrase_bank": "message_id",
    }

    for name, id_field in id_field_by_name.items():
        value = normalized.get(name)
        if isinstance(value, dict) and value:
            # Check if it looks like a dict-keyed-by-id (values are dicts)
            first_val = next(iter(value.values()))
            if isinstance(first_val, dict):
                converted = []
                for key, entry in value.items():
                    if isinstance(entry, dict):
                        if id_field not in entry:
                            entry = {**entry, id_field: key}
                        converted.append(entry)
                normalized[name] = converted

    return normalized


def _push_event(session_id: str, event: dict) -> None:
    """Push an event to the session's in-memory event list.

    Enforces a ring-buffer cap at MAX_EVENTS_IN_MEMORY entries. Events evicted
    from memory are NOT lost — every REPL step is persisted to Neon via
    log_repl_step(), so consumers can always replay the full trajectory by
    querying repl_steps. The in-memory buffer is for SSE catch-up when a
    client reconnects mid-run and for the frontend's event log pane.

    We also track `events_dropped` so the frontend can display "N earlier
    events not shown, full history in Neon".
    """
    session = _active_sessions.get(session_id)
    if session is None:
        return

    event["id"] = str(uuid.uuid4())
    event["timestamp"] = datetime.now(timezone.utc).isoformat()

    events = session["events"]
    events.append(event)

    # Evict oldest events if over the cap. Pop from the front in a loop so
    # we handle the (unusual) case where the cap shrinks at runtime.
    dropped = 0
    while len(events) > MAX_EVENTS_IN_MEMORY:
        events.pop(0)
        dropped += 1

    if dropped > 0:
        session["events_dropped"] = session.get("events_dropped", 0) + dropped
        # Adjust any SSE consumer's sent_count so the next yield picks up from
        # the new front of the buffer instead of crashing or skipping events.
        # We track this via a session-level offset that get_session_events
        # uses to translate sent_count into a live index.
        session["events_offset"] = session.get("events_offset", 0) + dropped


def _update_session_data(session_id: str, updates: dict) -> None:
    """Update session data in-memory and fire-and-forget to Redis."""
    if session_id not in _active_sessions:
        return
    data = _active_sessions[session_id]["session_data"]
    data.update(updates)
    # Best-effort Redis update (don't await, don't block)
    asyncio.ensure_future(
        redis_setex(f"session:{session_id}", SESSION_TTL, json.dumps(data))
    )


async def get_session_events(session_id: str) -> AsyncGenerator[dict, None]:
    """Stream session events as they arrive (for SSE).

    Uses a global index (events_seen_global) rather than a list index so the
    ring-buffer eviction in _push_event doesn't cause us to skip or repeat
    events. If older events have been evicted before we started streaming,
    we pick up from whatever's currently in the buffer and log a drop warning.
    """
    events_seen_global = 0
    while True:
        session = _active_sessions.get(session_id)
        if session is None:
            return

        events = session["events"]
        offset = session.get("events_offset", 0)
        # events[i] corresponds to global index (offset + i).
        # We want to yield every global index >= events_seen_global.
        if events_seen_global < offset:
            # We've missed some events due to eviction. Jump to the current
            # front of the buffer and log.
            missed = offset - events_seen_global
            logger.warning(
                "SSE session %s missed %d evicted events; resuming from current buffer",
                session_id,
                missed,
            )
            events_seen_global = offset

        # Translate global index to local index.
        local_start = events_seen_global - offset
        while local_start < len(events):
            yield events[local_start]
            local_start += 1
            events_seen_global += 1

        task = session.get("task")
        if task and task.done():
            # Final drain after task completes.
            events = session["events"]
            offset = session.get("events_offset", 0)
            local_start = max(0, events_seen_global - offset)
            while local_start < len(events):
                yield events[local_start]
                local_start += 1
                events_seen_global += 1
            return

        await asyncio.sleep(0.5)


async def get_session_status(session_id: str) -> dict | None:
    """Get current session status. In-memory first, Redis fallback.

    If the session is still running, this also merges the latest live
    cumulative usage from rlm_runner's `_run_usage` accumulator so the
    frontend polling sees cost/token updates on every LLM call, not just
    on iteration boundaries. Without this, a long iteration that fires
    many sub-calls via `llm_query_batched` would look frozen on the
    frontend for minutes at a time even though cost is accruing.
    """
    if session_id in _active_sessions:
        session_data = _active_sessions[session_id]["session_data"]

        # Live merge from the global usage accumulator (while running).
        # As of Gen 7 v2, the labeler feeds its direct-to-Anthropic calls
        # into _run_usage via accumulate_catcher_usage, and the REPL client's
        # monkey-patched _accumulate_run_usage catches everything else (REPL
        # turns + llm_query_batched sub-calls). So _run_usage is the
        # authoritative source for BOTH legacy and Gen 7 runs.
        if session_data.get("status") == "running":
            try:
                from backend.rlm_runner import _run_usage
                # The accumulator is process-global -- safe because we only
                # have ONE extraction running at a time (enforced by the
                # concurrent session limit in session_manager). If we ever
                # lift that restriction, this needs to become per-session.
                if _run_usage.get("cost_usd", 0) > session_data.get("cost_estimate_usd", 0):
                    session_data["cost_estimate_usd"] = _run_usage["cost_usd"]
                    session_data["input_tokens_total"] = _run_usage["input_tokens"]
                    session_data["output_tokens_total"] = _run_usage["output_tokens"]
                    session_data["cached_tokens_total"] = _run_usage["cached_tokens"]
                    session_data["cache_write_tokens_total"] = _run_usage.get("cache_write_tokens", 0)
                    session_data["calls_opus"] = _run_usage["calls_opus"]
                    session_data["calls_sonnet"] = _run_usage["calls_sonnet"]
                    session_data["calls_haiku"] = _run_usage.get("calls_haiku", 0)
                # Also merge call counts even when cost hasn't ticked up yet
                # (catchers are cheap -- a run of 3 Haiku calls can finish for
                # under a cent, which wouldn't trip the cost-threshold branch
                # above). This keeps the Haiku counter live on the frontend.
                elif _run_usage.get("calls_haiku", 0) > session_data.get("calls_haiku", 0):
                    session_data["calls_haiku"] = _run_usage["calls_haiku"]
                    session_data["cost_estimate_usd"] = _run_usage.get("cost_usd", 0.0)
                    session_data["input_tokens_total"] = _run_usage.get("input_tokens", 0)
                    session_data["output_tokens_total"] = _run_usage.get("output_tokens", 0)
                    session_data["cached_tokens_total"] = _run_usage.get("cached_tokens", 0)
                    session_data["cache_write_tokens_total"] = _run_usage.get("cache_write_tokens", 0)
            except Exception as exc:
                logger.debug("Failed to merge live usage into status: %s", exc)

        return session_data

    # Fallback to Redis
    data = await redis_get(f"session:{session_id}")
    if data:
        return json.loads(data)
    return None


async def get_session_artifacts(session_id: str) -> list[dict]:
    """Get download info for session artifacts.

    Returns the top-level files (scratchpad, clinical_logic, system prompt,
    DMN/Mermaid/XLSX/CSVs) plus the seven hybrid-plan intermediate artifacts
    under the artifacts/ subdirectory plus their validator sidecars plus the
    final DMN validator report.
    """
    if session_id not in _active_sessions:
        return []

    run_id = _active_sessions[session_id]["run_id"]
    artifact_dir = OUTPUT_DIR / run_id

    if not artifact_dir.exists():
        return []

    artifacts = []
    # Top-level files: one entry per relative path → (type_id, human label)
    file_types = {
        "scratchpad.md": ("journal", "Extraction Journal"),
        "clinical_logic.json": ("json", "Clinical Logic (JSON)"),
        "system_prompt.md": ("system_prompt", "System Prompt (Markdown)"),
        "clinical_logic.dmn": ("dmn", "DMN XML"),
        "form.xlsx": ("xlsx", "XLSForm (XLSX)"),
        "flowchart.md": ("mermaid", "Mermaid Flowchart (source)"),
        "flowchart.png": ("mermaid_png", "Mermaid Flowchart (rendered)"),
        "predicates.csv": ("predicates_csv", "Predicates (CSV)"),
        "phrases.csv": ("phrases_csv", "Phrase Bank (CSV)"),
        "final_dmn.validator.json": ("final_dmn_validator", "Final DMN Validator Report"),
        # Gen 7 specific artifacts
        "labeled_chunks.json": ("labeled_chunks", "Labeled Chunks (JSON)"),
        "chunk_difficulty.json": ("chunk_difficulty", "Chunk Difficulty Analysis (JSON)"),
        "test_suite.json": ("test_suite", "Test Suite (JSON)"),
        "deduped_labels.json": ("deduped_labels", "Deduped Label Inventory (JSON)"),
        "reconstructed_guide.txt": ("reconstructed_guide", "Reconstructed Guide Text"),
    }

    # Hybrid-plan intermediate artifacts in artifacts/ subdirectory
    intermediate_names = [
        "supply_list", "variables", "predicates", "modules",
        "router", "integrative", "phrase_bank",
    ]
    for name in intermediate_names:
        file_types[f"artifacts/{name}.json"] = (
            f"artifact_{name}",
            f"Artifact: {name.replace('_', ' ').title()}",
        )
        file_types[f"artifacts/{name}.validator.json"] = (
            f"artifact_{name}_validator",
            f"Validator: {name.replace('_', ' ').title()}",
        )

    # gen8 / gen8.5 additions. Probed by existence so older gen7 runs are
    # unaffected and we never advertise a file that wasn't emitted.
    gen8_file_types = {
        "manifest.json": ("manifest", "Run Manifest (Container Hash)"),
        "divergence_worklist.json": ("divergence_worklist", "Verifier Divergence Worklist"),
        "flowchart_index.md": ("flowchart_index", "Flowchart Index (source)"),
        "flowchart_index.png": ("flowchart_index_png", "Flowchart Index (rendered)"),
        "artifacts/data_flow.json": ("artifact_data_flow", "Artifact: Data Flow"),
        "artifacts/referential_integrity.json": ("artifact_ref_integrity", "Artifact: Referential Integrity"),
        "artifacts/predicates_validation.json": ("artifact_predicates_validation", "Artifact: Predicates Validation"),
        "artifacts/stockout_coverage.json": ("artifact_stockout_coverage", "Artifact: Stockout Coverage"),
    }
    file_types.update(gen8_file_types)

    for relative_path, (type_id, label) in file_types.items():
        path = artifact_dir / relative_path
        if path.exists():
            artifacts.append({
                "type": type_id,
                "filename": path.name,
                # relativePath is the path under OUTPUT_DIR/{run_id}/ that the
                # download endpoint uses to locate the file. Keeps the
                # artifacts/ subdirectory structure discoverable by server.py.
                "relativePath": relative_path,
                "label": label,
                "downloadUrl": f"/api/session/{session_id}/artifacts/{type_id}",
            })

    # Per-module xlsx + flowcharts (gen8/gen8.5). Glob-discovered so the
    # listing scales with however many modules the run produced.
    per_module_xlsx_dir = artifact_dir / "form_per_module"
    if per_module_xlsx_dir.exists():
        for path in sorted(per_module_xlsx_dir.glob("*.xlsx")):
            mid = path.stem.removeprefix("form_")
            type_id = f"form_module_{mid}"
            artifacts.append({
                "type": type_id,
                "filename": path.name,
                "relativePath": f"form_per_module/{path.name}",
                "label": f"XLSForm (module {mid})",
                "downloadUrl": f"/api/session/{session_id}/artifacts/{type_id}",
            })
    per_module_flow_dir = artifact_dir / "flowcharts_per_module"
    if per_module_flow_dir.exists():
        for path in sorted(per_module_flow_dir.glob("*.png")):
            mid = path.stem.removeprefix("flowchart_")
            type_id = f"flowchart_module_{mid}_png"
            artifacts.append({
                "type": type_id,
                "filename": path.name,
                "relativePath": f"flowcharts_per_module/{path.name}",
                "label": f"Flowchart (module {mid}, rendered)",
                "downloadUrl": f"/api/session/{session_id}/artifacts/{type_id}",
            })
        for path in sorted(per_module_flow_dir.glob("*.md")):
            mid = path.stem.removeprefix("flowchart_")
            type_id = f"flowchart_module_{mid}_md"
            artifacts.append({
                "type": type_id,
                "filename": path.name,
                "relativePath": f"flowcharts_per_module/{path.name}",
                "label": f"Flowchart (module {mid}, source)",
                "downloadUrl": f"/api/session/{session_id}/artifacts/{type_id}",
            })

    return artifacts


async def cancel_session(session_id: str) -> bool:
    """Cancel a running session."""
    if session_id not in _active_sessions:
        return False

    task = _active_sessions[session_id].get("task")
    if task and not task.done():
        task.cancel()

    run_id = _active_sessions[session_id]["run_id"]
    try:
        await update_extraction_run(run_id=run_id, status="HALTED")
    except Exception:
        pass

    _update_session_data(session_id, {"status": "halted"})
    logger.info("Session cancelled: %s", session_id)
    return True


async def cleanup_session(session_id: str) -> None:
    """Clean up session state."""
    await redis_delete(f"session:{session_id}")
    await redis_delete(f"session:{session_id}:key")
    _active_sessions.pop(session_id, None)
    logger.info("Session cleaned up: %s", session_id)
