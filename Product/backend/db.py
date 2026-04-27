"""Database layer using Prisma Client Python.

Handles all Neon Postgres interactions for logging extraction runs and REPL steps.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from prisma import Json, Prisma

logger = logging.getLogger(__name__)

# Global Prisma client instance
_db: Prisma | None = None


async def get_db() -> Prisma:
    """Get or create the Prisma client singleton."""
    global _db
    if _db is None:
        _db = Prisma()
        await _db.connect()
        logger.info("Prisma client connected to Neon Postgres")
    return _db


async def disconnect_db() -> None:
    """Disconnect the Prisma client."""
    global _db
    if _db is not None:
        await _db.disconnect()
        _db = None
        logger.info("Prisma client disconnected")


async def create_extraction_run(
    run_id: str,
    manual_name: str | None = None,
    model: str = "claude-opus-4-6",
    source_guide_id: str | None = None,
) -> dict:
    """Create a new extraction run record.

    If `source_guide_id` is provided, the run is linked to the SourceGuide
    row it was extracted from (foreign key on ExtractionRun.sourceGuideId).
    """
    db = await get_db()
    data: dict[str, Any] = {
        "runId": run_id,
        "manualName": manual_name,
        "model": model,
        "status": "RUNNING",
    }
    if source_guide_id is not None:
        data["sourceGuideId"] = source_guide_id
    run = await db.extractionrun.create(data=data)
    logger.info("Created extraction run: %s", run_id)
    return run.model_dump()


def _coerce_json_safe(obj: Any) -> Any:
    """Recursively coerce `obj` into something Prisma's Json wrapper accepts.

    Prisma Python rejects sets, tuples of mixed types, numeric dict keys,
    bytes, and anything else non-JSON-native with a vague type-match error.
    Run #8 hit this when the catcher result had a `set` of deduped issues
    nested inside a dict that reached the finalJson field — Prisma's
    validator refused the whole update and the run record never persisted.

    This coerces everything to plain JSON types:
      - sets/tuples → list
      - dict keys that aren't str → str(key)
      - bytes → decoded str (best-effort)
      - anything else unrecognized → str(obj)
    Recursively applied top-down.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_coerce_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {
            str(k): _coerce_json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.decode("utf-8", errors="replace")
    # Datetime, UUID, custom objects — coerce to str
    return str(obj)


async def update_extraction_run(
    run_id: str,
    status: str | None = None,
    total_iterations: int | None = None,
    total_subcalls: int | None = None,
    validation_errors: int | None = None,
    final_json: dict | None = None,
    cost_estimate_usd: float | None = None,
) -> dict:
    """Update an extraction run record."""
    db = await get_db()
    data: dict[str, Any] = {}
    if status is not None:
        data["status"] = status
    if total_iterations is not None:
        data["totalIterations"] = total_iterations
    if total_subcalls is not None:
        data["totalSubcalls"] = total_subcalls
    if validation_errors is not None:
        data["validationErrors"] = validation_errors
    if final_json is not None:
        # Prisma Python's Json wrapper requires JSON-native types only. Coerce
        # the dict to make sure sets/tuples/numeric keys/bytes/datetimes all
        # become valid JSON values before handing off to Prisma. This was
        # a silent data loss in Run #8 — the finalJson write was rejected
        # with an unhelpful type error and the run record never persisted.
        coerced = _coerce_json_safe(final_json)
        data["finalJson"] = Json(coerced)
    if cost_estimate_usd is not None:
        data["costEstimateUsd"] = cost_estimate_usd
    if status in ("PASSED", "FAILED", "HALTED"):
        data["completedAt"] = datetime.now(timezone.utc)

    run = await db.extractionrun.update(
        where={"runId": run_id},
        data=data,
    )
    logger.info("Updated extraction run: %s status=%s", run_id, status)
    return run.model_dump()


async def log_repl_step(
    run_id: str,
    step_number: int,
    step_type: str,
    code: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    prompt: str | None = None,
    response: str | None = None,
    execution_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> dict:
    """Log a single REPL step to Neon."""
    db = await get_db()
    step = await db.replstep.create(
        data={
            "runId": run_id,
            "stepNumber": step_number,
            "stepType": step_type,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "prompt": prompt,
            "response": response,
            "executionMs": execution_ms,
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
        }
    )
    logger.debug("Logged REPL step: run=%s step=%d type=%s", run_id, step_number, step_type)
    return step.model_dump()


async def get_run_steps(run_id: str) -> list[dict]:
    """Get all REPL steps for a run, ordered by step number."""
    db = await get_db()
    steps = await db.replstep.find_many(
        where={"runId": run_id},
        order={"stepNumber": "asc"},
    )
    return [s.model_dump() for s in steps]


async def get_run_status(run_id: str) -> dict | None:
    """Get the current status of an extraction run."""
    db = await get_db()
    run = await db.extractionrun.find_unique(where={"runId": run_id})
    return run.model_dump() if run else None


# ---------------------------------------------------------------------------
# Intermediate artifacts (Phase 4 of the hybrid plan)
#
# Each emit_artifact() call in the REPL persists one row here. The row
# captures both the artifact itself and the validator result so we can
# replay a run's quality trajectory offline without re-running the LLM.
# ---------------------------------------------------------------------------


async def create_intermediate_artifact(
    run_id: str,
    artifact_name: str,
    phase: int,
    artifact_json: dict | list,
    validator_passed: bool,
    critical_issues: list[str] | None = None,
    warnings: list[str] | None = None,
    catcher_outputs: dict | None = None,
) -> dict | None:
    """Persist one intermediate artifact + its validator result to Neon.

    Called from rlm_runner's emit_artifact custom tool after the phase
    validator has run. Multiple rows per (run_id, artifact_name) are allowed
    — if the model has to re-emit after fixing critical issues, the history
    is preserved so we can see the trajectory.

    This is best-effort: if the write fails, we log the specific error but
    return None instead of raising so the run continues. The artifact is
    already persisted to disk in rlm_runner.

    Two Prisma Client Python gotchas handled here:
      1. `IntermediateArtifact.run` is a relation to `ExtractionRun.runId`.
         We must use the `connect` pattern rather than passing `runId` as a
         raw string — Prisma rejects that with "data.run: A value is required".
      2. JSONB fields (`artifactJson`, `criticalIssues`, `warnings`,
         `catcherOutputs`) must be wrapped in `prisma.Json(...)` so the
         serializer sends them as raw JSON. Passing a bare list/dict fails
         with "should be of any of the following types: JsonNullValueInput,
         Json". This mirrors `backend/ingestion/cache.py`.
    """
    # Normalize the artifact payload — emit_artifact allows list or dict,
    # we store the list wrapped in {"items": ...} so the JSONB column is
    # always a JSON object at the top level.
    if isinstance(artifact_json, dict):
        artifact_payload: Any = artifact_json
    else:
        artifact_payload = {"items": artifact_json}

    try:
        db = await get_db()
        row = await db.intermediateartifact.create(
            data={
                "run": {"connect": {"runId": run_id}},
                "artifactName": artifact_name,
                "phase": phase,
                "artifactJson": Json(artifact_payload),
                "validatorPassed": validator_passed,
                "criticalIssues": Json(list(critical_issues or [])),
                "warnings": Json(list(warnings or [])),
                "catcherOutputs": Json(catcher_outputs or {}),
            }
        )
    except Exception as exc:
        # Best-effort: log the specific Prisma error and swallow so the run
        # can continue. The artifact JSON is already on disk.
        logger.warning(
            "intermediate artifact persist FAILED: run=%s name=%s phase=%d err=%s",
            run_id,
            artifact_name,
            phase,
            exc,
        )
        return None

    logger.info(
        "intermediate artifact persisted: run=%s name=%s phase=%d passed=%s critical=%d warnings=%d",
        run_id,
        artifact_name,
        phase,
        validator_passed,
        len(critical_issues or []),
        len(warnings or []),
    )
    return row.model_dump()


async def get_intermediate_artifacts(
    run_id: str,
    artifact_name: str | None = None,
) -> list[dict]:
    """Return all intermediate artifacts for a run.

    If artifact_name is given, filter to just that artifact (useful for
    fetching the latest version when the model re-emitted after fixing).
    Ordered by created_at ascending so the trajectory is in temporal order.
    """
    db = await get_db()
    where: dict = {"runId": run_id}
    if artifact_name is not None:
        where["artifactName"] = artifact_name
    rows = await db.intermediateartifact.find_many(
        where=where,
        order={"createdAt": "asc"},
    )
    return [r.model_dump() for r in rows]


async def get_latest_intermediate_artifacts(run_id: str) -> dict[str, dict]:
    """Return the LATEST version of each intermediate artifact for a run.

    Returns a dict keyed by artifact_name → artifact row (the most recently
    emitted version). Used by session_manager when bundling final artifacts
    — if the model re-emitted `predicates` three times, only the last one
    ships in the downloadable bundle.
    """
    all_rows = await get_intermediate_artifacts(run_id)
    latest: dict[str, dict] = {}
    for row in all_rows:
        name = row.get("artifactName") or row.get("artifact_name")
        if name:
            latest[name] = row  # overwrite — list is in ascending order
    return latest


# ---------------------------------------------------------------------------
# Gate runs (Apr 17 empirical gate — Phase 8 of the hybrid plan)
#
# Each run_gate() call persists ONE row here summarizing both modes across
# all their runs plus the verdict. Useful for a historical record of gate
# outcomes as the system prompt and catchers evolve.
# ---------------------------------------------------------------------------


async def create_gate_run(
    gate_id: str,
    guide_title: str,
    started_at: datetime,
    n_runs: int,
    modes: list[str],
    all_at_once_pass_rate: float,
    hybrid_repl_pass_rate: float,
    all_at_once_jaccard: float,
    hybrid_repl_jaccard: float,
    all_at_once_z3_pass_rate: float,
    hybrid_repl_z3_pass_rate: float,
    all_at_once_cost_usd: float,
    hybrid_repl_cost_usd: float,
    all_at_once_median_wall: float,
    hybrid_repl_median_wall: float,
    verdict: str,
    report_json: dict,
    source_guide_id: str | None = None,
    manual_name: str | None = None,
) -> dict:
    """Persist one gate run + its verdict to Neon.

    Called from backend/eval/gate_harness.py::run_gate() after both modes
    complete. The full report dict is stored in report_json so future
    analysis can replay the exact comparison without re-running the LLM.
    """
    db = await get_db()
    row = await db.gaterun.create(
        data={
            "gateId": gate_id,
            "guideTitle": guide_title,
            "sourceGuideId": source_guide_id,
            "manualName": manual_name,
            "nRuns": n_runs,
            "modes": modes,
            "allAtOncePassRate": all_at_once_pass_rate,
            "hybridReplPassRate": hybrid_repl_pass_rate,
            "allAtOnceJaccard": all_at_once_jaccard,
            "hybridReplJaccard": hybrid_repl_jaccard,
            "allAtOnceZ3PassRate": all_at_once_z3_pass_rate,
            "hybridReplZ3PassRate": hybrid_repl_z3_pass_rate,
            "allAtOnceCostUsd": all_at_once_cost_usd,
            "hybridReplCostUsd": hybrid_repl_cost_usd,
            "allAtOnceMedianWall": all_at_once_median_wall,
            "hybridReplMedianWall": hybrid_repl_median_wall,
            "verdict": verdict,
            "reportJson": report_json,
            "startedAt": started_at,
        }
    )
    logger.info(
        "gate run persisted: gate_id=%s verdict=%s hybrid_pass=%.2f hybrid_jaccard=%.2f hybrid_cost=$%.2f",
        gate_id,
        verdict,
        hybrid_repl_pass_rate,
        hybrid_repl_jaccard,
        hybrid_repl_cost_usd,
    )
    return row.model_dump()


async def get_gate_runs(limit: int = 20) -> list[dict]:
    """Return the most recent gate runs, newest first."""
    db = await get_db()
    rows = await db.gaterun.find_many(
        order={"completedAt": "desc"},
        take=limit,
    )
    return [r.model_dump() for r in rows]
