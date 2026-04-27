"""Arena orchestrator -- launches N REPL + N Sequential runs in parallel.

This is the core A/B test runner. Given a guide JSON, it launches both
pipelines simultaneously and streams progress to the frontend via SSE.

Respects Render Pro Plus limits:
- 8GB RAM total, ~500MB per REPL run, ~300MB per Sequential run
- 6 concurrent runs (3 REPL + 3 Sequential) = 2.7GB peak
- API rate limits: <3% Anthropic, <1% OpenAI (non-issue)
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from backend.db import (
    get_db,
    create_extraction_run,
    update_extraction_run,
    create_intermediate_artifact,
)
from backend.sequential.runner import run_sequential_pipeline
from backend.rlm_runner import run_extraction
from backend.session_manager import OUTPUT_DIR

logger = logging.getLogger(__name__)

# Max concurrent extractions per arena run. On Render Pro Plus (8GB):
# 3 REPL (500MB each) + 3 Sequential (300MB each) = 2.7GB + 300MB infra = 3GB
# Well within 8GB. Could go higher but 3+3 is the spec.
MAX_CONCURRENT_PER_SIDE = 3


class ArenaSession:
    """Manages a single arena comparison session."""

    def __init__(
        self,
        guide_json: dict,
        source_guide_id: str,
        manual_name: str,
        n_runs: int,
        anthropic_key: str,
        openai_key: str,
    ):
        self.arena_id = f"arena-{uuid.uuid4().hex[:12]}"
        self.guide_json = guide_json
        self.source_guide_id = source_guide_id
        self.manual_name = manual_name
        self.n_runs = min(n_runs, MAX_CONCURRENT_PER_SIDE)
        self.anthropic_key = anthropic_key
        self.openai_key = openai_key

        self.status = "pending"
        self.started_at: float | None = None
        self.completed_at: float | None = None

        # Results storage
        self.repl_results: list[dict] = []
        self.seq_results: list[dict] = []
        self.events: asyncio.Queue[dict] = asyncio.Queue()

        # Output directory
        self.output_dir = OUTPUT_DIR / self.arena_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> dict:
        """Run the full arena: N REPL + N Sequential in parallel."""
        self.status = "running"
        self.started_at = time.time()
        await self.events.put({
            "type": "arena_status",
            "data": {"status": "running", "arena_id": self.arena_id, "n_runs": self.n_runs},
        })

        # Launch all runs in parallel
        tasks = []

        for i in range(self.n_runs):
            run_num = i + 1
            # REPL run
            tasks.append(self._run_repl(run_num))
            # Sequential run
            tasks.append(self._run_sequential(run_num))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate REPL and Sequential results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Arena run failed: {result}")
                error_result = {"status": "FAILED", "error": str(result)}
                if i % 2 == 0:
                    self.repl_results.append(error_result)
                else:
                    self.seq_results.append(error_result)
            else:
                if i % 2 == 0:
                    self.repl_results.append(result)
                else:
                    self.seq_results.append(result)

        self.completed_at = time.time()
        self.status = "completed"
        wall_clock_ms = int((self.completed_at - self.started_at) * 1000)

        report = {
            "arena_id": self.arena_id,
            "status": self.status,
            "n_runs": self.n_runs,
            "manual_name": self.manual_name,
            "source_guide_id": self.source_guide_id,
            "wall_clock_ms": wall_clock_ms,
            "repl_results": self.repl_results,
            "sequential_results": self.seq_results,
            "repl_cost_usd": sum(r.get("cost_usd", 0) for r in self.repl_results if isinstance(r, dict)),
            "seq_cost_usd": sum(r.get("cost_usd", 0) for r in self.seq_results if isinstance(r, dict)),
        }

        # Save report
        report_path = self.output_dir / "arena_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        await self.events.put({"type": "arena_done", "data": report})
        return report

    async def _run_repl(self, run_number: int) -> dict:
        """Run a single REPL extraction."""
        run_id = f"{self.arena_id}-repl-{run_number}"
        run_dir = self.output_dir / f"repl-{run_number}"
        run_dir.mkdir(exist_ok=True)

        logger.info(f"[{run_id}] Starting REPL extraction (run {run_number}/{self.n_runs})")
        await self.events.put({
            "type": "run_start",
            "data": {"run_id": run_id, "pipeline": "repl", "run_number": run_number},
        })

        t0 = time.time()
        try:
            async def repl_on_step(step: dict):
                step_type = step.get("step_type", "exec")
                if step_type == "artifact":
                    await self.events.put({
                        "type": "repl_artifact",
                        "data": {"run_id": run_id, "run_number": run_number, "artifact_name": step.get("prompt", "").replace("emit_artifact(", "").rstrip("')\"")},
                    })
                else:
                    await self.events.put({
                        "type": "repl_step",
                        "data": {"run_id": run_id, "run_number": run_number, **step},
                    })

            result = await run_extraction(
                api_key=self.anthropic_key,
                guide_json=self.guide_json,
                run_id=run_id,
                manual_name=self.manual_name,
                log_dir=str(run_dir),
                on_step=repl_on_step,
            )

            wall_ms = int((time.time() - t0) * 1000)
            # Collect artifact names from the ExtractionResult
            artifact_names = []
            for name in ("supply_list", "variables", "predicates", "modules", "router", "integrative", "phrase_bank"):
                if getattr(result, name, None) is not None:
                    artifact_names.append(name)
                    # Save each artifact to disk for download
                    artifact_path = run_dir / f"{name}.json"
                    artifact_path.write_text(
                        json.dumps(getattr(result, name), indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

            result_dict = {
                "run_id": run_id,
                "pipeline_type": "REPL",
                "run_number": run_number,
                "status": result.status,
                "wall_clock_ms": wall_ms,
                "cost_usd": result.cost_estimate_usd,
                "total_iterations": result.total_iterations,
                "validation_errors": result.validation_errors,
                "artifacts": artifact_names,
            }

            await self.events.put({
                "type": "run_complete",
                "data": result_dict,
            })
            return result_dict

        except Exception as e:
            wall_ms = int((time.time() - t0) * 1000)
            logger.error(f"[{run_id}] REPL extraction failed: {e}")
            error_result = {
                "run_id": run_id,
                "pipeline_type": "REPL",
                "run_number": run_number,
                "status": "FAILED",
                "error": str(e),
                "wall_clock_ms": wall_ms,
            }
            await self.events.put({"type": "run_error", "data": error_result})
            return error_result

    async def _run_sequential(self, run_number: int) -> dict:
        """Run a single Sequential extraction."""
        run_id = f"{self.arena_id}-seq-{run_number}"
        run_dir = self.output_dir / f"seq-{run_number}"
        run_dir.mkdir(exist_ok=True)

        logger.info(f"[{run_id}] Starting Sequential extraction (run {run_number}/{self.n_runs})")
        await self.events.put({
            "type": "run_start",
            "data": {"run_id": run_id, "pipeline": "sequential", "run_number": run_number},
        })

        t0 = time.time()
        try:
            async def on_step(step: dict):
                await self.events.put({
                    "type": "seq_step",
                    "data": {"run_id": run_id, "run_number": run_number, **step},
                })

            async def on_artifact(name: str, artifact: Any, validation: Any):
                await self.events.put({
                    "type": "seq_artifact",
                    "data": {
                        "run_id": run_id,
                        "run_number": run_number,
                        "artifact_name": name,
                        "validation": validation,
                    },
                })

            result = await run_sequential_pipeline(
                guide_json=self.guide_json,
                run_id=run_id,
                anthropic_key=self.anthropic_key,
                openai_key=self.openai_key,
                on_step=on_step,
                on_artifact=on_artifact,
                output_dir=run_dir,
            )

            wall_ms = int((time.time() - t0) * 1000)
            # Convert artifacts dict to name list for frontend compatibility
            artifact_names = list(result.get("artifacts", {}).keys())
            result_event = {
                "run_id": run_id,
                "pipeline_type": "SEQUENTIAL",
                "run_number": run_number,
                "status": result.get("status", "unknown"),
                "wall_clock_ms": wall_ms,
                "cost_usd": result.get("cost_usd", 0),
                "artifacts": artifact_names,
                "stages_completed": result.get("stages_completed", 0),
                "total_stages": result.get("total_stages", 0),
            }

            await self.events.put({"type": "run_complete", "data": result_event})
            return result_event

        except Exception as e:
            wall_ms = int((time.time() - t0) * 1000)
            logger.error(f"[{run_id}] Sequential extraction failed: {e}")
            error_result = {
                "run_id": run_id,
                "pipeline_type": "SEQUENTIAL",
                "run_number": run_number,
                "status": "FAILED",
                "error": str(e),
                "wall_clock_ms": wall_ms,
            }
            await self.events.put({"type": "run_error", "data": error_result})
            return error_result

    async def stream_events(self) -> AsyncGenerator[dict, None]:
        """Yield SSE events as the arena progresses."""
        while True:
            try:
                event = await asyncio.wait_for(self.events.get(), timeout=2.0)
                yield event
                if event["type"] in ("arena_done", "arena_error"):
                    break
            except asyncio.TimeoutError:
                yield {"type": "heartbeat", "data": {"arena_id": self.arena_id}}
                if self.status == "completed":
                    break


# Module-level active arena sessions
_active_arenas: dict[str, ArenaSession] = {}


async def start_arena(
    guide_json: dict,
    source_guide_id: str,
    manual_name: str,
    n_runs: int,
    anthropic_key: str,
    openai_key: str,
) -> ArenaSession:
    """Create and start an arena session. Returns the session for SSE streaming."""
    session = ArenaSession(
        guide_json=guide_json,
        source_guide_id=source_guide_id,
        manual_name=manual_name,
        n_runs=n_runs,
        anthropic_key=anthropic_key,
        openai_key=openai_key,
    )
    _active_arenas[session.arena_id] = session

    # Start the arena run as a background task
    async def run_and_cleanup():
        try:
            await session.run()
        finally:
            # Keep the session in memory for 2 hours for result retrieval
            await asyncio.sleep(7200)
            _active_arenas.pop(session.arena_id, None)

    asyncio.create_task(run_and_cleanup())
    return session


def get_arena(arena_id: str) -> ArenaSession | None:
    return _active_arenas.get(arena_id)
