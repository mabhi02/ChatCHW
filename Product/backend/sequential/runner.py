"""Sequential pipeline runner -- top-level entry point.

Runs all stages in dependency order, emits artifacts at aligned checkpoints,
and reports results for arena comparison.

Memory budget: ~300MB peak per run (Render Pro Plus constraint: 8GB total,
6 concurrent runs = 1.8GB, plus 300MB infra = 2.1GB of 8GB).
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

from backend.sequential.llm_client import SequentialLLMClient
from backend.sequential.executor import StageExecutor
from backend.sequential.prompts import PARALLEL_GROUPS, STAGE_MAP
from backend.validators.phases import validate_artifact, valid_artifact_names

logger = logging.getLogger(__name__)


async def run_sequential_pipeline(
    guide_json: dict,
    run_id: str | None = None,
    anthropic_key: str | None = None,
    openai_key: str | None = None,
    on_step: Any = None,
    on_artifact: Any = None,
    output_dir: Path | None = None,
    catcher_api_key: str | None = None,
) -> dict:
    """Run the full sequential extraction pipeline.

    Args:
        guide_json: The pre-processed guide JSON (from ingestion)
        run_id: Unique run identifier (auto-generated if not provided)
        anthropic_key: Anthropic API key (falls back to env var)
        openai_key: OpenAI API key (falls back to env var)
        on_step: async callback(step_dict) for SSE streaming
        on_artifact: async callback(artifact_name, artifact_json, validation) for SSE
        output_dir: directory to save artifacts (optional)

    Returns:
        {
            run_id, status, pipeline_type, stages_completed, total_stages,
            artifacts: {name: json}, validation_results: {name: result},
            cost_usd, wall_clock_ms, step_log: [...]
        }
    """
    run_id = run_id or f"seq-{uuid.uuid4().hex[:12]}"
    logger.info(f"[{run_id}] Sequential pipeline starting")
    t0 = time.time()

    client = SequentialLLMClient(
        anthropic_key=anthropic_key,
        openai_key=openai_key,
    )
    executor = StageExecutor(client, guide_json)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    stages_completed = 0
    total_stages = sum(len(group) for group in PARALLEL_GROUPS)
    all_results = {}
    validation_results = {}
    errors = []

    for group in PARALLEL_GROUPS:
        # Run stages within each group (parallel where possible)
        if len(group) == 1:
            stage_id = group[0]
            stage = STAGE_MAP[stage_id]
            try:
                result = await executor.run_stage(stage, on_step=on_step)
                all_results[stage_id] = result

                # Validate and emit artifacts
                if result.get("artifact"):
                    for output_name in stage.outputs:
                        artifact_data = executor.artifacts.get(output_name)
                        if artifact_data is not None:
                            # Run shared validators if the artifact name is recognized
                            validation = None
                            if output_name in valid_artifact_names():
                                try:
                                    validation = await validate_artifact(
                                        output_name, artifact_data, guide_json,
                                        catcher_api_key or anthropic_key or "",
                                    )
                                    validation_results[output_name] = validation
                                except Exception as ve:
                                    logger.warning(
                                        f"[{run_id}] Validator error for {output_name}: {ve}"
                                    )

                            if on_artifact:
                                await on_artifact(output_name, artifact_data, validation)

                            # Save to disk
                            if output_dir:
                                artifact_path = output_dir / f"{output_name}.json"
                                artifact_path.write_text(
                                    json.dumps(artifact_data, indent=2, ensure_ascii=False),
                                    encoding="utf-8",
                                )

                stages_completed += 1
            except Exception as e:
                logger.error(f"[{run_id}] Stage {stage_id} failed: {e}")
                errors.append({"stage_id": stage_id, "error": str(e)})
                # Continue to next group rather than halting entirely
                stages_completed += 1
        else:
            # Parallel execution for independent stages (B2 + B3)
            tasks = []
            for stage_id in group:
                stage = STAGE_MAP[stage_id]
                tasks.append(executor.run_stage(stage, on_step=on_step))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, (stage_id, result) in enumerate(zip(group, results)):
                if isinstance(result, Exception):
                    logger.error(f"[{run_id}] Stage {stage_id} failed: {result}")
                    errors.append({"stage_id": stage_id, "error": str(result)})
                else:
                    all_results[stage_id] = result

                    stage = STAGE_MAP[stage_id]
                    if result.get("artifact"):
                        for output_name in stage.outputs:
                            artifact_data = executor.artifacts.get(output_name)
                            if artifact_data is not None:
                                validation = None
                                if output_name in valid_artifact_names():
                                    try:
                                        validation = await validate_artifact(
                                            output_name, artifact_data, guide_json
                                        )
                                        validation_results[output_name] = validation
                                    except Exception as ve:
                                        logger.warning(
                                            f"[{run_id}] Validator error for {output_name}: {ve}"
                                        )

                                if on_artifact:
                                    await on_artifact(output_name, artifact_data, validation)

                                if output_dir:
                                    artifact_path = output_dir / f"{output_name}.json"
                                    artifact_path.write_text(
                                        json.dumps(artifact_data, indent=2, ensure_ascii=False),
                                        encoding="utf-8",
                                    )

                stages_completed += 1

    wall_clock_ms = int((time.time() - t0) * 1000)
    stats = client.get_stats()

    status = "PASSED" if not errors else "FAILED"
    if stages_completed == total_stages and not errors:
        status = "PASSED"
    elif stages_completed > 0:
        status = "PARTIAL"

    result = {
        "run_id": run_id,
        "status": status,
        "pipeline_type": "SEQUENTIAL",
        "stages_completed": stages_completed,
        "total_stages": total_stages,
        "artifacts": {k: v for k, v in executor.artifacts.items()},
        "validation_results": validation_results,
        "errors": errors,
        "cost_usd": stats["total_cost_usd"],
        "wall_clock_ms": wall_clock_ms,
        "token_stats": stats,
        "step_log": executor.step_log,
    }

    logger.info(
        f"[{run_id}] Sequential pipeline completed: status={status}, "
        f"stages={stages_completed}/{total_stages}, "
        f"cost=${stats['total_cost_usd']:.2f}, wall={wall_clock_ms}ms"
    )

    return result


async def stream_sequential_events(
    guide_json: dict,
    run_id: str,
    anthropic_key: str,
    openai_key: str,
) -> AsyncGenerator[dict, None]:
    """Generator that yields SSE events as the sequential pipeline runs.

    Each event is a dict with:
      - type: "step" | "artifact" | "status" | "done"
      - data: the event payload
    """
    events: asyncio.Queue[dict] = asyncio.Queue()

    async def on_step(step: dict):
        await events.put({"type": "step", "data": step})

    async def on_artifact(name: str, artifact: Any, validation: Any):
        await events.put({
            "type": "artifact",
            "data": {
                "artifact_name": name,
                "validation": validation,
                "size_bytes": len(json.dumps(artifact)),
            },
        })

    async def run_pipeline():
        try:
            result = await run_sequential_pipeline(
                guide_json=guide_json,
                run_id=run_id,
                anthropic_key=anthropic_key,
                openai_key=openai_key,
                on_step=on_step,
                on_artifact=on_artifact,
                output_dir=Path(f"backend/output/{run_id}"),
            )
            await events.put({"type": "done", "data": result})
        except Exception as e:
            await events.put({"type": "error", "data": {"error": str(e)}})

    # Start pipeline in background
    task = asyncio.create_task(run_pipeline())

    while True:
        try:
            event = await asyncio.wait_for(events.get(), timeout=1.0)
            yield event
            if event["type"] in ("done", "error"):
                break
        except asyncio.TimeoutError:
            # Yield heartbeat to keep SSE alive
            yield {"type": "heartbeat", "data": {}}
            if task.done():
                break
