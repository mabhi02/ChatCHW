"""Apr 17 empirical gate harness.

Per the Levine memo (Apr 7), we committed to running both approaches on
WHO 2012 three times each, side-by-side, by Apr 17. If the hybrid-REPL
approach loses on reliability, we pivot back to a sequential pipeline.

What "all-at-once" vs "hybrid-REPL" means here:

  ALL_AT_ONCE mode: run_extraction() with the emit_artifact tool present BUT
    no per-phase enforcement. The system prompt does NOT require emit_artifact
    between phases (we use a variant system prompt that omits the checkpoint
    instructions). This is the baseline — closest to "one big LLM call that
    produces clinical_logic end-to-end". The tool is still callable so the
    model CAN choose to use it, but the validators won't block progress.

  HYBRID_REPL mode: the current default. The system prompt REQUIRES
    emit_artifact calls between phases, and the catcher validators block
    progress on critical_issues. This is the setup being evaluated.

Usage:

    from backend.eval.gate_harness import run_gate

    report = await run_gate(
        guide_json=my_who_2012_json,
        api_key=anthropic_key,
        n_runs=3,
        output_dir=Path("backend/eval/gate_results"),
    )
    print(report.summary())

Outputs per run:
  - raw ExtractionResult as JSON
  - clinical_logic structural diff vs the other runs (byte-level + field-level)
  - validator pass rates (how many of the catchers passed on each final artifact)
  - z3_check results
  - cost, wall clock, iteration count, subcall count

Aggregate metrics:
  - reliability: run-to-run consistency of clinical_logic (Jaccard over module keys)
  - validity: per-run pass rate of structural validators and z3_check
  - cost efficiency: dollars per successful run
  - wall clock median

Writes report.md + report.json to output_dir for reproducibility.
Persists each gate run to a new gate_runs Neon table (we'll add if/when we
decide we want permanent tracking).
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from backend.db import create_gate_run
from backend.rlm_runner import ExtractionResult, run_extraction
from backend.validators import run_all_validators
from backend.z3_verifier import verify_clinical_logic

logger = logging.getLogger(__name__)

Mode = Literal["all_at_once", "hybrid_repl"]

# ---------------------------------------------------------------------------
# Gate generation label
# ---------------------------------------------------------------------------
# Per the Levine memo, "gate generations" are named checkpoints where a
# frozen experimental surface (catcher prompts, validator code, model
# versions, chunk sizes, strict mode flags, etc.) is deliberately changed.
# A generation break INVALIDATES prior-generation baselines as direct
# comparison points — verdicts from old gen runs are not apples-to-apples
# with new gen runs because the measurement apparatus changed.
#
# When this label changes, a full 3-retest variance re-baseline on both
# the WHO sample and WHO full guide MUST run before any comparative
# experiment that references a pre-bump verdict.
#
# Generation history (most recent last):
#   2.1  initial chunked catcher layer
#   2.2  Citations API + dict-critical shape + contextual chunk headers
#   2.3  generator-side 2× repetition + Opus TOC preamble + two-stage prompts
#   2.4  strict tool mode on completeness catchers, Citations dropped,
#        _ARTIFACT_CACHE_MIN_CHARS bumped 16K → 20K to clear Haiku 4.5's
#        actual 4096-token minimum cache block (the earlier 16K figure
#        straddled the boundary and silently dropped cache markers on
#        small artifacts)
#
# The label is persisted into the gate report's `report_json` (Neon
# `gateRun.reportJson` column) so every stored gate run self-describes
# which generation produced its measurements.
_CURRENT_GEN = "2.4"


@dataclass
class GateRunResult:
    """One run of one mode."""
    mode: Mode
    run_index: int
    run_id: str
    status: str
    wall_clock_seconds: float
    total_iterations: int
    total_subcalls: int
    validation_errors: int
    cost_estimate_usd: float
    clinical_logic: dict | None
    validator_report: dict  # run_all_validators output
    z3_report: dict  # verify_clinical_logic output
    intermediate_artifact_names: list[str]  # which artifacts got emitted


@dataclass
class GateReport:
    """Aggregate across all runs of both modes."""
    guide_title: str
    started_at: str
    all_at_once_runs: list[GateRunResult] = field(default_factory=list)
    hybrid_repl_runs: list[GateRunResult] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-page summary."""
        lines = [
            f"Gate Report — {self.guide_title}",
            f"Generation: {_CURRENT_GEN}",
            f"Started: {self.started_at}",
            "",
            "ALL_AT_ONCE:",
            *self._summarize_mode(self.all_at_once_runs),
            "",
            "HYBRID_REPL:",
            *self._summarize_mode(self.hybrid_repl_runs),
            "",
            "VERDICT:",
            *self._verdict(),
            "",
            (
                "NOTE: Gate generations are frozen experimental surfaces. "
                "Verdicts from different generations are NOT directly "
                "comparable — a generation bump requires a full 3-retest "
                "re-baseline before any comparison experiment."
            ),
        ]
        return "\n".join(lines)

    def _summarize_mode(self, runs: list[GateRunResult]) -> list[str]:
        if not runs:
            return ["  (no runs)"]
        passed = sum(1 for r in runs if r.status == "passed")
        costs = [r.cost_estimate_usd for r in runs]
        walls = [r.wall_clock_seconds for r in runs]
        validator_pass = sum(1 for r in runs if r.validator_report.get("passed"))
        z3_pass = sum(1 for r in runs if r.z3_report.get("all_passed"))
        return [
            f"  runs={len(runs)}  status_passed={passed}/{len(runs)}",
            f"  validator_passed={validator_pass}/{len(runs)}  z3_passed={z3_pass}/{len(runs)}",
            f"  cost: median=${statistics.median(costs):.2f}  total=${sum(costs):.2f}",
            f"  wall clock: median={statistics.median(walls):.1f}s  "
            f"min={min(walls):.1f}s  max={max(walls):.1f}s",
            f"  reliability (jaccard of modules across runs): "
            f"{self._module_jaccard(runs):.2f}",
        ]

    def _module_jaccard(self, runs: list[GateRunResult]) -> float:
        """Average pairwise Jaccard similarity of module sets across runs."""
        module_sets = []
        for r in runs:
            if r.clinical_logic and isinstance(r.clinical_logic.get("modules"), dict):
                module_sets.append(set(r.clinical_logic["modules"].keys()))
        if len(module_sets) < 2:
            return 0.0
        scores = []
        for i in range(len(module_sets)):
            for j in range(i + 1, len(module_sets)):
                a, b = module_sets[i], module_sets[j]
                if not a and not b:
                    continue
                scores.append(len(a & b) / len(a | b))
        return statistics.mean(scores) if scores else 0.0

    def _verdict(self) -> list[str]:
        """Apply the Apr 17 gate rule. Calls _compute_verdict for the raw
        decision string; this method formats it for the text report."""
        if not self.all_at_once_runs or not self.hybrid_repl_runs:
            return ["  INCOMPLETE: need runs for both modes"]
        metrics = self.compute_metrics()
        verdict_lines = [
            f"  reliability: all_at_once={metrics['all_at_once_jaccard']:.2f}  "
            f"hybrid_repl={metrics['hybrid_repl_jaccard']:.2f}",
            f"  validator pass rate: all_at_once={metrics['all_at_once_pass_rate']:.2f}  "
            f"hybrid_repl={metrics['hybrid_repl_pass_rate']:.2f}",
            f"  z3 pass rate: all_at_once={metrics['all_at_once_z3_pass_rate']:.2f}  "
            f"hybrid_repl={metrics['hybrid_repl_z3_pass_rate']:.2f}",
        ]
        if metrics["verdict"] == "KEEP_HYBRID":
            verdict_lines.append("  DECISION: KEEP HYBRID-REPL (wins or ties on all metrics)")
        elif metrics["verdict"] == "PIVOT_SEQUENTIAL":
            verdict_lines.append("  DECISION: PIVOT TO SEQUENTIAL (hybrid-REPL lost on >=1 metric)")
        else:
            verdict_lines.append(f"  DECISION: {metrics['verdict']}")
        return verdict_lines

    def compute_metrics(self) -> dict:
        """Compute all aggregate metrics + verdict. Used by both the text
        report and the Neon persistence layer."""
        if not self.all_at_once_runs or not self.hybrid_repl_runs:
            return {
                "all_at_once_jaccard": 0.0, "hybrid_repl_jaccard": 0.0,
                "all_at_once_pass_rate": 0.0, "hybrid_repl_pass_rate": 0.0,
                "all_at_once_z3_pass_rate": 0.0, "hybrid_repl_z3_pass_rate": 0.0,
                "all_at_once_cost_usd": 0.0, "hybrid_repl_cost_usd": 0.0,
                "all_at_once_median_wall": 0.0, "hybrid_repl_median_wall": 0.0,
                "verdict": "INCOMPLETE",
            }
        a_jaccard = self._module_jaccard(self.all_at_once_runs)
        h_jaccard = self._module_jaccard(self.hybrid_repl_runs)
        a_validator = statistics.mean(
            1.0 if r.validator_report.get("passed") else 0.0
            for r in self.all_at_once_runs
        )
        h_validator = statistics.mean(
            1.0 if r.validator_report.get("passed") else 0.0
            for r in self.hybrid_repl_runs
        )
        a_z3 = statistics.mean(
            1.0 if r.z3_report.get("all_passed") else 0.0
            for r in self.all_at_once_runs
        )
        h_z3 = statistics.mean(
            1.0 if r.z3_report.get("all_passed") else 0.0
            for r in self.hybrid_repl_runs
        )
        a_cost = sum(r.cost_estimate_usd for r in self.all_at_once_runs)
        h_cost = sum(r.cost_estimate_usd for r in self.hybrid_repl_runs)
        a_wall = statistics.median(r.wall_clock_seconds for r in self.all_at_once_runs)
        h_wall = statistics.median(r.wall_clock_seconds for r in self.hybrid_repl_runs)
        verdict = (
            "KEEP_HYBRID"
            if (h_jaccard >= a_jaccard and h_validator >= a_validator and h_z3 >= a_z3)
            else "PIVOT_SEQUENTIAL"
        )
        return {
            "all_at_once_jaccard": a_jaccard,
            "hybrid_repl_jaccard": h_jaccard,
            "all_at_once_pass_rate": a_validator,
            "hybrid_repl_pass_rate": h_validator,
            "all_at_once_z3_pass_rate": a_z3,
            "hybrid_repl_z3_pass_rate": h_z3,
            "all_at_once_cost_usd": a_cost,
            "hybrid_repl_cost_usd": h_cost,
            "all_at_once_median_wall": a_wall,
            "hybrid_repl_median_wall": h_wall,
            "verdict": verdict,
        }


async def run_gate(
    *,
    guide_json: dict,
    api_key: str,
    n_runs: int = 3,
    output_dir: Path,
    manual_name: str = "gate-eval",
    source_guide_id: str | None = None,
    model_name: str = "claude-sonnet-4-6",
    subcall_model: str = "claude-haiku-4-5-20251001",
    max_iterations: int = 50,
) -> GateReport:
    """Run the full Apr 17 gate: n_runs of each mode, write report.

    Modes run SEQUENTIALLY (not in parallel) to isolate cost + latency
    measurements and avoid rate-limiting the backend API. Total wall clock
    is roughly 2 * n_runs * ~10 minutes for a 30-page test slice at the
    default model settings.

    IMPORTANT: the same model_name + subcall_model + max_iterations are used
    for BOTH all_at_once and hybrid_repl modes in a single gate. The gate
    measures the DELTA between modes, so any model-induced variance is
    held constant across both arms and cancels out of the verdict.

    Defaults:
      - model_name=claude-sonnet-4-6 (root model — does the main
        REPL iteration, the planning, the artifact emission)
      - subcall_model=claude-haiku-4-5-20251001 (sub-calls are bounded
        per-module work where the model asks a sub-LLM to extract a
        specific section's content into structured JSON. Haiku handles
        these well and the downstream catcher gates check the output.)
      - max_iterations=50 (full headroom; real efficiency comes from
        parallel sub-calls via llm_query_batched, not iteration caps)

    Why Haiku sub-calls are FINE for identification (the previous version of
    this comment block argued for Sonnet sub-calls; that argument was wrong
    on principle):

    1. The project's hypothesis is that *frontier-class LLMs* can extract
       DMN from CHW manuals. If the architecture is so brittle that swapping
       Haiku for Sonnet on bounded sub-calls produces materially different
       output, then we're not actually claiming "LLMs can do this" — we're
       claiming "this specific Sonnet checkpoint can do this." That's a
       much weaker claim and won't generalize to GPT-5 or Gemini either.
       At that point we may as well not be using LLMs.
    2. The gate harness measures the DELTA between hybrid and all-at-once
       modes. Both modes use the same sub-call model. Any sub-call model
       variance shows up identically in both arms and cancels out of the
       verdict. The gate is a within-subject comparison, not an absolute
       reliability measurement.
    3. Sub-calls are work-doing calls in the middle of the chain. Their
       output is checked by downstream phase validators (the catchers).
       Sub-call variance is bounded by the catcher gate that follows. This
       is different from catcher variance, which is at the END of the chain
       and has no downstream check — that's why catchers run with
       temperature=0 + majority vote and sub-calls do not.

    For capability-maximized runs (higher cost, higher wall clock):
        model_name="claude-opus-4-6"
        subcall_model="claude-sonnet-4-6"

    Cross-vendor roots (the real generalization test, requires rlm library support):
        model_name="gpt-5.4"  (OpenAI)
        model_name="gemini-2.5-pro"  (Google)

    If source_guide_id is provided, the persisted gate_run row in Neon links
    back to the SourceGuide for traceability. Otherwise the row stands alone.
    """
    import uuid as _uuid
    output_dir.mkdir(parents=True, exist_ok=True)
    # gate_id includes a uuid suffix so two gates launched in the same second
    # don't collide. Individual run_ids below are prefixed with gate_id so
    # all per-run rows in Neon trace back to the parent gate unambiguously.
    gate_id = f"gate_{int(time.time())}_{_uuid.uuid4().hex[:8]}"
    started_at_dt = datetime.now(timezone.utc)
    report = GateReport(
        guide_title=guide_json.get("metadata", {}).get("title", "unknown"),
        started_at=started_at_dt.isoformat(),
    )

    # All-at-once mode: we temporarily swap the system prompt to one that
    # does NOT require emit_artifact between phases. The tool is still
    # registered so the model CAN call it, but the prompt doesn't enforce it.
    for i in range(n_runs):
        logger.info("Gate: all_at_once run %d/%d", i + 1, n_runs)
        run_id = f"{gate_id}_allatonce_{i}"
        run_log_dir = output_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = await _run_all_at_once(
                guide_json=guide_json,
                api_key=api_key,
                run_id=run_id,
                manual_name=manual_name,
                log_dir=str(run_log_dir),
                model_name=model_name,
                subcall_model=subcall_model,
                max_iterations=max_iterations,
            )
            validator_report = (
                run_all_validators(result.clinical_logic)
                if result.clinical_logic
                else {"passed": False, "errors": ["no clinical_logic"]}
            )
            z3_report = (
                verify_clinical_logic(result.clinical_logic)
                if result.clinical_logic
                else {"all_passed": False, "checks": []}
            )
            report.all_at_once_runs.append(
                GateRunResult(
                    mode="all_at_once",
                    run_index=i,
                    run_id=run_id,
                    status=result.status,
                    wall_clock_seconds=time.time() - t0,
                    total_iterations=result.total_iterations,
                    total_subcalls=result.total_subcalls,
                    validation_errors=result.validation_errors,
                    cost_estimate_usd=result.cost_estimate_usd,
                    clinical_logic=result.clinical_logic,
                    validator_report=validator_report,
                    z3_report=z3_report,
                    intermediate_artifact_names=[
                        name for name in (
                            "supply_list", "variables", "predicates", "modules",
                            "router", "integrative", "phrase_bank"
                        )
                        if getattr(result, name, None) is not None
                    ],
                )
            )
        except Exception as exc:
            logger.exception("Gate all_at_once run %d failed: %s", i, exc)

    # Hybrid-REPL mode: the default run_extraction path, unchanged.
    for i in range(n_runs):
        logger.info("Gate: hybrid_repl run %d/%d", i + 1, n_runs)
        run_id = f"{gate_id}_hybrid_{i}"
        run_log_dir = output_dir / run_id
        run_log_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            result = await run_extraction(
                guide_json=guide_json,
                api_key=api_key,
                run_id=run_id,
                manual_name=manual_name,
                log_dir=str(run_log_dir),
                model_name=model_name,
                subcall_model=subcall_model,
                max_iterations=max_iterations,
            )
            validator_report = (
                run_all_validators(result.clinical_logic)
                if result.clinical_logic
                else {"passed": False, "errors": ["no clinical_logic"]}
            )
            z3_report = (
                verify_clinical_logic(result.clinical_logic)
                if result.clinical_logic
                else {"all_passed": False, "checks": []}
            )
            report.hybrid_repl_runs.append(
                GateRunResult(
                    mode="hybrid_repl",
                    run_index=i,
                    run_id=run_id,
                    status=result.status,
                    wall_clock_seconds=time.time() - t0,
                    total_iterations=result.total_iterations,
                    total_subcalls=result.total_subcalls,
                    validation_errors=result.validation_errors,
                    cost_estimate_usd=result.cost_estimate_usd,
                    clinical_logic=result.clinical_logic,
                    validator_report=validator_report,
                    z3_report=z3_report,
                    intermediate_artifact_names=[
                        name for name in (
                            "supply_list", "variables", "predicates", "modules",
                            "router", "integrative", "phrase_bank"
                        )
                        if getattr(result, name, None) is not None
                    ],
                )
            )
        except Exception as exc:
            logger.exception("Gate hybrid_repl run %d failed: %s", i, exc)

    # Write report files (disk)
    _write_report(output_dir, report)

    # Persist gate run to Neon (best-effort — don't fail the gate on a DB hiccup)
    try:
        metrics = report.compute_metrics()
        report_dict = _report_to_dict(report)
        await create_gate_run(
            gate_id=gate_id,
            guide_title=report.guide_title,
            started_at=started_at_dt,
            n_runs=n_runs,
            modes=["all_at_once", "hybrid_repl"],
            all_at_once_pass_rate=metrics["all_at_once_pass_rate"],
            hybrid_repl_pass_rate=metrics["hybrid_repl_pass_rate"],
            all_at_once_jaccard=metrics["all_at_once_jaccard"],
            hybrid_repl_jaccard=metrics["hybrid_repl_jaccard"],
            all_at_once_z3_pass_rate=metrics["all_at_once_z3_pass_rate"],
            hybrid_repl_z3_pass_rate=metrics["hybrid_repl_z3_pass_rate"],
            all_at_once_cost_usd=metrics["all_at_once_cost_usd"],
            hybrid_repl_cost_usd=metrics["hybrid_repl_cost_usd"],
            all_at_once_median_wall=metrics["all_at_once_median_wall"],
            hybrid_repl_median_wall=metrics["hybrid_repl_median_wall"],
            verdict=metrics["verdict"],
            report_json=report_dict,
            source_guide_id=source_guide_id,
            manual_name=manual_name,
        )
        logger.info("Gate run persisted to Neon: gate_id=%s verdict=%s", gate_id, metrics["verdict"])
    except Exception as exc:
        logger.warning("Failed to persist gate run to Neon: %s", exc)

    return report


async def _run_all_at_once(
    *,
    guide_json: dict,
    api_key: str,
    run_id: str,
    manual_name: str,
    log_dir: str,
    model_name: str,
    subcall_model: str,
    max_iterations: int,
) -> ExtractionResult:
    """Run extraction with a minimal system prompt that does NOT require
    per-phase emit_artifact. The tool is still registered so the model can
    optionally use it, but there's no enforcement and the prompt describes
    the task as one continuous extraction rather than phased checkpoints."""
    # Temporarily monkey-patch build_system_prompt to return the minimal variant.
    # This is a dirty-but-scoped hack that ensures the gate harness does NOT
    # permanently alter the default system prompt.
    from backend import system_prompt as sp

    original_build = sp.build_system_prompt

    def _all_at_once_prompt() -> str:
        # Everything from the real system prompt EXCEPT the emit_artifact
        # checkpoint instructions. The catcher tools still exist, just no
        # requirement that the model use them.
        sections = [
            sp.load_fragment("role_knowledge_engineer.txt"),
            sp.load_fragment("role_dmn_architect.txt"),
            sp.REPL_INSTRUCTIONS.replace(
                "MANDATORY between phases. Persists",
                "AVAILABLE as an optional tool. Persists",
            ).replace(
                "You MUST call it between phases. Skipping checkpoints will halt the run.",
                "You MAY call it for debugging but it is not required.",
            ),
            sp.NAMING_CODEBOOK,
            sp.PREDICATE_CONVENTION,
            sp.load_fragment("std_logic_integrity.txt"),
            sp.load_fragment("std_safe_endpoints.txt"),
            sp.load_fragment("std_missingness_model.txt"),
            sp.load_fragment("std_antigravity_data.txt"),
            sp.load_fragment("std_queue_management.txt"),
            sp.load_fragment("std_dmn_subset.txt"),
            # All-at-once strategy: describe the phases as guidance, not checkpoints
            """
EXTRACTION STRATEGY (all-at-once mode):

Read the guide, extract the clinical logic, and return the final
clinical_logic dict. You may use sub-calls to help with per-module extraction.
Validate with validate() and z3_check() before returning. Call FINAL_VAR
when you are done.

There is no per-phase checkpoint requirement. You may structure your work
however you find most effective for the manual at hand.
""",
            sp.load_fragment("footer_safety.txt"),
        ]
        return "\n\n---\n\n".join(sections)

    try:
        sp.build_system_prompt = _all_at_once_prompt
        result = await run_extraction(
            guide_json=guide_json,
            api_key=api_key,
            run_id=run_id,
            manual_name=manual_name,
            log_dir=log_dir,
            model_name=model_name,
            subcall_model=subcall_model,
            max_iterations=max_iterations,
        )
    finally:
        sp.build_system_prompt = original_build
    return result


def _run_to_dict(r: GateRunResult) -> dict:
    """Serialize one GateRunResult into a JSON-safe dict."""
    return {
        "mode": r.mode,
        "run_index": r.run_index,
        "run_id": r.run_id,
        "status": r.status,
        "wall_clock_seconds": r.wall_clock_seconds,
        "total_iterations": r.total_iterations,
        "total_subcalls": r.total_subcalls,
        "validation_errors": r.validation_errors,
        "cost_estimate_usd": r.cost_estimate_usd,
        "clinical_logic_module_count": (
            len(r.clinical_logic.get("modules", {}))
            if r.clinical_logic and isinstance(r.clinical_logic.get("modules"), dict)
            else 0
        ),
        "validator_passed": r.validator_report.get("passed"),
        "z3_passed": r.z3_report.get("all_passed"),
        "intermediate_artifacts_emitted": r.intermediate_artifact_names,
    }


def _report_to_dict(report: GateReport) -> dict:
    """Serialize a GateReport into a JSON-safe dict. Used by both the file
    writer and the Neon persistence layer.

    Includes the current gate generation label so every stored report
    self-describes which measurement regime produced it. Cross-generation
    comparisons must be treated as apples-to-oranges — see _CURRENT_GEN
    docstring above for the generation-break discipline."""
    return {
        "generation": _CURRENT_GEN,
        "guide_title": report.guide_title,
        "started_at": report.started_at,
        "metrics": report.compute_metrics(),
        "all_at_once": [_run_to_dict(r) for r in report.all_at_once_runs],
        "hybrid_repl": [_run_to_dict(r) for r in report.hybrid_repl_runs],
    }


def _write_report(output_dir: Path, report: GateReport) -> None:
    """Write report.md + report.json to output_dir."""
    (output_dir / "report.md").write_text(report.summary(), encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(_report_to_dict(report), indent=2), encoding="utf-8"
    )
    logger.info("Gate report written to %s", output_dir)
