"""Stage executor for the sequential pipeline.

Each stage runs: maker -> red team (3x majority) -> repair (if needed, max 2 retries).
Emits intermediate artifacts at aligned checkpoints for arena comparison.
"""

import asyncio
import json
import logging
import time
from typing import Any

from backend.sequential.llm_client import SequentialLLMClient
from backend.sequential.prompts import (
    REDTEAM_PROMPT,
    REPAIR_PROMPT,
    StagePrompt,
)

logger = logging.getLogger(__name__)

MAX_REPAIR_RETRIES = 2


def _build_guide_context(guide_json: dict) -> str:
    """Build a compact guide context string from the guide JSON.

    Includes metadata and all section content, capped at ~90K tokens worth
    of text to stay within context windows.
    """
    parts = []
    metadata = guide_json.get("metadata", {})
    parts.append(f"MANUAL: {metadata.get('title', 'Unknown')} ({metadata.get('total_pages', '?')} pages)")
    parts.append("")

    # Include page-level text for grounding
    pages = guide_json.get("pages", {})
    for page_num in sorted(pages.keys(), key=lambda x: int(x)):
        page = pages[page_num]
        raw_text = ""
        if isinstance(page, dict):
            raw_text = page.get("raw_text", "")
        elif isinstance(page, str):
            raw_text = page
        if raw_text and raw_text.strip():
            parts.append(f"--- Page {page_num} ---")
            parts.append(raw_text.strip())
            parts.append("")

    # Also include section content if available. No per-section cap on
    # content_items — the old [:50] silently dropped items 51+ from any
    # section with more than 50 bullet points (drug lists, exhaustive
    # symptom enumerations), which is a recall hole for sequential runs.
    sections = guide_json.get("sections", {})
    if isinstance(sections, dict):
        for slug, section in sections.items():
            if isinstance(section, dict):
                title = section.get("title", slug)
                content_items = section.get("content", [])
                if content_items:
                    parts.append(f"## Section: {title}")
                    for item in content_items:  # was [:50] — removed for full coverage
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if text:
                                parts.append(text)
                    parts.append("")

    result = "\n".join(parts)
    # Cap raised from 300K → 700K chars (~175K tokens) to fit Sonnet 4.6's
    # 200K token context while leaving headroom for the system prompt, user
    # message, and output. On full WHO (1.9M chars pretty) this is still
    # truncation but the sequential pipeline never fit a full WHO guide in
    # one call anyway — it was designed for smaller manuals. Log the
    # truncation so we know when the cap is biting.
    if len(result) > 700_000:
        logger.warning(
            "_build_guide_context: truncating guide context %d → 700000 chars "
            "for Sequential pipeline. Consider splitting the extraction into "
            "per-section passes for guides this large.",
            len(result),
        )
        result = result[:700_000] + "\n\n[TRUNCATED -- guide exceeds 700K chars]"
    return result


def _build_system_prompt(guide_context: str) -> str:
    """Build the system prompt with the guide embedded for caching.

    The guide is placed in the system block so Anthropic can cache it
    with 1h TTL across all calls in this pipeline run.
    """
    return f"""You are a clinical decision logic extractor for the WHO CHW (Community Health Worker) manual system.

You extract clinical logic faithfully from the manual below. You do not invent clinical rules, thresholds, or treatments. Every piece of output must trace back to a specific page and quote in the manual.

Always respond with valid JSON only. No markdown, no explanation outside the JSON.

SOURCE MANUAL
=============
{guide_context}
"""


class StageExecutor:
    """Runs a single stage through the maker -> red team -> repair cycle."""

    def __init__(self, client: SequentialLLMClient, guide_json: dict):
        self._client = client
        self._guide_json = guide_json
        self._guide_context = _build_guide_context(guide_json)
        self._system_prompt = _build_system_prompt(self._guide_context)
        # Artifact store: maps artifact name -> JSON dict
        self.artifacts: dict[str, Any] = {}
        # Step log: list of all LLM calls made
        self.step_log: list[dict] = []

    async def run_stage(
        self,
        stage: StagePrompt,
        on_step: Any = None,
    ) -> dict:
        """Run a complete stage: maker -> red team -> repair loop.

        Args:
            stage: StagePrompt with maker prompt and metadata
            on_step: optional async callback(step_dict) for SSE streaming

        Returns:
            {artifact_name: artifact_json, passed: bool, retries: int, steps: [...]}
        """
        logger.info(f"[{stage.stage_id}] Starting: {stage.display_name}")
        t0 = time.time()
        steps = []

        # Build user message for the maker
        user_msg = self._build_maker_user_message(stage)

        # --- MAKER ---
        maker_result = await self._client.call_anthropic(
            system_prompt=self._system_prompt,
            user_message=user_msg,
            model="claude-sonnet-4-6",
            max_tokens=16384,
            cache_system=True,
        )

        step = {
            "stage_id": stage.stage_id,
            "stage_type": "maker",
            "step_number": len(steps) + 1,
            "retry_number": 0,
            "model": maker_result["model"],
            "input_tokens": maker_result["input_tokens"],
            "output_tokens": maker_result["output_tokens"],
            "cached_tokens": maker_result["cached_tokens"],
            "duration_ms": maker_result["duration_ms"],
            "cost_usd": maker_result["cost_usd"],
        }
        steps.append(step)
        self.step_log.append(step)
        if on_step:
            await on_step(step)

        # Parse maker output
        artifact = self._parse_json_response(maker_result["content"], stage.stage_id)
        if artifact is None:
            logger.error(f"[{stage.stage_id}] Maker produced invalid JSON")
            return {
                "artifact_name": stage.artifact_name,
                "artifact": None,
                "passed": False,
                "retries": 0,
                "steps": steps,
                "error": "Maker produced invalid JSON",
                "duration_ms": int((time.time() - t0) * 1000),
            }

        # --- RED TEAM + REPAIR LOOP ---
        prior_redteam_content: str | None = None
        for retry in range(MAX_REPAIR_RETRIES + 1):
            # Red team (3x majority vote on gpt-5.4-mini)
            redteam_result = await self._run_redteam(stage, artifact, prior_redteam_content=prior_redteam_content)
            rt_step = {
                "stage_id": stage.stage_id,
                "stage_type": "redteam",
                "step_number": len(steps) + 1,
                "retry_number": retry,
                "model": redteam_result["model"],
                "input_tokens": redteam_result["input_tokens"],
                "output_tokens": redteam_result["output_tokens"],
                "cached_tokens": redteam_result["cached_tokens"],
                "duration_ms": redteam_result["duration_ms"],
                "cost_usd": redteam_result["cost_usd"],
                "passed": redteam_result["passed"],
            }
            steps.append(rt_step)
            self.step_log.append(rt_step)
            if on_step:
                await on_step(rt_step)

            if redteam_result["passed"]:
                logger.info(f"[{stage.stage_id}] Red team PASSED (retry {retry})")
                break

            # Save this round's red team content so the next round can
            # reference it, preventing the "moving goalpost" problem where
            # the red team invents entirely new issues after each repair.
            prior_redteam_content = redteam_result["content"]

            if retry >= MAX_REPAIR_RETRIES:
                logger.warning(
                    f"[{stage.stage_id}] Red team FAILED after {MAX_REPAIR_RETRIES} retries, "
                    f"accepting artifact as-is"
                )
                break

            # Repair
            logger.info(f"[{stage.stage_id}] Red team FAILED, running repair (retry {retry + 1})")
            repair_result = await self._run_repair(stage, artifact, redteam_result["content"])
            rp_step = {
                "stage_id": stage.stage_id,
                "stage_type": "repair",
                "step_number": len(steps) + 1,
                "retry_number": retry + 1,
                "model": repair_result["model"],
                "input_tokens": repair_result["input_tokens"],
                "output_tokens": repair_result["output_tokens"],
                "cached_tokens": repair_result["cached_tokens"],
                "duration_ms": repair_result["duration_ms"],
                "cost_usd": repair_result["cost_usd"],
            }
            steps.append(rp_step)
            self.step_log.append(rp_step)
            if on_step:
                await on_step(rp_step)

            repaired = self._parse_json_response(repair_result["content"], f"{stage.stage_id}_repair")
            if repaired is not None:
                artifact = repaired
            else:
                logger.warning(f"[{stage.stage_id}] Repair produced invalid JSON, keeping original")

        # Store artifact(s)
        self._store_artifacts(stage, artifact)

        duration_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[{stage.stage_id}] Completed in {duration_ms}ms, "
            f"{len(steps)} steps, artifact stored as '{stage.artifact_name}'"
        )

        return {
            "artifact_name": stage.artifact_name,
            "artifact": artifact,
            "passed": True,
            "retries": max(0, len([s for s in steps if s["stage_type"] == "repair"])),
            "steps": steps,
            "duration_ms": duration_ms,
        }

    def _build_maker_user_message(self, stage: StagePrompt) -> str:
        """Build the user message for the maker call, including prior artifacts as context."""
        parts = [stage.maker_prompt, ""]

        # Include prior artifacts this stage depends on. Cap raised from
        # 50K → 200K chars per prior artifact. On full WHO the modules
        # artifact can reach ~100K chars, and the sequential pipeline needs
        # to see the full prior artifacts to build consistent downstream
        # stages. 50K silently truncated half of the modules content
        # when predicates/router stages referenced it.
        for input_name in stage.inputs:
            if input_name == "guide_json":
                continue  # guide is in system prompt
            if input_name in self.artifacts:
                artifact_json = json.dumps(self.artifacts[input_name], indent=2)
                if len(artifact_json) > 200_000:
                    logger.warning(
                        "Maker: prior artifact %s truncated %d → 200000 chars",
                        input_name, len(artifact_json),
                    )
                    artifact_json = artifact_json[:200_000] + "\n... [truncated]"
                parts.append(f"PRIOR ARTIFACT: {input_name}")
                parts.append(artifact_json)
                parts.append("")

        return "\n".join(parts)

    async def _run_redteam(self, stage: StagePrompt, artifact: Any, prior_redteam_content: str | None = None) -> dict:
        """Run qualitative red team audit on Sonnet 4.6.

        Sonnet's reasoning depth is better for the qualitative work the red
        team does: cross-referencing artifacts against a clinical manual,
        distinguishing safety-critical omissions from granularity preferences,
        and producing structured reports with citations. Classification gates
        (pass/fail on checklist items) use 3x GPT-5.4-mini majority vote
        instead -- see backend/validators/phases.py.
        """
        quality_standards = "\n".join(f"- {qs}" for qs in stage.quality_standards) if stage.quality_standards else "No specific standards for this stage."

        # On retries, inject the prior red team report so the auditor
        # focuses on remaining/new issues instead of re-flagging fixed ones.
        # Cap raised 15K → 50K so a long prior report isn't silently cut
        # — red team reports on full WHO runs routinely exceed 15K chars.
        if prior_redteam_content:
            prior_section = (
                "\nPREVIOUSLY FLAGGED ISSUES (already repaired -- do NOT re-flag these or close variants):\n"
                f"{prior_redteam_content[:50_000]}\n"
                "Focus ONLY on issues that remain unfixed or are genuinely new. "
                "Do not re-flag issues from the previous report that were addressed by the repair.\n"
            )
        else:
            prior_section = ""

        redteam_system = REDTEAM_PROMPT.format(
            stage_id=stage.stage_id,
            artifact_name=stage.artifact_name,
            quality_standards=quality_standards,
            prior_feedback_section=prior_section,
        )

        # Artifact cap raised 80K → 200K chars for the same reason as
        # _build_maker_user_message above.
        artifact_json = json.dumps(artifact, indent=2)
        if len(artifact_json) > 200_000:
            logger.warning(
                "Red team: artifact %s truncated %d → 200000 chars",
                stage.artifact_name, len(artifact_json),
            )
            artifact_json = artifact_json[:200_000] + "\n... [truncated]"

        user_msg = f"""ARTIFACT TO AUDIT ({stage.artifact_name}):
{artifact_json}

Review this artifact against the source manual in the system prompt. Return your assessment as JSON."""

        # Guide context in system prompt for Anthropic prompt caching.
        # Cap raised 100K → 500K chars. The _guide_context itself is now
        # capped at 700K (up from 300K), and this cap on top of it was
        # slicing the red team's view of the guide down to the first 20%.
        # 500K leaves room for the redteam_system prompt and artifact +
        # response within Sonnet 4.6's 200K token budget (~800K chars).
        full_system = (
            f"{redteam_system}\n\nSOURCE MANUAL CONTEXT (for cross-reference):\n"
            f"{self._guide_context[:500_000]}"
        )

        result = await self._client.call_anthropic(
            system_prompt=full_system,
            user_message=user_msg,
            model="claude-sonnet-4-6",
            max_tokens=4096,
            temperature=0,
            cache_system=True,
        )

        # Parse the pass/fail from the JSON response to match the interface
        # that run_stage expects (same shape as call_openai_majority returns)
        passed = False
        try:
            parsed = json.loads(result["content"])
            passed = bool(parsed.get("passed", parsed.get("pass", False)))
        except (json.JSONDecodeError, KeyError):
            content_lower = result["content"].lower()
            passed = "pass" in content_lower and "fail" not in content_lower

        result["passed"] = passed
        result["votes"] = [passed]  # single vote for compatibility
        result["n_calls"] = 1
        return result

    async def _run_repair(self, stage: StagePrompt, artifact: Any, redteam_report: str) -> dict:
        """Run repair on Sonnet with the red team report."""
        # Red team report cap raised 20K → 60K chars. The repair stage needs
        # to see all the red team findings, not just the first few. Reports
        # on full WHO runs routinely exceed 20K when a stage has dozens of
        # issues — the old cap silently hid half the findings from repair.
        repair_system = REPAIR_PROMPT.format(
            stage_id=stage.stage_id,
            artifact_name=stage.artifact_name,
            redteam_report=redteam_report[:60_000],
        )

        # Artifact cap raised 80K → 200K chars (same rationale as above).
        artifact_json = json.dumps(artifact, indent=2)
        if len(artifact_json) > 200_000:
            logger.warning(
                "Repair: artifact %s truncated %d → 200000 chars",
                stage.artifact_name, len(artifact_json),
            )
            artifact_json = artifact_json[:200_000] + "\n... [truncated]"

        user_msg = f"""CURRENT ARTIFACT ({stage.artifact_name}):
{artifact_json}

Repair this artifact based on the red team findings. Return the COMPLETE repaired artifact as valid JSON."""

        return await self._client.call_anthropic(
            system_prompt=self._system_prompt + "\n\n" + repair_system,
            user_message=user_msg,
            model="claude-sonnet-4-6",
            max_tokens=16384,
            cache_system=True,
        )

    def _parse_json_response(self, content: str, label: str) -> Any:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = content.strip()
        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"[{label}] JSON parse error: {e}")
            # Try to find JSON in the response
            for start_char in ["{", "["]:
                idx = text.find(start_char)
                if idx >= 0:
                    try:
                        return json.loads(text[idx:])
                    except json.JSONDecodeError:
                        continue
            return None

    def _store_artifacts(self, stage: StagePrompt, artifact: Any) -> None:
        """Store extracted artifacts in the artifact store."""
        if isinstance(artifact, dict):
            # Some stages produce multiple named outputs within a single response
            for output_name in stage.outputs:
                if output_name in artifact:
                    self.artifacts[output_name] = artifact[output_name]
                    logger.info(f"[{stage.stage_id}] Stored artifact: {output_name}")

            # Also store the full response under the stage's primary artifact name
            self.artifacts[stage.artifact_name] = artifact
        else:
            self.artifacts[stage.artifact_name] = artifact
