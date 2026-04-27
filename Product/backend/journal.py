"""Extraction Journal -- real-time human-readable scratchpad.

Maintains a markdown file that accumulates as the RLM processes.
Each REPL step is parsed into a readable entry showing what the model
discovered, built, or fixed. Streamed to the frontend via SSE alongside
the raw REPL events.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JournalWriter:
    """Writes and maintains a live extraction journal (scratchpad.md).

    Call `record_step()` after each REPL iteration. The writer parses
    the code and stdout to produce a human-readable journal entry,
    appends it to the markdown file, and returns the entry for SSE streaming.
    """

    def __init__(self, output_dir: Path, manual_name: str = "Unknown Guide"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = output_dir / "scratchpad.md"
        self.manual_name = manual_name
        self._step_count = 0
        self._current_phase = "Initializing"
        self._modules_found: list[str] = []
        self._predicates_found: list[str] = []
        self._phrases_found: int = 0
        self._validation_runs: int = 0
        self._z3_runs: int = 0
        self._subcall_count = 0

        # Write header
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._write(f"# Extraction Journal: {manual_name}\n")
        self._write(f"Started: {now}\n")
        self._write("---\n")

    def _write(self, text: str) -> None:
        """Append text to the journal file."""
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def record_step(
        self,
        step_number: int,
        step_type: str,
        code: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        prompt: str | None = None,
        response: str | None = None,
        validation_result: dict | None = None,
        z3_result: dict | None = None,
        **kwargs,
    ) -> dict | None:
        """Record a REPL step and return a journal entry for SSE.

        Returns a dict with the journal entry, or None if the step
        is not interesting enough to journal.
        """
        self._step_count += 1
        entry = None

        if step_type == "exec" and code:
            entry = self._parse_exec_step(step_number, code, stdout, stderr)
        elif step_type == "exec" and not code:
            # Iteration callback with no code (fired by rlm library per-turn)
            entry = self._parse_iteration_tick(step_number, kwargs)
        elif step_type == "status" and stdout:
            entry = self._parse_status_step(step_number, stdout)
        elif step_type == "artifact":
            entry = self._parse_artifact_step(step_number, prompt, response)
        elif step_type == "subcall":
            entry = self._parse_subcall(step_number, prompt, response)
        elif step_type == "validate" and validation_result:
            entry = self._parse_validation(step_number, validation_result)
        elif step_type == "z3" and z3_result:
            entry = self._parse_z3(step_number, z3_result)
        elif step_type == "final":
            entry = self._parse_final(step_number, stdout)

        if entry:
            self._write(entry["markdown"])
            return entry

        return None

    def _parse_exec_step(
        self, step_number: int, code: str, stdout: str | None, stderr: str | None
    ) -> dict | None:
        """Parse a code execution step into a journal entry."""
        code_lower = code.lower()
        summary_parts: list[str] = []
        phase_changed = False

        # Detect phase from code patterns
        if self._step_count <= 2 or "keys()" in code or "sections" in code_lower and "print" in code_lower:
            if self._current_phase != "Scanning":
                self._current_phase = "Scanning"
                phase_changed = True
            summary_parts.append("Examining guide structure")

        if "convention" in code_lower or "predicate" in code_lower and ("=" in code or "append" in code_lower):
            if "convention" in code_lower:
                if self._current_phase != "Schema":
                    self._current_phase = "Schema"
                    phase_changed = True
                summary_parts.append("Building shared naming conventions")

        if "llm_query" in code:
            self._subcall_count += 1

        if "validate(" in code:
            summary_parts.append("Running validators on current output")

        if "z3_check(" in code:
            summary_parts.append("Running Z3 exhaustiveness proofs")

        if "FINAL_VAR" in code:
            self._current_phase = "Complete"
            phase_changed = True
            summary_parts.append("Returning final validated output")

        # Parse stdout for discoveries
        if stdout:
            stdout_discoveries = self._parse_stdout_discoveries(stdout)
            summary_parts.extend(stdout_discoveries)

        # Parse code for variable assignments that indicate module building
        module_match = re.findall(r"['\"]module_id['\"]:\s*['\"](\w+)['\"]", code)
        for mid in module_match:
            if mid not in self._modules_found:
                self._modules_found.append(mid)
                summary_parts.append(f"Building module: **{mid}**")

        predicate_match = re.findall(r"['\"]predicate_id['\"]:\s*['\"](\w+)['\"]", code)
        for pid in predicate_match:
            if pid not in self._predicates_found:
                self._predicates_found.append(pid)

        if not summary_parts:
            # Generic step
            lines = code.strip().split("\n")
            first_line = lines[0][:80] if lines else "..."
            summary_parts.append(f"Executing: `{first_line}`")

        # Build markdown
        md_lines: list[str] = []
        if phase_changed:
            phase_label = self._get_phase_label()
            md_lines.append(f"\n## {phase_label}")

        md_lines.append(f"**[Step {step_number}]** {' | '.join(summary_parts)}")

        if stderr:
            md_lines.append(f"  - Error: {stderr[:200]}")

        # Add running tallies when interesting
        if self._modules_found and phase_changed:
            md_lines.append(f"  - Modules so far: {', '.join(self._modules_found)}")
        if self._predicates_found and len(self._predicates_found) % 5 == 0:
            md_lines.append(f"  - Predicates defined: {len(self._predicates_found)}")

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": self._current_phase,
            "summary": " | ".join(summary_parts),
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_iteration_tick(self, step_number: int, extra: dict) -> dict:
        """Parse a bare iteration tick (no code, just the callback firing)."""
        iteration = extra.get("iteration", step_number)
        duration_ms = extra.get("executionMs", 0)
        duration_s = round(duration_ms / 1000, 1) if duration_ms else 0

        summary = f"Iteration {iteration} completed"
        if duration_s:
            summary += f" ({duration_s}s)"

        markdown = f"**[Step {step_number}]** {summary}"

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": self._current_phase,
            "summary": summary,
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_status_step(self, step_number: int, stdout: str) -> dict:
        """Parse a status event (labeling progress, phase transitions) into a journal entry."""
        markdown = f"**[Step {step_number}]** {stdout.strip()}"

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": self._current_phase,
            "summary": stdout.strip()[:120],
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_artifact_step(
        self, step_number: int, prompt: str | None, response: str | None
    ) -> dict:
        """Parse an emit_artifact checkpoint into a journal entry."""
        # prompt contains e.g. "emit_artifact('supply_list')"
        artifact_name = "unknown"
        if prompt:
            match = re.search(r"emit_artifact\(['\"](\w+)", prompt)
            if match:
                artifact_name = match.group(1)

        # response contains JSON with passed, critical_issues, warnings, phase
        passed = False
        critical_count = 0
        warning_count = 0
        phase_num = 0
        if response:
            try:
                data = json.loads(response)
                passed = data.get("passed", False)
                critical_count = len(data.get("critical_issues", []))
                warning_count = len(data.get("warnings", []))
                phase_num = data.get("phase", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        # Update current phase based on artifact
        phase_map = {
            "supply_list": "Schema", "variables": "Schema",
            "predicates": "Extracting", "modules": "Extracting",
            "router": "Assembling", "integrative": "Assembling",
            "phrase_bank": "Phrases",
        }
        new_phase = phase_map.get(artifact_name, self._current_phase)
        phase_changed = new_phase != self._current_phase
        self._current_phase = new_phase

        status = "PASSED" if passed else f"FAILED ({critical_count} critical, {warning_count} warnings)"
        icon = "+" if passed else "!"

        md_lines = []
        if phase_changed:
            md_lines.append(f"\n## {self._get_phase_label()}")
        md_lines.append(f"**[Step {step_number}]** [{icon}] Checkpoint `{artifact_name}` (Phase {phase_num}): **{status}**")
        if not passed and critical_count > 0:
            md_lines.append(f"  - Model must fix {critical_count} critical issue(s) before proceeding")

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": self._current_phase,
            "summary": f"Checkpoint {artifact_name}: {status}",
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_subcall(
        self, step_number: int, prompt: str | None, response: str | None
    ) -> dict:
        """Parse a sub-call (llm_query) into a journal entry."""
        self._subcall_count += 1

        # Try to detect what the sub-call was for
        purpose = "Sub-call to extraction model"
        if prompt:
            prompt_lower = prompt[:500].lower()
            # Look for module-specific extraction
            mod_match = re.search(r"mod_(\w+)", prompt)
            if mod_match:
                purpose = f"Extracting module **mod_{mod_match.group(1)}**"
            elif "predicate" in prompt_lower:
                purpose = "Extracting predicates and thresholds"
            elif "phrase" in prompt_lower or "message" in prompt_lower:
                purpose = "Extracting phrase bank entries"
            elif "activator" in prompt_lower:
                purpose = "Building activator table"
            elif "router" in prompt_lower:
                purpose = "Building router table"
            elif "danger" in prompt_lower:
                purpose = "Extracting danger signs"

        # Parse response for what was extracted
        response_summary = ""
        if response:
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    if "rules" in parsed:
                        response_summary = f"{len(parsed['rules'])} rules extracted"
                    elif "module_id" in parsed:
                        response_summary = f"Module {parsed['module_id']} with {len(parsed.get('rules', []))} rules"
                elif isinstance(parsed, list):
                    response_summary = f"{len(parsed)} items extracted"
            except (json.JSONDecodeError, TypeError):
                # Response is plain text
                if response:
                    response_summary = f"Response: {len(response)} chars"

        md_lines = [f"**[Step {step_number}]** Sub-call #{self._subcall_count}: {purpose}"]
        if response_summary:
            md_lines.append(f"  - Result: {response_summary}")

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": self._current_phase,
            "summary": f"Sub-call: {purpose}",
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_validation(self, step_number: int, result: dict) -> dict:
        """Parse a validation result into a journal entry."""
        self._validation_runs += 1
        passed = result.get("passed", False)
        error_count = result.get("error_count", 0)
        errors = result.get("errors", [])

        status = "PASSED" if passed else f"FAILED ({error_count} errors)"
        md_lines = [f"**[Step {step_number}]** Validation run #{self._validation_runs}: **{status}**"]

        if not passed and errors:
            # Show top 5 errors
            for err in errors[:5]:
                validator = err.get("validator", "?")
                message = err.get("message", "?")
                severity = err.get("severity", "error")
                icon = "X" if severity == "error" else "!"
                md_lines.append(f"  - [{icon}] [{validator}] {message}")
            if len(errors) > 5:
                md_lines.append(f"  - ... and {len(errors) - 5} more")

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": "Validating",
            "summary": f"Validation: {status}",
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_z3(self, step_number: int, result: dict) -> dict:
        """Parse a Z3 result into a journal entry."""
        self._z3_runs += 1
        all_passed = result.get("all_passed", False)
        checks = result.get("checks", [])
        failed = [c for c in checks if not c.get("passed")]

        status = "ALL PROVED" if all_passed else f"{len(failed)} FAILURES"
        md_lines = [f"**[Step {step_number}]** Z3 verification #{self._z3_runs}: **{status}**"]

        if all_passed:
            md_lines.append(f"  - {len(checks)} checks passed (domain SAT, reachability, exhaustiveness)")
        else:
            for f in failed[:3]:
                md_lines.append(f"  - FAIL: [{f.get('testId', '?')}] {f.get('message', '?')}")
            if len(failed) > 3:
                md_lines.append(f"  - ... and {len(failed) - 3} more failures")

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": "Z3 Verification",
            "summary": f"Z3: {status}",
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_final(self, step_number: int, stdout: str | None) -> dict:
        """Parse the FINAL_VAR step."""
        md_lines = [
            "\n## Extraction Complete",
            f"**[Step {step_number}]** `FINAL_VAR(clinical_logic)` called",
            "",
            "### Final Statistics",
            f"- Modules extracted: {len(self._modules_found)} ({', '.join(self._modules_found)})",
            f"- Predicates defined: {len(self._predicates_found)}",
            f"- Sub-calls made: {self._subcall_count}",
            f"- Validation runs: {self._validation_runs}",
            f"- Z3 verification runs: {self._z3_runs}",
            f"- Total REPL steps: {self._step_count}",
        ]

        markdown = "\n".join(md_lines)

        return {
            "type": "journal",
            "stepNumber": step_number,
            "phase": "Complete",
            "summary": "Extraction complete",
            "markdown": markdown,
            "stats": self._get_stats(),
        }

    def _parse_stdout_discoveries(self, stdout: str) -> list[str]:
        """Extract interesting facts from stdout."""
        discoveries: list[str] = []
        stdout_lower = stdout.lower()

        # Section/key discoveries
        if "sections" in stdout_lower or "keys" in stdout_lower:
            # Try to find list-like output
            key_match = re.findall(r"'(\w+)'", stdout[:500])
            if key_match and len(key_match) <= 20:
                discoveries.append(f"Found keys: {', '.join(key_match[:10])}")

        # Module count discoveries
        mod_matches = re.findall(r"mod_\w+", stdout)
        if mod_matches:
            unique_mods = list(dict.fromkeys(mod_matches))
            for m in unique_mods:
                if m not in self._modules_found:
                    self._modules_found.append(m)
            if unique_mods:
                discoveries.append(f"Modules referenced: {', '.join(unique_mods[:8])}")

        # Predicate discoveries
        pred_matches = re.findall(r"p_\w+", stdout)
        if pred_matches:
            unique_preds = list(dict.fromkeys(pred_matches))
            new_preds = [p for p in unique_preds if p not in self._predicates_found]
            if new_preds:
                self._predicates_found.extend(new_preds)
                discoveries.append(f"New predicates: {', '.join(new_preds[:5])}")

        # Size/count discoveries
        size_match = re.search(r"(\d{3,})\s*(char|byte|token)", stdout_lower)
        if size_match:
            discoveries.append(f"Data size: {size_match.group(0)}")

        return discoveries

    def _get_phase_label(self) -> str:
        """Get a descriptive label for the current phase."""
        labels = {
            "Initializing": "Phase 0: Initializing",
            "Scanning": "Phase 1: Scanning Guide Structure",
            "Schema": "Phase 2: Building Shared Schema",
            "Extracting": "Phase 3: Per-Module Extraction",
            "Assembling": "Phase 4: Assembling Tables",
            "Phrases": "Phase 5: Building Phrase Bank",
            "Validating": "Phase 6: Self-Validation",
            "Z3 Verification": "Phase 6b: Z3 Verification",
            "Complete": "Extraction Complete",
        }
        return labels.get(self._current_phase, f"Phase: {self._current_phase}")

    def _get_stats(self) -> dict:
        """Get current extraction statistics."""
        return {
            "modules_found": len(self._modules_found),
            "modules": self._modules_found[:],
            "predicates_found": len(self._predicates_found),
            "subcalls": self._subcall_count,
            "validation_runs": self._validation_runs,
            "z3_runs": self._z3_runs,
            "total_steps": self._step_count,
            "current_phase": self._current_phase,
        }

    def get_journal_text(self) -> str:
        """Read the full journal markdown."""
        if self.journal_path.exists():
            return self.journal_path.read_text(encoding="utf-8")
        return ""
