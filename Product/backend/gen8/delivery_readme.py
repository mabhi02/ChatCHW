"""gen8 / gen8.5 delivery README builder.

Adapted from `backend/gen7/delivery_readme.py`. Adds a verification
status callout and a header line for the container hash. When the
verifier ran in same-family-fallback mode, the README steers reviewers
to `divergence_worklist.json` as the first stop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _size(path: Path) -> str:
    try:
        b = path.stat().st_size
    except FileNotFoundError:
        return "(missing)"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b/1024:.1f} KB"
    return f"{b/(1024*1024):.1f} MB"


def _rel(output_dir: Path, relative: str) -> str:
    return _size(output_dir / relative)


def build_readme(
    output_dir: Path,
    run_id: str,
    manual_name: str,
    git_sha: str,
    started_at: str,
    wall_clock_sec: float,
    labeled_chunks: list[dict],
    stats: dict[str, Any],
    pipeline: str = "gen8",
    labeler: str = "opus",
    container_sha: str = "",
    verification_summary: dict[str, Any] | None = None,
    stage3_artifact: dict[str, Any] | None = None,
) -> str:
    total_chunks = len(labeled_chunks)
    total_labels = sum(len(c.get("labels", []) or []) for c in labeled_chunks)
    vsum = verification_summary or {}
    vmodel = vsum.get("verifier_model", "none")
    vindep = vsum.get("verifier_independence", "")
    verified = vsum.get("verified", 0)
    failed = vsum.get("verification_failed", 0)
    unverified = vsum.get("unverified", 0)

    L: list[str] = []
    L.append(f"# CHW Navigator ({pipeline}): {manual_name}")
    L.append("")
    L.append("Open this README first. It is the navigation file for everything in the zip.")
    L.append("")
    L.append("| Field | Value |")
    L.append("|-------|-------|")
    L.append(f"| Run ID | `{run_id}` |")
    L.append(f"| Source manual | {manual_name} |")
    L.append(f"| Pipeline | `{pipeline}` (labeler: `{labeler}`) |")
    L.append(f"| Container hash | `{container_sha}` |")
    L.append(f"| Git SHA | `{git_sha}` |")
    L.append(f"| Started | {started_at} |")
    L.append(f"| Wall clock | {wall_clock_sec:.1f}s |")
    L.append(f"| Chunks | {total_chunks} |")
    L.append(f"| Total labels (raw) | {total_labels:,} |")
    if vmodel and vmodel != "none":
        indep_note = f" ({vindep})" if vindep else ""
        L.append(f"| Verifier | `{vmodel}`{indep_note} |")
        L.append(f"| Verification status | {verified} verified / {failed} failed / {unverified} unverified |")
    if "total_cost_usd" in stats:
        L.append(f"| Total cost | ${stats['total_cost_usd']:.2f} |")
    L.append("")

    if vindep == "same-family-fallback" or failed > 0:
        L.append("> **Reviewers should start with `divergence_worklist.json`.** "
                 "Items are signal-ranked: errors first, then high-frequency divergences.")
        L.append("")

    L.append("## Quick start (where to look first)")
    L.append("")
    L.append("- **Decision logic the model produced:** `clinical_logic.json` plus `flowchart_index.png` (module overview).")
    L.append("- **Deployable artifacts:** `form.xlsx` (aggregate XLSForm) and `clinical_logic.dmn`. Per-module review workbooks in `form_per_module/`.")
    L.append("- **Per-module flowcharts:** `flowcharts_per_module/flowchart_<mid>.png`.")
    L.append("- **Verifier flagged issues:** `divergence_worklist.json` (errors first, then high-frequency patterns).")
    L.append("- **What the LLM saw and said:** `system_prompt.md` (multi-stage bundle), `stage3_repl.json` (final Stage 3 input/output).")
    L.append("- **Per-variable scope and lifecycle:** `artifacts/data_flow.json`.")
    L.append("- **Stockout coverage gaps:** `artifacts/stockout_coverage.json`.")
    L.append("")

    L.append("## How this bundle was produced")
    L.append("")
    L.append(f"Pipeline `{pipeline}` with labeler `{labeler}`. All generator calls at temperature 0 "
             "against the same content-hashed PDF.")
    L.append("")
    L.append("1. **Chunk** (deterministic Python). Cached PDF JSON is split into ~2K-token chunks with page ranges threaded from the ingestion layer.")
    if labeler == "sonnet7way":
        L.append("2. **Label (7-way Sonnet).** Each chunk runs 7 narrow-type Sonnet passes in parallel; a deterministic reconciler dedups and ground-truth-filters spans.")
    else:
        L.append("2. **Label (Opus mono).** Each chunk runs one Opus Stage 1A labeling call followed by a Stage 1B QC pass on the candidate labels.")
    L.append("3. **Dedupe** (deterministic Python). Exact-string dedup on `(id, type)`.")
    L.append("4. **Compile** (Opus REPL). Reads cached deduped labels + reconstructed guide and emits seven artifacts in DAG order.")
    L.append("5. **Reconcile** (deterministic Python). Referential integrity audit + Tier-0 predicate grammar validation + triplet dedup + Tier-2 stockout coverage + extended data_flow.")
    L.append("6. **Sidecar** (provenance). Every artifact gets a `.provenance.json` sidecar; the run-level `manifest.json` rolls them into a container hash.")
    if vmodel and vmodel != "none":
        L.append(f"7. **Verify.** An independent pass (`{vmodel}`) reads each artifact and emits `.verification.json` sidecars. Disagreements never edit the artifact; they land in `divergence_worklist.json`.")
    L.append("")

    L.append("## File map")
    L.append("")
    L.append("### Reviewer / deploy outputs")
    L.append("")
    L.append("| File | What it is | Size |")
    L.append("|------|------------|------|")
    for name, desc in [
        ("manifest.json", "Run-level container hash + per-artifact SHAs."),
        ("clinical_logic.json", "Decision model source of truth."),
        ("clinical_logic.dmn", "DMN 1.3 XML. Deployable to any DMN engine."),
        ("form.xlsx", "Aggregate XLSForm deployable to CHT."),
        ("form_per_module/", "Per-module XLSForm workbooks for focused review."),
        ("flowchart_index.png", "Top-level module map with inter-module routes."),
        ("flowcharts_per_module/", "Per-module flowcharts with explicit Entry/Exit bands."),
        ("divergence_worklist.json", "Verifier's signal-ranked review queue."),
    ]:
        L.append(f"| `{name}` | {desc} | {_rel(output_dir, name)} |")
    L.append("")

    L.append("### Per-phase artifacts (with provenance + verification sidecars)")
    L.append("")
    L.append("| File | What it is |")
    L.append("|------|------------|")
    for name, desc in [
        ("artifacts/supply_list.json", "Equipment + consumables the CHW must have."),
        ("artifacts/variables.json", "Runtime inputs: symptoms, exam, measurements, demographics."),
        ("artifacts/predicates.json", "Boolean predicates with Tier-0 strict schema (units, missingness, domain, provenance)."),
        ("artifacts/modules.json", "Decision modules (assess, diarrhoea, fever, etc.)."),
        ("artifacts/router.json", "Two-traffic-cop router: `cop1_queue_builder` (COLLECT) + `cop2_next_module` (UNIQUE)."),
        ("artifacts/integrative.json", "Cross-module merge rules."),
        ("artifacts/phrase_bank.json", "Every phrase the CHW says."),
        ("artifacts/data_flow.json", "Per-variable scope (universal, module_local, cross_module, derived, conditionally_collected, orphan)."),
        ("artifacts/referential_integrity.json", "Cross-reference audit + auto-reconcile report."),
        ("artifacts/predicates_validation.json", "Tier-0 grammar validation + triplet-dedup report."),
        ("artifacts/stockout_coverage.json", "Gap list: (module, rule, supply) triples missing a stockout fallback."),
    ]:
        L.append(f"| `{name}` | {desc} |")
    L.append("")
    L.append("Every JSON artifact above has a sibling `.provenance.json` "
             "(who/when/what/SHA) and, if Tier-3 ran, a `.verification.json` "
             "(verifier agreement + divergences).")
    L.append("")

    L.append("### Logging and traceability")
    L.append("")
    L.append("| File | What it is |")
    L.append("|------|------------|")
    for name, desc in [
        ("system_prompt.md", "Full multi-stage system prompt bundle."),
        ("stage3_repl.json", "Stage 3 REPL session transcript."),
        ("labeled_chunks.json", "Stage 1 per-chunk labels + _labeling_meta."),
        ("deduped_labels.json", "Consolidated labels after exact-string dedup."),
        ("reconstructed_guide.txt", "Full text the labeler saw."),
    ]:
        L.append(f"| `{name}` | {desc} |")
    L.append("")

    return "\n".join(L)
