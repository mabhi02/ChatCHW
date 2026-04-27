"""gen8 orchestrator: Phase 0 (chunk) -> Phase 1 (label) -> Phase 2 (REPL)
                    -> Tier-0/1/2 post-process -> Tier-3 verify.

`run(..., labeler="opus"|"sonnet7way")` selects between gen8 and gen8.5.
The two labelers share this pipeline; everything downstream is identical.

Writes every artifact through `provenance.write_with_provenance` so each
JSON gets a sibling `.provenance.json`. After all writes land, the
verifier produces `.verification.json` sidecars and the worklist builder
emits `divergence_worklist.json`. `manifest.json` at the run root rolls
the whole thing up into one container hash.
"""

from __future__ import annotations

import asyncio
import ast as _ast
import json
import logging
import time
from pathlib import Path
from typing import Any

from backend.gen8.chunker import micro_chunk_guide
from backend.gen8 import system_prompt_bundle as gen8_prompt
from backend.gen8 import traffic_cops
from backend.gen8.data_flow import derive as data_flow_derive
from backend.provenance.container import get_git_sha, write_manifest
from backend.provenance.schema import ParentRef, SourceManual
from backend.provenance.sidecar import (
    write_with_provenance,
    write_binary_with_provenance,
    compute_sha256,
)
from backend.validators.predicate_grammar import (
    validate_all as validate_predicates_all,
    dedup_predicate_triplets,
)
from backend.validators.predicate_completeness import (
    build_report as predicate_completeness_report,
    to_divergences as predicate_completeness_divergences,
)
from backend.validators.dmn_predicate_only import (
    check as dmn_predicate_only_check,
    to_divergences as dmn_predicate_only_divergences,
)
from backend.validators.imci_canonical import (
    check as imci_canonical_check,
    to_divergences as imci_canonical_divergences,
)
from backend.validators.stockout_coverage import build_report as stockout_report
from backend.verifier import write_verification_sidecar, update_provenance_status
from backend.verifier.runner import verify_artifact
from backend.verifier.worklist import build_worklist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REPL system prompt: copied from gen7, extended with Tier-0/2 prompt blocks
# ---------------------------------------------------------------------------

def _build_repl_system_prompt(codebook: str, publication_year: str | None = None) -> str:
    from backend.gen7.pipeline import _build_repl_system_prompt as _gen7_repl
    base = _gen7_repl(codebook)

    # Tier-0 prompt with the actual publication year stitched in (replaces
    # the example "WHO 2012 page <N>" so Opus doesn't guess the year).
    tier0 = gen8_prompt.TIER0_PREDICATE_RULES
    if publication_year:
        tier0 = tier0.replace("\"WHO 2012 page <N>\"",
                              f"\"WHO {publication_year} page <N>\"")
        tier0 += (
            f"\n## Publication year (USE THIS EXACTLY)\n\n"
            f"The source manual was published in {publication_year}. Every predicate's\n"
            f"`provenance` field MUST cite that year, e.g. `\"WHO {publication_year} page 23\"`.\n"
            f"Do NOT guess a different year.\n"
        )

    extras = (
        "\n\n================================================================\n"
        "gen8 ADDITIONAL RULES (Tier 0, Tier 2)\n"
        "================================================================\n\n"
        + tier0
        + "\n"
        + gen8_prompt.TIER0_NO_STUB_PREDICATES
        + "\n"
        + gen8_prompt.TIER0_PREDICATE_ONLY_DMN
        + "\n"
        + gen8_prompt.TIER2_MANUAL_ORDERING
        + "\n"
        + gen8_prompt.TIER2_INPUT_COMPLETENESS
        + "\n"
        + gen8_prompt.TIER2_TRAFFIC_COPS
    )
    return base + extras


# ---------------------------------------------------------------------------
# Source manual fetch
# ---------------------------------------------------------------------------

def _build_source_manual(
    guide_json: dict,
    manual_name_hint: str | None,
    source_guide_meta: dict | None = None,
) -> SourceManual:
    """Construct the `SourceManual` record for provenance sidecars.

    `source_guide_meta` (when provided by session_manager from a Neon
    SourceGuide row) takes precedence over `guide_json.metadata`.  The
    cached guide JSON does not always carry the SHA in its metadata, so
    relying on the row directly is the only way to anchor the manifest
    container hash to the real PDF bytes.
    """
    meta = (guide_json.get("metadata") or {}) if isinstance(guide_json, dict) else {}
    sg = source_guide_meta or {}
    name = (
        manual_name_hint
        or sg.get("manualName")
        or meta.get("manual_name")
        or meta.get("filename")
        or "unknown"
    )
    sha = (
        sg.get("sha256")
        or meta.get("sha256")
        or meta.get("pdf_sha256")
        or "unknown"
    )
    page_count = int(
        sg.get("pageCount")
        or meta.get("page_count")
        or meta.get("pageCount")
        or len(guide_json.get("pages", {}) or {})
        or 0
    )
    publisher = meta.get("publisher") or "unknown"

    # Publication year: prefer explicit field, then extract a 4-digit year
    # from the manual name (e.g. "WHO CHW guide 2012") or the publisher.
    import re
    year = (
        sg.get("publication_year")
        or meta.get("publication_year")
        or meta.get("year")
    )
    if not year:
        # Probe every reasonable string field; the manual_name_hint may
        # override the source-guide name and lose the year, so we still
        # try the original source-guide manualName + filename as fallbacks.
        for s in (name, sg.get("manualName"), sg.get("filename"), publisher):
            m = re.search(r"\b(19|20)\d{2}\b", str(s or ""))
            if m:
                year = m.group(0)
                break

    return SourceManual(
        name=name,
        publisher=publisher,
        pdf_sha256=sha,
        page_count=page_count,
        publication_year=str(year) if year else None,
        neon_guide_id=(
            sg.get("neon_guide_id")
            or meta.get("neon_guide_id")
            or meta.get("source_guide_id")
        ),
    )


# ---------------------------------------------------------------------------
# Post-Stage-3 deterministic reconcile
# ---------------------------------------------------------------------------

def _post_stage3_reconcile(clinical_logic: dict) -> dict:
    """Run Tier-0/1/2 deterministic post-processing IN MEMORY only.

    Mutates `clinical_logic` in place (predicate dedup, router migration,
    referential-integrity auto-fills). Returns a dict of per-check reports
    so the caller can persist them with the correct (post-reconcile) parent
    SHA.

    Order matters:
      1. Triplet dedup runs FIRST so user-emitted duplicates collapse before
         the auto-reconciler sees them.
      2. Router migration + stockout coverage run on the dedup'd dict.
      3. Auto-reconciler (`referential_integrity`) backfills missing ids.
      4. **Predicate grammar validation runs AFTER auto-reconcile** so newly
         auto-registered predicates are also grammar-checked. Without this,
         `predicates_validation.json` reflected only the model-emitted subset
         (~25 of 53 on the previous WHO 2012 run).
      5. data_flow runs last so its scope classifier sees the final
         module/predicate/variable lists.
    """
    # Step 1: triplet dedup
    preds = list(clinical_logic.get("predicates", []) or [])
    kept, dropped = dedup_predicate_triplets(preds)
    clinical_logic["predicates"] = kept

    # Step 2: router migration + stockout
    router = clinical_logic.get("router") or {}
    router = traffic_cops.migrate_flat_router_to_cops(router)
    router_errors = traffic_cops.validate_router(router)
    clinical_logic["router"] = router

    stock_report = stockout_report(clinical_logic)

    # Step 3: legacy auto-reconcile (may add predicates / variables / etc)
    audit_report: dict = {}
    try:
        from backend.validators.referential_integrity import audit_and_reconcile
        audit_report = audit_and_reconcile(clinical_logic)
        patched = audit_report.get("patched", {}) or {}
        for key in ("variables", "predicates", "supply_list", "phrase_bank"):
            if key in patched:
                clinical_logic[key] = patched[key]
    except Exception as exc:
        logger.warning("gen8 referential_integrity failed: %s", exc, exc_info=True)

    # Step 4: grammar-check the FINAL predicate list (post-reconcile)
    final_preds = list(clinical_logic.get("predicates", []) or [])
    pred_validation = validate_predicates_all(final_preds)
    pred_validation["validated_at"] = "post_reconcile"
    pred_validation["pre_reconcile_count"] = len(kept)
    pred_validation["post_reconcile_count"] = len(final_preds)
    if pred_validation["with_errors"] > 0:
        logger.warning("gen8 Tier 0 (post-reconcile): %d predicates have schema errors",
                       pred_validation["with_errors"])

    # Step 5: data_flow scope classification (sees fully reconciled state)
    data_flow = data_flow_derive(clinical_logic)

    # Step 6: clinical-content catchers (run AFTER reconcile so they see
    # the full final picture). These produce additional reports and
    # supplemental divergences the worklist picks up.
    completeness = predicate_completeness_report(clinical_logic.get("predicates") or [])
    imci_coverage = imci_canonical_check(clinical_logic)

    return {
        "predicate_validation": pred_validation,
        "predicate_dropped": dropped,
        "router_errors": router_errors,
        "stockout_gaps": stock_report,
        "data_flow": data_flow,
        "referential_integrity": audit_report,
        "predicate_completeness": completeness,
        "imci_coverage": imci_coverage,
    }


def _write_reconcile_reports(
    reports: dict,
    output_dir: Path,
    prov_kwargs: dict,
    clinical_logic_sha: str,
) -> None:
    """Persist the reconcile reports with the post-reconcile parent SHA."""
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    parent = [ParentRef(kind="clinical_logic", content_sha256=clinical_logic_sha)]

    write_with_provenance(
        artifacts_dir / "predicates_validation.json",
        {"report": reports["predicate_validation"],
         "dropped_dups": reports["predicate_dropped"]},
        artifact_kind="predicates_validation",
        parents=parent,
        **prov_kwargs,
    )
    write_with_provenance(
        artifacts_dir / "stockout_coverage.json",
        reports["stockout_gaps"],
        artifact_kind="stockout_coverage",
        parents=parent,
        **prov_kwargs,
    )
    write_with_provenance(
        artifacts_dir / "data_flow.json",
        reports["data_flow"],
        artifact_kind="data_flow",
        parents=parent,
        **prov_kwargs,
    )
    audit = reports.get("referential_integrity") or {}
    if audit:
        write_with_provenance(
            artifacts_dir / "referential_integrity.json",
            {
                "audit_before": audit.get("audit_before", {}),
                "reconciliation": audit.get("reconciliation", {}),
                "audit_after": audit.get("audit_after", {}),
            },
            artifact_kind="referential_integrity",
            parents=parent,
            **prov_kwargs,
        )

    # gen8 catchers added in this iteration: stub-predicate detection +
    # IMCI canonical danger sign coverage.
    if "predicate_completeness" in reports:
        write_with_provenance(
            artifacts_dir / "predicate_completeness.json",
            reports["predicate_completeness"],
            artifact_kind="predicate_completeness",
            parents=parent,
            **prov_kwargs,
        )
    if "imci_coverage" in reports:
        write_with_provenance(
            artifacts_dir / "imci_canonical_coverage.json",
            reports["imci_coverage"],
            artifact_kind="imci_canonical_coverage",
            parents=parent,
            **prov_kwargs,
        )


# ---------------------------------------------------------------------------
# Verifier loop
# ---------------------------------------------------------------------------

def _propagate_clinical_logic_status(output_dir: Path, summary: dict) -> None:
    """Roll the per-artifact verifier verdicts up to clinical_logic.json's
    provenance status.

    Logic:
      - any per-artifact verification with severity=error -> verification_failed
      - all per-artifact verifications agree (only warn/info) -> verified
      - else -> unverified (default; e.g. no verifier ran)
    """
    cl_prov = output_dir / "clinical_logic.json.provenance.json"
    if not cl_prov.exists():
        return

    artifacts_dir = output_dir / "artifacts"
    has_error = False
    all_agree = True
    any_seen = False
    for vpath in sorted(artifacts_dir.glob("*.json.verification.json")):
        try:
            v = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        any_seen = True
        if not v.get("agree", False):
            all_agree = False
        for d in v.get("divergences", []) or []:
            if d.get("severity") == "error":
                has_error = True

    if not any_seen:
        return

    if has_error:
        new_status = "verification_failed"
    elif all_agree:
        new_status = "verified"
    else:
        new_status = "verification_failed"

    try:
        prov = json.loads(cl_prov.read_text(encoding="utf-8"))
        prov["status"] = new_status
        cl_prov.write_text(json.dumps(prov, indent=2), encoding="utf-8")
        summary["clinical_logic_status"] = new_status
    except Exception as exc:
        logger.warning("gen8: failed to propagate clinical_logic status: %s", exc)


def _verify_all_artifacts(
    output_dir: Path,
    generator_model: str,
    reconstructed_guide: str,
    stage3_system_prompt: str,
) -> dict:  # noqa: positional-arg signature is intentional (asyncio.to_thread)
    """Run the verifier over every per-artifact JSON and emit sidecars."""
    artifacts_dir = output_dir / "artifacts"
    verify_kinds = [
        "supply_list", "variables", "predicates", "modules",
        "router", "integrative", "phrase_bank",
    ]

    summary: dict[str, Any] = {
        "verifier_model": None,
        "verifier_independence": None,
        "verified": 0,
        "verification_failed": 0,
        "unverified": 0,
        "per_artifact": [],
    }

    for kind in verify_kinds:
        apath = artifacts_dir / f"{kind}.json"
        prov_path = apath.with_suffix(apath.suffix + ".provenance.json")
        if not apath.exists() or not prov_path.exists():
            summary["unverified"] += 1
            continue
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            vblock = verify_artifact(
                apath,
                artifact_kind=kind,
                artifact_content_sha=prov["content_sha256"],
                source_manual_text=reconstructed_guide,
                generator_prompts={"stage3_prompt": stage3_system_prompt},
                generator_model=generator_model,
            )
            write_verification_sidecar(apath, vblock)
            update_provenance_status(apath, vblock)
            if vblock.agree and not any(d.severity == "error" for d in vblock.divergences):
                summary["verified"] += 1
            else:
                summary["verification_failed"] += 1
            summary["verifier_model"] = vblock.verifier_model
            summary["verifier_independence"] = vblock.verifier_independence
            summary["per_artifact"].append({
                "kind": kind,
                "agree": vblock.agree,
                "divergence_count": len(vblock.divergences),
            })
        except Exception as exc:
            logger.warning("gen8 verify %s failed: %s", kind, exc)
            summary["unverified"] += 1

    _propagate_clinical_logic_status(output_dir, summary)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = (
    "(original text as provided)",
    "(content unavailable)",
    "(no text extracted)",
)


def _is_real_chunk(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) < 20:
        return False
    cleaned = t
    for p in _PLACEHOLDER_PATTERNS:
        cleaned = cleaned.replace(p, "").replace(p.lower(), "").strip()
    return len(cleaned) >= 50


async def run(
    guide_json: dict,
    anthropic_key: str,
    naming_codebook: str = "",
    output_dir: Path | None = None,
    on_progress: Any = None,
    manual_name_hint: str | None = None,
    labeler: str = "opus",
    run_verifier: bool = True,
    source_guide_meta: dict | None = None,
) -> dict[str, Any]:
    """Run gen8 (labeler=opus) or gen8.5 (labeler=sonnet7way) end-to-end."""
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pipeline_name = "gen8.5" if labeler == "sonnet7way" else "gen8"
    run_id = output_dir.name
    source_manual = _build_source_manual(guide_json, manual_name_hint, source_guide_meta)
    git_sha = get_git_sha()
    model_tag = "claude-opus-4-6" if labeler == "opus" else "claude-sonnet-4-6"
    base_prov = dict(
        run_id=run_id,
        pipeline=pipeline_name,
        labeler=labeler,
        pipeline_git_sha=git_sha,
        model=model_tag,
        source_manual=source_manual,
    )

    try:
        from backend.rlm_runner import _reset_run_usage
        _reset_run_usage()
    except Exception:
        pass

    start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start))

    # ---------- Phase 0: chunk ----------
    if on_progress:
        await on_progress({"phase": "chunk", "status": "starting"})
    chunks = micro_chunk_guide(guide_json)
    if on_progress:
        await on_progress({"phase": "chunk", "status": "done", "count": len(chunks)})

    # ---------- Phase 1: label ----------
    if labeler == "sonnet7way":
        from backend.gen8.labeler_sonnet7way import label_all_chunks, deduplicate_labels
    else:
        from backend.gen8.labeler_opus import label_all_chunks, deduplicate_labels

    if on_progress:
        await on_progress({"phase": "label", "status": "starting", "model": model_tag})

    async def _on_chunk_labeled(event: dict) -> None:
        if on_progress:
            await on_progress({**event, "phase": "label"})

    labeled_chunks = await label_all_chunks(
        chunks, naming_codebook, anthropic_key,
        on_chunk_labeled=_on_chunk_labeled,
    )
    total_labels = sum(len(c.get("labels", []) or []) for c in labeled_chunks)
    label_errors = sum(
        1 for c in labeled_chunks
        if c.get("_labeling_meta", {}).get("error")
    )
    if on_progress:
        await on_progress({
            "phase": "label", "status": "done",
            "total_labels": total_labels, "errors": label_errors,
        })

    deduped_labels = deduplicate_labels(labeled_chunks)

    reconstructed_guide = "\n\n".join(
        c.get("text", "") for c in sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        if _is_real_chunk(c.get("text", ""))
    )

    labeled_sha = write_with_provenance(
        output_dir / "labeled_chunks.json",
        labeled_chunks,
        artifact_kind="labeled_chunks",
        parents=[],
        **base_prov,
    )
    deduped_sha = write_with_provenance(
        output_dir / "deduped_labels.json",
        deduped_labels,
        artifact_kind="deduped_labels",
        parents=[ParentRef(kind="labeled_chunks", content_sha256=labeled_sha)],
        **base_prov,
    )
    reconstructed_bytes = reconstructed_guide.encode("utf-8")
    write_binary_with_provenance(
        output_dir / "reconstructed_guide.txt",
        reconstructed_bytes,
        artifact_kind="reconstructed_guide",
        parents=[],
        **base_prov,
    )
    reconstructed_sha = compute_sha256(reconstructed_bytes)

    write_with_provenance(
        output_dir / "chunk_difficulty.json",
        {
            "total_chunks": len(chunks),
            "distribution": {
                d: sum(1 for c in chunks if c.get("difficulty") == d)
                for d in ["trivial", "easy", "medium", "hard", "extreme"]
            },
            "chunks": [
                {"index": c["chunk_index"], "section": c["section_id"],
                 "difficulty": c["difficulty"], "score": c.get("difficulty_score", 0),
                 "manual_page_start": c.get("manual_page_start"),
                 "manual_page_end": c.get("manual_page_end")}
                for c in chunks
            ],
        },
        artifact_kind="chunk_difficulty",
        parents=[],
        **base_prov,
    )

    # ---------- Phase 2: REPL compile ----------
    if on_progress:
        await on_progress({"phase": "compile", "status": "starting"})

    system_prompt = _build_repl_system_prompt(
        naming_codebook,
        publication_year=source_manual.publication_year,
    )
    system_prompt_runtime = system_prompt.replace("{", "{{").replace("}", "}}")
    system_prompt_runtime += "\n\n{custom_tools_section}"

    import sys
    rlm_path = str(Path(__file__).parent.parent.parent / "rlm")
    if rlm_path not in sys.path:
        sys.path.insert(0, rlm_path)
    from rlm import RLM
    from rlm.logger import RLMLogger

    repl_context = [
        {
            "chunk_index": c.get("chunk_index", -1),
            "section_id": c.get("section_id", ""),
            "section_title": c.get("section_title", ""),
            "text": c.get("text", ""),
            "labels": c.get("labels", []),
            "manual_page_start": c.get("manual_page_start"),
            "manual_page_end": c.get("manual_page_end"),
        }
        for c in labeled_chunks
    ]

    rlm_logger = RLMLogger()

    from backend.rlm_runner import _set_gen7_cached_context
    _set_gen7_cached_context(reconstructed_guide, deduped_labels)

    initial_message = (
        f"The labeled guide has {len(repl_context)} chunks with {total_labels} raw labels "
        f"and {len(deduped_labels)} deduped entries. Each chunk includes manual_page_start/end "
        f"so every predicate you emit MUST have provenance = \"WHO <year> page <N>\".\n\n"
        "Follow the gen8 rules (predicate strict schema, two-traffic-cop router, manual ordering)."
    )

    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": "claude-opus-4-6",
            "api_key": anthropic_key,
            "max_tokens": 16384,
        },
        environment="local",
        environment_kwargs={"context_payload": repl_context},
        max_iterations=30,
        custom_system_prompt=system_prompt_runtime,
        compaction=False,
        max_budget=30.0,
        max_errors=5,
        max_timeout=1800.0,
        logger=rlm_logger,
        verbose=False,
    )

    stage3_artifact: dict[str, Any] = {
        "phase": "stage3_repl",
        "model": "claude-opus-4-6",
        "system_prompt": system_prompt,
        "cached_source_guide": reconstructed_guide,
        "cached_deduped_labels": deduped_labels,
        "initial_user_message": initial_message,
        "response_text": None,
        "total_iterations": None,
    }

    try:
        result = await asyncio.to_thread(rlm.completion, initial_message)
    finally:
        _set_gen7_cached_context(None, None)

    response_text = getattr(result, "response", None)
    stage3_artifact["response_text"] = str(response_text or "")
    try:
        traj = rlm_logger.get_trajectory()
        stage3_artifact["total_iterations"] = len((traj or {}).get("iterations", []))
    except Exception:
        stage3_artifact["total_iterations"] = None

    clinical_logic = _parse_repl_result(response_text) or {}

    if on_progress:
        await on_progress({"phase": "compile", "status": "done",
                           "elapsed": round(time.time() - start, 1)})

    stage3_sha = compute_sha256(json.dumps(stage3_artifact, sort_keys=True, default=str).encode())
    write_with_provenance(
        output_dir / "stage3_repl.json",
        stage3_artifact,
        artifact_kind="stage3_repl",
        parents=[
            ParentRef(kind="deduped_labels", content_sha256=deduped_sha),
            ParentRef(kind="reconstructed_guide", content_sha256=reconstructed_sha),
        ],
        **base_prov,
    )

    clinical_logic_sha = write_with_provenance(
        output_dir / "clinical_logic.json",
        clinical_logic,
        artifact_kind="clinical_logic",
        parents=[ParentRef(kind="stage3_repl", content_sha256=stage3_sha)],
        **base_prov,
    )

    for kind in ("supply_list", "variables", "predicates", "modules",
                 "router", "integrative", "phrase_bank"):
        default = {} if kind in ("router", "integrative") else []
        write_with_provenance(
            artifacts_dir / f"{kind}.json",
            clinical_logic.get(kind, default),
            artifact_kind=kind,
            parents=[ParentRef(kind="clinical_logic", content_sha256=clinical_logic_sha)],
            **base_prov,
        )

    # ---------- Tier-0 / Tier-1 / Tier-2 deterministic post-process ----------
    # In-memory only; reports are persisted AFTER the post-reconcile
    # `clinical_logic.json` write so their `parents[0]` SHA references the
    # bytes the verifier will actually read.
    reports = _post_stage3_reconcile(clinical_logic)

    # Re-emit clinical_logic + per-artifact files so on-disk bytes match
    # what the verifier and downstream converters see.
    clinical_logic_sha = write_with_provenance(
        output_dir / "clinical_logic.json",
        clinical_logic,
        artifact_kind="clinical_logic",
        parents=[ParentRef(kind="stage3_repl", content_sha256=stage3_sha)],
        **base_prov,
    )
    for kind in ("supply_list", "variables", "predicates", "modules",
                 "router", "integrative", "phrase_bank"):
        default = {} if kind in ("router", "integrative") else []
        write_with_provenance(
            artifacts_dir / f"{kind}.json",
            clinical_logic.get(kind, default),
            artifact_kind=kind,
            parents=[ParentRef(kind="clinical_logic", content_sha256=clinical_logic_sha)],
            **base_prov,
        )

    # Now persist the four reconcile reports with the post-reconcile SHA.
    _write_reconcile_reports(reports, output_dir, base_prov, clinical_logic_sha)

    # ---------- Converters ----------
    try:
        from backend.converters import convert_to_dmn, convert_to_csv
        from backend.converters.xlsform_split import emit_all as emit_xlsx_all
        from backend.converters.mermaid_split import emit_all as emit_mermaid_all
    except Exception as exc:
        logger.warning("gen8: converter imports failed: %s", exc)
        convert_to_dmn = None
        convert_to_csv = None
        emit_xlsx_all = None
        emit_mermaid_all = None

    if convert_to_dmn:
        try:
            dmn_xml = convert_to_dmn(clinical_logic)
            write_binary_with_provenance(
                output_dir / "clinical_logic.dmn",
                dmn_xml.encode("utf-8"),
                artifact_kind="clinical_logic_dmn",
                parents=[ParentRef(kind="clinical_logic", content_sha256=clinical_logic_sha)],
                **base_prov,
            )
        except Exception as exc:
            logger.warning("gen8: DMN conversion failed: %s", exc)

    if emit_xlsx_all:
        try:
            emit_xlsx_all(clinical_logic, output_dir)
        except Exception as exc:
            logger.warning("gen8: xlsform_split failed: %s", exc)

    if emit_mermaid_all:
        try:
            emit_mermaid_all(clinical_logic, output_dir)
        except Exception as exc:
            logger.warning("gen8: mermaid_split failed: %s", exc)

    if convert_to_csv:
        try:
            csvs = convert_to_csv(clinical_logic)
            for name, content in csvs.items():
                (output_dir / f"{name}.csv").write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("gen8: CSV conversion failed: %s", exc)

    # ---------- Tier 3: verify ----------
    verification_summary: dict[str, Any] = {
        "verifier_model": "none",
        "verifier_independence": None,
        "verified": 0,
        "verification_failed": 0,
        "unverified": 0,
        "per_artifact": [],
    }
    if run_verifier:
        try:
            # Verifier issues blocking HTTP calls; run it in a worker thread
            # so the SSE keep-alive / Redis heartbeat keep firing during the
            # multi-minute verification pass.
            verification_summary = await asyncio.to_thread(
                _verify_all_artifacts,
                output_dir,
                model_tag,
                reconstructed_guide,
                system_prompt,
            )
            # Build supplemental catcher divergences AFTER the DMN exists so
            # the predicate-only check can scan its <input> expressions.
            extra_rows: list[dict] = []
            preds = clinical_logic.get("predicates") or []
            for d in predicate_completeness_divergences(preds):
                extra_rows.append({"artifact_kind": "predicates", **d})
            for d in imci_canonical_divergences(reports.get("imci_coverage") or {}):
                extra_rows.append({"artifact_kind": "imci_coverage", **d})
            dmn_only = dmn_predicate_only_check(
                clinical_logic,
                dmn_path=output_dir / "clinical_logic.dmn",
            )
            for d in dmn_predicate_only_divergences(dmn_only):
                extra_rows.append({"artifact_kind": "router", **d})
            # Persist the DMN-only report alongside the rest of the catcher reports
            try:
                write_with_provenance(
                    artifacts_dir / "dmn_predicate_only.json",
                    dmn_only,
                    artifact_kind="dmn_predicate_only",
                    parents=[ParentRef(kind="clinical_logic", content_sha256=clinical_logic_sha)],
                    **base_prov,
                )
            except Exception as exc:
                logger.warning("gen8: dmn_predicate_only sidecar failed: %s", exc)

            worklist_path = build_worklist(output_dir, extra_rows=extra_rows)
            logger.info("gen8 verify: worklist at %s (catcher rows: %d)",
                        worklist_path, len(extra_rows))
            verification_summary["catcher_divergence_count"] = len(extra_rows)
        except Exception as exc:
            logger.warning("gen8 verifier loop failed: %s", exc, exc_info=True)

    # ---------- Multi-stage sendable system prompt bundle ----------
    try:
        if labeler == "sonnet7way":
            from backend.gen8.labeler_sonnet7way import _pass_prompt as _sv_pass
            label_prompt = _sv_pass("supply_list", naming_codebook)
            distill_prompt = ("(gen8.5 does not run a Stage 1B distillation; each of 7 "
                              "narrow passes is its own QC.)")
        else:
            from backend.gen8.labeler_opus import (
                _build_labeling_system_prompt,
                _build_distillation_system_prompt,
            )
            label_prompt = _build_labeling_system_prompt(naming_codebook)
            distill_prompt = _build_distillation_system_prompt(naming_codebook)
        sendable = gen8_prompt.build_sendable_system_prompt(
            labeling_prompt=label_prompt,
            distillation_prompt=distill_prompt,
            repl_prompt=system_prompt_runtime,
            naming_codebook=naming_codebook,
            run_id=run_id,
            manual_name=source_manual.name,
            pipeline=pipeline_name,
            labeler=labeler,
            container_sha="",
            verifier_model=verification_summary.get("verifier_model") or "",
            verifier_independence=verification_summary.get("verifier_independence") or "",
        )
        (output_dir / "system_prompt.md").write_text(sendable, encoding="utf-8")
    except Exception as exc:
        logger.warning("gen8 sendable system prompt failed: %s", exc)

    # ---------- Manifest (container hash) ----------
    artifact_shas: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name == "manifest.json":
            continue
        if name.endswith(".provenance.json") or name.endswith(".verification.json"):
            continue
        try:
            artifact_shas[str(path.relative_to(output_dir)).replace("\\", "/")] = compute_sha256(path.read_bytes())
        except Exception:
            continue
    container_sha = write_manifest(
        output_dir,
        artifact_shas=artifact_shas,
        source_pdf_sha=source_manual.pdf_sha256,
        git_sha=git_sha,
        verifier_model=verification_summary.get("verifier_model") or "none",
        pipeline=pipeline_name,
        labeler=labeler,
        run_id=run_id,
    )

    # ---------- README ----------
    try:
        from backend.gen8.delivery_readme import build_readme
        wall_clock = time.time() - start
        stats = _assemble_stats(labeled_chunks, wall_clock, len(chunks), total_labels,
                                label_errors, pipeline_name, labeler)
        readme = build_readme(
            output_dir=output_dir,
            run_id=run_id,
            manual_name=source_manual.name,
            git_sha=git_sha,
            started_at=started_at,
            wall_clock_sec=wall_clock,
            labeled_chunks=labeled_chunks,
            stats=stats,
            pipeline=pipeline_name,
            labeler=labeler,
            container_sha=container_sha,
            verification_summary=verification_summary,
            stage3_artifact=stage3_artifact,
        )
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
    except Exception as exc:
        logger.warning("gen8 README build failed: %s", exc, exc_info=True)

    elapsed = time.time() - start
    status = "passed" if clinical_logic and isinstance(clinical_logic, dict) and clinical_logic else "failed"

    stats = _assemble_stats(labeled_chunks, elapsed, len(chunks), total_labels,
                            label_errors, pipeline_name, labeler)

    return {
        "clinical_logic": clinical_logic,
        "labeled_chunks": labeled_chunks,
        "chunk_difficulty": [
            {"index": c["chunk_index"], "section": c["section_id"],
             "difficulty": c["difficulty"]}
            for c in chunks
        ],
        "stats": stats,
        "status": status,
        "container_sha": container_sha,
        "verification_summary": verification_summary,
        "reports": reports,
    }


def _parse_repl_result(response_text: Any) -> dict | None:
    """Parse the REPL FINAL_VAR result back into a dict.

    Tries JSON first, then `ast.literal_eval` (safe: parses Python
    literals only; never executes arbitrary code). Same strategy as the
    gen7 pipeline.
    """
    if isinstance(response_text, dict):
        return response_text
    if not isinstance(response_text, str):
        return None
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = _ast.literal_eval(response_text)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def _assemble_stats(labeled_chunks, elapsed, n_chunks, total_labels, label_errors,
                    pipeline_name, labeler) -> dict:
    label_in = sum(c.get("_labeling_meta", {}).get("input_tokens", 0) for c in labeled_chunks)
    label_out = sum(c.get("_labeling_meta", {}).get("output_tokens", 0) for c in labeled_chunks)
    try:
        from backend.rlm_runner import _snapshot_run_usage
        snap = _snapshot_run_usage()
        total_in = snap.get("input_tokens", 0)
        total_out = snap.get("output_tokens", 0)
        total_cost = snap.get("cost_usd", 0.0)
    except Exception:
        total_in = label_in
        total_out = label_out
        total_cost = 0.0
    return {
        "pipeline": pipeline_name,
        "labeler": labeler,
        "total_chunks": n_chunks,
        "total_labels": total_labels,
        "label_errors": label_errors,
        "label_input_tokens": label_in,
        "label_output_tokens": label_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round(total_cost, 4),
        "elapsed_sec": round(elapsed, 1),
    }
