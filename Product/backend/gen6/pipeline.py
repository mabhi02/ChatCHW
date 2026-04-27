"""Gen 6 pipeline: micro-chunk -> codebook label -> REPL compile.

Phase 0: Re-chunk to ~1-1.5K tokens
Phase 1+2: GPT-4o-mini swarm + Sonnet disagreement resolution
Phase 3: RLM REPL loads labeled chunks, queries structured data, builds 7 artifacts
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from backend.gen6.chunker import micro_chunk_guide
from backend.gen6.labeler import label_all_chunks

logger = logging.getLogger(__name__)


def _build_repl_system_prompt(codebook: str) -> str:
    """Build the system prompt for the REPL compilation phase."""
    return (
        "You are compiling clinical decision logic from PRE-LABELED guide chunks.\n\n"
        "The `context` variable contains an array of labeled chunks. Each chunk has:\n"
        "  - section_id, section_title, text (original guide text)\n"
        "  - labels: array of dicts with keys: span, id, type, quote_context\n"
        "    where 'type' is one of: supply_list, variables, predicates, modules,\n"
        "    phrase_bank, router, integrative\n"
        "    and 'id' follows the codebook naming conventions\n\n"
        "YOUR JOB: Query the labeled chunks to build 7 artifacts. The labels are\n"
        "already assigned -- you just need to compile them into the correct schemas.\n\n"
        "For each artifact, write Python code to:\n"
        "1. Filter labels by type\n"
        "2. Deduplicate by ID\n"
        "3. Format into the artifact schema\n"
        "4. Resolve cross-references (modules reference predicate IDs, etc.)\n"
        "5. Call emit_artifact(name, value) for each\n\n"
        "ARTIFACT SCHEMAS:\n"
        "  supply_list: list of (id, display_name, kind, source_quote, source_section_id)\n"
        "  variables: list of (id, display_name, kind, unit, data_type, source_quote, source_section_id)\n"
        "  predicates: list of (id, threshold_expression, fail_safe, source_vars, human_label, source_quote, source_section_id)\n"
        "  modules: dict keyed by module_id, each with (module_id, display_name, hit_policy, inputs, outputs, rules)\n"
        "  router: dict with (hit_policy, rows list of (priority, condition, output_module, description))\n"
        "  phrase_bank: list of (id, category, text, module_context, source_quote, source_section_id)\n"
        "  integrative: dict with rules list of (id, modules_involved, combine_rule, referral_priority, treatment_combination)\n\n"
        "DAG ORDER: supply_list + variables first, then predicates, modules, router + phrase_bank, integrative.\n"
        "Each later artifact can reference IDs from earlier ones.\n\n"
        f"NAMING CODEBOOK:\n{codebook[:3000]}\n\n"
        "RULES:\n"
        "  a) Emit all 7 artifacts in DAG order. System handles validation.\n"
        "  b) The labels are pre-verified -- trust them. Your job is compilation, not extraction.\n"
        "  c) For predicates: derive threshold_expression from the quote_context.\n"
        "  d) For modules: group related labels into decision tables with rules.\n"
        "  e) Router row 0 must be the danger sign short-circuit.\n"
        "  f) When done, call FINAL_VAR(clinical_logic) with all 7 artifacts.\n"
    )


async def run_gen6_extraction(
    guide_json: dict,
    anthropic_key: str,
    openai_key: str,
    naming_codebook: str = "",
    output_dir: Optional[Path] = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run the full Gen 6 pipeline."""
    start = time.time()

    # Phase 0: Micro-chunk
    logger.info("Gen6: Phase 0 -- micro-chunking guide")
    if on_progress:
        await on_progress({"phase": "chunk", "status": "starting"})

    chunks = micro_chunk_guide(guide_json)
    logger.info("Gen6: %d micro-chunks created", len(chunks))

    # Phase 1+2: Label
    logger.info("Gen6: Phase 1+2 -- codebook labeling")
    if on_progress:
        await on_progress({"phase": "label", "status": "starting"})

    labeled_chunks = await label_all_chunks(chunks, naming_codebook, openai_key, anthropic_key)

    total_labels = sum(len(c.get("labels", [])) for c in labeled_chunks)
    sonnet_escalations = sum(
        1 for c in labeled_chunks
        if c.get("_labeling_meta", {}).get("sonnet_escalated")
    )
    agreement_rate = sum(
        1 for c in labeled_chunks
        if c.get("_labeling_meta", {}).get("agreed")
    ) / len(labeled_chunks) if labeled_chunks else 0

    logger.info(
        "Gen6: labeling complete -- %d labels, %d chunks, %d Sonnet, %.0f%% agree",
        total_labels, len(labeled_chunks), sonnet_escalations, agreement_rate * 100,
    )

    # Save labeled chunks
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "labeled_chunks.json").write_text(
            json.dumps(labeled_chunks, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "chunk_difficulty.json").write_text(json.dumps({
            "total_chunks": len(chunks),
            "distribution": {
                d: sum(1 for c in chunks if c.get("difficulty") == d)
                for d in ["trivial", "easy", "medium", "hard", "extreme"]
            },
            "chunks": [
                {"index": c["chunk_index"], "section": c["section_id"],
                 "difficulty": c["difficulty"], "score": c.get("difficulty_score", 0)}
                for c in chunks
            ],
        }, indent=2), encoding="utf-8")

    # Phase 3: RLM REPL compilation
    logger.info("Gen6: Phase 3 -- RLM REPL compilation from labeled chunks")
    if on_progress:
        await on_progress({"phase": "compile", "status": "starting"})

    repl_context = [
        {
            "chunk_index": c.get("chunk_index", -1),
            "section_id": c.get("section_id", ""),
            "section_title": c.get("section_title", ""),
            "text": c.get("text", ""),
            "labels": c.get("labels", []),
        }
        for c in labeled_chunks
    ]

    system_prompt = _build_repl_system_prompt(naming_codebook)

    from rlm import RLM
    from rlm.logger import RLMLogger

    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": "claude-opus-4-6",
            "api_key": anthropic_key,
            "max_tokens": 16384,
        },
        environment="local",
        environment_kwargs={
            "context_payload": repl_context,
        },
        max_iterations=30,
        custom_system_prompt=system_prompt,
        compaction=False,
        max_budget=30.0,
        max_errors=5,
        max_timeout=1800.0,
        logger=RLMLogger(),
        verbose=False,
    )

    initial_message = (
        f"The labeled guide has {len(repl_context)} chunks with {total_labels} total labels.\n\n"
        f"The `context` variable is a list of {len(repl_context)} chunks. Each chunk has 'labels' array.\n"
        f"Each label has: 'span' (text), 'id' (codebook name), 'type' (artifact type), 'quote_context'.\n\n"
        f"START by extracting all labels grouped by type:\n"
        f"```python\n"
        f"all_labels = [l for c in context for l in c.get('labels', [])]\n"
        f"print(f'Total labels: {{len(all_labels)}}')\n"
        f"by_type = {{}}\n"
        f"for l in all_labels:\n"
        f"    t = l.get('type', 'unknown')\n"
        f"    by_type.setdefault(t, []).append(l)\n"
        f"for t, items in sorted(by_type.items()):\n"
        f"    print(f'  {{t}}: {{len(items)}} labels')\n"
        f"    for item in items[:3]:\n"
        f"        print(f'    {{item.get(\"id\", \"?\")}} -> {{item.get(\"span\", \"\")[:60]}}')\n"
        f"```\n\n"
        f"Then for each artifact type, deduplicate by id and build the schema. "
        f"Example for supply_list:\n"
        f"```python\n"
        f"supply_labels = by_type.get('supply_list', [])\n"
        f"seen = set()\n"
        f"supply_list = []\n"
        f"for l in supply_labels:\n"
        f"    if l['id'] not in seen:\n"
        f"        seen.add(l['id'])\n"
        f"        supply_list.append({{'id': l['id'], 'display_name': l.get('span',''), "
        f"'kind': 'consumable' if l['id'].startswith('supply_') else 'equipment', "
        f"'source_quote': l.get('quote_context',''), 'source_section_id': l.get('_section_id','')}})\n"
        f"emit_artifact('supply_list', supply_list)\n"
        f"```\n\n"
        f"Do this for ALL 7 types in DAG order. Then call FINAL_VAR with all artifacts combined."
    )

    result = await asyncio.to_thread(
        rlm.completion,
        initial_message,
    )

    elapsed = time.time() - start

    # Parse FINAL_VAR result
    response_text = getattr(result, "response", None)
    clinical_logic = None
    if isinstance(response_text, dict):
        clinical_logic = response_text
    elif isinstance(response_text, str):
        try:
            clinical_logic = json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            # ast.literal_eval is safe: only parses Python literals
            # (dict, list, str, int, float, bool, None), no code execution.
            # Used because the REPL model returns Python dict repr with
            # single quotes and True/False instead of JSON syntax.
            import ast
            try:
                clinical_logic = ast.literal_eval(response_text)
            except (ValueError, SyntaxError):
                pass

    status = "passed" if clinical_logic and isinstance(clinical_logic, dict) else "failed"

    if output_dir and clinical_logic and isinstance(clinical_logic, dict):
        # 1. Save clinical_logic.json (the combined output)
        (output_dir / "clinical_logic.json").write_text(
            json.dumps(clinical_logic, indent=2, default=str), encoding="utf-8"
        )

        # 2. Save individual artifact JSONs in artifacts/ subdirectory
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        for at, artifact in clinical_logic.items():
            (artifacts_dir / f"{at}.json").write_text(
                json.dumps(artifact, indent=2, default=str), encoding="utf-8"
            )

        # 3. Save system prompt
        (output_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")

        # 4. Run converters: DMN XML, XLSX, Mermaid, CSV
        try:
            from backend.converters import convert_to_dmn, convert_to_xlsx, convert_to_mermaid
            from backend.converters.json_to_csv import convert_to_csv
        except ImportError:
            logger.warning("Gen6: converters not available, skipping DMN/XLSX/Mermaid/CSV")
            convert_to_dmn = convert_to_xlsx = convert_to_mermaid = convert_to_csv = None

        if convert_to_dmn:
            try:
                dmn_xml = convert_to_dmn(clinical_logic)
                (output_dir / "clinical_logic.dmn").write_text(dmn_xml, encoding="utf-8")
                logger.info("Gen6: saved DMN XML")
            except Exception as exc:
                logger.warning("Gen6: DMN conversion failed: %s", exc)

        if convert_to_xlsx:
            try:
                xlsx_path = str(output_dir / "form.xlsx")
                convert_to_xlsx(clinical_logic, xlsx_path)
                logger.info("Gen6: saved XLSX")
            except Exception as exc:
                logger.warning("Gen6: XLSX conversion failed: %s", exc)

        if convert_to_mermaid:
            try:
                mermaid_src = convert_to_mermaid(clinical_logic)
                (output_dir / "flowchart.md").write_text(mermaid_src, encoding="utf-8")
                logger.info("Gen6: saved Mermaid flowchart")
            except Exception as exc:
                logger.warning("Gen6: Mermaid conversion failed: %s", exc)

        if convert_to_csv:
            try:
                csvs = convert_to_csv(clinical_logic)
                for csv_name, csv_content in csvs.items():
                    (output_dir / f"{csv_name}.csv").write_text(csv_content, encoding="utf-8")
                logger.info("Gen6: saved %d CSVs", len(csvs))
            except Exception as exc:
                logger.warning("Gen6: CSV conversion failed: %s", exc)

        # 5. Save test suite (frozen dedup items from labeled chunks)
        all_labels = []
        for c in labeled_chunks:
            for label in c.get("labels", []):
                label["_chunk_index"] = c.get("chunk_index", -1)
                label["_section_id"] = c.get("section_id", "")
                all_labels.append(label)

        (output_dir / "test_suite.json").write_text(json.dumps({
            "total_labels": len(all_labels),
            "labels_by_type": {
                t: [l for l in all_labels if l.get("type") == t]
                for t in ["supply_list", "variables", "predicates", "modules",
                          "phrase_bank", "router", "integrative"]
            },
        }, indent=2, default=str), encoding="utf-8")

        # 6. Per-chunk info is already saved above (labeled_chunks.json, chunk_difficulty.json)

        # List all output files
        logger.info("Gen6: output files:")
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                logger.info("  %s (%d bytes)", f.relative_to(output_dir), f.stat().st_size)

    logger.info(
        "Gen6: complete in %.0fs -- status=%s chunks=%d labels=%d",
        elapsed, status, len(chunks), total_labels,
    )

    return {
        "clinical_logic": clinical_logic,
        "labeled_chunks": labeled_chunks,
        "chunk_difficulty": [
            {"index": c["chunk_index"], "section": c["section_id"],
             "difficulty": c["difficulty"]}
            for c in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "total_labels": total_labels,
            "sonnet_escalations": sonnet_escalations,
            "agreement_rate": round(agreement_rate, 3),
            "elapsed": round(elapsed, 1),
        },
        "status": status,
    }
