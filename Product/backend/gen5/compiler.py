"""Gen 5 Phase 2: DAG-ordered schema compilation.

Takes deduplicated raw items and compiles them into final artifact JSON,
resolving cross-references between artifact types in DAG order.

Each compilation step is 1 Opus call.
"""

import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# DAG levels: artifacts at the same level compile in parallel
DAG = [
    ["supply_list", "variables"],       # Level 0: no dependencies
    ["predicates"],                      # Level 1: needs variable IDs
    ["modules"],                         # Level 2: needs variables, predicates, supply IDs
    ["router", "phrase_bank"],           # Level 3: needs module IDs
    ["integrative"],                     # Level 4: needs module + predicate IDs
]

SCHEMA_TEMPLATES = {
    "supply_list": "Array of objects: {id, display_name, kind ('equipment'|'consumable'), source_quote, source_section_id}",
    "variables": "Array of objects: {id (with prefix q_/ex_/v_/lab_/demo_/hx_), display_name, kind, unit (if numeric), data_type, source_quote, source_section_id}",
    "predicates": "Array of objects: {id (p_ prefix), threshold_expression, fail_safe (1 for equipment-dependent, 0 for self-reported), source_vars (array of variable IDs), human_label, source_quote, source_section_id}",
    "modules": "Dict keyed by module_id: {module_id, display_name, hit_policy ('FIRST'), inputs (array of {id, label}), outputs (array of {id, label}), rules (array of {rule_id, description, inputs (dict of predicate->value), outputs (dict), provenance})}",
    "router": "Dict with hit_policy='FIRST', rows array. Row 0 = danger sign short-circuit. Each row: {priority, condition (predicate expression), output_module, description}",
    "phrase_bank": "Array of objects: {id (m_ prefix), category ('question'|'diagnosis'|'treatment'|'advice'|'referral'|'follow_up'), text (verbatim phrase), module_context, source_quote, source_section_id}",
    "integrative": "Dict with rules for cross-module combinations. Each rule: {id, modules_involved (array), combine_rule ('additive'|'highest_referral_wins'), referral_priority, treatment_combination, follow_up_rule, source_quote}",
}


async def compile_artifact(
    artifact_type: str,
    raw_items: list[dict],
    upstream_artifacts: dict[str, Any],
    api_key: str,
    naming_codebook: str = "",
) -> Any:
    """Compile one artifact type from raw extracted items.

    Uses Opus to format raw items into the correct schema,
    resolving cross-references against upstream_artifacts.
    """
    import anthropic

    # Build the prompt
    schema = SCHEMA_TEMPLATES.get(artifact_type, "JSON object or array")

    # Summarize upstream artifacts for cross-referencing
    upstream_summary = ""
    for name, artifact in upstream_artifacts.items():
        if isinstance(artifact, list):
            ids = [e.get("id", "?") for e in artifact[:30] if isinstance(e, dict)]
            upstream_summary += f"\n{name} IDs available: {ids}"
        elif isinstance(artifact, dict):
            upstream_summary += f"\n{name} keys: {list(artifact.keys())[:20]}"

    # Serialize raw items (cap to fit in context)
    items_json = json.dumps(raw_items[:100], indent=1)
    if len(raw_items) > 100:
        items_json += f"\n... and {len(raw_items) - 100} more items"

    prompt = (
        f"You are compiling the '{artifact_type}' artifact from raw extracted items.\n\n"
        f"SCHEMA:\n{schema}\n\n"
    )
    if naming_codebook:
        prompt += f"NAMING CONVENTIONS:\n{naming_codebook[:2000]}\n\n"
    if upstream_summary:
        prompt += f"UPSTREAM ARTIFACTS (use these IDs for cross-references):{upstream_summary}\n\n"
    prompt += (
        f"RAW EXTRACTED ITEMS ({len(raw_items)} items):\n{items_json}\n\n"
        f"Compile these into the final '{artifact_type}' artifact following the schema above. "
        f"Deduplicate any remaining overlaps. Use IDs from upstream artifacts for cross-references. "
        f"Return ONLY valid JSON. No markdown fences. No explanation."
    )

    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        r = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16384,
            temperature=0.0,
            system="You compile clinical decision logic artifacts. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        artifact = json.loads(text)
        logger.info("Gen5 compile: %s = %s (%d items compiled from %d raw)",
                     artifact_type,
                     type(artifact).__name__,
                     len(artifact) if isinstance(artifact, (list, dict)) else 0,
                     len(raw_items))
        return artifact
    except Exception as exc:
        logger.error("Gen5 compile failed for %s: %s", artifact_type, exc)
        return [] if artifact_type != "modules" else {}


async def compile_all(
    deduped_items: dict[str, list[dict]],
    api_key: str,
    naming_codebook: str = "",
    on_progress: Any = None,
) -> dict[str, Any]:
    """Phase 2: Compile all artifacts in DAG order.

    Returns {artifact_type: compiled_artifact}.
    """
    compiled: dict[str, Any] = {}

    for level_idx, level_types in enumerate(DAG):
        logger.info("Gen5 compile: DAG level %d — %s", level_idx, level_types)

        # Build upstream context from already-compiled artifacts
        upstream = {name: art for name, art in compiled.items()}

        # Compile all types at this level in parallel
        coros = []
        for at in level_types:
            raw = deduped_items.get(at, [])
            coros.append(compile_artifact(at, raw, upstream, api_key, naming_codebook))

        results = await asyncio.gather(*coros, return_exceptions=True)

        for at, result in zip(level_types, results):
            if isinstance(result, Exception):
                logger.error("Gen5 compile: %s failed: %s", at, result)
                compiled[at] = [] if at not in ("modules", "router", "integrative") else {}
            else:
                compiled[at] = result

        if on_progress:
            try:
                await on_progress({
                    "phase": "compile",
                    "level": level_idx,
                    "types": level_types,
                    "status": "done",
                })
            except Exception:
                pass

    return compiled
