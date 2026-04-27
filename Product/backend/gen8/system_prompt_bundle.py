"""gen8 multi-stage system prompt bundle.

Adapted from `backend/gen7/system_prompt_bundle.py`. The Stage 3 prompt
gains three new sections that deliver Tier 0-2:
  * Tier 0 -- predicate strict schema (operators, fields, provenance format)
  * Tier 2 -- manual ordering rule (every list-shaped artifact carries
              `_order_source` and `_manual_page`)
  * Tier 2 -- traffic-cop split (router emits `cop1_queue_builder` +
              `cop2_next_module` instead of a single `rows` list)

The sendable bundle adds a `Pipeline: gen8|gen8.5` header plus a
container-hash line filled in by the caller after the run completes.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Tier-0 / Tier-2 prompt fragments (spliced into the Stage 3 prompt)
# ---------------------------------------------------------------------------

TIER0_PREDICATE_RULES = (
    "## Predicate-table requirements (gen8 strict)\n\n"
    "When emitting `predicates`, every entry MUST have ALL of:\n"
    "- `id` -- `p_<concept>` lowercase. Examples: `p_fever`, `p_fast_breathing_2_to_12mo`, "
    "`p_chest_indrawing`. Stable across runs.\n"
    "- `description_clinical` -- plain-language clinician-readable, e.g. "
    "\"Axillary temperature above 37.5C indicating fever\".\n"
    "- `inputs_used` -- list of canonical raw variable ids referenced.\n"
    "- `formal_definition` -- the boolean expression. Allowed operators ONLY: "
    "`>`, `<`, `>=`, `<=`, `=`, `and`, `or`, `not`, parentheses. No `==`, no "
    "uppercase logic ops, no quoted string literals. For categorical values use enum "
    "predicates: instead of `v_muac_color = \"red\"` define `p_muac_red` whose "
    "`formal_definition` is `v_muac_color_is_red` and have a separate computed boolean.\n"
    "- `units` -- REQUIRED even when obvious. Examples: \"C\", \"breaths/min\", "
    "\"months\", \"(boolean)\".\n"
    "- `missingness_rule` -- exactly one of: `FALSE_IF_MISSING`, `TRIGGER_REFERRAL`, "
    "`BLOCK_DECISION`, `ALERT_NO_RULE_SPECIFIED`. Use `ALERT_NO_RULE_SPECIFIED` only "
    "when the source manual gives no guidance.\n"
    "- `allowed_input_domain` -- explicit range or set. Examples: \"34 to 43 C\", "
    "\"0 to 200 breaths/min\", \"{red, green, yellow}\".\n"
    "- `rounding_parsing_rule` -- explicit. Default for numerics: \"parse as float; "
    "no rounding before comparison\". Default for booleans: \"parse as boolean\".\n"
    "- `provenance` -- citation in form \"WHO 2012 page <N>\" using the page number "
    "from the source chunk (provided in the chunk metadata as `manual_page_start` "
    "and `manual_page_end`).\n\n"
    "DO NOT emit duplicate predicates. If two predicates would name the same "
    "clinical concept (e.g., `p_fever` and `p_has_fever`), emit only ONE. Prefer the "
    "shorter id. The pipeline will reject duplicates downstream.\n\n"
    "DO NOT emit predicates with empty `formal_definition`. Every predicate MUST be "
    "computable.\n"
)

TIER2_MANUAL_ORDERING = (
    "## Ordering rule (MANDATORY)\n\n"
    "Every list-shaped artifact (`variables`, `predicates`, `modules`, `phrase_bank`, "
    "`supply_list`) must order items by manual position. Each item MUST carry "
    "`_order_source` set to one of:\n"
    "- `\"manual\"`: ordering follows source manual page sequence. Include "
    "`_manual_page` field.\n"
    "- `\"explicit_hint\"`: source manual gives explicit ordering instruction (e.g., "
    "\"ask cough before fever\"). Include `_order_quote` field with the source quote.\n"
    "- `\"alphabetical_fallback\"`: only when (1) and (2) provide no signal.\n\n"
    "Within `modules`, `routes_to`, `rules`, and `inputs` lists, ordering follows "
    "manual sequence. Sort items by `_manual_page` ascending; ties broken by "
    "`_order_source == \"explicit_hint\" > manual > alphabetical_fallback`.\n"
)

TIER0_NO_STUB_PREDICATES = (
    "## Every predicate referenced in any rule MUST be defined here\n\n"
    "If a module rule, router rule, or integrative rule mentions `p_X`, you\n"
    "MUST emit a full predicate `p_X` with all 9 required fields\n"
    "(id, description_clinical, inputs_used, formal_definition, units,\n"
    "missingness_rule, allowed_input_domain, rounding_parsing_rule, provenance).\n"
    "It is a HARD FAILURE to reference a predicate id that has no definition\n"
    "in the predicate table. The downstream auto-reconciler will fill missing\n"
    "predicates with placeholder stubs and the verifier will flag every stub.\n\n"
    "Before emitting `clinical_logic`, walk every rule's `condition` and confirm\n"
    "each `p_X` token has a matching entry in `predicates[]`.\n"
)


TIER0_PREDICATE_ONLY_DMN = (
    "## DMN must reference ONLY predicates (no raw inputs)\n\n"
    "Per the Predicate Table specification: 'DMN must only reference p_* and\n"
    "c_* variables.' That means rule conditions MUST NOT contain raw\n"
    "`q_*`, `ex_*`, `v_*`, `lab_*`, `hx_*`, or `demo_*` variables. Every raw\n"
    "input that appears in a rule should first be wrapped in a predicate.\n\n"
    "Examples:\n"
    "  WRONG:  `condition: q_fever = true and v_temperature_c > 38`\n"
    "  RIGHT:  `condition: p_fever_high` (define p_fever_high in predicates[])\n\n"
    "  WRONG:  `condition: v_respiratory_rate_bpm >= 50`\n"
    "  RIGHT:  `condition: p_fast_breathing_2mo_to_12mo`\n"
)


TIER2_INPUT_COMPLETENESS = (
    "## Input completeness rule (MANDATORY -- prevents orphan variables)\n\n"
    "Every variable id that appears in any rule's `condition` (or in the\n"
    "`conditions` dict of a two-cop rule) MUST also appear in the enclosing\n"
    "`module.inputs[]` list. This includes predicate ids -- if a predicate\n"
    "`p_X` is referenced in a rule, list it in `inputs[]` even though it is\n"
    "not a raw variable.\n\n"
    "Example (CORRECT):\n"
    "```\n"
    "{\n"
    '  "id": "mod_fever",\n'
    '  "inputs": ["q_fever", "v_temperature_c", "p_fever"],\n'
    '  "rules": [\n'
    '    {"id": "r_high_fever", "condition": "p_fever", "outputs": {...}}\n'
    "  ]\n"
    "}\n"
    "```\n\n"
    "The downstream `data_flow` validator counts any variable referenced but\n"
    "never collected as `orphan`. Orphan count > 0 fires a Tier-2 divergence.\n"
)


TIER2_TRAFFIC_COPS = (
    "## Router: two-traffic-cop split (MANDATORY)\n\n"
    "Emit `router` as a dict with EXACTLY two keys (NOT a single rows list):\n"
    "- `cop1_queue_builder` -- `hit_policy: \"COLLECT\"`. Rules here ADD modules to "
    "the queue based on symptom flags. Multiple rules can fire for one visit.\n"
    "- `cop2_next_module` -- `hit_policy: \"UNIQUE\"`. Rules here PICK the next "
    "module to run from the queue. Exactly one rule fires at a time.\n\n"
    "Example shape:\n"
    "```\n"
    "{\n"
    "  \"cop1_queue_builder\": {\n"
    "    \"hit_policy\": \"COLLECT\",\n"
    "    \"description\": \"Add modules to queue based on symptom flags\",\n"
    "    \"rules\": [\n"
    "      {\"id\": \"r_at_start\", \"conditions\": {\"at_start\": true},\n"
    "       \"actions\": {\"add_to_queue\": [\"mod_startup\", \"mod_integrative\"]}},\n"
    "      {\"id\": \"r_fever\", \"conditions\": {\"q_has_fever\": true},\n"
    "       \"actions\": {\"add_to_queue\": [\"mod_fever\"]}}\n"
    "    ]\n"
    "  },\n"
    "  \"cop2_next_module\": {\n"
    "    \"hit_policy\": \"UNIQUE\",\n"
    "    \"description\": \"Pick next module to run from queue\",\n"
    "    \"rules\": [\n"
    "      {\"id\": \"r_queue_empty\", \"conditions\": {\"queue_empty\": true},\n"
    "       \"actions\": {\"end_visit\": true}},\n"
    "      {\"id\": \"r_urgent_referral\", \"conditions\": {\"is_urgent_referral\": true},\n"
    "       \"actions\": {\"next_module\": \"mod_integrative\"}}\n"
    "    ]\n"
    "  }\n"
    "}\n"
    "```\n"
    "A flat `{\"rows\": [...]}` router is REJECTED downstream.\n"
)


def build_sendable_system_prompt(
    labeling_prompt: str,
    distillation_prompt: str,
    repl_prompt: str,
    naming_codebook: str = "",
    run_id: str = "",
    manual_name: str = "",
    pipeline: str = "gen8",
    labeler: str = "opus",
    container_sha: str = "",
    verifier_model: str = "",
    verifier_independence: str = "",
) -> str:
    """Assemble the full multi-stage prompt bundle written to `system_prompt.md`."""
    repl_clean = repl_prompt.replace("{{", "{").replace("}}", "}")
    repl_clean = repl_clean.replace("{custom_tools_section}", "").rstrip()

    header = [
        "# CHW Navigator System Prompts",
        "",
        "_Full multi-stage prompt bundle: every prompt the labeler/compiler sees in this pipeline._",
    ]
    if run_id:
        header.append(f"_Run ID: `{run_id}`_")
    if manual_name:
        header.append(f"_Manual: {manual_name}_")
    header.append(f"_Pipeline: `{pipeline}` (labeler: `{labeler}`)_ ")
    if container_sha:
        header.append(f"_Container hash: `{container_sha}`_")
    if verifier_model:
        indep = f" (`{verifier_independence}`)" if verifier_independence else ""
        header.append(f"_Verifier: `{verifier_model}`{indep}_")
    header += ["_Generator model: Claude at temperature 0 for every stage._", ""]

    overview = [
        "## Pipeline overview",
        "",
        "Three (or more) generator calls against the same content-hashed manual. "
        "Chunking is deterministic Python, label dedup is exact-string, validators are "
        "Python, XLSForm / DMN / flowchart assembly is deterministic. Verification "
        "is an independent read-only pass that flags disagreements in a sidecar; it "
        "never edits the artifact.",
        "",
        "| Stage | Phase | Input | Output |",
        "|-------|-------|-------|--------|",
        "| 1 | Per-chunk labeling (Call 1) | one ~2K-token chunk | labels array |",
        "| 2 | Per-chunk QC (Call 2) | Stage 1 candidate labels | verified labels |",
        "| 3 | REPL compilation | deduped labels + reconstructed guide | 7 DMN artifacts |",
        "| V | Verification | each artifact + prompts | agree/divergences sidecar |",
        "",
        "The naming codebook is injected into all three generator stages so ids are "
        "compatible across stages.",
        "",
    ]

    stage1 = [
        "## Stage 1: Chunk labeling system prompt",
        "",
        "Used once per micro-chunk. Anthropic prompt caching is set on the system block "
        "so the codebook and instructions are read at cache-read rates for every chunk "
        "after the first.",
        "",
        "```",
        labeling_prompt.strip(),
        "```",
        "",
    ]

    stage2 = [
        "## Stage 2: Label QC / distillation system prompt",
        "",
        "Second pass per chunk. Verifies each Stage 1 candidate label is grounded in "
        "the input text and has a codebook-compliant ID. Drops hallucinations; never "
        "adds new labels.",
        "",
        "```",
        distillation_prompt.strip(),
        "```",
        "",
    ]

    stage3 = [
        "## Stage 3: REPL compilation system prompt",
        "",
        "Drives the Python REPL for the final compile session. Two blocks attach to "
        "this prompt at runtime (the full reconstructed guide text, and the deduped "
        "label inventory) so every REPL turn reads them at cache-read rates. "
        "The Tier-0/Tier-2 rules (predicate grammar, manual ordering, two-traffic-cop "
        "split) are appended to this prompt.",
        "",
        "```",
        repl_clean.strip(),
        "```",
        "",
    ]

    codebook_section: list[str] = []
    if naming_codebook:
        codebook_section = [
            "## Naming codebook (injected into all three generator stages)",
            "",
            "```",
            naming_codebook.strip(),
            "```",
            "",
        ]

    return "\n".join(header + overview + codebook_section + stage1 + stage2 + stage3)
