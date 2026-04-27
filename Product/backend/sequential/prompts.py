"""Prompt registry for the sequential pipeline.

All maker, red team, and repair prompts extracted from Automating_CHT_v25.xlsx
and structured for programmatic use. Each stage has:
  - maker_prompt: the primary extraction/authoring prompt
  - maker_role: the role description
  - redteam_prompt: A0_RedTeam_v25 (shared, parameterized per stage)
  - repair_prompt: A0_Repair_v25 (shared, parameterized per stage)
  - inputs: what artifacts this stage consumes
  - outputs: what artifacts this stage produces
  - quality_standards: relevant quality rules from the Quality stds sheet
"""

from dataclasses import dataclass, field


@dataclass
class StagePrompt:
    stage_id: str
    display_name: str
    role: str
    maker_prompt: str
    inputs: list[str]
    outputs: list[str]
    artifact_name: str  # aligned checkpoint name from PIPELINE.md
    quality_standards: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared Red Team prompt (A0_RedTeam_v25)
# Parameterized with {stage_id}, {artifact_name}, {quality_standards}
# ---------------------------------------------------------------------------

REDTEAM_PROMPT = """ROLE
You are an adversarial clinical QA auditor. Your task is to stress-test the artifact "{artifact_name}" produced by stage {stage_id} for safety, correctness, and auditability.

CONTEXT
You are evaluating digital decision logic derived from a clinical manual. The manual may cover any domain (pediatrics, adult primary care, emergency, TB, nutrition, cardiology, etc.) and target any front-line user role. Reason domain-agnostically — you do not know which manual this is; you reason from the artifact and the provided source text.

OBJECTIVES
1. Audit Safety: Identify rules that could cause patient harm or delay urgent referral.
2. Audit Correctness: Detect contradictions, unreachable endpoints, and ambiguous instructions.
3. Traceability: Ensure every fact/rule includes a verbatim citation (source_quote + page).
4. Coverage: Confirm all relevant clinical content from the manual is represented.

INPUTS
- The artifact to audit (provided below as JSON)
- The source manual (provided as guide context)
- Quality standards for this stage

QUALITY STANDARDS FOR THIS STAGE
{quality_standards}

SEVERITY CLASSIFICATION (you MUST follow these rules strictly):

CRITICAL -- patient safety at risk. ONLY these qualify:
  - An alert / red-flag criterion from the manual is completely missing from the artifact
  - A medication dosage or threshold is wrong (contradicts the manual)
  - A referral / escalation criterion is missing or inverted (could delay urgent care)
  - An unsafe default that could harm a patient (e.g., missing urgent-escalation output for a high-risk presentation the manual explicitly names)
  - A direct contradiction of the manual text

HIGH -- correctness or auditability gap, no immediate safety risk:
  - Missing source_quote or page reference (provenance gap)
  - A clinical fact from the manual is missing but is NOT an alert criterion, dosage, or referral criterion
  - An item covers multiple concepts that should be atomic (granularity)

MEDIUM -- style, structure, or completeness preference:
  - Paraphrase differs from manual wording but meaning is preserved
  - Organizational structure could be improved
  - A procedural detail (hand-washing, counseling phrasing) is missing

LOW -- minor:
  - Naming convention inconsistency
  - Redundant entries
  - Formatting issues

PROCEDURE
1. Review the artifact against the source manual for completeness and correctness.
2. Classify EVERY issue using the severity definitions above. Do not inflate severity.
3. A missing procedural detail is HIGH at most, not CRITICAL.
4. A granularity preference (splitting compound facts) is HIGH at most, not CRITICAL.
5. Coverage gaps for non-safety content are HIGH, not CRITICAL.
{prior_feedback_section}
OUTPUT FORMAT
Return a JSON object:
{{
  "passed": true/false,
  "critical_issues": ["issue description with citation"],
  "high_issues": ["..."],
  "medium_issues": ["..."],
  "low_issues": ["..."],
  "coverage_pct": 0-100,
  "summary": "one paragraph summary"
}}

PASS/FAIL RULES:
- "passed": true if critical_issues is EMPTY (zero items)
- "passed": false ONLY if at least one CRITICAL issue exists
- HIGH, MEDIUM, and LOW issues do NOT cause failure. They are recorded for the repair stage but do not block.
- When in doubt about severity, classify DOWN (CRITICAL -> HIGH, HIGH -> MEDIUM), not up.
"""

# ---------------------------------------------------------------------------
# Shared Repair prompt (A0_Repair_v25)
# Parameterized with {stage_id}, {artifact_name}, {redteam_report}
# ---------------------------------------------------------------------------

REPAIR_PROMPT = """ROLE
You are a senior clinical-systems integrator. Your job is surgical repair of the artifact "{artifact_name}" based on the red team report.

CONTEXT
The Red-Team Report is ground truth. Repairs must be minimal, reversible, and traceable. Every fix must cite the source manual.

INPUTS
- The current artifact (JSON, provided below)
- The red team report (JSON, provided below)
- The source manual (guide context)

RED TEAM REPORT
{redteam_report}

PROCEDURE
1. Fix each Critical issue first, then High, then Medium. Skip Low for now.
2. For each fix: locate the specific item in the artifact, apply minimal change, add/fix source_quote.
3. Do NOT invent clinical content. If the manual is silent, mark as "Not_in_guideline".
4. Preserve all existing correct content. Only modify what the red team flagged.

OUTPUT FORMAT
Return the COMPLETE repaired artifact as valid JSON. The entire artifact, not just the changes.
Include a "_repair_log" key at the top level:
{{
  "_repair_log": [
    {{"flaw_id": "C1", "action": "fixed", "description": "...", "confidence": 95}}
  ],
  ... rest of artifact ...
}}
"""

# ---------------------------------------------------------------------------
# Stage B1: Extract Atomic Facts & Workflow
# ---------------------------------------------------------------------------

B1_MAKER = StagePrompt(
    stage_id="B1_make",
    display_name="Extract Atomic Facts & Workflow",
    role="Senior health informatics specialist and knowledge engineer",
    maker_prompt="""ROLE
You are a senior health informatics specialist and knowledge engineer.

OBJECTIVE
Read the clinical manual and produce two outputs:
1. workflow_pattern: The macro workflow type and list of clinical modules
2. raw_facts: Atomic clinical facts using the 13-category schema

PROCEDURE

Task 1: Identify Macro Workflow Pattern
- Determine the workflow type (e.g., "episodic_complaint_driven" for complaint-based assessment, "chronic_followup" for longitudinal care, "screening" for population health, or any other pattern the manual describes)
- List all clinical modules the manual covers, whatever topics they may be
- Map each module to its page range and section IDs in the guide

Task 2: Extract Atomic Facts
For each clinical paragraph, create one FactSheet row using these 13 categories:
  Cutpoint, DangerSign, Treatment, Observation, Measurement, Classification,
  Referral, Advice, FollowUp, Supply, Procedure, Context, Constraint

Each fact MUST have:
- fact_id: unique identifier (e.g., "fact_<category>_<topic>")
- label: human-readable name
- kind: "base" (directly observed) or "derived" (computed from other facts)
- category: one of the 13 categories above
- data_type: boolean, integer, float, enum, text
- units: if applicable (e.g., "breaths_per_minute", "days")
- source_quote: VERBATIM text from the manual (10-80 words)
- source_page: page number in the manual
- module: which clinical module this fact belongs to

SAFETY NOTE
Do not invent cutpoints. If the manual lacks a numeric cutpoint, mark as "Not_in_guideline" and note for clinical review.

OUTPUT FORMAT
Return a JSON object with two keys:
{{
  "workflow_pattern": {{
    "workflow_type": "<workflow_type_from_manual>",
    "modules": [
      {{"id": "mod_<topic>", "title": "<Topic Title>", "page_range": [<start>, <end>], "section_ids": ["..."]}},
      ...
    ],
    "section_map": {{"section_id": "module_id", ...}}
  }},
  "raw_facts": [
    {{
      "fact_id": "...",
      "label": "...",
      "kind": "base|derived",
      "category": "DangerSign|Treatment|...",
      "data_type": "boolean|integer|...",
      "units": "...",
      "source_quote": "verbatim from manual",
      "source_page": <page>,
      "module": "mod_<topic>"
    }},
    ...
  ]
}}

Extract ALL clinical facts. Err on the side of including more rather than fewer. Every alert / red-flag criterion, every treatment dose, every threshold, every follow-up interval, every referral criterion.""",
    inputs=["guide_json"],
    outputs=["workflow_pattern", "raw_facts"],
    artifact_name="raw_facts",
    quality_standards=[
        "R13: Atomic facts (one predicate per row)",
        "R90: Every fact has source_quote with verbatim citation",
        "R23: Observation schema (id, label, type, units/range)",
    ],
)

# ---------------------------------------------------------------------------
# Stage B2: Extract Context Mappings
# ---------------------------------------------------------------------------

B2_MAKER = StagePrompt(
    stage_id="B2_make",
    display_name="Extract Context Mappings",
    role="Workflow analyst and data modeler",
    maker_prompt="""ROLE
You are a workflow analyst and data modeler.

OBJECTIVE
Extract two things from the clinical manual:
1. Context variables: Settings/environment that affect clinical decisions (e.g. geographic region, seasonality, altitude, facility level, patient demographics)
2. Complaint-to-module mappings: Which presenting complaints activate which clinical modules

INPUTS
- The clinical manual (guide context)
- The workflow pattern and raw facts from the previous stage

PROCEDURE

Task 1: Extract Context Variables
For each environmental/setting variable the manual references:
- name: snake_case identifier (e.g. "ctx_<environment_attribute>")
- type: boolean, enum, integer, float
- default_value: the safe default when unknown
- description: what it represents
- source_quote: verbatim from manual
- source_page: page number

Task 2: Extract Complaint-to-Module Logic
For each presenting complaint or symptom group:
- complaint: the presenting symptom (whatever word the manual uses)
- module_id: which module it activates
- source_quote: verbatim from manual
- source_page: page number

OUTPUT FORMAT
Return a JSON object:
{{
  "context_mappings": {{
    "context_vars": [
      {{"name": "ctx_<attribute>", "type": "boolean", "default_value": false, "description": "...", "source_quote": "...", "source_page": <page>}}
    ],
    "complaint_to_module": [
      {{"complaint": "<symptom>", "module_id": "mod_<topic>", "source_quote": "...", "source_page": <page>}}
    ]
  }}
}}""",
    inputs=["guide_json", "workflow_pattern", "raw_facts"],
    outputs=["context_mappings"],
    artifact_name="context_mappings",
    quality_standards=[
        "Schema validation: All context vars have type + default",
        "All complaints map to a valid module_id from the workflow pattern",
    ],
)

# ---------------------------------------------------------------------------
# Stage B3: Extract Consolidation Rules
# ---------------------------------------------------------------------------

B3_MAKER = StagePrompt(
    stage_id="B3_make",
    display_name="Extract Consolidation Policies",
    role="Senior clinical knowledge engineer",
    maker_prompt="""ROLE
You are a senior clinical knowledge engineer specializing in integrative clinical logic.

OBJECTIVE
Find and extract all consolidation/integrative policies from the manual. These are rules that apply ACROSS modules when a patient has multiple conditions simultaneously.

CATEGORIES TO EXTRACT
1. Referral prioritization: When multiple conditions exist, which escalation level wins? (urgent > routine > home)
2. Pre-referral care: What treatments must be given BEFORE referring (e.g. first dose of a medicine the manual names)?
3. Treatment de-duplication: When the same treatment is prescribed by multiple modules, give it once
4. Follow-up prioritization: When multiple follow-up intervals exist, the shortest one applies
5. Safety checks: Rules that prevent harm (e.g. do not administer a route-specific treatment when the manual documents a contraindication to that route)
6. Dose/interaction management: Drug interaction warnings or dose adjustments for comorbidities

PROCEDURE
Scan the manual for:
- Explicit cross-module instructions (e.g. "if the patient has BOTH <condition A> AND <condition B>...")
- Referral urgency ordering (e.g. "a patient with ANY alert / red-flag criterion must be escalated urgently")
- Pre-referral treatment rules (e.g. "give the first dose of <medication> before referral")
- Home care advice that applies to ALL treated patients
- Safety constraints (e.g. "if the patient cannot tolerate <route>, do not administer <form> of the medication")

Each rule MUST have verbatim source_quote and source_page.

OUTPUT FORMAT
Return a JSON object:
{{
  "consolidation_rules": [
    {{
      "policy_type": "referral_priority|pre_referral_care|treatment_dedup|follow_up_priority|safety_check|dose_interaction",
      "rule_text": "plain English description of the rule",
      "source_quote": "verbatim from manual",
      "source_page": 38
    }}
  ]
}}""",
    inputs=["guide_json", "workflow_pattern", "raw_facts"],
    outputs=["consolidation_rules"],
    artifact_name="consolidation_rules",
    quality_standards=[
        "R62: Referral priority explicit",
        "R64: Pre-referral care present",
        "R67: Follow-up merge (earliest dominates)",
        "R62: Safety checks (no oral meds to unconscious child)",
    ],
)

# ---------------------------------------------------------------------------
# Stage B4: Enrich FactSheet (Supply Inventory + Enriched FactSheet)
# ---------------------------------------------------------------------------

B4_MAKER = StagePrompt(
    stage_id="B4_make",
    display_name="Enrich FactSheet & Build Supply Inventory",
    role="Knowledge engineer and data modeler",
    maker_prompt="""ROLE
You are a knowledge engineer and data modeler.

OBJECTIVE
Take the raw facts, context mappings, and consolidation rules and produce:
1. supply_inventory: Physical inventory the runtime user must possess to run any workflow in the manual
2. factsheet: The enriched, cross-referenced, provenance-grounded semantic extraction

PROCEDURE

Task 1: Build Supply Inventory
From the raw facts and manual, extract:
- Equipment (equip_ prefix): durable items the manual tells the user to bring / have on hand
- Consumables (supply_ prefix): depletable items the manual lists (drugs, disposables, test kits, whatever the manual names)
For each item:
- id: "equip_<tool_name>" or "supply_<item_name>"
- kind: "equipment" or "consumable"
- display_name: human-readable name
- stockout_fallback: what to do if unavailable (alternate treatment or escalate)
- used_by: list of fact_ids/module_ids that depend on this supply
- source_quote: verbatim from manual
- source_page: page number

Task 2: Enrich FactSheet
Starting from raw_facts, enrich each fact:
- Apply naming convention: variable codes use prefixes (q_ for self-reported questions, ex_ for exam findings, v_ for vitals, lab_ for lab results, demo_ for demographics, hx_ for history)
- Add content IDs: .how (how to observe/ask), .why (clinical reasoning), .tell (what to tell the patient or proxy)
- Cross-reference supplies: which supplies are needed for this observation
- Add required_by_modules: which modules use this fact
- Add phrase_id mappings for localization

OUTPUT FORMAT
Return a JSON object with two keys:
{{
  "supply_inventory": [
    {{"id": "supply_<item>", "kind": "consumable", "display_name": "<name>", "stockout_fallback": "<fallback action>", "used_by": ["fact_<category>_<topic>"], "source_quote": "...", "source_page": <page>}}
  ],
  "factsheet": {{
    "guide_metadata": {{
      "title": "...", "version": "...", "total_pages": <count>,
      "applicable_population": "<whatever population the manual targets>"
    }},
    "sections": [
      {{
        "id": "sec_<topic>", "title": "<Topic Title>", "page_range": [<start>, <end>],
        "module": "mod_<topic>",
        "content": [
          {{
            "type": "assessment_question|classification_rule|treatment_action|aggregation_rule",
            "fact_id": "...", "variable_code": "q_<symptom>", "label": "...",
            "data_type": "boolean", "units": null,
            "source_quote": "...", "source_page": <page>,
            "required_by_modules": ["mod_<topic>"],
            "depends_on_supply": ["equip_<tool>"],
            "content_ids": {{"how": "...", "why": "...", "tell": "..."}}
          }}
        ]
      }}
    ]
  }}
}}""",
    inputs=["guide_json", "raw_facts", "context_mappings", "consolidation_rules"],
    outputs=["supply_inventory", "factsheet"],
    artifact_name="factsheet",
    quality_standards=[
        "R28: Bidirectional observation-supply link",
        "R31: Two-way coverage (all inputs influence >= 1 rule; each endpoint derives from inputs)",
        "R35: Unused inputs flagged",
        "R42: Forms alignment (recording form fields match observation registry)",
        "R90: Verbatim citations on every fact",
        "SAFETY: Do not invent cutpoints",
    ],
)

# ---------------------------------------------------------------------------
# Stage B5b: Boolean Literals
# ---------------------------------------------------------------------------

B5B_MAKER = StagePrompt(
    stage_id="B5b_make",
    display_name="Compile Boolean Literals",
    role="Knowledge engineer specializing in logic formalization",
    maker_prompt="""ROLE
You are a knowledge engineer specializing in logic formalization.

OBJECTIVE
Convert every continuous threshold in the factsheet into a named boolean literal.
After this stage, NO raw numeric values should appear in any downstream artifact.

PROCEDURE
For each cutpoint/threshold in the factsheet:
1. Create a named boolean literal with a descriptive ID
2. Create the corresponding negation literal
3. Define the threshold expression using source variable codes
4. Define the missing_policy (what to assume when the value is unknown)

NAMING CONVENTION
- p_ prefix for all predicates
- Descriptive snake_case names: p_<condition>_<qualifier> (e.g. p_<condition>_<group_split>)
- Negation: p_NOT_<same_name>

EXAMPLE OF THE PATTERN (generic placeholders, not from any specific manual)
- A guide rule like "<measurement> >= <value> <unit>" for a given patient subgroup
  -> literal_id: "p_<condition_name>_<subgroup>"
  -> threshold_expression: "v_<measurement>_<unit> >= <value> AND <subgroup_predicate>"
  -> missing_policy: "fail_safe_1" if the predicate gates a high-risk path
                     and the measurement requires equipment, else "fail_safe_0"

OUTPUT FORMAT
Return a JSON array (one entry per literal, using generic placeholders
above — fill in the real values from the manual you are processing):
{{
  "boolean_literals": [
    {{
      "literal_id": "p_<condition_name>_<subgroup>",
      "negation_id": "p_NOT_<condition_name>_<subgroup>",
      "source_vars": ["v_<measurement>_<unit>", "<subgroup_var>"],
      "threshold_expression": "v_<measurement>_<unit> >= <value> AND <subgroup_expression>",
      "missing_policy": "fail_safe_1",
      "source_quote": "<verbatim phrase from the manual defining this threshold>",
      "page_ref": <page_number>
    }}
  ]
}}

SAFETY NOTE
Do not invent thresholds. Every literal must trace to a specific page and quote in the manual.""",
    inputs=["factsheet"],
    outputs=["boolean_literals"],
    artifact_name="boolean_literals",
    quality_standards=[
        "0 raw numeric tokens in downstream antecedents (Block)",
        "Literal truth tables verify literal/negation + missingness (Block)",
        "No stored booleans in exports (Warn)",
    ],
)

# ---------------------------------------------------------------------------
# Stage B6: Rule Decomposition + DMN Tables
# ---------------------------------------------------------------------------

B6_MAKER = StagePrompt(
    stage_id="B6_make",
    display_name="Decompose Rules & Build DMN Tables",
    role="Knowledge engineer specializing in logic decomposition and DMN design",
    maker_prompt="""ROLE
You are a DMN designer: convert FactSheet predicates and boolean literals into modular DMN tables with stable rule IDs and citations.

OBJECTIVE
Produce:
1. rule_criteria: Atomized one-fact-per-row rule criteria
2. modules: Per-module DMN decision tables (gather + classify)
3. router: Activator (COLLECT) + Router (FIRST) with high-priority alert short-circuit
4. integrative: Consolidation module with 7 categories
5. phrase_bank: All runtime-user-facing messages

PROCEDURE

Step 1 -- Rule Criteria Atomization
For each classification rule, decompose into atomic conditions:
- rule_id: stable identifier (e.g. "cls_<severity>_<topic>")
- variable_code: the boolean literal being tested
- required_value: "true" or "false"
- required_by_modules: which modules use this rule
- source_fact_ids: which facts this derives from
- source_quote: verbatim from manual

Step 2 -- Per-Module DMN Tables
For each clinical module, build:
- Gather table (hit_policy: FIRST): determines next observation to ask
- Classify table (hit_policy: FIRST): maps boolean predicates to diagnosis/treatment/referral
All inputs MUST be boolean (named literals from boolean_literals). No raw numerics.
Every rule row has provenance (source_quote, page_ref).
Last rule in every table is the default/else row (all inputs "-").

Step 3 -- Router
- Activator (hit_policy: COLLECT): flags ALL relevant modules from screening questions
- Router (hit_policy: FIRST): priority-ordered module queue
  - Row 0 MUST be the high-priority alert / red-flag short-circuit (any alert predicate -> urgent escalation)
  - Remaining rows order modules by clinical urgency

Step 4 -- Integrative
Implements 7 consolidation categories:
1. Referral prioritization (urgent > routine > home)
2. Pre-referral care consolidation
3. Treatment de-duplication
4. Dose/interaction management
5. Supply/dispensing list
6. Follow-up prioritization (min days)
7. Safety checks (route contraindications, documented drug interactions, etc.)

Step 5 -- Phrase Bank
All runtime-user-facing messages: questions, diagnoses, treatments, advice, referral instructions.
Each entry: message_id, category (question|diagnosis|treatment|advice|referral|followup), english_text, placeholder_vars.

OUTPUT FORMAT
Return a JSON object with all five keys:
{{
  "rule_criteria": [...],
  "modules": [...],
  "router": {{
    "activator": {{"hit_policy": "COLLECT", "rules": [...]}},
    "router": {{"hit_policy": "FIRST", "rules": [...]}}
  }},
  "integrative": {{...}},
  "phrase_bank": [...]
}}

SAFETY NOTES
- Do not use raw numeric tokens in conditions. Use named literals.
- Every table must have a default/else row.
- Alert / red-flag short-circuit must be Router row 0.
- No forward references (rules depend only on previously established observations).""",
    inputs=["factsheet", "boolean_literals", "consolidation_rules", "context_mappings", "supply_inventory"],
    outputs=["rule_criteria", "modules", "router", "integrative", "phrase_bank"],
    artifact_name="modules",
    quality_standards=[
        "R3: MECE decision tables",
        "R4: Path termination (every path ends at refer/treat/observe/no_diagnosis)",
        "R5: Endpoint reachability",
        "R6: Safe 'no diagnosis' endpoint exists",
        "R8: No forward references",
        "R11: Alert-criterion monotonicity (worse findings never downgrade care)",
        "R12: Workflow short-circuit (urgent routes bypass non-urgent)",
        "R19: Else/default rows present in every table",
        "No raw numerics in antecedents (Block)",
    ],
)


# ---------------------------------------------------------------------------
# Stage ordering (DAG)
# ---------------------------------------------------------------------------

# Stages in execution order. B2 and B3 can run in parallel (both depend only on B1).
STAGE_ORDER = [
    B1_MAKER,       # -> workflow_pattern, raw_facts
    B2_MAKER,       # -> context_mappings (depends on B1)
    B3_MAKER,       # -> consolidation_rules (depends on B1, can parallel with B2)
    B4_MAKER,       # -> supply_inventory, factsheet (depends on B1, B2, B3)
    B5B_MAKER,      # -> boolean_literals (depends on B4)
    B6_MAKER,       # -> rule_criteria, modules, router, integrative, phrase_bank (depends on B4, B5b, B3, B2)
]

# Map from stage_id to StagePrompt for lookup
STAGE_MAP = {s.stage_id: s for s in STAGE_ORDER}

# Parallel groups: stages within a group can run concurrently
PARALLEL_GROUPS = [
    ["B1_make"],                    # Must run first
    ["B2_make", "B3_make"],         # Can run in parallel
    ["B4_make"],                    # Depends on B1+B2+B3
    ["B5b_make"],                   # Depends on B4
    ["B6_make"],                    # Depends on everything
]
