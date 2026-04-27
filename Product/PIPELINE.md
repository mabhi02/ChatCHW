# CHW Navigator: Full Pipeline Architecture

> **Date:** 2026-04-09
> **Author:** Atharva Patel (synthesized by Claude from xlsx, codebase, experiments, and design docs)
> **Grounded in:** `Automating_CHT_v25.xlsx` (Professor Levine), `arena_style.md`, postmortem experiments, existing codebase

---

## Shared Vocabulary

Every term below is canonical. Use these names in all code, prompts, docs, and conversations.

| Term | What It Is | Format | Produced By | Consumed By |
|---|---|---|---|---|
| **RAW_PDF** | Original unmodified PDF (e.g., WHO CHW guide 2012, 141 pages). Fixed input. | `.pdf` | External (WHO, MoH) | Stage 1 |
| **PROVENANCE_JSON** | Layout-level extraction: every PDF element typed (paragraph, table, image, heading, list) with page numbers, bounding box coordinates, raw text. Tables refined to structured rows via GPT vision. Images described. Flowcharts parsed to nodes/edges. This is a DOCUMENT representation, not clinical content. Training narratives, exercises, and clinical protocol are undifferentiated. | `RLMGuide` JSON: `{metadata, sections: {slug: RLMSection}, pages: {num: RLMPage}}` | Stage 1 (Unstructured + GPT vision) | Stage 2 |
| **CONTENT_TRIAGE** | Element-level classification: each PROVENANCE_JSON element tagged as `clinical_protocol`, `training_exercise`, `narrative`, or `administrative`. Filters the signal from the noise in training manuals. | Tagged element list with `relevance_class` per element | Stage 2a | Stage 2b-2g |
| **WORKFLOW_PATTERN** | Macro workflow type (e.g., episodic complaint-driven) plus the full list of clinical modules the manual covers, with section-to-module mapping. | JSON: `{workflow_type, modules: [{id, title, page_range}], section_map}` | Stage 2b (xlsx B1) | Stage 2c-2g |
| **RAW_FACTS** | Atomic clinical facts extracted using the 13-category schema: Cutpoint, DangerSign, Treatment, Observation, Measurement, Classification, Referral, Advice, FollowUp, Supply, Procedure, Context, Constraint. Each fact: one predicate or measurement only. | CSV/JSON rows: `{fact_id, label, kind, category, data_type, units, source_quote, source_page, module}` | Stage 2c (xlsx B1 + B5b_FactSheet) | Stage 2d-2g |
| **CONTEXT_MAPPINGS** | Context variables (e.g., `malaria_area: bool`, `altitude: enum`) plus complaint-to-module mapping (e.g., "cough" activates `mod_cough_breathing`). | JSON: `{context_vars: [{name, type, default}], complaint_to_module: [{complaint, module_id}]}` | Stage 2d (xlsx B2) | Stage 2g, Stage 4 |
| **SUPPLY_INVENTORY** | Physical inventory: equipment (`equip_` prefix, durable) and consumables (`supply_` prefix, depletable). Each item has `stockout_fallback` (alternate treatment or refer). Cross-referenced to observations that depend on them. | JSON array: `[{id, kind, display_name, stockout_fallback, used_by, source_quote, source_page}]` | Stage 2e (xlsx B1 partial + B5b) | Stage 2g, Stage 4 |
| **CONSOLIDATION_RULES** | Integrative-phase policies extracted verbatim from the manual: referral priority ordering, follow-up prioritization, treatment de-duplication rules, pre-referral care rules. | CSV/JSON: `[{policy_type, rule_text, source_quote, source_page}]` | Stage 2f (xlsx B3) | Stage 4c |
| **FACTSHEET** | The enriched, cross-referenced, provenance-grounded semantic extraction. Every clinical concept tagged by type (`assessment_question`, `classification_rule`, `treatment_action`, `aggregation_rule`) with full provenance (`source_quote`, `page`, `line`, `section_id`). Naming conventions applied, content IDs assigned (.how, .why, .tell), supplies cross-referenced to observations, `required_by_modules` populated, phrase_id mappings added. This matches `guide_full_example.json`. | JSON: `{guide_metadata, sections: [{id, title, page_range, module, instructions, content: [{type, ...provenance}]}]}` | Stage 2g (xlsx B4 + B5b_FactSheet) | Stage 3 |
| **BOOLEAN_LITERALS** | Named boolean predicates replacing every continuous threshold. Each literal has an ID, negation ID, threshold expression, and missing_policy. "RR >= 50 for age < 12mo" becomes `p_fast_breathing_under_12mo`. No raw numerics pass this point. | CSV/JSON: `[{literal_id, negation_id, source_vars, threshold_expression, missing_policy, source_quote}]` | Stage 3a (xlsx B5b) | Stage 3b, Stage 4 |
| **RULE_CRITERIA** | Atomized one-fact-per-row rule criteria. Each complex classification rule decomposed into atomic conditions. Module linkage added (which modules need which observations). | CSV/JSON: `[{rule_id, variable_code, required_value, required_by_modules, source_fact_ids, source_quote}]` | Stage 3b (xlsx B6 + B6_v25) | Stage 4 |
| **DMN_JSON** | Compiled decision model: `modules` (per-module Gather + Classify tables), `activator` (COLLECT), `router` (FIRST with danger-sign short-circuit), `integrative` (7 consolidation categories), `predicates` (boolean-only with fail-safe), `phrase_bank` (localization messages). All inputs boolean. All antecedents reference named literals. Full provenance on every rule row. | JSON: `{modules, activator, router, integrative, predicates, phrase_bank}` matching `valid_logic.json` schema | Stage 4 (A/B test target) | Stage 5 |
| **DMN_XML** | OMG DMN 1.3 XML serialization. Deterministic conversion from DMN_JSON. | `.dmn` XML | Stage 5 (converter) | External tools, Stage 6 |
| **XLSFORM** | CHT-deployable ODK XLSForm with survey/choices/settings sheets, workflow engine calculates (module queue, urgent referral short-circuit), per-rule `calc_*_citation`, and Why-trace JSON payload per diagnosis. | `.xlsx` (3 sheets) | Stage 5 (compiler, xlsx B9) | CHT deployment |
| **MERMAID** | Flowchart visualization of the DMN. Distinct node shapes per table kind. | `.md` with `graph TD` | Stage 5 (converter) | Documentation |
| **PATIENT_SET** | Synthetic or clinician-validated patient scenarios. T=3 across modules, t=2 within, boundary triplets {c-1,c,c+1} for continuous thresholds. External input to the gate harness; NEVER seen by the pipelines. | CSV/XLSX: one row per patient, columns are variable names | External (xlsx D1b or clinician) | Stage 6 |
| **COMPARISON_REPORT** | Test-retest reliability (within-pipeline) + clinical equivalence (cross-pipeline). Patient-level outcome comparison. Cost and timing metrics. | JSON + Markdown | Stage 6 (gate harness) | Decision-making |

---

## Stage Architecture

### Stage 0: RAW_PDF
- **Input:** PDF file
- **Output:** RAW_PDF (bytes, unchanged)
- **Tool:** None
- **Status:** N/A (fixed input)

### Stage 1: PROVENANCE_JSON (physical extraction)
- **Input:** RAW_PDF
- **Output:** PROVENANCE_JSON
- **Tool:** Unstructured hi_res API + GPT vision (table refinement, image description, flowchart parsing)
- **Status:** Already built (`backend/ingestion/`)
- **Time:** 5-30 min for 141 pages
- **xlsx provenance:**
  - Sheet `"Quality stds"` row R24: "OCR legibility: page numbers, tables, annexes machine-readable. OCR confidence >= 0.9 required; for dosing tables, confidence >= 0.99."
  - Sheet `"Software_Libraries_v23"` row R3: OCR validator tool mapping (pytesseract, pdf2image, OpenCV)
  - Sheet `"A2 reviewmanual"` row R24: "OCR legibility" quality standard for the manual review stage
- **Checkpoints:**
  - `emit("layout_elements")` -- pages + typed elements + coordinates
  - `emit("vision_enrichment")` -- table refinements + image descriptions + flowchart structures
- **Quality gates:**
  - IngestionManifest `critical_count == 0`
  - Hierarchy quality: `good` or `sparse` (not `fallback`)
  - All Table elements have `text_as_html` (hi_res requirement)
  - Coordinates present on all elements (vision crop requirement)

### Stage 2: FACTSHEET (semantic extraction)
- **Input:** PROVENANCE_JSON
- **Output:** FACTSHEET
- **Tool:** LLM (the core unsolved problem)
- **Status:** NOT BUILT
- **xlsx provenance:**
  - Sheet `"b1,b2,b3,b4,b6"` -- Summary of all B-phase stages. Row R13 has column headers (B1_make through B8_redteam). Rows R14-R22 have: role, description, objectives, inputs, tasks, outputs, footer, version per stage. **Start here** for the overview of what each B-stage does.
  - Sheet `"PR_B5b_make_v23"` -- Contains TWO prompts: (1) the v23 Boolean Literals maker spec, and (2) the embedded **B5b_FactSheet_v25** prompt which is the full FactSheet authoring specification. The FactSheet prompt starts at the "B5b_FactSheet_v25" header ~halfway down the sheet. It defines: ROLE (clinical content author), CONTEXT (FactSheet drives all DMN predicates), OBJECTIVES (produce FactSheet.csv with one row per atomic fact), INPUTS, 8-step TASKS/PROCEDURE, OUTPUTS, embedded QUALITY STANDARDS, and SAFETY NOTE ("Do not invent cutpoints").
  - Sheet `"A2 reviewmanual"` -- The manual review prompt (A2_make). Rows R9-R35 contain 20+ quality sub-standards for content triage: MECE decision tables, contiguous cutpoints, stock-out backup, "no diagnosis" endpoint, referral hierarchy, pre-referral care, OCR legibility, drug dosing coverage, workflow clarity, cross-module interactions, re-measurement rules, follow-up intervals, counselling extraction, version notation. **This is the content triage checklist.**
  - Sheet `"A2 repair"` -- The manual repair prompt (A3_make). Proposes WHO/national-policy-aligned fixes for each issue found by A2. Red-teams its own fixes before presenting.
  - Sheet `"Quality stds"` rows R3-R44 -- Quality rules for Decision Logic (R3-R22), Observations & Measurements (R23-R25), Supplies & Stock-outs (R27-R29), and Fact Sheet QA (R31-R44). Each row has: Domain, Title, Rule, Check, Check_Category, Rationale.
- **Intermediate artifacts and checkpoints:**

| Sub-step | Checkpoint | xlsx Sheet(s) | What to Read | Key Validation |
|---|---|---|---|---|
| 2a. Content triage | `emit("content_triage")` | `"A2 reviewmanual"` rows R9-R35 | 20+ quality sub-standards as classification checklist | Coverage: every element classified; no clinical content in `training_exercise` bucket |
| 2b. Workflow pattern | `emit("workflow_pattern")` | `"b1,b2,b3,b4,b6"` column B1_make, row R19 (Tasks) | "Task 1: Identify Macro Workflow Pattern. Task 2: Extract Atomic Facts using 13 categories." | Completeness: module list covers all clinical topics in the manual |
| 2c. Raw fact extraction | `emit("raw_facts")` | `"b1,b2,b3,b4,b6"` column B1_make + `"PR_B5b_make_v23"` section B5b_FactSheet_v25 | B1 Tasks: "Extract Atomic Facts using 13 categories." B5b_FactSheet: 8-step procedure for creating atomic FactSheet rows | R13: Atomic (one predicate per row). R90: Every fact has source_quote. |
| 2d. Context mappings | `emit("context_mappings")` | `"b1,b2,b3,b4,b6"` column B2_make, row R19 | "Task 1: Extract Context Variables. Task 2: Extract Complaint-to-Module Logic." | Schema validation. All context vars have type + default. |
| 2e. Supply inventory | `emit("supply_inventory")` | `"PR_B5b_make_v23"` section B5b_FactSheet_v25, Task 4 | "Stockout Policy: For every treatment add stockout_fallback (alternate treatment or refer)." | R28: Bidirectional observation-supply link. |
| 2f. Consolidation rules | `emit("consolidation_rules")` | `"b1,b2,b3,b4,b6"` column B3_make, row R19 | "Extract consolidation policies (referral, follow-up, safety) into CSV." | R62-R64: Referral priority explicit, pre-referral care present. |
| 2g. Enriched factsheet | `emit("factsheet")` | `"b1,b2,b3,b4,b6"` column B4_make + `"PR_B5b_make_v23"` section B5b_FactSheet_v25 | B4: "Apply naming convention, add content IDs (.how, .why, .tell), cross-reference supplies." B5b_FactSheet: quality gates, phrase_id mappings. | R31: Two-way coverage. R42: Forms alignment. Provenance grounded. |

- **Quality gates (from xlsx `"Quality stds"`):**
  - R13 (row 13): Atomic facts (one predicate per row)
  - R23 (row 23): Observation schema (id, label, type, units/range, help_text)
  - R28 (row 28): Bidirectional observation-supply link
  - R31 (row 31): Two-way coverage (all inputs influence >= 1 rule; each endpoint derives from some inputs)
  - R35 (row 35): Unused inputs flagged
  - R42 (row 42): Forms alignment (recording form fields match observation registry)
  - R90 (row 90): Verbatim citations on every fact
  - `"PR_B5b_make_v23"` SAFETY NOTE: "Do not invent cutpoints. If manual lacks numeric cutpoint, mark Not_in_guideline and escalate."

### Stage 3: RULE_CRITERIA (predicate compilation)
- **Input:** FACTSHEET
- **Output:** BOOLEAN_LITERALS + RULE_CRITERIA
- **Tool:** LLM + deterministic validation
- **Status:** NOT BUILT
- **xlsx provenance:**
  - Sheet `"PR_B5b_make_v23"` -- The v23 section (top of sheet) is the Boolean Literals maker prompt: "Emit Boolean_Literals.csv + negations + missingness; generate Python & XLSForm calculate templates; booleans are runtime only." Unit tests: "Literal truth tables (Block, difficulty 2)" and "No stored booleans (Warn, difficulty 2)."
  - Sheet `"PR_B6_make_v23"` -- Contains TWO prompts: (1) the v23 rule decomposition spec: "Decompose rules; enforce boolean-only antecedents; emit Boolean Literal Spec; no raw numerics; safe defaults." (2) the embedded **B6_DMNs_v25** prompt: "Convert FactSheet predicates and rule_criteria into modular DMN tables with stable rule IDs and citations." The v25 section has the full ROLE, CONTEXT, OBJECTIVES, 7-step TASKS/PROCEDURE, OUTPUTS, and SAFETY NOTE ("Do not use raw numeric tokens in conditions").
  - Sheet `"v25 supplements"` section "PR_B6_make_v25": "rule_criteria ATOMIZATION -- For each decision_rule, add one rule_criteria row per atomic condition: rule_id, variable_code, required_value." And: "required_by_modules -- Enrich every observation_definition with required_by_modules. Base linkage on exact quotes + page numbers."
  - Sheet `"V 25c sup"` section "PR_B6_make_v25": Structured copy-paste version with unit tests table (Criteria coverage Block/3, No raw numerics Block/3, Link completeness Block/3).
  - Sheet `"Quality stds"` rows R3-R22: Decision Logic domain rules. Especially R13 (atomic facts), R8 (no forward references), R20 (unit typing enforced).
- **Checkpoints:**

| Sub-step | Checkpoint | xlsx Sheet(s) | What to Read | Key Validation |
|---|---|---|---|---|
| 3a. Boolean literals | `emit("boolean_literals")` | `"PR_B5b_make_v23"` top section (v23 Boolean Literals) | Maker prompt + unit tests (literal truth tables, no stored booleans) | 0 raw numeric tokens downstream (Block). Literal truth tables verify literal/negation + missingness (Block). |
| 3b. Rule criteria + linkage | `emit("rule_criteria")` | `"PR_B6_make_v23"` v23 section + `"v25 supplements"` PR_B6_make_v25 | v23: "Decompose rules, no raw numerics." v25: "One rule_criteria row per atomic condition. Enrich with required_by_modules." | Every rule decomposed to 1+ criteria (Block). Every observation linked to module(s) (Block). |

- **Quality gates (from xlsx):**
  - `"PR_B5b_make_v23"` unit tests: Literal truth tables (Block, difficulty 2); No stored booleans (Warn, difficulty 2)
  - `"PR_B6_make_v23"` unit tests: No raw numerics (Block, difficulty 3); Defaults present (Block, difficulty 2)
  - `"v25 supplements"` PR_B6_make_v25 unit tests: Criteria coverage (Block, 3); No raw numerics (Block, 3); Link completeness (Block, 3)
  - `"Quality stds"` R13: Atomic facts; R8: No forward references; R20: Unit typing enforced

### Stage 4: DMN_JSON (decision model compilation)
- **Input:** FACTSHEET + BOOLEAN_LITERALS + RULE_CRITERIA + CONSOLIDATION_RULES
- **Output:** DMN_JSON
- **Tool:** LLM -- THIS IS THE A/B TEST (REPL vs Sequential)
- **Status:** Partially built (existing REPL pipeline in `backend/`; needs adaptation to consume FACTSHEET instead of PROVENANCE_JSON)
- **xlsx provenance:**
  - Sheet `"PR_B6_make_v23"` embedded section **B6_DMNs_v25** -- The DMN authoring prompt. ROLE: "DMN designer: convert FactSheet predicates and rule_criteria into modular DMN tables (gather, classify, integrative) with stable rule IDs and citations." 7-step procedure: rule atomization, gather tables (FIRST), classify tables (ordered IF cascade), integrative DMN, citations (calc_citation), trace fields (calc_facts_used), notes. SAFETY NOTE: "Do not use raw numeric tokens in conditions."
  - Sheet `"b7 DMN Integrative"` -- Parts 3 and 4 of the integrative maker prompt. Part 3: 7 consolidation categories (referral priority, pre-referral care, treatment de-dup, dose/interaction, supply/dispensing, follow-up min days, safety checks). Part 4: Follow-Up Plan DMN with concrete examples (dX.diarrhea + Tx.ORS + "Severe" -> days=1).
  - Sheet `"makecompiler"` -- Contains THREE specs: (1) B9_make v23 compiler prompt, (2) embedded **B9_Compiler_v25** (compiler spec with literal emits, gather calculates, queue calculates, classification, integrative mapping, why-trace, unit tests), (3) embedded **Workflow_Engine_v25** (Initial Module Selection COLLECT, Next Action Engine FIRST, overrides/short-circuits, gather table pattern example).
  - Sheet `"v25 supplements"` sections: PR_B9_make_v25 (workflow engine + queue, gather tables example with Cough_gather 4-rule table, decision trace JSON schema example, 7 unit tests all Block gate). PR_Workflow_Engine_v25 (DMN_Workflow_Engine.dmn decisions 1-2, override rule, 4 unit tests). PR_B6_make_v25 (rule_criteria atomization + required_by_modules, 3 unit tests).
  - Sheet `"V 25c sup"` -- Structured copy-paste versions of all v25 supplements with exact column headers and unit test tables.
  - Sheet `"F1 phrase bank"` -- F1_make: "Generate English Phrase Bank. Collate all IDs, generate Phrase option 1 & 2 with confidence. Include rule-specific 'why' IDs."
  - Sheet `"Quality stds"` rows R3-R22 (Decision Logic), R46-R60 (DMN QA), R62-R71 (Integration Phase).
- **Checkpoints (existing emit_artifact names):**

| Sub-step | Checkpoint | xlsx Sheet(s) | What to Read | Key Validation |
|---|---|---|---|---|
| 4a. Module tables | `emit("modules")` | `"PR_B6_make_v23"` section B6_DMNs_v25 + `"v25 supplements"` PR_B6_make_v25 | DMN authoring: gather (FIRST) + classify tables. Rule atomization. Citations per row. | Each row has rule_id, facts_used, citation. |
| 4b. Activator + Router | `emit("router")` | `"makecompiler"` section Workflow_Engine_v25 + `"v25 supplements"` PR_Workflow_Engine_v25 | Decision 1: Initial Module Selection (COLLECT). Decision 2: Next Action Engine (FIRST). Override: danger sign -> clear queue. Gather table example (Cough_gather). | Activator COLLECT, Router FIRST. Row 0 = danger sign short-circuit. |
| 4c. Integrative | `emit("integrative")` | `"b7 DMN Integrative"` parts 3-4 + `"v25 supplements"` PR_B9_make_v25 part 3 | 7 consolidation categories. Follow-Up Plan DMN with (diagnosis, treatment, severity) -> (days, instruction). | R62-R71: All 10 Integration Phase quality rules. |
| 4d. Phrase bank | `emit("phrase_bank")` | `"F1 phrase bank"` | "Collate all IDs, generate Phrase option 1 & 2 with confidence. Include rule-specific 'why' IDs." | All CHW-facing messages with placeholder vars. |
| 4e. Self-validation | `emit("supply_list")`, `emit("variables")`, `emit("predicates")` | (assembled from earlier artifacts; validated by `"Quality stds"` R3-R22 + R46-R60) | Structural validators + Z3 proofs. | Z3: exhaustiveness, reachability, monotonicity. |

- **Quality gates (from xlsx `"Quality stds"`):**
  - R3 (row 3): MECE decision tables (Z3/pyDMNrules)
  - R4 (row 4): Path termination (every path ends at refer/treat/observe/no_diagnosis)
  - R5 (row 5): Endpoint reachability (each endpoint reachable by >= 1 input assignment)
  - R6 (row 6): Safe "no diagnosis" endpoint exists
  - R8 (row 8): No forward references
  - R11 (row 11): Danger-sign monotonicity (worse findings never downgrade care)
  - R12 (row 12): Workflow short-circuit (urgent routes bypass non-urgent)
  - R19 (row 19): Else/default rows present in every table
  - R51 (row 51): Hit policy valid (U/F/P only)
  - R62 (row 62): Safety checks (no oral meds to unconscious child)
  - R63 (row 63): Cross-module interaction check
  - R64 (row 64): Pre-referral care consolidated
  - R66 (row 66): Integrative de-duplication
  - R67 (row 67): Follow-up merge (earliest dominates)

### Stage 5: DEPLOYMENT_FORMATS (deterministic conversion)
- **Input:** DMN_JSON
- **Output:** DMN_XML, XLSFORM, MERMAID, CSV
- **Tool:** Deterministic converters (no LLM)
- **Status:** Mostly built (`backend/converters/`). Missing: Why-trace JSON, module queue workflow engine in XLSForm, compiler patch log.
- **xlsx provenance:**
  - Sheet `"makecompiler"` -- B9_make v23 + embedded B9_Compiler_v25. The compiler spec: "Translate DMN tables and FactSheet literals into an XLSForm survey. Emit xlsform_survey.csv and xlsform_calculates.csv that implement gather/classify/integrative logic." 8-step procedure: literal emits, gather calculates, queue calculates (module_queue, current_module, push/pop), classification (ordered IF cascade to calc_matched_rule), integrative mapping, Why-trace JSON (compact JSON array per diagnosis with facts_used + selected_rule_id), unit tests emission, TODO_literal handling. SAFETY NOTE: "Never drop citations. If cannot map a DMN row, emit compiler_warning and failing test case."
  - Sheet `"C1 make xlsform"` -- C1_XLSForm_v25: "Create the final survey CSVs that implement module groups, calculates, and the integrative step; include hidden fields for Why-trace JSON." Tasks: group design (begin_group per module, relevant = current_module), questions mapping, calculates import, hidden trace fields (why_trace_<module>), stockout/dispense UI, localization columns.
  - Sheet `"PR_C2_make_v23"` -- C2_make: "Insert boolean calculates before clinical logic; run pyxform + Validate; ensure language coverage; no undefined references." Tasks: generate calc fields for matched_rule, diagnosis_code, citation note. Integrative logic. Content linkers (why_id, how_id, tell_id).
  - Sheet `"Software_Libraries_v23"` rows R2-R5: Tool mappings for B9 (pyxform), C2 (pyxform + ODK Validate), D1c/D4 (SpiffWorkflow, pyxform-runner).
  - Sheet `"WriteOnce_Code_v23"` row for dmn_to_xlsform_compiler.py: "Compiler for boolean-only antecedents; emits calculates & patch log."
  - Sheet `"v25 supplements"` PR_B9_make_v25: Decision trace JSON example schema: `{"decision_id":"visit-uuid/dx.Pneumonia","dmn_table":"imci.respiratory.classify","selected_rule_id":"R3","facts_used":[...],"policy":{"dmn_hit_policy":"FIRST","confidence":1.0}}`. 7 unit tests (all Block gate).
- **Quality gates:**
  - `"Quality stds"` R46 (row 46): DMN XML schema validation (OMG 1.3)
  - `"PR_C2_make_v23"` unit tests: pyxform clean (Block, 2); Calculate order -- no forward references (Block, 3)
  - `"v25 supplements"` PR_B9_make_v25 unit tests: DMN-to-XForm equivalence (Block, 4); Why-trace JSON schema (Block, 3); Equivalence SMOKE/QUICK/SAFETY (Block, 2/3/4); Follow-Up DMN cases (Block, 3); Safety nets (Block, 4)
  - `"makecompiler"` unit tests: DMN<->XForm equivalence (Block, 4); Why-trace injected (Block, 3); Patch log integrity (Block, 3)

### Stage 6: PATIENT_TESTING (measurement instrument)
- **Input:** DMN_JSON + PATIENT_SET (external)
- **Output:** COMPARISON_REPORT
- **Tool:** Deterministic DMN executor (no LLM)
- **Status:** NOT BUILT
- **xlsx provenance:**
  - Sheet `"PR_D1b_make_v23"` -- Synthetic patient generation spec. v23 section (rows R3-R51): "Mixed-strength coverage (T=3 modules / t=2 intra); boundary triplets (c-1,c,c+1) replace continuous; modes; repeats." Detailed YAML metadata (risk_level: High). Unit tests: Coverage >= 95% (Warn, 3), threshold flips (Block, 3), determinism (Block, 2). Embedded **D1b_make_v25** in `"v25 supplements"`: expanded coverage policy (T=3 across module selectors, t=2 within, context toggles one-at-a-time in QUICK, all relevant in SAFETY), boundary triplets for age_months {c-1,c,c+1}, resp_rate, temp_c {c-0.2,c,c+0.2}, muac_mm. "Predicates remain transient at runtime; EMR stores raw continuous values only."
  - Sheet `"PR_D1c_make_v23"` -- Robot execution spec. v23: "Robot executes DMN; compute coverage & compare to gold; determinism checks." Unit tests: Exec correctness -- all cases match gold (Block, 3); Coverage report (Warn, 2). Embedded **D1_Tests_v25** (rows R21-R88): Full QA test author spec. "Testing uses synthetic patients run through compiled XLSForm (XML) by a robot harness." Grading: Dx_correct (0/1), Action_correct (0/1), Safety_violation (0/1). Includes comorbidity coverage, edge/negative cases, repeat determinism.
  - Sheet `"PR_D4_make_v23"` -- End-to-end equivalence testing. "Robot executes form.xml; compute PASS/FAIL; ensure equivalence with DMN; coverage & repeats." Unit tests: Equivalence with DMN -- exact match on outputs (Block, 4); Determinism (Block, 2).
  - Sheet `"V 25c sup"` section PR_D1b_make_v25: Structured version with unit tests table (Module triples >= 95% Warn/3, Age 6 mo triplet Block/3, RR 50 bpm triplet Block/3, Determinism repeats Block/2).
  - Sheet `"Software_Libraries_v23"` rows R5-R6: Tool mappings for D1b (ACTS/AllPairs, synthetic_gen.py), D1c/D4 (SpiffWorkflow, pyxform-runner, pytest).
  - Sheet `"WriteOnce_Code_v23"` row for synthetic_gen.py: "Mixed-strength generator with boundary triplets."
  - Sheet `"Quality stds"` rows R73-R80 (Testing & CI): synthetic test cases >= 5 per module, golden test per rule row, mutation testing, edge cases, consistency/determinism, coverage metric. Rows R108-R112 (Patient Simulation): happy paths, edge cases, multiple triggers, invalid/missing data, conflicting rules.
- **Patient generation spec (from xlsx D1b + D1b_v25):**
  - T=3 across module selectors
  - t=2 within each active module's locals
  - Boundary triplets: {c-1, c, c+1} for age, RR, temp, MUAC
  - Context toggles: one-at-a-time in QUICK; all relevant in SAFETY
  - Determinism check: 2 sets of 10 repeats, byte-identical outputs
  - Three modes: SMOKE (t=1), QUICK (T=3/t=2), SAFETY (full + boundaries + repeats)
- **Quality gates:**
  - `"PR_D1b_make_v23"` + `"V 25c sup"`: Module triples >= 95% (Warn, 3); Threshold flips at boundary (Block, 3); Determinism across repeats (Block, 2)
  - `"PR_D4_make_v23"`: Equivalence with DMN (Block, 4); Determinism (Block, 2)
  - `"Quality stds"` R73: >= 5 edge/negative/scenario cases per module; R74: golden test per rule row; R75: mutation testing; R78: consistency (same case = same output); R80: coverage metric reported

---

## A/B Equivalence Principle: Same Knowledge, Different Packaging

The REPL's single system prompt IS the aggregate of the Sequential pipeline's ~45 separate prompts. The difference being tested is **architecture** (one continuous session vs many isolated calls), not **content** (what the LLM is told to do).

The mapping is exact:

| Sequential Prompt | REPL System Prompt Section | Content (identical in both) |
|---|---|---|
| B1_make (Extract Atomic Facts) | Phase 2 instructions | "Extract atomic facts using 13-category schema: Cutpoint, DangerSign, Treatment, Observation, Measurement, Classification, Referral, Advice, FollowUp, Supply, Procedure, Context, Constraint" |
| B2_make (Extract Mappings) | Phase 2 context extraction | "Extract context variables (malaria_area, etc.) and complaint-to-module mappings" |
| B3_make (Consolidation Policies) | Phase 2 consolidation instructions | "Extract referral priority, follow-up prioritization, treatment de-duplication policies" |
| B4_make (Enrich FactSheet) | Phase 2 enrichment pass | "Apply naming conventions, add content IDs (.how, .why, .tell), cross-reference supplies" |
| B5b_make (Boolean Literals) | Predicate convention section | "Convert every continuous threshold to a named boolean literal. No raw numerics downstream." |
| B6_make (Rule Decomposition) | Phase 3 instructions | "Atomize each decision rule to one-fact-per-row rule criteria with variable_code, required_value" |
| B6_DMNs_v25 (DMN Tables) | Phase 4 module extraction | "Produce per-module DMN: Gather (FIRST), Classify (FIRST/PRIORITY), Outcome tables" |
| B7 (Integrative) | Phase 4 integrative instructions | "7 consolidation categories: referral priority, pre-referral care, treatment de-dup, dose/interaction, supply/dispensing, follow-up min, safety checks" |
| Workflow_Engine_v25 | Phase 4 router instructions | "Activator (COLLECT), Router (FIRST), emergency short-circuit row 0" |
| F1_make (Phrase Bank) | Phase 5 instructions | "All CHW-facing messages with placeholder variables, rule-specific 'why' sentences" |
| A0_redteam_v25 (Red Team) | Frozen catcher validators at each `emit_artifact` | "Severity matrix: Critical/High/Medium/Low. Evidence: verbatim citation <= 120 words. Coverage >= 95%." |
| Quality Standards (160 rules) | Embedded in catcher validator logic | Same rules checked by same criteria |

**This equivalence is a design invariant.** When the Sequential pipeline's B6_make prompt is edited, the corresponding Phase 3 section of the REPL system prompt must be updated to match, and vice versa. If they diverge, the A/B test measures content differences, not architectural differences, which invalidates the comparison.

**What differs between the two architectures (the independent variable):**

| Property | Sequential | REPL |
|---|---|---|
| Context visibility | Each stage sees only what you explicitly pass to it. The predicate extractor sees the supply list and the manual, but NOT the reasoning the model used to produce the supply list. | Full context of everything done so far. If predicate extraction reveals a missing variable, the model can fix the variable list before proceeding. |
| Backward fixes | Require `PROPAGATE_FIX_NEEDED`: re-run upstream stages from scratch with new prompts that have no memory of the original reasoning. | In-session variable edit + re-emit artifact. Original reasoning still in context. |
| Red team degrees of freedom | LLM red team per stage: can be "creative" about what it checks. Strength: catches novel issues. Weakness: introduces variance. | Frozen catcher validators: fixed criteria, deterministic Python + LLM at temperature=0 with 3x majority vote. Strength: zero variance. Weakness: only catches what they're programmed to check. |
| Prompt count (tunable variables) | ~45 prompts. Editing one can contaminate downstream stages' inputs. | 1 system prompt + catcher suite. Editing the prompt has one total derivative. |
| Context window per call | Fresh, small (~15-20K tokens). Fast time-to-first-token. | Growing (~100-200K tokens by iteration 50). Slower per iteration but no information loss. |

---

## Artifact Inventory (17 Checkpoints)

**Everything is JSON until Stage 5.** The REPL works in Python dicts; `emit_artifact` serializes to JSON. The Sequential pipeline's xlsx specifies CSV outputs (FactSheet.csv, Boolean_Literals.csv, etc.) but for the A/B test both pipelines emit JSON for consistent comparison. Stage 5 is the only stage that produces non-JSON formats (XML, XLSX, Markdown).

**Count: 17 checkpoints. 16 JSON. 1 mixed (XML + XLSX + MD + CSV).**

### Stage 1: Physical Extraction (2 artifacts)

| # | Artifact | Format | Shape | Contents |
|---|---|---|---|---|
| 1 | `layout_elements` | JSON | `{pages: {num: {page_number, elements: [{element_type, text, text_as_html, page_number, element_id, coordinates}], raw_text}}}` | Every element from the PDF typed by Unstructured (paragraph, table, image, heading, list, title, header, footer). Page numbers, bounding box coordinates, element IDs. Space-joined raw text per page. |
| 2 | `vision_enrichment` | JSON | `{table_refinements: {eid: TableRefinement}, image_descriptions: {eid: ImageDescription}, flowchart_structures: {eid: FlowchartStructure}}` | Tables: structured rows, headers, severity colors, translation flag. Images: caption, description, is_flowchart/is_photograph/is_diagram. Flowcharts: nodes (id, label, type), edges (from, to, condition), English walkthrough. |

**Validator:** Deterministic IngestionManifest (critical_count == 0, hierarchy quality).

### Stage 2: Semantic Extraction (7 artifacts)

| # | Artifact | Format | Shape | Contents |
|---|---|---|---|---|
| 3 | `content_triage` | JSON | `[{element_id, page_number, relevance_class, confidence}]` | Each PROVENANCE_JSON element classified as `clinical_protocol`, `training_exercise`, `narrative`, or `administrative`. Filters the WHO 2012 training manual's stories, role plays, video exercises, and fill-in-the-blank worksheets from the actual clinical protocol. |
| 4 | `workflow_pattern` | JSON | `{workflow_type, modules: [{id, title, page_range, section_ids}], section_map: {section_id: module_id}}` | Macro workflow type (e.g., "episodic_complaint_driven"). Complete list of clinical modules the manual covers. Mapping from PROVENANCE_JSON section slugs to module IDs. |
| 5 | `raw_facts` | JSON | `[{fact_id, label, kind, category, data_type, units, source_quote, source_page, module}]` | Atomic clinical facts using the 13-category schema from xlsx B1: Cutpoint, DangerSign, Treatment, Observation, Measurement, Classification, Referral, Advice, FollowUp, Supply, Procedure, Context, Constraint. One predicate per row (R13 quality standard). Every row has `source_quote` (R90). |
| 6 | `context_mappings` | JSON | `{context_vars: [{name, type, default_value, description}], complaint_to_module: [{complaint, module_id, source_quote}]}` | Context variables (e.g., `malaria_area: bool, default: false`). Complaint-to-module activation mapping (e.g., "cough" activates `mod_cough_breathing`). |
| 7 | `supply_inventory` | JSON | `[{id, kind, display_name, stockout_fallback, used_by, source_quote, source_page}]` | Equipment (`equip_` prefix: thermometer, MUAC tape, timer) and consumables (`supply_` prefix: ORS packets, amoxicillin, RDT kits). Each item has `stockout_fallback` (alternate treatment or refer). Cross-referenced to variables and predicates that depend on them. |
| 8 | `consolidation_rules` | JSON | `[{policy_type, rule_text, source_quote, source_page}]` | Integrative policies extracted verbatim from the manual: referral priority ordering, follow-up prioritization, treatment de-duplication, pre-referral care rules, safety checks (e.g., "do not give oral medicine if child cannot drink"). |
| 9 | `factsheet` | JSON | `{guide_metadata: {title, version, country, language, total_pages, applicable_age_range}, sections: [{id, title, page_range, module, instructions, content: [{type, ...fields, source_quote, page, line}]}]}` | The enriched, cross-referenced, provenance-grounded semantic extraction. Every clinical concept tagged by type (`assessment_question`, `classification_rule`, `treatment_action`, `aggregation_rule`). Naming conventions applied, content IDs assigned, supplies cross-referenced to observations, `required_by_modules` populated. **Matches `guide_full_example.json` format.** |

**Validators:** Mix of deterministic (schema, cross-ref integrity, R13 atomic, R28 bidirectional supply link) and LLM catchers (provenance grounding against raw PDF text, completeness, clinical review).

### Stage 3: Predicate Compilation (2 artifacts)

| # | Artifact | Format | Shape | Contents |
|---|---|---|---|---|
| 10 | `boolean_literals` | JSON | `[{literal_id, negation_id, source_vars, threshold_expression, missing_policy, source_quote, page_ref}]` | Named boolean predicates replacing every continuous threshold. "Respiratory rate >= 50 breaths/min for age < 12 months" becomes `{literal_id: "p_fast_breathing_under_12mo", threshold_expression: "v_resp_rate_per_min >= 50 AND demo_age_months < 12", missing_policy: "fail_safe_1"}`. No raw numerics pass this point. Includes negation IDs and missingness handling. |
| 11 | `rule_criteria` | JSON | `[{rule_id, variable_code, required_value, required_by_modules, source_fact_ids, source_quote}]` | Atomized one-fact-per-row rule criteria. Each complex classification rule decomposed into atomic conditions. Module linkage added (which modules need which observations). E.g., rule "severe pneumonia" decomposes to 3 rows: `{rule_id: "cls_severe_pneumonia", variable_code: "p_chest_indrawing", required_value: "true"}`, `{..., variable_code: "p_danger_sign_present", required_value: "true"}`, etc. |

**Validators:** Deterministic only. 0 raw numeric tokens in antecedents (Block). Every rule decomposed to 1+ criteria (Block). Every observation linked to module(s) (Block). Literal truth tables pass (Block).

### Stage 4: DMN Compilation (6 artifacts)

| # | Artifact | Format | Shape | Contents |
|---|---|---|---|---|
| 12 | `modules` | JSON | `[{module_id, display_name, hit_policy, input_columns, output_columns, rules: [{inputs, outputs, provenance: {page, quote}}]}]` | Per-module decision tables. Gather tables (FIRST policy: picks next observation to ask) and Classify tables (FIRST/PRIORITY: maps boolean predicates to diagnosis/treatment/referral). All inputs are boolean (named literals from Stage 3). Every rule row has provenance. |
| 13 | `router` | JSON | `{activator: {input_columns, rules: [{inputs, module_id}]}, router: {rules: [{condition, next_module, priority}]}}` | Activator (COLLECT policy: flags all relevant modules from screening questions). Router (FIRST policy: priority-ordered module queue with emergency short-circuit as row 0). |
| 14 | `integrative` | JSON | `{description, rules: [...], uncovered_combinations: [...]}` | Consolidation module implementing 7 categories from xlsx B7: (1) referral prioritization, (2) pre-referral care, (3) treatment de-duplication, (4) dose/interaction management, (5) supply/dispensing list, (6) follow-up prioritization (min days), (7) safety checks. |
| 15 | `phrase_bank` | JSON | `[{message_id, category, english_text, placeholder_vars, page_ref}]` | All CHW-facing messages: questions, diagnoses, treatments, advice, referral instructions, follow-up instructions. Category is one of: question, diagnosis, treatment, advice, referral, followup, instruction. Placeholder variables for dynamic values (e.g., `{dose_mg}`, `{days}`). |
| 16 | `supply_list` + `variables` + `predicates` | JSON | Three sub-artifacts assembled from earlier stages | `supply_list`: equip_ + supply_ items with used_by cross-refs. `variables`: q_, ex_, v_, lab_, hx_, demo_ inputs with data types and units. `predicates`: p_ boolean predicates with threshold_expression, fail_safe (0 or 1), source_vars. Validated together by structural validators + Z3. |

**Validators:** Existing catcher suite. Deterministic: provenance, consistency (duplicate IDs), module architecture (hit policies, router row 0). LLM: completeness, clinical review, comorbidity coverage. Plus Z3: exhaustiveness, reachability, danger-sign monotonicity, mutual exclusivity.

**Note:** Artifact #16 is three logical artifacts (`supply_list`, `variables`, `predicates`) that are emitted separately via `emit_artifact` but validated together as the assembled `clinical_logic` dict during Phase 6 (self-validation) of the REPL session.

### Stage 5: Deployment Formats (1 checkpoint, 4 output files)

| # | Artifact | Format | Shape | Contents |
|---|---|---|---|---|
| 17 | `deployment_formats` | Mixed | 4 files | **DMN XML** (`.dmn`): OMG DMN 1.3 with `<inputData>` per predicate, `<decision>` per module, `<decisionTable>` with hit policies, provenance in `<description>`. **XLSForm** (`.xlsx`): 3 sheets (survey, choices, settings) with workflow engine calculates, module queue, Why-trace JSON payload, fail-safe defaults. **Mermaid** (`.md`): `graph TD` flowchart with distinct node shapes per table kind, emergency paths in red. **CSV**: predicates table (6 columns) + phrases table (5 columns). |

**Validators:** Deterministic. pyxform validation (no Critical errors). DMN schema validation (OMG 1.3). DMN-to-XForm equivalence (same dx/referral/follow-up outputs). Why-trace JSON schema compliance. 0 raw numerics in compiled antecedents.

### Artifact Flow Diagram

```
RAW_PDF
  │
  ├─► [1] layout_elements ─► [2] vision_enrichment
  │                                    │
  │                        PROVENANCE_JSON (combined 1+2)
  │                                    │
  ├─► [3] content_triage ─────────────┐
  │                                    │
  ├─► [4] workflow_pattern ───────────┤
  │                                    │
  ├─► [5] raw_facts ──────────────────┤
  │                                    │
  ├─► [6] context_mappings ───────────┤
  │                                    │
  ├─► [7] supply_inventory ───────────┤
  │                                    │
  ├─► [8] consolidation_rules ────────┤
  │                                    │
  └─► [9] factsheet (assembles 3-8) ──┘
                    │
                    ├─► [10] boolean_literals
                    │
                    └─► [11] rule_criteria
                              │
                    ┌─────────┘
                    │
                    ├─► [12] modules
                    │
                    ├─► [13] router
                    │
                    ├─► [14] integrative
                    │
                    ├─► [15] phrase_bank
                    │
                    └─► [16] supply_list + variables + predicates
                              │
                         DMN_JSON (combined 12-16)
                              │
                    └─► [17] deployment_formats (XML, XLSX, MD, CSV)
```

---

## Generic Red Team / Repair Framework (xlsx A0)

The A0_redteam_v25 and A0_repair_v25 prompts are **reusable wrappers** that apply to any stage.

**xlsx provenance:**
- Sheet `"PR_A0_redteam_v23"` -- Contains BOTH the v23 skeleton (rows R1-R14: YAML metadata, maker prompt, unit tests) AND the embedded **A0_RedTeam_v25** (rows R21-R115: full expanded prompt). The v25 prompt defines: ROLE (adversarial clinical QA auditor), CONTEXT (WHO/UNICEF community-health manuals, front-line CHWs with limited literacy), OBJECTIVES (audit safety, correctness, traceability, coverage, documentation), INPUTS (7 inputs, 3 required + 4 optional), HALT CONDITIONS, TASKS (Round 0 setup, Cyclical Audit Loop max 3 rounds: Round 1 Contradictions & Omissions, Round 2 Thresholds & Safety Nets, Round 3 Regression Check), Evidence Collection (verbatim <= 120 words), Severity Matrix (Critical/High/Medium/Low with actions), Coverage Audit (>= 95% cutpoints and modules), Deliverables (JSON + Markdown + optional QA Annex). Embedded quality standards sections 1, 2, 5.
- Sheet `"PR_A0_repair_v23"` -- Contains BOTH the v23 skeleton (rows R1-R14) AND the embedded **A0_Repair_v25** (rows R20-R89). The v25 prompt defines: ROLE (senior clinical-systems integrator), CONTEXT (Red-Team report is ground truth, surgical edits only), OBJECTIVES (fix by severity, repair log, patch instructions, unit tests, self-check), TASKS (5 tasks: Repair Log with confidence 0-100, Patch Instructions with find_text OR location never both, Unit Tests as runnable pytest, Process Improvement analysis, Post-Repair Self-Check). Also contains embedded **B6_DMNs_v25** (rows R154-R213) which is a copy of the DMN authoring prompt.
- Sheet `"Quality stds"` rows R82-R89: Red-team Process domain (JSON issue log, compare new vs old, severity tagging, trace linkage, edge-case sims, golden test per issue, mutation testing, auto-close stale issues).

**Red Team (A0_redteam):**
- Role: Adversarial clinical QA auditor
- Procedure: max 3 audit rounds; Cyclical Repair Audit -> New Flaws -> Coverage Audit
- Severity: Critical (may cause harm) > High (major workflow failure) > Medium (minor logic error) > Low (formatting)
- Evidence: verbatim citation <= 120 words + page/para reference
- Output: `redteam_results.json` + `redteam_summary.md`
- Halt conditions: missing standards or artifacts
- Coverage: >= 95% of cutpoints and modules

**Repair (A0_repair):**
- Role: Senior clinical-systems integrator
- Procedure: surgical patches only; find_text OR structured location (never both); auto-generate pytest per fix; post-repair self-check
- Confidence scores 0-100 per fix
- Output: `repair_log.csv`, `patches.json`, `unit_tests.py`, `post_fix_status.json`
- Safety: "If a patch introduces new ambiguity, mark regression and revert."

**In the REPL pipeline:** These map to the frozen catcher validators at each `emit_artifact` checkpoint. The REPL has 0 degrees of freedom in the red team step (deterministic Python catchers + LLM catchers at temperature=0 with N=3 majority vote).

**In the Sequential pipeline:** These run as separate LLM calls after each maker stage. The sequential pipeline has full LLM degrees of freedom in the red team step (the LLM can be "creative" about what it checks). This is both a strength (catches novel issues) and a weakness (introduces variance).

---

## The A/B Test

**What is being compared:** Architecture, not model quality. Both pipelines use the same model, same temperature, same input.

**Pipeline A (Sequential):** ~45 prompts (maker + red team + repair per stage). Each stage is a fresh context window. Decisions are frozen once they pass red team. Backward fixes require PROPAGATE_FIX_NEEDED (rerun upstream stages from scratch).

**Pipeline B (REPL):** 1 system prompt, 1 continuous session, 17 checkpoints. Full context visibility. Backward fixes are in-session variable edits + re-emit.

**Where the A/B test applies:** Stage 4 (DMN_JSON compilation) is the primary comparison per the arena spec. Stages 2-3 are candidates for the same comparison pattern.

**Success metric:** Clinical equivalence on synthetic patients. NOT text equivalence. `if age_mo > 12` and `if not age_mo <= 12` are the same.

**xlsx provenance for the A/B test framework:**
- Sheet `"orchestrator_pseudo"` -- 110 lines of pseudocode for the DAG runner. Defines the BOOTSTRAP (init git, artifact registry, load DAG, load checkpoint), EXECUTOR (parallel_map with pop_ready, SUCCESS/ESCALATE_HUMAN/PROPAGATE_FIX_NEEDED/ERROR handling), and FINALIZE (build manifest) loops. Also defines `run_stage()` inner function: resolve inputs, cache check, execute by kind (make/audit/finalize), validate outputs + compute hashes, evaluate gates, human gate check.
- Sheet `"YAML DAG Schema for orch"` -- Full pipeline_definition.yaml with stage contracts. Defines: policy.yaml (mode QUICK/SMOKE/SAFETY, LLM assignments, block/warn thresholds), stage definitions (id, deps, kind, inputs with name/path/type/hash, outputs, llm_prompt, human_gate, tool). Example stages: A1_make, B1_make, B8_redteam, C1_make, D1b_make, F3b_make, G1_make. Shows how stages reference each other via `from: stage.output_name`.
- Sheet `"WriteOnce_Code_v23"` -- 5 write-once files: orchestrator.py (DAG runner), dmn_to_xlsform_compiler.py, synthetic_gen.py, translation_qc_linter.py, pipeline_definition.yaml.
- Sheet `"Prompt_Directory_v23"` -- Index of all prompts with Phase, ID, Name, Core Output, Human Gate, Owner. Rows R2-R28 list all stages from A through G.
- Sheet `"README_v23"` -- Overview with TODO items (rows R15-R35) noting: "Many prompts have cut&paste a few versions", "Lots of the fact sheet to DMN prompt is in v25 sup or v25c sup", "Unit tests for as many quality standards as we have time for."

---

## Build Priority

| Priority | What | Stage | Status | Blocks |
|---|---|---|---|---|
| **P0** | Stage 2 system prompt (REPL: PROVENANCE_JSON to FACTSHEET) | 2 | Not built | Everything downstream |
| **P0** | Stage 2 validators (content_triage, raw_facts provenance grounding, factsheet completeness) | 2 | Not built | Quality gates |
| **P1** | Stage 3 system prompt (FACTSHEET to BOOLEAN_LITERALS + RULE_CRITERIA) | 3 | Not built | Stage 4 |
| **P1** | Stage 3 validators (0 raw numerics, atomization completeness, linkage) | 3 | Partially built | Quality gates |
| **P2** | Stage 4 system prompt update (consume FACTSHEET + RULE_CRITERIA instead of PROVENANCE_JSON) | 4 | Partially built | A/B test |
| **P2** | Sequential pipeline orchestrator (xlsx DAG runner for A/B comparison) | 4 | Not built | A/B test |
| **P3** | DMN executor (deterministic patient routing) | 6 | Not built | Measurement |
| **P3** | Synthetic patient generator (xlsx D1b spec) | 6 | Not built | Measurement |
| **P4** | Stage 5 compiler enrichment (Why-trace JSON, module queue, compiler patch log) | 5 | Partially built | CHT deployment |
| **P4** | Bidirectional converters (DMN-to-IR, XLSX-to-IR) from CHW-Chat-Interface | 5 | Not ported | Roundtrip validation |

---

## Infrastructure

### Render Tier: Pro Plus ($175/mo)

**Specs:** 8 GB RAM, 4 CPU, zero downtime, SSH, persistent disks, one-off jobs.

**Why Pro Plus over Pro ($85/mo):** The A/B gate runs 6 concurrent extractions (3 REPL + 3 Sequential). Peak combined memory is ~2.7 GB. Pro (4GB) fits but has no headroom for spikes. Pro Plus (8GB) runs at 34% utilization with comfortable margin.

**Why not Pro Max ($225/mo):** The bottleneck is API latency, not compute. During extraction, CPU usage is under 10% (the server waits on Anthropic/OpenAI 95% of the time). More CPU/RAM does not make LLMs respond faster. Pro Max only makes sense if processing multiple guides simultaneously in production.

| Resource | Peak Usage (6 concurrent) | Pro Plus Budget | Utilization |
|---|---|---|---|
| RAM | 2,700 MB (3 REPL x 500MB + 3 Sequential x 300MB + 300MB infra) | 8,192 MB | 33% |
| CPU | <1.5 cores (API-bound, minimal compute) | 4 cores | <38% |
| Outbound connections | ~21 concurrent HTTPS at peak | No hard limit | Fine |
| HTTP timeout | N/A (extractions run as background asyncio tasks, not HTTP requests) | 30 min on Pro plans | Not a constraint |

**Render HTTP timeout note:** The gate harness must NOT run via HTTP endpoint (would timeout at 30 min for a 20-min gate). Run as either:
1. Background `asyncio.Task` dispatched from an endpoint that returns immediately
2. CLI management command: `python -c "import asyncio; from backend.eval.gate_harness import run_gate; asyncio.run(run_gate(...))"`
3. Render Cron Job for scheduled overnight runs

### API Keys and Providers

| Provider | Key Source | Tier | RPM Limit | TPM Limit | Role |
|---|---|---|---|---|---|
| **Anthropic** | Server-side `.env` for gate; BYOK per-session for user extractions | Tier 2+ | 1,000 RPM (Sonnet) | 450K ITPM | REPL root, Sequential maker/repair |
| **OpenAI** | Server-side `.env` | Tier 3+ | 5,000 RPM | 4M+ TPM | Sub-calls, catchers, red team, vision |
| **Unstructured** | Server-side `.env` | Pay-as-you-go | Serverless auto-scale | -- | PDF parsing (Stage 1) |

**BYOK boundary (unchanged):** Only Anthropic keys are BYOK for user-facing extractions. OpenAI and Unstructured are server-side keys the platform owner pays for. For the gate run, all keys are server-side (no user is involved).

**One key per provider is sufficient for the gate.** Rate limit utilization at 6 concurrent extractions: <3% Anthropic, <1% OpenAI. Two keys per provider is insurance for future scaling but not required.

---

## Model Configuration (Hybrid Strategy)

### Model Assignment

The hybrid strategy uses Anthropic for reasoning-heavy calls and OpenAI for high-volume workhorse calls. This balances cost and speed: Sonnet/Opus for quality where it matters, gpt-5.4-mini for throughput where it matters.

| Role | Model | Why This Model | Calls/Run |
|---|---|---|---|
| **REPL root** (Stages 2, 3, 4) | **Claude Sonnet 4.6** | Best reasoning-per-dollar for long-context iterative work. 200K context window. Prompt caching on system prompt. | 125 iterations |
| **Complex sub-calls** (integrative assembly, consolidation reasoning, comorbidity logic) | **Claude Opus 4.6** | 1.67x Sonnet's price, but safety-critical: wrong referral priority = patient harm. Used for 3-7 calls per run only. | 3-7 calls |
| **Standard sub-calls** (per-module extraction, fact extraction, simple transforms) | **gpt-5.4-mini** | 25% cheaper than Haiku, 5x higher RPM ceiling, excellent at focused narrow tasks. | 25-30 calls |
| **Catcher validators** (LLM catchers, 3x majority vote) | **gpt-5.4-mini** | The token hog role. 105+ calls per run at 30K tokens each. OpenAI's higher RPM (5,000 vs 1,000) means catchers complete faster. Auto-cache handles repeated guide context without explicit `cache_control`. | 45-105 calls |
| **Sequential maker/repair** | **Claude Sonnet 4.6** | Maker calls need strong reasoning to produce correct artifacts. Repair calls need surgical precision. | 16 calls |
| **Sequential red team** (3x majority vote) | **gpt-5.4-mini** | Red team is structured classification (find flaws, assign severity). Does not need Sonnet-level reasoning. Higher RPM means 3x majority vote calls return faster. | 36 calls |
| **Vision: table refinement** | **gpt-5.4** | High reasoning effort for complex clinical tables with merged cells, color coding, multi-language content. | 25 calls |
| **Vision: image description** | **gpt-5.4-mini** | Fast/cheap model sufficient for image classification and description. | 20 calls |
| **Vision: flowchart parsing** | **gpt-5.4** | High reasoning effort for node/edge extraction from complex clinical flowcharts. | 5 calls |

### Pricing Table (April 2026)

| Model | Input/MTok | Cached Read/MTok | Output/MTok | Cache Write Cost | Batch Discount |
|---|---|---|---|---|---|
| **Claude Sonnet 4.6** | $3.00 | $0.30 (90% off) | $15.00 | 5min: 1.25x ($3.75); 1h: 2x ($6.00) | 50% off |
| **Claude Opus 4.6** | $5.00 | $0.50 (90% off) | $25.00 | 5min: 1.25x ($6.25); 1h: 2x ($10.00) | 50% off |
| **Claude Haiku 4.5** | $1.00 | $0.10 (90% off) | $5.00 | 5min: 1.25x ($1.25); 1h: 2x ($2.00) | 50% off |
| **gpt-5.4** | $2.50 | $0.63 (75% off, free write) | $15.00 | $0 (auto-cache) | 50% off |
| **gpt-5.4-mini** | $0.75 | $0.19 (75% off, free write) | $4.50 | $0 (auto-cache) | 50% off |
| **Unstructured hi_res** | $0.01/page | -- | -- | -- | -- |

**Key pricing difference:** Anthropic charges 1.25-2x input price to WRITE a cache entry. OpenAI caches automatically with zero write surcharge. For high-call-count workloads (catchers: 105+ calls sharing the same guide context), OpenAI's auto-cache is structurally simpler and avoids the write cost.

### When to Use Opus (Surgical Deployment)

Opus costs 1.67x Sonnet. Use it only for calls where correctness directly affects patient safety:

| Call | Why Opus | Frequency | Extra Cost vs Sonnet |
|---|---|---|---|
| Integrative assembly (Stage 4c) | Cross-module reasoning over 7 consolidation categories. Wrong referral priority = patient harm. | 1-2 calls/run | ~$0.30 |
| Consolidation rule extraction (Stage 2f) | Implicit policies scattered across manual sections. Requires synthesis across distant context. | 1-2 calls/run | ~$0.25 |
| Complex comorbidity sub-calls | Drug interactions, contraindications, dose conflicts across simultaneously-active modules. | 1-3 calls/run | ~$0.40 |
| **Total Opus overhead** | | **3-7 calls/run** | **~$0.60-$1.00/run** |

---

## Prompt Caching Strategy

### What to Cache and Where

| Content | Size | Where Cached | TTL | Provider | Savings Per Gate |
|---|---|---|---|---|---|
| **REPL system prompt** | 17K tokens | System message block, `cache_control: {"type": "ephemeral", "ttl": "1h"}` | 1 hour | Anthropic | $6.72 (already built in `rlm_runner.py`) |
| **Sequential guide context** | 90K tokens | System message block (move guide from user message to system block) | 1 hour | Anthropic | **$31.26** (NOT yet built, highest-value optimization) |
| **Catcher guide context** | 30K tokens per call | System message block on catcher calls (move guide from user message) | Auto (5-10 min) | OpenAI | **$12-15** (NOT yet built, second-highest-value) |
| **OpenAI vision prompts** | 400 tokens each | Auto-cached by OpenAI across 50 calls | Auto (5-10 min) | OpenAI | $0.45 |
| **Catcher system prompts** | 400 tokens each | System message block, 1h TTL | 1 hour | Anthropic (if using Haiku) or OpenAI (if using gpt-5.4-mini) | $0.10-0.20 |

### Caching Implementation Priority

1. **Sequential guide context as system block** (saves $31/gate): Each sequential stage call includes the guide JSON as context. Currently this would go in the user message (uncacheable). Moving it to the system message block enables 90% cache discount on Anthropic. First call: 90K tokens at $6/MTok write = $0.54. Remaining 44 calls: 90K at $0.30/MTok read = $1.19 total. Without caching: 45 calls x 90K x $3/MTok = $12.15. **Net: $0.54 + $1.19 = $1.73 vs $12.15 uncached.**

2. **Catcher guide in system block** (saves $12-15/gate): Each LLM catcher call sends the guide JSON for cross-referencing (provenance verification, completeness checking). With the hybrid model strategy (catchers on gpt-5.4-mini), OpenAI auto-caches the system prefix with 75% discount and zero write cost. 105+ calls sharing the same guide prefix.

3. **REPL system prompt** (saves $7/gate): Already implemented. The 17K-token system prompt is cached with 1h TTL across all 50 iterations per run. The 1h TTL survives the entire gate (6 runs in ~20 min).

### Total Caching Impact

| Scenario | Cost Per Gate |
|---|---|
| No caching | ~$100-110 |
| REPL system prompt only (current state) | ~$95 |
| + Sequential guide context | ~$64 |
| + Catcher guide context | ~$50 |
| + Hybrid model strategy (gpt-5.4-mini for workhorse) | **~$36** |

---

## Cost Model

### Measured Baseline Data

| Artifact | Characters | Tokens (approx) |
|---|---|---|
| WHO 2012 PDF raw text (141 pages) | 237,917 | 59,500 |
| PROVENANCE_JSON (Unstructured + vision, estimated) | ~360,000 | 90,000 |
| FACTSHEET (guide_full_example.json) | 85,600 | 21,400 |
| DMN_JSON for WHO 2012 (estimated 10-15 modules) | ~45,000 | 11,000 |
| Assembled system prompt (14 sections) | 28,000 | 7,000 |

### Cost Per Single Run (Hybrid Model Strategy, With Caching)

**REPL Pipeline (1 run):**

| Component | Model | Calls | Input Tok | Output Tok | Cost |
|---|---|---|---|---|---|
| Root iterations (50) | Sonnet 4.6 | 50 | 441K | 75K | $2.45 |
| Complex sub-calls | Opus 4.6 | 5 | 40K | 15K | $0.58 |
| Standard sub-calls | gpt-5.4-mini | 25 | 140K | 60K | $0.38 |
| Catchers LLM (3x majority) | gpt-5.4-mini | 45-105 | 1,350K-3,150K | 90K-210K | $1.42-$3.31 |
| Catchers deterministic | Python | 12 | 0 | 0 | $0 |
| **REPL per run** | | **137-197** | **1.97M-3.77M** | **240K-360K** | **$4.83-$6.72** |

**Sequential Pipeline (1 run):**

| Component | Model | Calls | Input Tok | Output Tok | Cost |
|---|---|---|---|---|---|
| Maker + repair | Sonnet 4.6 | 16 | 320K | 64K | $1.92 |
| Red team (3x majority) | gpt-5.4-mini | 36 | 720K | 54K | $0.78 |
| Catchers LLM (3x majority) | gpt-5.4-mini | 45-105 | 1,350K-3,150K | 90K-210K | $1.42-$3.31 |
| **Sequential per run** | | **97-157** | **2.39M-4.19M** | **208K-328K** | **$4.12-$6.01** |

**Shared (run once per gate):**

| Component | Model | Calls | Cost |
|---|---|---|---|
| Unstructured hi_res | -- | 141 pages | $1.41 |
| Vision: tables | gpt-5.4 | 25 | $1.43 |
| Vision: images | gpt-5.4-mini | 20 | $0.06 |
| Vision: flowcharts | gpt-5.4 | 5 | $0.87 |
| **Stage 1 total** | | **51** | **$3.77** |

### Full A/B Gate (6 Runs: 3 REPL + 3 Sequential)

| Component | Cost |
|---|---|
| Stage 1 (shared, run once) | $3.77 |
| 3 REPL runs (avg $5.78 each) | $17.34 |
| 3 Sequential runs (avg $5.07 each) | $15.21 |
| **Total API cost** | **~$36** |

### Cost Comparison: Strategies

| Strategy | Cost Per Gate | Notes |
|---|---|---|
| All-Anthropic, no caching | ~$100-110 | Baseline worst case |
| All-Anthropic, with caching | ~$70 | REPL system prompt + guide caching |
| Hybrid (Anthropic reasoning + OpenAI workhorse), no caching | ~$55 | gpt-5.4-mini saves 25-50% on workhorse calls |
| **Hybrid + full caching** | **~$36** | **Recommended configuration** |
| Hybrid + full caching + Anthropic Batch API for catchers | ~$24 | 50% batch discount on catchers, but 24h turnaround (not suitable for interactive gate runs) |

---

## Parallelism and Execution

### REPL Pipeline: Internal Parallelism

The REPL is **iteration-sequential** (iteration N+1 depends on N). You cannot parallelize across iterations. But within each iteration and at checkpoints, there are parallel windows:

| Window | When | Concurrent Calls | Model | Duration |
|---|---|---|---|---|
| Phase 2c batched extraction | ~iter 10-15 | 6-8 calls | gpt-5.4-mini | 15-25s |
| Phase 3B module extraction | ~iter 25-30 | 8-12 calls | gpt-5.4-mini | 20-30s |
| Phase 4 integrative sub-calls | ~iter 35-40 | 3-5 calls | Opus / gpt-5.4-mini | 15-20s |
| Each emit_artifact checkpoint | 7 times/run | 3-9 calls (LLM catchers x3 majority) | gpt-5.4-mini | 10-20s |

**Peak concurrent API calls in a REPL run:** ~12 (Phase 3B batched extraction).

**REPL wall clock per run: ~17 minutes** (50 iterations x ~20s avg, including catcher validation time).

### Sequential Pipeline: Internal Parallelism

The sequential pipeline is **stage-serial** (B2 depends on B1). But the YAML DAG reveals:

- **B2 and B3 can run in parallel** (both depend only on B1, not each other)
- Within each stage, 3x majority vote calls are parallel
- Maker and red team are serial within a stage (red team needs maker output)

**Sequential wall clock per run: ~10 minutes** (12 stages x ~50s avg, with B2/B3 parallelized).

Sequential is faster per run because each call has a fresh, small context window (~15-20K tokens) vs REPL's growing context (100-200K tokens). Smaller context = faster time-to-first-token.

### Combined: The 20-Minute Gate

Run all 6 extractions at t=0 as concurrent asyncio tasks. The longest run (REPL, ~17 min) sets the wall clock.

```
t=0                              t=10        t=17        t=22
|────────────────────────────────|───────────|───────────|
[REPL run 1 ───────────────────────────────── 17 min ──]
[REPL run 2 ───────────────────────────────── 17 min ──]
[REPL run 3 ───────────────────────────────── 17 min ──]
[Sequential run 1 ──────────── 10 min ──]
[Sequential run 2 ──────────── 10 min ──]
[Sequential run 3 ──────────── 10 min ──]
                                                        [Gate 5m]
                                                              ↓
                                                        ~22 min total
```

**Pre-requisite:** Stage 1 (ingestion) runs once before the gate (~15 min). This is amortized; the PROVENANCE_JSON is cached by content hash. Re-running the gate on the same PDF skips Stage 1 entirely.

### Combined Rate Limits (Hybrid, 6 Concurrent Extractions)

| Provider | Model | Role | Peak RPM (6 concurrent) | Provider Limit | Utilization |
|---|---|---|---|---|---|
| Anthropic | Sonnet | REPL root + Sequential maker | ~8 RPM | 1,000 | <1% |
| Anthropic | Opus | Complex sub-calls | <2 RPM | 1,000 | <0.2% |
| OpenAI | gpt-5.4-mini | Sub-calls + catchers + red team | ~27 RPM burst | 5,000 | <0.5% |
| OpenAI | gpt-5.4 | Vision | 0 during gate (Stage 1 pre-computed) | 5,000 | 0% |

**Combined: <3% Anthropic, <1% OpenAI.** Rate limits are a non-issue. You could run 30+ concurrent extractions before approaching any limit.

### Measurement Isolation on Pro Plus

The original gate harness ran sequentially to avoid resource contention confounding measurements. On Pro Plus with the hybrid strategy:

- **CPU-independent:** >95% time waiting on API. No CPU contention.
- **Memory-independent:** Each session has its own heap (500MB REPL, 300MB Sequential). 2.7GB total out of 8GB. No GC pressure interference.
- **API-independent:** <3% rate limit utilization. No throttling interference.
- **Network-independent:** 21 concurrent HTTPS connections. Negligible bandwidth.

**Parallelism does NOT confound the measurement on Pro Plus.** The runs are effectively isolated.

### Wall Clock Comparison

| Configuration | Infra | Wall Clock | API Cost | Monthly Infra |
|---|---|---|---|---|
| Serial (current gate harness) | Pro ($85/mo) | ~108 min | $36 | $85 |
| Paired parallel (1 REPL + 1 Seq at a time) | Pro ($85/mo) | ~53 min | $36 | $85 |
| **Full parallel (all 6 at once)** | **Pro Plus ($175/mo)** | **~22 min** | **$36** | **$175** |
| Full parallel + prompt engineering (reduce iterations) | Pro Plus ($175/mo) | ~17 min | $30 | $175 |

**Recommended: Full parallel on Pro Plus.** The $90/mo uplift buys 86 minutes back per gate run. If running 2-3 gates per week during prompt iteration, that's 3-4 hours saved per week.

### Reducing Wall Clock Below 20 Minutes

The floor is set by the slowest REPL run (~17 min = 50 iterations x ~20s). To push lower:

| Optimization | Time Saved | Implementation |
|---|---|---|
| Combine Stages 2+3 into one REPL session | ~3 min (avoid session restart, context reload) | System prompt change |
| Pre-compute content triage deterministically | ~2 min (skip LLM triage iterations) | Heuristic classifier on Unstructured element types |
| More aggressive `llm_query_batched` | ~1 min (fewer iterations for module extraction) | System prompt instruction to batch more aggressively |
| Prompt caching latency benefit | ~1 min (15-30% faster TTFT on cached 17K prefix, across 50 iters) | Already implemented |
| **Combined** | **~7 min** | **REPL run drops to ~12-14 min; gate to ~17-19 min** |

---

## Scaling Guide

### When to Upgrade

| Scenario | Infrastructure | Wall Clock | Monthly Cost |
|---|---|---|---|
| **Development: iterate prompts, run gates** | Pro Plus, 1 Anthropic key + 1 OpenAI key | 20 min/gate | $175 + ~$36/gate |
| **Multiple guides simultaneously** | Pro Max ($225, 16GB, 4CPU) | 20 min/guide, 2 guides parallel | $225 + ~$72/2 guides |
| **CI/CD: gate on every system prompt PR** | Pro Plus + Render Cron Job | 20 min/PR, queued | $175 + ~$36/PR |
| **Production: user-facing extractions** | Pro Plus, MAX_CONCURRENT_SESSIONS=4 | 15 min/user extraction | $175 + ~$6/extraction |
| **Scale: 10+ concurrent users** | Pro Max or 2x Pro Plus behind load balancer | 15 min/user, 10 concurrent | $450+ |

### Cost Per Production Extraction (Single Guide, No Test-Retest)

| Component | Cost |
|---|---|
| Stage 1 (ingestion, cached after first run per PDF) | $0 (cache hit) or $3.77 (first time) |
| Stage 2+3 (FACTSHEET + RULE_CRITERIA) | ~$3-5 |
| Stage 4 (DMN_JSON) | ~$3-5 |
| Stage 5 (deterministic converters) | $0 |
| **Total per extraction** | **~$6-10** |

Processing 5 guides per month = ~$30-50 in API costs. Monthly total: ~$205-225 (Render + API).

### Batch API for Overnight Gate Runs

Anthropic's Batch API gives 50% off with 24-hour turnaround. OpenAI's Batch API gives 50% off with 24-hour turnaround. Useful for:

| Use Case | Worth Batching? | Savings |
|---|---|---|
| Interactive gate run (watching results live) | No (need real-time feedback) | N/A |
| Overnight scheduled gate | Yes (submit before sleep, results in morning) | ~$18 saved (50% off the $36 gate) |
| Catcher validators during REPL | No (REPL blocks waiting for catcher results) | N/A |
| Bulk re-processing multiple guides | Yes (submit all, collect next day) | 50% off all API costs |

---

## Executive Summary: What to Tell Stakeholders

### Why 3 Runs Per Architecture (Test-Retest Reliability)

A single A/B comparison (1 REPL run vs 1 Sequential run) shows that two outputs differ, but cannot distinguish a systematic architectural difference from a one-time fluke. LLMs are stochastic; the same input can produce cosmetically different outputs on successive runs (the CHW-Chat-Interface postmortem measured 0% string identity at temperature=0 for full-guide extraction, but 100% clinical equivalence across 28 synthetic patients).

**Three runs is the minimum for test-retest reliability.** Run the same input through the same pipeline 3 times, then route synthetic patients through all 3 outputs. If all 3 produce identical clinical outcomes (diagnosis, referral level, treatment, follow-up), the architecture is reliable. If not, the variance is the failure mode being measured.

This is the standard from the psychology measurement literature (Professor Levine's framing): "re-run pipeline is test-retest reliability."

| Depth | Runs | Cost | Wall Clock | What It Measures |
|---|---|---|---|---|
| Single A/B (1v1) | 2 | ~$12 | 17 min | "These two outputs differ here" -- anecdote, not evidence |
| **Test-retest A/B (3v3)** | **6** | **~$36** | **22 min** | **"Pipeline A agrees with itself X% of the time; Pipeline B agrees Y%"** -- the reliability measurement |
| Deep reliability (5v5) | 10 | ~$58 | 22 min | Tighter confidence intervals; worth doing once the winner is clear |

The 3v3 costs 3x per pipeline but the wall clock only increases from 17 to 22 minutes because all 6 runs execute in parallel. The extra 5 minutes is the gate harness computing Jaccard similarity and patient routing comparisons.

### Numbers for Professor Levine

**A/B gate (the research deliverable):**
- **$36** per full head-to-head comparison (3 runs of each architecture on one guide)
- **22 minutes** wall clock (all 6 runs in parallel)
- Produces: test-retest reliability score per pipeline, cross-pipeline clinical equivalence rate, per-patient outcome comparison, cost and timing metrics
- Re-running on the same PDF costs ~$32 (Stage 1 ingestion is cached)

**Prompt iteration (the development loop):**
- **$6-10** per single extraction (one pipeline, one run)
- **15-17 minutes** wall clock
- 2-3 test extractions can run in parallel to compare prompt variants

**Production (after the A/B winner is chosen):**
- **$6-10** per guide extraction (single run, no test-retest)
- **15 minutes** wall clock per extraction
- **4 concurrent users** on Pro Plus infrastructure
- First-time PDF ingestion adds ~$4 and ~15 min (cached by content hash after that)

**Monthly operating cost:**
- $175/month infrastructure (Render Pro Plus, fixed)
- ~$6-10 per extraction (variable, API usage)
- 5 guides/month = ~$225/month all-in
- 20 guides/month = ~$350/month all-in

### One-Liner

> A full head-to-head reliability comparison costs **$36 and takes 22 minutes**. A single production extraction costs **$6-10 and takes 15 minutes**. The platform supports **4 concurrent users** at **$175/month** infrastructure.

---

## Production Patterns (from Generative-Flywheel)

Proven patterns from the Generative-Flywheel production system (`C:\Users\athar\Documents\GitHub\Generative-Flywheel`). These are battle-tested on a pipeline that makes 50-200 LLM calls per run with real users on Render.

### Pattern 1: Sequential Stages, Parallel Within (Critical)

**Source:** `Generative-Flywheel/backend/api/taskgen_v2.py`

Run pipeline stages in order (Stage 2 before Stage 3 before Stage 4), but within each stage fire all independent LLM calls in parallel via `asyncio.gather`:

```python
# All module extraction sub-calls fire simultaneously
task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)
```

After `gather` completes, iterate results and handle per-task exceptions. Failed tasks go into a `retry_candidates` list. Call `gc.collect()` between stages to reclaim memory.

**Apply to CHW pipeline:** Phase 3B module extraction (8-12 concurrent sub-calls), catcher majority votes (3 concurrent per LLM catcher), and Sequential B2+B3 parallel stages all use this pattern.

### Pattern 2: ModelRouter with Tier-Based Routing (Critical)

**Source:** `Generative-Flywheel/backend/src/core/openai_client.py`

A `ModelRouter` class with `call_cheap()`, `call_reasoning()`, and `call_auto(task_type)` methods. Task types are mapped to model tiers in a static config:

```python
MODEL_CONFIGS = {
    "cheap": ModelConfig(model="gpt-5.4-mini", use_cases=["classification", "filtering", ...]),
    "reasoning": ModelConfig(model="claude-sonnet-4", use_cases=["synthesis", "extraction", ...]),
}
```

Every call returns a `CallResult` dataclass with `content`, `model`, `tokens`, `cost_usd`, `latency_ms`. Downstream code is provider-agnostic.

**Apply to CHW pipeline:** Map task types to models: `call_reasoning()` for REPL root and maker calls (Sonnet), `call_opus()` for safety-critical integrative calls, `call_cheap()` for catchers, sub-calls, and red team (gpt-5.4-mini).

### Pattern 3: Per-Call Cost and Token Tracking (Critical)

**Source:** `Generative-Flywheel/backend/src/core/openai_client.py`

Every API call returns exact cost calculated from model-specific pricing:

```python
@dataclass
class CallResult:
    content: str
    model: str
    model_tier: str
    tokens: TokenUsage  # prompt_tokens, completion_tokens, reasoning_tokens
    cost_usd: float
    latency_ms: int
```

Aggregate `UsageStats` tracked by model and by task_type. At 200+ calls per run, this is essential for budget monitoring and optimization.

### Pattern 4: Semaphore-Based Concurrency Limiting (Critical)

**Source:** `Generative-Flywheel/backend/src/mcp/lotus_filter.py`

`asyncio.Semaphore` caps concurrent outbound API calls. Configurable via env var:

```python
semaphore = asyncio.Semaphore(config.semaphore_limit)  # Default: 30

async def score_with_semaphore(batch):
    async with semaphore:
        return await self._score_batch_async(query, batch)

results = await asyncio.gather(*[score_with_semaphore(b) for b in batches])
```

**Apply to CHW pipeline:** Use separate semaphores per provider. Anthropic semaphore = 10 (conservative for Tier 2). OpenAI semaphore = 30 (gpt-5.4-mini has 5,000 RPM). This prevents rate limit errors without manual backoff.

### Pattern 5: Cancelable HTTP Clients (Critical)

**Source:** `Generative-Flywheel/backend/src/core/openai_client.py`

Each cancelable LLM call gets a dedicated `httpx.AsyncClient`. A `cancel_fn` closure can close the HTTP connection to immediately abort an in-flight request:

```python
def get_cancelable_async_client(self):
    http_client = httpx.AsyncClient(timeout=timeout)
    client = AsyncOpenAI(api_key=self.api_key, http_client=http_client)
    def cancel():
        loop.create_task(http_client.aclose())
    return client, cancel
```

**Apply to CHW pipeline:** When a user cancels a running extraction, you need to kill 10-15 in-flight API calls immediately. Without per-call HTTP clients, you'd have to wait for all of them to complete (potentially 30+ seconds).

### Pattern 6: Multi-Layer Cancellation (High)

**Source:** `Generative-Flywheel/backend/api/cancel_utils.py`, `taskgen_v2.py`

Five layers of cancellation:
1. **In-memory flag** (fastest): `v2_jobs[job_id]["cancelled"] = True`
2. **Redis flag** (survives restart): `redis.cancel_pipeline(job_id)`
3. **CancelChecker polling** (250ms): Thread-safe poller checks both 1 and 2
4. **Piggyback cancel** on status poll: `GET /status/{job_id}?cancel=true`
5. **asyncio.Task.cancel()**: Direct task cancellation

The `check_job_cancelled()` function checks all three sources (in-memory, Redis job meta, Redis pipeline state) and returns true if any is set.

**Apply to CHW pipeline:** Check cancellation before each `emit_artifact` call and before each sub-call batch. The piggyback pattern (cancel=true on status poll) is more reliable than a separate cancel endpoint.

### Pattern 7: Background Tasks with asyncio.create_task (High)

**Source:** `Generative-Flywheel/backend/api/taskgen_v2.py`

Use `asyncio.create_task()` instead of FastAPI `BackgroundTasks` for async pipelines:

```python
v2_job_tasks: Dict[str, asyncio.Task] = {}

task = asyncio.create_task(_run_with_exception_logging(job_id))
v2_job_tasks[job_id] = task  # Store reference for cancellation
```

`BackgroundTasks` does not properly `await` async functions. Store task references in a dict for cancellation.

**Apply to CHW pipeline:** The existing `_background_tasks: set[asyncio.Task]` pattern in `backend/ingestion/router.py` already does this correctly. Extend to extraction sessions.

### Pattern 8: SSE with Named Events and Heartbeats (High)

**Source:** `Generative-Flywheel/backend/api/taskgen_v2.py`, `sse_streaming.py`

Named SSE events (`event: task`, `event: run`, `event: judge`) with incremental delivery tracking:

```python
async def event_generator():
    while True:
        yield format_sse(step_data, event="artifact_emitted")
        yield format_sse({"status": status}, event="heartbeat")
        await asyncio.sleep(0.5)
```

Critical headers: `X-Accel-Buffering: no` (disables nginx/Render proxy buffering), `Cache-Control: no-cache`, `Connection: keep-alive`.

**Apply to CHW pipeline:** Emit named events for each `emit_artifact` checkpoint (`event: artifact`), each catcher result (`event: catcher`), and each REPL iteration (`event: iteration`). The existing SSE infrastructure in `backend/session_manager.py` already uses this pattern.

### Pattern 9: Stale Job Auto-Cleanup (High)

**Source:** `Generative-Flywheel/backend/api/processes.py`

Auto-detect and delete ghost jobs on status poll:

```python
STALE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes for pipeline jobs

def is_job_stale(job_meta) -> bool:
    age_seconds = (now - last_update).total_seconds()
    return age_seconds > STALE_THRESHOLD_SECONDS
```

Aggressively delete stale jobs (not just mark cancelled). This prevents ghost jobs after server restarts.

**Apply to CHW pipeline:** An extraction that hasn't emitted an artifact or REPL iteration in 30 minutes is dead. Clean it up so the concurrent session slot is freed.

### Pattern 10: Redis Key Schema with TTLs (High)

**Source:** `Generative-Flywheel/backend/src/storage/redis_schema.py`

Namespaced keys with documented TTLs:

```python
TTL = {
    "job": 604800,            # 7 days
    "pipeline_state": 7200,   # 2 hours (active execution)
    "pipeline_result": 86400, # 24 hours
}
```

User-scoped keys: `user:{username}:{resource}`. Pipeline-scoped: `pipeline:{job_id}:state`.

**Apply to CHW pipeline:** Adapt as `chw:run:{run_id}:artifact:{name}` with 24h TTL for gate results, 2h TTL for active extraction state. Already compatible with the existing Upstash REST wrapper in `backend/redis_client.py`.

### Pattern 11: Safe Async Cancellation on Windows (High)

**Source:** `Generative-Flywheel/backend/src/mcp/lotus_filter.py`

Use `asyncio.shield(task)` + 250ms timeout polling instead of `task.cancel()`:

```python
while True:
    if cancel_check and cancel_check():
        cancel_fn()  # Close HTTP client to abort
        return None   # Don't cancel task (corrupts event loop on Windows)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
    except asyncio.TimeoutError:
        continue
```

`task.cancel()` on Windows can corrupt the asyncio event loop. The `shield + timeout + HTTP close` pattern is the safe alternative.

**Apply to CHW pipeline:** Essential for local development on Windows. The existing `rlm_runner.py` uses `asyncio.to_thread()` which has similar risks.

### Pattern 12: Singleton Stream Manager (Frontend) (Medium)

**Source:** `Generative-Flywheel/frontend/lib/pipeline-stream-manager.ts`

A singleton that lives outside React component lifecycle. State survives page navigation:

```typescript
class PipelineStreamManager {
    private static instance: PipelineStreamManager
    private isPageVisible: boolean = true
    // Polling pauses when tab is backgrounded, resumes on focus
}
```

Triple-redundancy: SSE stream (primary) + polling fallback (500ms) + exponential backoff on errors. Heartbeat timeout detection: force reconnect if no heartbeat for 15 seconds.

### Pattern 13: GZip + Keep-Alive (Medium)

**Source:** `Generative-Flywheel/backend/app/main.py`

```python
app.add_middleware(GZipMiddleware, minimum_size=500)
```

GZip reduces JSON payloads 60-80%. Important when streaming 17 artifact JSONs (some up to 85KB). Keep-alive on frontend reuses TCP connections for the high-frequency polling pattern.

**Exception:** Do NOT GZip SSE streams. SSE must stream unbuffered. The `minimum_size=500` threshold naturally excludes SSE heartbeats.

### Pattern 14: Docker Memory Optimization (Medium)

**Source:** `Generative-Flywheel/backend/Dockerfile`

```dockerfile
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
CMD ["uvicorn", "app.main:app", "--workers", "1", "--timeout-keep-alive", "120"]
HEALTHCHECK --interval=60s --timeout=30s --start-period=180s --retries=3
```

Single worker (reduces memory), thread limits (prevents numpy/torch parallelism from spawning threads), 120s keep-alive (for long SSE streams), generous health check start period (180s for cold starts with model downloads).

### Pattern 15: Windows asyncio Policy (Medium)

**Source:** `Generative-Flywheel/backend/app/main.py`

```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

Already implemented in the CHW pipeline. Confirmed necessary for stable local development.

### Patterns NOT to Reuse (Gaps in Flywheel)

| Gap | Flywheel Status | CHW Pipeline Needs |
|---|---|---|
| Prompt caching (LLM-level) | Not implemented | Critical (saves $50+/gate). Already built in `rlm_runner.py`. |
| Exponential backoff on retries | Not implemented (single retry only) | Should add for transient API errors across 200+ calls. Use `tenacity` library. |
| Reactive rate limit handling (429 detection + backoff) | Passive only (concurrency limits) | Should add 429 detection with exponential backoff. The semaphore pattern prevents most 429s but doesn't handle bursts. |
| Distributed locking | Not implemented | May need for concurrent extractions writing to same guide_id in Neon. Prisma handles this via row-level locking. |
| Batch writes to Redis | Individual setex per artifact | Consider MULTI/EXEC or single JSON blob for 17 artifacts per extraction. |
