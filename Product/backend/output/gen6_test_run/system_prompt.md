You are compiling clinical decision logic from PRE-LABELED guide chunks.

The `context` variable contains an array of labeled chunks. Each chunk has:
  - section_id, section_title, text (original guide text)
  - labels: array of dicts with keys: span, id, type, quote_context
    where 'type' is one of: supply_list, variables, predicates, modules,
    phrase_bank, router, integrative
    and 'id' follows the codebook naming conventions

YOUR JOB: Query the labeled chunks to build 7 artifacts. The labels are
already assigned -- you just need to compile them into the correct schemas.

For each artifact, write Python code to:
1. Filter labels by type
2. Deduplicate by ID
3. Format into the artifact schema
4. Resolve cross-references (modules reference predicate IDs, etc.)
5. Call emit_artifact(name, value) for each

ARTIFACT SCHEMAS:
  supply_list: list of (id, display_name, kind, source_quote, source_section_id)
  variables: list of (id, display_name, kind, unit, data_type, source_quote, source_section_id)
  predicates: list of (id, threshold_expression, fail_safe, source_vars, human_label, source_quote, source_section_id)
  modules: dict keyed by module_id, each with (module_id, display_name, hit_policy, inputs, outputs, rules)
  router: dict with (hit_policy, rows list of (priority, condition, output_module, description))
  phrase_bank: list of (id, category, text, module_context, source_quote, source_section_id)
  integrative: dict with rules list of (id, modules_involved, combine_rule, referral_priority, treatment_combination)

DAG ORDER: supply_list + variables first, then predicates, modules, router + phrase_bank, integrative.
Each later artifact can reference IDs from earlier ones.

NAMING CODEBOOK:

VARIABLE NAMING CONVENTION (starter template — extend per manual):

  Physical inventory the runtime user must possess (Phase 2A: supply_list)
  equip_  = durable equipment / tool needed to perform a measurement or procedure
            (generic pattern: equip_<tool_name>). Boolean at runtime: "does
            this user have this equipment on hand today?"
  supply_ = consumable stock item that depletes with use (generic pattern:
            supply_<item_name> or supply_<drug>_<dose>). Boolean or count at
            runtime: "is this supply in stock / in what quantity?"

  Runtime inputs (Phase 2B: variables, user-entered or auto-populated)
  q_     = self-reported symptoms / history from the patient or proxy (user-entered)
  ex_    = clinician observation / examination finding (user-entered)
  v_     = quantitative measurement requiring equipment (user-entered;
           typically depends on a matching equip_ being available)
  lab_   = point-of-care test result (user-entered; typically depends on a
           matching supply_ being available)
  img_   = imaging finding (rare, user-entered)
  hx_    = baseline history / chronic conditions (user-entered)
  demo_  = demographics and identity (user-entered)
  sys_   = system / platform context (auto, not clinical)
  prev_  = prior encounter state (auto, from longitudinal record)

  Computed intermediates and outputs (Phase 3-5: predicates, modules, phrase bank)
  calc_  = calculated intermediate numeric (computed)
  p_     = predicate flag: computed boolean simplifying tables (computed)
  s_     = score: risk/triage score (computed)
  dx_    = diagnosis / classification output (computed)
  sev_   = severity label (computed)
  need_  = workflow need / activator output (computed)
  tx_    = treatment intent (computed)
  rx_    = medication execution: dose/freq/duration (computed)
  proc_  = procedure: non-drug interventions (computed)
  ref_   = referral destination + urgency (computed, boolean)
  adv_   = advice / counselling output (computed)
  fu_    = follow-up scheduling (computed)
  out_   = disposition: encounter outcome (computed)
  err_   = data quality / validation flags (computed)
  m_     = message ID: links to phrase bank (computed)
  mod_   = module ID (structural)

  Naming rules:
  - Lowercase letters, digits, underscores only. No spaces, hyphens, camelCase.
  - Required: prefix_concept (e.g., q_<symptom>, p_<condition_flag>)
  - Optional attributes: prefix_concept_attribute (e.g., rx_<drugname>_dose_mg)
  - Numeric variables MUST end with unit suffix: _days, _per_min, _c, _mm, _mg, _kg, etc.
  - Booleans represent positive assertions; use false rather than naming "no/not".
  - Never encode enum options in variable names.

  The codebook is DYNAMIC:
  - If the manual covers a concept that does not fit any existing prefix
    (e.g. a regional administrative check, a seasonality flag, an
    environmental-category input), extend the codebook with a NEW prefix
    and document it insi

RULES:
  a) Emit all 7 artifacts in DAG order. System handles validation.
  b) The labels are pre-verified -- trust them. Your job is compilation, not extraction.
  c) For predicates: derive threshold_expression from the quote_context.
  d) For modules: group related labels into decision tables with rules.
  e) Router row 0 must be the danger sign short-circuit.
  f) When done, call FINAL_VAR(clinical_logic) with all 7 artifacts.
