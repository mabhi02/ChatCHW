# ORCHESTRATOR.md -- CHW Navigator Extraction Pipeline

> **Status:** Architecture spec v1
> **Last updated:** 2026-03-31
> **Audience:** Claude Code build instance + ChatCHW development team
>
> This document specifies both pieces of the demo Professor Levine wants to show the CHT community:
> 1. A minimal Python orchestrator that moves from Manual text to DMN (boolean-only) + predicates table + phrase bank
> 2. Deterministic converters that render DMN XML, Mermaid flowcharts, and XLSForms from the same JSON
>
> The converters (piece #2) are baked into this pipeline, not a separate system. They run deterministically on the validated JSON that piece #1 produces.

---

## 0. How It Works (8 Steps)

1. **PDF --> JSON.** The clinical manual PDF is converted to structured JSON with full provenance (section IDs, page numbers, source quotes, extracted tables). This step is external to this pipeline (assumed done).

2. **REPL session.** The JSON guide is loaded as a variable in a REPL (Read-Eval-Print Loop) programming environment. The reasoning model (Claude Opus) receives ONLY the system prompt and metadata about the guide. It never sees the full guide in its context window. Instead, it writes Python code to programmatically examine, decompose, and extract from the guide variable, launching recursive sub-calls for per-module extraction.

3. **JSON output.** The model builds up a structured JSON containing all clinical decision logic: Activator/Router tables, per-module decision tables (boolean-only), predicates with fail-safe defaults, and a full phrase bank. This JSON is stored in a REPL variable, not generated as free text.

4. **Validation + Z3.** Deterministic Python validators check the JSON for structural correctness (architecture, completeness, clinical safety, naming conventions). Z3 proofs verify exhaustiveness (no input gaps) and reachability (no dead rules). These run INSIDE the REPL -- the model calls `validate()` and `z3_check()` and sees results in stdout.

5. **Error feedback.** If validation fails, the model sees the specific errors in its REPL output and fixes them in the same session. No separate "repair call" -- the model iterates within one continuous REPL trajectory.

6. **Model fixes.** The model patches the JSON variable, re-runs validators, repeats until all checks pass.

7. **Final JSON.** The model calls `FINAL_VAR(clinical_logic)` to return the validated JSON. This is the source of truth.

8. **Convert.** Deterministic Python converts JSON --> DMN XML + Mermaid flowcharts + CHT XLSForm XLSX + flat CSVs. No AI. Same JSON source, multiple output formats.

---

## 1. Recursive Language Models: What and Why

### 1.1 The Problem with Long-Context Extraction

CHW clinical manuals range from 50 pages (WHO 2012) to 500+ pages (national MOH guidelines with appendices). Even frontier reasoning models exhibit **context rot**: quality degrades as prompts get longer and tasks get more complex. A model that perfectly extracts cough classification rules from page 25 may hallucinate or omit fever rules from page 140, even when both fit within its context window.

The naive approach -- stuff the entire manual into one prompt and ask for structured output -- works for small manuals but fails as manuals grow. Multi-prompt pipelines (extract section by section) lose cross-section consistency. Neither scales reliably.

### 1.2 The RLM Insight

Recursive Language Models (Zhang, Kraska, Khattab -- MIT CSAIL, January 2026) propose a simple but powerful idea: **don't feed the long document into the model's context window. Instead, load it as a variable in a code execution environment and let the model programmatically examine, decompose, and process it.**

The naive approach -- stuff the entire manual into one prompt -- exhibits context rot: quality degrades as inputs get longer and tasks get more complex. Prompt caching (which the existing chat interface uses) helps with repeated calls on the same document but doesn't solve the fundamental problem: as manuals exceed cache limits or grow beyond the context window, extraction quality degrades regardless of caching. The traffic cop checker pipeline (multi-agent red-teaming with multiple prompt stages) addresses accuracy but adds enormous infrastructure complexity.

The RLM approach sidesteps all three limitations. The model gets:
- A REPL (Read-Eval-Print Loop) Python environment
- A `context` variable containing the document (or a reference to it)
- A `llm_query()` function that lets it invoke itself recursively on slices of the document
- The ability to write code that filters, chunks, and transforms the document programmatically

This means:
- The model's context window holds only its system prompt + current working state + truncated stdout, never the full document
- The model can process documents 10-100x longer than its context window
- The model decides HOW to decompose the problem: chunk by section, search by keyword, process per-module
- Sub-calls are programmatic (in loops, with variables) rather than verbalized (one at a time in conversation)

### 1.3 Why This Matters for CHW Navigator

The pipeline needs to:
- Extract ALL clinical modules from a manual (4 modules for WHO 2012, potentially 15+ for larger manuals)
- Produce cross-referenced artifacts (predicates referenced in DMN must exist in predicates table)
- Handle comorbidity logic (Activator/Router pattern requires global view)
- Maintain provenance (every rule traces back to a page and quote)

The REPL/RLM approach lets a reasoning model:
1. First scan the full guide structure (table of contents, section headings)
2. Extract shared schema (predicate names, module list, variable conventions)
3. Launch per-module sub-calls with the conventions locked in
4. Build up the output JSON incrementally in REPL variables
5. Self-validate by calling validators exposed in the REPL
6. Fix errors in the same session without losing state

This is fundamentally more capable than a single-shot prompt, and fundamentally simpler than the traffic cop checker pipeline. One model, one session, programmatic decomposition.

### 1.4 RLM Results (From the Paper)

On benchmarks scaling from 8K to 1M+ tokens:
- RLM(GPT-5) maintained 80-95% accuracy where base GPT-5 degraded to 0-40%
- Costs were comparable to or cheaper than base model calls at the median
- The approach is model-agnostic (worked with GPT-5, Qwen3-Coder, and fine-tuned Qwen3-8B)
- Even without sub-calls, just having the REPL (code execution over the context variable) dramatically improved performance on long inputs

---

## 2. Architecture Overview

### 2.1 The Full Loop

```
Step 1: PDF --> JSON (external, assumed done)
        Full provenance: section IDs, page numbers, source quotes, tables, flowcharts

Step 2: REPL Session (this pipeline)
        Claude Opus as root model
        JSON guide loaded as REPL variable
        Model writes code to examine, decompose, extract
        Sub-calls (llm_query) for per-module extraction
        Validators available inside REPL as validate()
        Model self-corrects within the session

Step 3: Output JSON (source of truth)
        clinical_logic.json with:
          - modules (per-condition decision tables, boolean-only)
          - activator (COLLECT policy: symptoms -> modules)
          - router (FIRST policy: priority-ordered module selection)
          - integrative (combine module outputs)
          - predicates (boolean definitions with fail-safes)
          - phrase_bank (all CHW-facing messages)

Step 4: Deterministic Validation
        Python validators on JSON (structural)
        Z3 exhaustiveness / mutual exclusivity proofs (algorithmic)
        If errors: feed back into REPL (step 2 continues)

Step 5: Deterministic Conversion
        JSON --> DMN XML (for clinician review, DMN tooling)
        JSON --> Mermaid (for flowchart review)
        JSON --> CHT XLSForm XLSX (for phone deployment)

Step 6: External Test Harness (out of scope)
        XLSX tested against 1000+ synthetic patients
```

### 2.4 What Runs Where

```
Browser (user's machine):
  React frontend (Vercel)
    |
    +--> API key input (held in memory, never persisted)
    +--> Guide JSON upload
    +--> SSE stream of REPL trajectory
    +--> Download buttons for output artifacts

Backend (Render):
  FastAPI server
    |
    +--> POST /api/session/start { api_key, guide_json }
    |     Creates REPL session, begins extraction
    |
    +--> GET /api/session/{id}/stream (SSE)
    |     Streams REPL steps, stdout, validation results live
    |
    +--> GET /api/session/{id}/artifacts
    |     Download completed JSON, DMN, XLSX, Mermaid, CSVs
    |
    +--> Python REPL sandbox (subprocess per session)
    |     |
    |     +--> context variable (JSON guide)
    |     +--> llm_query() (calls Claude API with user's key)
    |     +--> validate() (structural validators)
    |     +--> z3_check() (exhaustiveness proofs)
    |
    +--> Claude API (user's BYOK key, passed per request)
    |     Root model: Claude Opus
    |     Sub-calls: Claude Sonnet (cheaper, same key)
    |
    +--> Neon Postgres (shared logging)
          Every REPL step, sub-call, validation, Z3 check logged
          Team can query trajectories across all users/runs
```

### 2.3 Web Interface (Basic, BYOK Auth)

The pipeline runs through a basic web interface. No auth system -- the user's Anthropic API key IS the authentication. Paste your key, upload a JSON guide, watch the REPL trajectory stream in real time, download outputs when done.

**Prior art:** The BYOK session pattern is adapted from CRM-RecSys (another project I built). In that system, the user pastes an API key to start a session, the key is held in memory for the session lifetime (never persisted to disk or database), all LLM calls use that key, and the session ends when the user closes the tab or the key is cleared. This pattern works well for developer tooling where you want zero friction but full visibility. The CRM-RecSys implementation should be referenced for the session management patterns, specifically how the API key flows from the frontend to the backend without being stored.

**What the interface shows:**
- Upload area for the JSON guide
- API key input field (paste to start, cleared on close)
- Live REPL trajectory stream (every code block, every stdout, every sub-call)
- Validation results inline (green/red per check as they run)
- Z3 proof results
- Download buttons for all output artifacts (JSON, DMN XML, XLSX, Mermaid, CSVs)
- Cost tracker (tokens used, estimated cost, sub-call count)

**What the interface does NOT have:**
- User accounts or login
- Persistent sessions across browser closes
- Shared state between users
- Rate limiting (your key, your budget)

**Why this matters for CHT:** When we hand this to the CHT community, any developer can try it immediately. No onboarding, no access requests, no shared API key management. They paste their key, upload a manual, and see if it works. The barrier to evaluation is literally zero.

**Infrastructure:**
- Frontend: React (or even plain HTML), deployed to Vercel
- Backend: FastAPI on Render, receives API key per request, executes REPL, streams results via SSE
- Neon Postgres: shared logging (the one persistent resource), stores trajectories for team review
- The API key flows: frontend --> backend request header --> REPL sandbox env var --> llm_query() calls

The key is NEVER written to Neon or any persistent store. It lives in the request context and the REPL subprocess environment for the duration of the session.

---

## 3. The REPL Implementation

### 3.1 Core Loop

Claude Opus doesn't have a native REPL (Read-Eval-Print Loop). We build one: Claude generates code blocks in its responses, our orchestrator parses them out, executes them in a sandboxed Python subprocess, captures stdout, feeds the (truncated) output back as the next user message, and loops until the model calls FINAL_VAR.

**Initialization:** The REPL state is set up with the JSON guide loaded as the `guide` variable, plus three helper functions: `llm_query()`, `validate()`, and `z3_check()`. Claude receives only metadata about the guide (total size, top-level keys, section count, a short preview) as its first message, not the full guide content. This keeps the context window clear for the system prompt and working state.

**Iteration:** Each loop iteration calls Claude Opus with the conversation history. Claude responds with reasoning text and code blocks. The orchestrator extracts and executes the code blocks, truncates stdout, logs to Neon, and appends the output as the next user message. If Claude calls FINAL_VAR(variable_name), the loop ends and that REPL variable is returned as the result.

**Termination:** The session ends when the model calls FINAL_VAR (success) or when MAX_ITERATIONS is reached (halt, human review needed). A generous iteration budget (e.g. 50) gives the model room to explore, extract, validate, fix, and re-validate.

### 3.2 REPL Sandbox

The sandbox restricts what the model's code can do. Blocked imports: os, sys, subprocess, shutil, pathlib, socket, http, urllib, requests, importlib. No `open()`, `exec()`, or `eval()` in builtins. Allowed: json, re, math, collections, and the exposed helper functions.

Code is executed with a per-step timeout (120 seconds). The sandbox maintains a shared variable namespace: anything the model assigns to a variable persists across execution steps. This is how the model builds up the `clinical_logic` JSON incrementally.

For the dev team context, this level of sandboxing is sufficient. Production deployment would use Docker or a subprocess jail.

### 3.3 Helper Functions Exposed to the REPL

**`llm_query(prompt: str) -> str`** -- Invokes Claude Sonnet on a sub-prompt. The sub-call sees only the prompt string, not the full REPL history. This is the recursive call mechanism from the RLM paper. Root model is Claude Opus; sub-calls use Claude Sonnet to manage costs. Same API key. A max sub-call budget (default 50) prevents cost explosion. If exceeded, the function returns an error message telling the model to consolidate queries.

**`validate(logic: dict) -> dict`** -- Runs all structural validators on the JSON and returns a report: `{"passed": bool, "error_count": int, "errors": [...]}`. The model calls this mid-trajectory, sees errors in stdout, and fixes them in the next code block. This is what collapses generate/validate/repair into one continuous session.

**`z3_check(logic: dict) -> dict`** -- Runs Z3 exhaustiveness and mutual exclusivity proofs. Returns `{"all_passed": bool, "checks": [...]}` with counterexamples for any failing check.

### 3.4 Stdout Truncation

Critical for context window management. If the model prints the entire guide JSON (200K+ chars), it fills Claude's context window and displaces the system prompt. Every stdout is truncated to ~3000 chars: show the first half and last half, with a message in the middle indicating how much was truncated and advising the model to use slicing to see more.

The model is told in the system prompt: "You will only see truncated REPL output. Use slicing to examine specific sections. Do not print the entire guide."

### 3.5 History Trimming

As the REPL session progresses, the conversation history grows. Claude Opus has a 200K token window. We reserve ~50K for the system prompt, ~100K for history, and ~50K for output. When history exceeds the budget, trim from the middle: keep the first exchange (initial prompt + first response) and the last N exchanges (recent state). Insert a message noting that earlier iterations were trimmed and key state is in REPL variables.

---

## 4. JSON Schema (Output Format)

The REPL session produces a JSON dict with this structure. Validators check against this schema.

### 4.1 Top Level

```json
{
  "modules": [...],        // Per-condition decision tables
  "activator": {...},      // COLLECT: symptoms -> module list
  "router": {...},         // FIRST: priority-ordered module selection
  "integrative": {...},    // Combine module outputs
  "predicates": [...],     // Boolean definitions with fail-safes
  "phrase_bank": [...]     // All CHW-facing messages
}
```

### 4.2 Module Table

```json
{
  "module_id": "mod_cough",
  "display_name": "Cough / Pneumonia Assessment",
  "hit_policy": "FIRST",
  "input_columns": ["p_cough_gte_14d", "p_chest_indrawing", "p_fast_breathing"],
  "output_columns": ["dx", "tx", "ref_hospital_urgent", "message_ids"],
  "rules": [
    {
      "inputs": ["true", "-", "-"],
      "outputs": ["chronic_cough", "tx_refer_tb", "false", "m_dx_chronic_cough,m_ref_tb"],
      "provenance": {"page": "p.38", "quote": "cough for 14 days or more is a danger sign"}
    },
    {
      "inputs": ["-", "true", "-"],
      "outputs": ["severe_pneumonia", "tx_pre_referral_amox", "true", "m_dx_severe_pneumonia,m_ref_urgent"],
      "provenance": {"page": "p.22", "quote": "chest indrawing lower chest wall when child breathes in"}
    },
    {
      "inputs": ["false", "false", "true"],
      "outputs": ["pneumonia", "tx_oral_amoxicillin", "false", "m_dx_pneumonia,m_tx_amoxicillin"],
      "provenance": {"page": "p.45", "quote": "fast breathing without chest indrawing"}
    },
    {
      "inputs": ["-", "-", "-"],
      "outputs": ["common_cold", "tx_fluids_rest", "false", "m_dx_common_cold,m_adv_home_care"],
      "provenance": {"page": "p.46", "quote": "no pneumonia simple cough or cold"}
    }
  ]
}
```

Rules are boolean-only. Every input is "true", "false", or "-". No numeric thresholds. Last rule is always the default (all inputs "-").

### 4.3 Activator (COLLECT)

```json
{
  "input_columns": ["q_cough", "q_diarrhea", "q_fever", "v_muac_color"],
  "rules": [
    {"inputs": ["true", "-", "-", "-"], "module_id": "mod_cough"},
    {"inputs": ["-", "true", "-", "-"], "module_id": "mod_diarrhea"},
    {"inputs": ["-", "-", "true", "-"], "module_id": "mod_fever"},
    {"inputs": ["-", "-", "-", "red"], "module_id": "mod_malnutrition_severe"},
    {"inputs": ["-", "-", "-", "yellow"], "module_id": "mod_malnutrition_moderate"}
  ]
}
```

COLLECT: ALL matching rules fire. Patient with cough + diarrhea triggers both modules.

### 4.4 Router (FIRST)

```json
{
  "rules": [
    {"condition": "p_danger_sign_present == true", "next_module": "mod_emergency_referral", "priority": 0},
    {"condition": "mod_cough in required AND mod_cough not in completed", "next_module": "mod_cough", "priority": 1},
    {"condition": "mod_diarrhea in required AND mod_diarrhea not in completed", "next_module": "mod_diarrhea", "priority": 2},
    {"condition": "mod_fever in required AND mod_fever not in completed", "next_module": "mod_fever", "priority": 3},
    {"condition": "always", "next_module": "mod_integrative", "priority": 99}
  ]
}
```

FIRST: scan top to bottom, fire first match. Row 0 is emergency short-circuit.

### 4.5 Predicate

```json
{
  "predicate_id": "p_fast_breathing",
  "source_vars": ["v_resp_rate_per_min", "demo_age_months"],
  "threshold_expression": "(age_mo < 12 AND rr >= 50) OR (age_mo >= 12 AND rr >= 40)",
  "human_label": "Fast breathing for age",
  "fail_safe": 1,
  "page_ref": "p.26"
}
```

- `fail_safe: 1` = if source data is missing, assume true (conservative). Used for equipment-dependent measurements.
- `fail_safe: 0` = if source data is missing, assume false. Used for caregiver-reported symptoms (absence = not reported).

### 4.6 Phrase Bank Entry

```json
{
  "message_id": "m_tx_amoxicillin",
  "category": "treatment",
  "english_text": "Give amoxicillin {dose} tablet(s) twice daily for {days} days.",
  "placeholder_vars": ["dose", "days"],
  "page_ref": "p.45"
}
```

Categories: question, diagnosis, treatment, advice, referral, followup, instruction.

---

## 5. System Prompt

The system prompt tells Claude Opus how to operate as an RLM for clinical extraction. **The prompt must be fully manual-agnostic.** No hardcoded danger sign lists, no specific thresholds, no module names, no WHO 2012 references. The model discovers all clinical content from whatever JSON guide is loaded into the REPL. The same prompt must work on any CHW manual (WHO, national MOH guidelines, NGO protocols, or invented test manuals).

The prompt defines:
- The REPL environment and how to use it
- The output JSON structure (Activator/Router/Module/Integrative pattern)
- The extraction strategy (scan, schema, per-module, assemble, validate)
- Conventions (boolean-only DMN, predicate fail-safes, variable naming, provenance)
- What "conservative default" means (higher risk classification when ambiguous)

The prompt does NOT define:
- Which modules exist (the model finds them in the guide)
- Which danger signs exist (the model extracts them from the guide)
- What thresholds apply (the model reads them from the guide)
- How many rules each table should have (the model determines this from the guide)

### 5.1 Role and REPL Instructions

```
You are a clinical decision logic extractor operating in a REPL environment. Your goal is to read a CHW clinical manual (loaded as the `guide` variable) and produce a structured JSON containing all clinical decision logic.

Your REPL environment provides:
1. `guide` -- the full manual as a JSON object with sections, provenance, page references
2. `llm_query(prompt)` -- invoke a sub-model on a specific question (handles ~100K chars)
3. `validate(json_dict)` -- run structural validators on your output, returns errors
4. `z3_check(json_dict)` -- run exhaustiveness/consistency proofs, returns counterexamples
5. `print()` -- output will be truncated to ~3000 chars. Use slicing to inspect specific parts.

When you write Python code in ```repl blocks, it will be executed and you'll see the output.
When you are done, call FINAL_VAR(variable_name) to return your result.

IMPORTANT: You will only see truncated REPL output. Do NOT print the entire guide. Use slicing (e.g., print(json.dumps(guide["sections"][list(guide["sections"].keys())[0]], indent=2)[:2000])) to examine specific parts.
```

### 5.2 Extraction Strategy (Manual-Agnostic)

```
Recommended extraction strategy:

Phase 1: Scan the guide structure
  - Print top-level keys and section names
  - Identify all clinical modules the manual covers (the model discovers these, not hardcoded)
  - Identify all assessment questions (ASK phase)
  - Identify all examination findings (LOOK phase)
  - Identify all danger signs described in the manual

Phase 2: Extract shared schema
  - Define all predicate names (p_ prefix) with thresholds and fail-safes
  - Define all variable names following the naming convention
  - Build the composite danger sign predicate from whatever danger signs the manual defines
  - Store this as a `conventions` variable in the REPL

Phase 3: Extract per-module logic
  - For each module identified in Phase 1, use llm_query() with the relevant guide section + conventions
  - The sub-call prompt should include the variable naming conventions to prevent drift
  - Parse the sub-call response and build the module's decision table
  - Add to the growing `clinical_logic` variable

Phase 4: Build Activator, Router, and Integrative tables
  - Activator: which symptoms/findings trigger each module (discovered from the manual)
  - Router: priority ordering based on clinical urgency described in the manual, emergency short-circuit as row 0
  - Integrative: how to combine outputs (highest referral wins, additive treatments, shortest follow-up)

Phase 5: Assemble phrase bank
  - Every question, diagnosis, treatment, advice, referral, and follow-up message from the manual
  - Include placeholder variables for dynamic values (doses, durations)
  - All text comes from the manual, not invented

Phase 6: Self-validate
  - Call validate(clinical_logic) and review errors
  - Fix any errors in the REPL
  - Call z3_check(clinical_logic) for exhaustiveness
  - Repeat until both pass

Phase 7: Return
  - FINAL_VAR(clinical_logic)
```

### 5.3 Clinical Architecture (Agnostic Pattern)

```
The output JSON must contain:

ACTIVATOR (hit_policy: COLLECT)
  Looks at ALL patient symptoms/findings, flags EVERY module that needs to run.
  A patient with multiple complaints triggers all relevant modules.
  The model must identify all triggering conditions from the manual.

ROUTER (hit_policy: FIRST)
  Given required modules and completed modules, picks the SINGLE next module.
  Row 0 MUST be the emergency/danger sign short-circuit (referral).
  The danger signs come from the manual, not from the prompt.
  Priority ordering reflects clinical urgency as described in the manual.
  Last row: fallback to Integrative when all modules done.

PER-MODULE TABLES (hit_policy: FIRST)
  Clinical classification for each condition the manual covers.
  ALL input cells must be boolean only: "true", "false", or "-".
  No numeric thresholds in rule inputs. All thresholds go in predicates.
  Last rule must be default (all inputs "-").

INTEGRATIVE TABLE
  Combines outputs from all modules into a single care plan.
  Highest referral level wins. Treatments are additive unless the manual
  specifies interactions. Shortest follow-up interval applies.
```

### 5.4 Predicate and Naming Conventions (Agnostic)

```
PREDICATE CONVENTION (CRITICAL):
  DMN tables use ONLY boolean inputs. Thresholds live in the predicates array.
  The model must extract all thresholds from the manual and create predicates for them.
  
  fail_safe rules (universal, not manual-specific):
    1 (assume true / high risk) for measurements requiring equipment (thermometer, timer, scale)
    0 (assume false / low risk) for caregiver-reported symptoms (asked but not reported = no)
    1 for composite danger sign predicates (if unsure, refer)
  
  The model determines which fail_safe value based on whether the source variable
  requires equipment (v_ prefix -> likely 1) or is caregiver-reported (q_ prefix -> likely 0).

VARIABLE NAMING (universal convention, not manual-specific):
  q_   = caregiver-reported symptoms and history
  ex_  = CHW observation / examination finding
  v_   = quantitative measurement requiring equipment
  lab_ = point-of-care test result
  p_   = computed predicate boolean (derived from raw inputs)
  dx_  = diagnosis / classification output
  tx_  = treatment intent output
  ref_ = referral output
  adv_ = advice / counseling output
  fu_  = follow-up scheduling output
  m_   = message ID (links to phrase bank)
  mod_ = module ID
  demo_ = demographics

  Numeric variables MUST end with unit suffix: _days, _per_min, _c, _mm, _mg, _kg

PROVENANCE (universal requirement):
  Every rule must have provenance: {"page": "p.XX", "quote": "5-15 words from manual"}
  Every predicate must have page_ref.
  Every phrase bank entry must have page_ref.
  If a rule is inferred (not directly stated), annotate as "Inferred from [description]".
```

---

## 6. Validators (Inside the REPL)

The model calls `validate(clinical_logic)` and sees errors in stdout. All validators operate on the JSON dict. No AI.

### 6.1 Architecture Validator

- `activator` exists with at least one rule
- `router` exists, rule[0] condition references `p_danger_sign_present`
- Router has a fallback/default last rule
- Every activator `module_id` appears in a router condition
- Every activator `module_id` has a matching entry in `modules`
- No `module_id` is a substring of another
- `integrative` exists

### 6.2 Completeness Validator

- Every `p_` variable in any module's `input_columns` exists in `predicates`
- Every `m_` value in any module's `outputs` exists in `phrase_bank`
- `p_danger_sign_present` exists in predicates
- Every predicate has non-empty `source_vars`, `threshold_expression`, `page_ref`
- Every phrase entry has non-empty `english_text`, `page_ref`

### 6.3 Clinical Validator

- Every module's last rule has all inputs = "-" (default/else row)
- Rule input array lengths match `input_columns` length
- Rule output array lengths match `output_columns` length
- All rule inputs are strictly "true", "false", or "-"
- No raw numeric thresholds appear in rule inputs
- Every predicate `fail_safe` is 0 or 1
- Fail-safe sanity: predicates with `v_` source vars should have fail_safe=1

### 6.4 Naming Validator

- predicate_id matches `^p_[a-z][a-z0-9_]*$`
- module_id matches `^mod_[a-z][a-z0-9_]*$`
- message_id matches `^m_[a-z][a-z0-9_]*$`
- source_vars match `^(q|ex|v|lab|demo|hx)_[a-z][a-z0-9_]*$`
- output columns match `^(dx|tx|ref|adv|fu|m|sev|out)_[a-z][a-z0-9_]*$`

---

## 7. Z3 Verification (Inside the REPL)

The model calls `z3_check(clinical_logic)` to mathematically prove properties of the decision tables.

### 7.1 What Z3 Checks

For each module table:
- **Exhaustiveness:** Every combination of boolean inputs matches at least one rule. Z3 tries to find an input that matches NO rule. If UNSAT, the table is exhaustive.
- **Mutual exclusivity (UNIQUE tables):** No input combination triggers two rules. Z3 tries to find an input matching two rules simultaneously. If UNSAT, the table is conflict-free.
- **Reachability:** Every rule can be triggered by some valid input. Z3 finds a satisfying assignment for each rule. If UNSAT, the rule is dead code.
- **Danger sign monotonicity:** Adding a danger sign never downgrades the disposition. If p_danger_sign_present becomes true, the disposition must be referral or higher.

For the Router:
- **Termination:** The completed_modules set always grows. No module is routed to twice.

### 7.2 What Z3 Also Does: Synthetic Patient Generation

For each rule in each table, Z3 produces a concrete patient that triggers that rule. These are boundary-correct synthetic patients that can feed the external test harness without manual authoring.

---

## 8. Converters (Piece #2)

All converters take the validated JSON and produce output formats. Pure Python, deterministic.

### 8.1 json_to_dmn.py

Maps JSON to DMN 1.3 XML:
- Each module --> `<decision>` with `<decisionTable>`
- hitPolicy attribute from module's `hit_policy`
- input_columns --> `<input>` elements with typeRef="boolean"
- output_columns --> `<output>` elements
- rules --> `<rule>` with `<inputEntry>`, `<outputEntry>`, `<description>` (provenance)
- predicates --> `<inputData>` elements

Uses `xml.etree.ElementTree`.

### 8.2 json_to_xlsx.py (Critical)

Maps JSON + predicates to CHT XLSForm:

**Survey sheet:**
1. Patient metadata (demo_ variables)
2. Universal screening group (all q_ variables)
3. Examination group (all ex_ and v_ variables)
4. Hidden predicate calculates (with fail-safe XPath)
5. Activator calculate
6. Module groups (relevance-gated by router logic)
7. Completion tracking (completed_modules bitmask)
8. Emergency group
9. Integrative group

**Predicate compilation to XPath:**
```
p_fast_breathing:
  source_vars: [v_resp_rate_per_min, demo_age_months]
  threshold: (age<12 AND rr>=50) OR (age>=12 AND rr>=40)
  fail_safe: 1

-->

calculate: p_fast_breathing_measured
  = string-length(${v_resp_rate_per_min}) > 0

calculate: p_fast_breathing
  = if(${p_fast_breathing_measured},
       if(${demo_age_months} < 12,
          if(${v_resp_rate_per_min} >= 50, 1, 0),
          if(${v_resp_rate_per_min} >= 40, 1, 0)),
       1)
```

**Module routing:**
```
begin_group: grp_mod_cough
  relevant: contains(${required_modules}, 'mod_cough')
            and not(contains(${completed_modules}, '|mod_cough|'))
            and ${p_danger_sign_present} = 0
```

Uses `openpyxl`.

### 8.3 json_to_mermaid.py

Modules as subgraphs, rules as decision paths, emergency path highlighted in red.

### 8.4 json_to_csv.py

Renders predicates and phrase_bank arrays as flat CSVs for human review. Trivial conversion.

---

## 9. Known Failure Modes and Mitigations

### 9.1 FINAL Answer Divergence (from RLM paper)

**Risk:** Model correctly builds clinical_logic in REPL variables across 8 steps, then on step 9 "verifies" by regenerating, overwrites the correct variable with a wrong version, returns the wrong one.

**Clinical impact:** Correct DMN built up programmatically gets corrupted in the final step. A danger sign rule might be dropped.

**Mitigation:** FINAL_VAR returns the REPL variable by reference, not by regeneration. The system prompt explicitly says: "Do NOT regenerate your result. Call FINAL_VAR on the variable you built up." Validators run inside the REPL BEFORE the model calls FINAL_VAR, so the returned variable has already passed validation.

### 9.2 Sub-Call Schema Drift

**Risk:** Root model extracts conventions (predicate names, variable naming). Sub-call for mod_cough uses `fast_breathing` instead of `p_fast_breathing`. Sub-call for mod_diarrhea uses `blood_stool` instead of `p_blood_in_stool`. Final JSON is internally inconsistent.

**Clinical impact:** Predicates referenced in module tables don't exist in predicates array. XLSForm compilation fails or produces broken logic.

**Mitigation:** The root model constructs sub-call prompts that include the conventions dict as context. The system prompt instructs: "When calling llm_query() for per-module extraction, always include the variable naming conventions in the prompt." The completeness validator catches any remaining drift (predicate referenced but not defined).

### 9.3 Cost Explosion from Excessive Sub-Calls

**Risk:** Model decides to sub-call per-line of the guide JSON, launching hundreds or thousands of calls. The RLM paper documented Qwen3-Coder making 1000+ sub-calls for a single task.

**Clinical impact:** None clinical, but a single extraction run could cost $20+ instead of $2.

**Mitigation:** `llm_query()` has a max_subcalls budget (default 50). If exceeded, returns an error message. The system prompt says: "Batch information into sub-calls. Aim for one sub-call per module, not per line." Root model is Claude Opus (disciplined about sub-calls); sub-calls use Claude Sonnet (cheaper).

### 9.4 REPL Stdout Overflow

**Risk:** Model prints the entire guide JSON. 200K+ chars fill the context window, displacing the system prompt and earlier REPL state.

**Clinical impact:** Model loses track of what it already extracted, re-extracts incorrectly, or halts.

**Mitigation:** All stdout is truncated to 3000 chars with a message indicating truncation. System prompt explicitly warns against printing large objects.

### 9.5 Clinically Plausible Wrong Thresholds

**Risk:** Model writes `>= 45` instead of `>= 40` for fast breathing cutoff. Structurally valid, medically wrong.

**Clinical impact:** Children with respiratory rate 40-44 classified as common cold instead of pneumonia. No antibiotics.

**Mitigation:** Validators CANNOT catch this. The predicates array makes all thresholds human-readable for clinician review. The external test harness catches via patient outcomes. Z3 can generate boundary patients (resp_rate = 39, 40, 41) that expose threshold errors when run through the XLSX.

### 9.6 Context Rot on Long Manuals

**Risk:** Model correctly extracts early modules but degrades on later ones.

**Clinical impact:** Fever and malnutrition modules might have missing or incorrect rules while cough and diarrhea are perfect.

**Mitigation:** The REPL approach inherently mitigates this. Each module is extracted via a separate sub-call that sees only the relevant section + conventions. The root model's context window holds only the working state, not the full manual. Completeness validator catches missing modules.

### 9.7 Non-Determinism Across Runs

**Risk:** Even at temp=0, reasoning traces can vary, producing different outputs across runs.

**Clinical impact:** Can't certify non-deterministic output.

**Mitigation:** Run the pipeline 3 times, diff the JSON outputs structurally. The `--determinism-check` flag automates this. REPL trajectories may differ but final validated JSON should be equivalent.

### 9.8 Sandbox Escape

**Risk:** Model writes code that accesses the filesystem, network, or system resources.

**Clinical impact:** Security issue, not clinical.

**Mitigation:** REPL sandbox blocks dangerous imports (os, sys, subprocess, socket, etc.). No open(), exec(), eval() in builtins. For the dev team context, this is sufficient. Production deployment would use Docker.

---

## 10. Neon Logging

Every REPL session is logged to shared Neon Postgres for team visibility.

### 10.1 Schema

Two tables in Neon:

**extraction_runs** -- One row per pipeline run. Columns: run_id, manual_name, model, started_at, completed_at, status (running/passed/failed/halted), total_iterations, total_subcalls, validation_errors count, final_json (JSONB, the output), cost_estimate_usd.

**repl_steps** -- One row per REPL iteration or sub-call. Columns: run_id (FK), step_number, step_type (exec/subcall/validate/z3/final), code (for exec steps), stdout (truncated), stderr, prompt (for sub-call steps), response (for sub-call steps), execution_ms, input_tokens, output_tokens, created_at. Indexed on run_id for fast trajectory retrieval.

### 10.2 What Gets Logged

- Every REPL code execution (code + stdout + stderr + timing)
- Every sub-call (prompt + response + tokens)
- Every validation run (errors list)
- Every Z3 check (results)
- Final output JSON
- Total cost estimate

The team can query: "Show me all runs on WHO 2012 this week, sorted by error count." Or: "Show me the REPL trajectory for the run that produced the best output."

---

## 11. Directory Structure

```
chw-navigator/
  frontend/                     # Vercel deployment
    src/
      App.tsx                   # Main layout
      components/
        ApiKeyInput.tsx         # BYOK paste field (session memory only)
        GuideUpload.tsx         # JSON file upload
        ReplStream.tsx          # Live SSE stream of REPL trajectory
        ValidationPanel.tsx     # Green/red check results
        ArtifactDownloads.tsx   # Download buttons for all outputs
        CostTracker.tsx         # Token count + estimated cost
      hooks/
        useSession.ts           # Session management (key in memory, never persisted)
        useSSE.ts               # SSE stream consumption
      api/
        client.ts               # Backend API calls (key in Authorization header)
    package.json
    tsconfig.json

  backend/                      # Render deployment
    run.py                      # CLI entry point (also works standalone without web)
    server.py                   # FastAPI server (web interface backend)
    repl.py                     # REPL session loop + sandbox
    llm_client.py               # Claude API wrapper (root + sub-calls)
    schema.py                   # JSON schema definitions
    logging_db.py               # Neon logging
    validators/
      __init__.py               # run_all_validators()
      architecture.py
      completeness.py
      clinical.py
      naming.py
    z3_verifier/
      __init__.py
      exhaustiveness.py
      reachability.py
      termination.py
      patient_gen.py
    converters/
      json_to_dmn.py
      json_to_xlsx.py           # Critical: predicate compilation + state management
      json_to_mermaid.py
      json_to_csv.py
    prompts/
      system_generate.txt
    output/                     # Generated artifacts (per session)
    tests/
      test_validators.py
      test_converters.py
      test_z3.py
      test_repl_sandbox.py
      fixtures/
        valid_logic.json
        invalid_logic.json
    reference/
      who_2012_guide.json
      grading_rubric.csv
      patientbot_1000.csv
```

**Note on CRM-RecSys:** The frontend session management (API key in memory, never persisted, cleared on tab close, passed via Authorization header) follows the same pattern as the CRM-RecSys project. Reference that codebase for the `useSession` hook implementation, the key-in-header pattern, and the session lifecycle (start on paste, end on close/clear).

---

## 12. Build Order

| # | Task | What It Produces | Effort |
|---|------|------------------|--------|
| 1 | JSON schema definitions | `schema.py` with type specs | 0.5 day |
| 2 | All 4 validators + unit tests | `validators/` passing on fixtures | 2 days |
| 3 | REPL sandbox + execution loop | `repl.py` with sandboxing, truncation, history trimming | 2 days |
| 4 | Claude API client (root + sub-call) | `llm_client.py` with BYOK key passthrough | 0.5 day |
| 5 | Neon logging | `logging_db.py` + schema migration | 1 day |
| 6 | FastAPI server + SSE streaming | `server.py` with session endpoints | 1.5 days |
| 7 | Frontend (BYOK session + REPL stream + downloads) | `frontend/` -- follow CRM-RecSys patterns for session management | 2 days |
| 8 | CLI entry point (non-web, for scripting) | `run.py` standalone mode | 0.5 day |
| 9 | System prompt | `prompts/system_generate.txt` -- iterate against validators | 2-3 days |
| 10 | Z3 verifier (basic: exhaustiveness + reachability) | `z3_verifier/` | 2 days |
| 11 | json_to_xlsx converter | With predicate compilation + state management | 2-3 days |
| 12 | json_to_dmn converter | Standard DMN 1.3 generation | 1 day |
| 13 | json_to_mermaid converter | Flowchart generation | 0.5 day |
| 14 | Run on WHO 2012 + iterate prompt | Working extraction for one manual | 2-3 days |
| 15 | Determinism check (3 runs) | Confirm reproducibility | 0.5 day |
| **Total** | | | **~18-22 days** |

**Build order matters:**
1. Schema + validators first (you can't iterate on anything without knowing what "correct" looks like)
2. REPL sandbox second (the core infrastructure)
3. Backend server + frontend third (follow CRM-RecSys BYOK patterns -- session hook, key-in-header, SSE streaming)
4. System prompt fourth (iterate against validators in the REPL, visible in the web UI)
5. Converters can be built in parallel with prompt iteration
6. Z3 can be added after basic extraction works

---

## 13. Evidence for the Professor

### Demo 1: Extraction Quality
```bash
export ANTHROPIC_API_KEY=sk-...
python run.py reference/who_2012_guide.json --model claude-opus-4
# --> clinical_logic.json (validated, z3-checked)
# --> dmn_tables.xml, form.xlsx, flowchart.md, predicates.csv, phrase_bank.csv
```

### Demo 2: Determinism
```bash
python run.py reference/who_2012_guide.json --determinism-check 3
# --> "3/3 runs produced structurally equivalent output"
```

### Demo 3: REPL Trajectory Visibility
The team can query Neon to see the full extraction trajectory for any run: every REPL step, every sub-call, every validation result, ordered by step number. This makes debugging straightforward -- when a run produces wrong output, you trace the exact step where it diverged.

### Demo 4: Clinical Correctness (external harness)
```bash
python test_harness.py output/form.xlsx reference/patientbot_1000.csv
# --> 1000/1000 correct diagnoses + treatments
```

### Demo 5: Scalability Argument
"The model never sees the full manual in its context window. It programmatically examines sections, extracts per-module, and self-validates. This scales to manuals 10x longer than WHO 2012 with the same architecture."

---

## 14. What This Does NOT Include

- PDF-to-JSON extraction (assumes JSON guide already exists with provenance)
- User accounts or persistent auth (BYOK is the auth, keys live in session memory only)
- Translation / localization of phrase bank
- CHT deployment integration (the XLSX output feeds into CHT tooling)
- VLM extraction from PDF images/flowcharts
- Multi-tenant infrastructure (each user's session is independent)
