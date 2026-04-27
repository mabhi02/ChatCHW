# CHW Navigator: Arena-Style A/B Test Specification

> **Purpose:** Hand this file + `Automating_CHT_v25.xlsx` to Claude Code. It should scaffold both pipelines, align their intermediate artifacts, and produce a gate harness that compares outputs on synthetic patients.
>
> **Date:** 2026-04-09
> **Author:** Atharva Patel

---

## PROFESSOR LEVINE'S EMAILS (verbatim, for context)

### Email 1 (responding to architecture memo + postmortem)

Atharva,

Thanks for these memos. Your concerns with the sequential method are well grounded.  At the same time, we do not know if they mean it will fail until we try.

1. I am not convinced if the sequential prompts will fail

The sequential prompts partly address your "hard to diagnose" problem because each output has a set of quality standards for that red team LLM.  If the maker & repair loop cannot hit the standards, we know roughly where the problem is.  This approach may not work for the reasons you outline: fixing one phase affects others.  Nevertheless, it seems far more likely to diagnose problems than "red team at end" as you propose.  

A key issue you highlight is that LLMs have randomness.  We can turn that "bug" into a feature: If we run the sequential pipeline 3 times, the DMN should function the same.  The specifics may differ slightly, but if we run 1000 patients they should have 1000 identical outcomes.  This is a simpler variation of the test you proposed earlier: Check if multiple pipelines (varying LLM roles, prompts, etc.) achieve identical outputs.  (In the psychology measurement literature, the re-run pipeline is called "test-retest reliability."  Different pipelines might be called "convergent validity" -- but I think of it as closer to a reliability check as we are not comparing to clinical truth.)

As you note, "unrecorded user inputs" would be a disaster. In my approach any clinical decisions or clarifications become appendices to the manual and must be recorded and logged. 

Note: The CHW Navigator proposal appears to have mangled the order and maybe content of prompts.  A similar and somehwat easier to read version is at https://docs.google.com/spreadsheets/d/1MlzqI4Maa7Ej_CQLT-c9rrtj_m-kHclKQqzd2nVc9wI/edit?usp=drive_link

Also: It is fine to combine some of my many proposed prompts, shift the intermediate artifacts, etc.  I am not defending their current form.


2. I do not see why the Zhang et al. helps

Your integrated prompt still has 7 phases.  

A) Putting multiple sequential prompts into an integrated prompt with phases does not resolve the "hard to diagnose" issue.  For example, if we change the sub-prompt for phase 3, that change can still affect the operation of other phases.  I do not see what is solved by combining the phases into an integrated prompt. 

B) Your prompt does not red team till the end.  That approach almost surely makes diagnosis more difficult. 

3. Conclusion

Again: I'd like to try the sequential prompts before we give up.  I am open to the possibility of failure, but I need evidence. 

Is it a bigger job than I think to write the Python and markdown files described in my CHW Navigator proposal?  A week ago you said it would be running this week. Now what is the timeline to do so? 

Thanks,

-David

### Email 2 (after reading the postmortem more carefully)

Atharva,

I) A big win : 7 Deterministic IR converters
CHT was planning on having MDs write some IR and then programmers convert to XLSForm.  It is great that we have working demos of IR converters. 

II) Your report 
I did not read your email carefully enough.  I now see that you did get a pipeline working. Unfortunately, I do not understand your report on the results.

1. Did you use sequential steps + red team & repair loops?

My proposed sequential procedure has a number of prompts to break "Make a DMN" into multiple intermediate artifacts.  Each artifact has a red team & repair loop. Only after they all pass are the artifacts then assembled into DMN. 

I read: 
> A generator prompt running on Anthropic Claude (routed through a provider abstraction that also supports Google Gemini) reads the guide from a prompt-cached context block and produces an initial DMN artifact..  

But elsewhere you wrote: 
> Ten distinct prompts control the system's behavior.

So I am confused.  Are most of those prompts your quality checkers?


2. Human input 

A) You point out human decisions are not logged. 
Key fix: Log all decisions.  Who, when, what information they were shown, what they decided, and (if relevant) automated fix.

B) Separate out forms of failure. 

i. There are defects in the manual such as, "Does not tell what to do for stockouts."  Those require clinical review.  

ii. There are defects in the Maker prompt / LLM / LLM output.  My proposal for each phase was a red team & repair loop.  This is related to your "fix automatically", but only makes a repair when the manual is complete and the LLM messed up, but another LLM is able to figure it out.  
     a) If the LLM discovers a new incompleteness, it SHOULD go to the human review. 
     b) If the LLM cannot repair, it SHOULD go to the human review.   We hope these cases are not hundreds / manual 

Can you easily classify the hundreds of defects or show me a dozen? 

3. What is failure? 

Failure is NOT "A DIFF routine on DMN1 and DMN2 finds deviations". 

i. DMN can differ in ways that are not clinically relevant.  "If age_mo > 12" or "if not age_mo <=12" are the same, as are "tx.ORS tx.zinc" and "tx.zinc tx.ORS".  
We need a clinical comparison such as run our synthetic patients through both and see results. 

ii. If clinical different DMN are due to different human inputs, that is also not a failure of the automated pipeline. 


III) Details (I think)

Is this a fixable bug?  "all calls to Claude Haiku regardless of configured provider"


Is this a fixable bug?  "The caching investment was real; the realized discount was zero."
Note that Gemini tells me it keeps the cache longer than claude. 


Is this a constraint? " And guides larger than the cache would be impossible turn into a DMN."
    LLMs run caches much larger than 100,000 tokens these days.  I believe folks have written code to compress documents to save input tokens -- though we would want to check that no dosage, danger signs, etc. are compressed. 
   Am I missing something about the limit here?

---

## 0. CONTEXT: WHAT THIS IS AND WHY

We have two competing architectures for extracting clinical decision logic (DMN tables) from CHW (Community Health Worker) manuals. Both take the same input (a PDF clinical manual) and produce the same output (structured DMN tables + XLSForm). Neither has been tested head-to-head.

This is a Chatbot Arena-style blind comparison. Both pipelines run on the same input simultaneously, emit the same intermediate artifacts at aligned checkpoints, and their final DMN outputs are compared for clinical equivalence using synthetic patients.

The question being answered: **which architecture produces more reliable DMN extraction from CHW manuals?**

Reliability = clinical equivalence across repeated runs. NOT text equivalence. `if age_mo > 12` and `if not age_mo <= 12` are the same clinically. The correct metric is: run synthetic patients through both DMNs, compare clinical outcomes (diagnosis, referral level, treatment, follow-up days). Identical clinical outcomes = reliable. Different clinical outcomes = defect.

---

## 1. THE TWO PIPELINES

### Pipeline A: Sequential (Professor Levine's proposal)

**Architecture:** Many separate LLM calls chained in order. Each stage has a MAKER prompt, a RED TEAM prompt, and a REPAIR prompt. Each call is a fresh context window. The maker produces an artifact, the red team checks it against quality standards, the repair fixes issues the red team found. Only after passing the red team gate does the next stage begin.

**Key property:** Each prompt sees only what you explicitly pass to it. The variable extractor sees the supply list and the manual, but NOT the full reasoning the model used to produce the supply list. Decisions are fixed: once the supply list passes its red team, it is frozen. If the predicate extractor discovers the supply list missed a variable, it cannot go back. It has to work with what it received. Backward fixes require re-running upstream stages from scratch.

**Source:** All prompts, quality standards, inputs/outputs, and orchestrator pseudocode are in `Automating_CHT_v25.xlsx`. The spreadsheet has ~17 sheets covering phases A through G, plus Quality Standards (160 rules), Software Libraries, WriteOnce Code, orchestrator pseudocode, and a YAML DAG schema.

**Prompt count:** ~45 separate prompts (3 per stage x ~15 stages). Each is a tunable variable.

### Pipeline B: REPL / RLM (Atharva's proposal, based on Zhang, Kraska, Khattab 2026)

**Architecture:** One system prompt (~7,000 tokens) runs in a single continuous session with REPL tools. The model navigates the manual programmatically, builds artifacts incrementally in Python variables, and emits them at mandatory checkpoints. Frozen catcher validators (3 deterministic Python + 5 LLM at temperature=0 with N=3 majority vote) run at each checkpoint.

**Key property:** The model has full context of everything it has done so far as it does each step. If during predicate extraction it realizes the variable list is incomplete, it can go back and fix the variable list before proceeding. The catcher gates still run between phases, but the model has the ability to self-correct across phase boundaries, not just within them.

**Source:** `system_prompt.md` (included in this repo, also pasted below in the appendix). The REPL scaffold is already built in `backend/`.

**Prompt count:** 1 system prompt + catcher suite. Two tunable variables total.

---

## 2. WHAT CLAUDE CODE NEEDS TO BUILD

### 2.1 Sequential Pipeline (`sequential/`)

Build a Python orchestrator that implements Professor Levine's pipeline from `Automating_CHT_v25.xlsx`. The input to every stage is `guide.json` (the pre-processed manual, see Section 6.6). The orchestrator should:

1. **Parse the xlsx** to extract all maker prompts, red team prompts, repair prompts, inputs/outputs, quality standards, and stage dependencies. Each sheet in the xlsx is a stage or supporting resource.

2. **Implement the stage executor** following the orchestrator pseudocode in the `orchestrator_pseudo` sheet. Each stage:
   - Resolves inputs from prior stage outputs or filesystem
   - Sends the maker prompt to the LLM API with the resolved inputs
   - Sends the red team prompt with the maker output
   - If red team finds issues: sends repair prompt, loops (max 2 retries)
   - Validates output schema
   - Saves artifact to disk
   - Moves to next stage

3. **Use the YAML DAG schema** from the `YAML DAG Schema for orch` sheet to determine stage ordering and dependencies. Stages that don't depend on each other can run in parallel.

4. **Emit intermediate artifacts at aligned checkpoints** (see Section 3 below). The sequential pipeline MUST emit artifacts with the same names and at the same logical points as the REPL pipeline so we can compare intermediate outputs, not just final outputs.

5. **Skip Phase A (manual review/repair) for the A/B test.** Both pipelines should receive the raw manual as input. Phase A improves the manual before extraction, which would confound the architecture comparison with input quality. Phase A can be added back after the A/B test.

6. **Skip Phases E, F, G for the A/B test.** These are downstream of DMN extraction (human feedback, translation, governance). They don't affect extraction reliability.

7. **API configuration:**
   - Model: `claude-sonnet-4-20250514` for maker/repair, `claude-sonnet-4-20250514` for red team (matching cost profile to RLM)
   - Temperature: 0 for all calls
   - Provider abstraction: support both Anthropic and Google Gemini APIs via env var
   - BYOK pattern: API key from environment variable, never hardcoded

### 2.2 REPL Pipeline (`repl/`)

The REPL scaffold already exists in `backend/`. The REPL receives `guide.json` (same pre-processed JSON as the sequential pipeline) as the `guide` variable. What needs to happen:

1. **Update the system prompt** to align intermediate artifact names with the sequential pipeline (see Section 3). The current system prompt already has 7 emit_artifact checkpoints. These need to map 1:1 to the sequential pipeline's intermediate outputs.

2. **Update the system prompt** to incorporate any extraction guidance from the xlsx that the current prompt is missing. Specifically:
   - The xlsx has detailed quality standards (160 rules across 10 domains) that the current REPL prompt doesn't reference. The most important ones should be folded into the system prompt or the catcher validators.
   - The xlsx specifies boolean-only antecedents, named literals (no raw numerics), and the `Boolean_Literals.csv` spec (sheet `PR_B5b_make_v23`). The current REPL prompt already requires boolean-only inputs but doesn't have the literal naming convention.
   - The xlsx's B7 integrative specification (referral prioritization, pre-referral care, treatment de-duplication, dose/interaction management, follow-up prioritization, safety checks) is more detailed than the current REPL Phase 4 instructions. Incorporate the specifics.
   - The xlsx's D1b synthetic patient generation spec (T=3 modules, t=2 intra, boundary triplets) should inform the gate harness, not the extraction prompt.

3. **Do NOT change the fundamental architecture.** The REPL pipeline must remain a single continuous session with one system prompt. Adding more prompts or breaking it into separate calls would defeat the purpose of the A/B test.

4. **API configuration:** Same as sequential. Same model, same temperature, same provider abstraction.

### 2.3 Gate Harness (`harness/`)

Build a comparison harness that operates AFTER both pipelines have finished. The harness never runs during extraction. It is a post-hoc measurement tool.

1. **Collects intermediate artifacts** from both pipelines at each aligned checkpoint (read from disk/database)

2. **Runs the reliability test:** Compare outputs across runs WITHIN each pipeline (test-retest reliability) and BETWEEN pipelines (convergent validity). This is purely structural and can run without a patient set.

3. **Synthetic patient comparison (runs when patient set is provided):**
   - Accept a patient set as CSV or XLSX upload (either clinician-validated or algorithmically generated)
   - The patient set is an INPUT TO THE HARNESS, not to the pipelines. The pipelines never see it.
   - Run each patient through all DMN outputs (3 sequential runs + 3 REPL runs = 6 DMNs)
   - Compare clinical outcomes: diagnosis, referral level, treatment plan, follow-up days
   - Report: which patients got different outcomes, and what specifically differed
   - This is the PRIMARY reliability metric

4. **Text-level comparison (secondary diagnostic, no patient set needed):**
   - Jaccard similarity on variable names across runs
   - Jaccard similarity on predicate IDs across runs  
   - Structural diff on module decision tables
   - These are SECONDARY. They help diagnose WHY clinical outcomes differ, but they are not the success metric.

5. **Report output:** A single comparison report showing:
   - Per-pipeline test-retest reliability (3 runs each)
   - Cross-pipeline clinical equivalence (if patient set provided)
   - Per-checkpoint artifact diffs
   - Patient-level outcome comparison (if patient set provided)
   - Cost per run (API tokens used)
   - Wall-clock time per run

### 2.4 Patient Set Input (`harness/patients/`)

The gate harness accepts a patient set as input. It does NOT generate patients. The A/B test only takes this information in, nothing else.

The patient set is a CSV or XLSX file uploaded after both pipelines have finished. Each row is one patient. Columns are variable names, values are the patient's clinical data. The file comes from one of two sources, both external to the arena:

1. **Clinician-validated patient set** (preferred): Created and approved by the clinical team. Each patient has a known correct diagnosis, referral level, treatment, and follow-up. This is the gold standard.

2. **PatientBot/GraderBot-generated patient set** (fallback): Created algorithmically by our existing bot infrastructure. Sufficient for test-retest reliability but not for clinical validity.

Either way, the arena does not care where the patient set came from. It takes in the CSV, runs each patient through both pipelines' DMNs, and reports which patients got different clinical outcomes. The patient set is an input to the harness, never an output of it.

### 2.5 DMN Execution Engine (`harness/dmn_executor/`)

This is the hardest engineering problem in the harness and it needs to be addressed explicitly: **how do you actually run a patient through a DMN?**

A "patient" is a row in a CSV/XLSX: columns are variable names, values are the patient's data (age=8mo, fever=true, chest_indrawing=false, etc.). A "DMN" is the JSON artifact each pipeline produces: decision tables with boolean inputs, hit policies, and output actions. "Running a patient through a DMN" means: evaluate the patient's values against the DMN's predicates, fire the matching rules, and produce clinical outputs (diagnosis, referral level, treatment, follow-up days).

**This execution must be fully deterministic.** No LLM calls. No randomness. Given the same patient row and the same DMN JSON, the executor produces the same clinical output every time. If it doesn't, the executor is broken, not the pipeline.

**The column mapping problem is the critical engineering challenge.** The two pipelines may produce different variable names for the same clinical concept. Pipeline A might call it `v_temp_c`, Pipeline B might call it `temperature_celsius`. The patient CSV has its own column names. If the executor does naive column matching, every naming difference looks like a clinical difference. This is the "column drift" problem.

The executor must handle this in layers:

1. **Canonical variable registry.** Before running patients, build a mapping from each pipeline's variable names to a shared canonical namespace. This can be done automatically by matching on `source_quote` and `source_section_id` (both pipelines cite the same manual), or semi-automatically by fuzzy-matching on display names. The mapping must be reviewed and frozen before any patient runs. This is a one-time cost per A/B comparison.

2. **Patient CSV adaptation.** The patient CSV uses the canonical namespace. The executor translates canonical column names to each pipeline's variable names using the mapping from step 1. If a pipeline uses a variable name that has no canonical mapping, that variable is flagged as "unmapped" and the executor reports it (this is itself diagnostic data about the pipeline).

3. **Predicate evaluation.** The executor reads the pipeline's `predicates` artifact, evaluates each predicate against the patient's translated values, and produces a set of boolean flags. This is pure arithmetic: `v_temp_c >= 38.0` -> `p_high_fever = true`. No interpretation. No LLM.

4. **Table traversal.** The executor walks the DMN tables in order (activator -> router -> per-module -> integrative) using the evaluated predicates. For FIRST-policy tables, it scans top to bottom and fires the first matching rule. For COLLECT-policy tables, it fires all matching rules. For UNIQUE-policy tables, it verifies exactly one rule matches (and flags an error if zero or multiple match). This is a deterministic state machine.

5. **Output normalization.** The executor normalizes clinical outputs to a canonical format before comparison: diagnosis codes sorted alphabetically, referral level mapped to an ordinal (none < routine < urgent), treatments sorted alphabetically, follow-up days as integer. This prevents false positives from ordering differences (`tx.ORS tx.zinc` vs `tx.zinc tx.ORS` are the same, as Levine noted).

6. **Comparison.** After running the same patient through both pipelines' DMNs, compare the normalized outputs. Report: same/different for each of (diagnosis, referral_level, treatments, follow_up_days). A "clinical difference" is any difference in diagnosis OR referral_level. Treatment ordering differences and follow-up day ties (e.g., 2 vs 3 days) are flagged as warnings, not failures.

**What if the two pipelines produce structurally incompatible DMNs?** For example, Pipeline A has 5 modules and Pipeline B has 7 (because it split one module into sub-modules). The executor must still run patients through both. The canonical variable mapping handles input-side differences. Output-side differences (different module granularity, different diagnosis codes) are handled by the output normalization layer: map both pipelines' diagnosis codes to the manual's original classification categories. If the manual says "severe pneumonia," both pipelines should produce an output that maps to "severe pneumonia" regardless of what they internally called it.

**The executor is the most important piece of infrastructure in this project.** Without it, we're back to eyeballing JSON diffs. With it, we can run any patient set (clinician-provided or algorithmically generated) through any DMN (from either pipeline, from any run) and get a deterministic clinical comparison in seconds.

---

## 3. ARTIFACT ALIGNMENT TABLE

Both pipelines must emit artifacts at these checkpoints with these names. The sequential pipeline maps its xlsx stages to these names. The REPL pipeline already uses these names in `emit_artifact()`.

| Checkpoint | REPL artifact name | Sequential stage(s) | Content |
|---|---|---|---|
| 1 | `supply_list` | B1_make (partial) | Physical inventory: equipment + consumables with `equip_` and `supply_` prefixes |
| 2 | `variables` | B1_make + B2_make | Runtime inputs: symptoms, findings, vitals, labs with `q_`, `ex_`, `v_`, `lab_` prefixes. Includes `depends_on_supply` cross-refs. |
| 3 | `predicates` | B4_make + B5b_make + B6_make (partial) | Boolean predicates with threshold expressions, fail-safe values, source quotes. Named literals (no raw numerics). |
| 4 | `modules` | B6_make (DMN tables) | Per-module decision tables: gather + classify. Hit policies FIRST/UNIQUE/COLLECT. All inputs boolean. |
| 5 | `router` | B6_make (workflow) + B9_make (compiler, partial) | Activator (COLLECT) + Router (FIRST). Emergency short-circuit row 0. Module queue management. |
| 6 | `integrative` | B7 + B3_make | Consolidation: referral priority, pre-referral care, treatment de-duplication, follow-up merge, safety checks. |
| 7 | `phrase_bank` | F1_make | English phrase bank: questions, diagnoses, treatments, advice, referral messages, follow-up instructions. |

**For the sequential pipeline:** After each mapped stage completes (post red-team, post-repair), save the output with the artifact name from this table. This is the comparison point.

**For the REPL pipeline:** The `emit_artifact` calls already use these names. No change needed.

---

## 4. WHAT TO DO WITH THE XLSX

The xlsx `Automating_CHT_v25.xlsx` has these sheets. Here's what each one is for:

### Prompt sheets (build sequential pipeline from these):
- `A2 reviewmanual` -- A2_make: Clinical rough-edge audit. **SKIP for A/B test.**
- `A2 repair` -- A3_make: Draft manual repairs. **SKIP for A/B test.**
- `b1,b2,b3,b4,b6` -- Summary of B-phase stages (B1 through B8). Contains role, context, objectives, inputs, outputs, tasks for each. **This is the extraction core.**
- `PR_B5b_make_v23` -- Boolean literals specification. Detailed prompt + unit tests + the FactSheet authoring prompt.
- `PR_B6_make_v23` -- Rule decomposition + DMN authoring prompt. Detailed prompt + unit tests.
- `b7 DMN Integrative` -- Integrative specification (7 consolidation categories).
- `makecompiler` -- B9_make: DMN-to-XLSForm compiler + Workflow Engine DMN. **Important for understanding the output format but the compiler itself is a deterministic code step, not an LLM call.**
- `C1 make xlsform` -- C1_make: XLSForm assembly. **Downstream of DMN extraction. Include if time permits, skip for initial A/B.**
- `PR_C2_make_v23` -- C2_make: Clinical logic + pyxform validation. **Downstream.**
- `PR_D1b_make_v23` -- Synthetic patient generation spec. **Use for building the patient generator in the gate harness, NOT as an LLM prompt.**
- `PR_D1c_make_v23` -- Robot execution + coverage spec. **Use for building the gate harness.**
- `PR_D4_make_v23` -- End-to-end equivalence testing. **Use for gate harness.**
- `E1 Human to JSON` -- Human feedback ingestion. **SKIP for A/B test.**
- `F1 phrase bank` -- English phrase bank generation. **Include as final extraction step.**
- `F2 translate` -- Translation. **SKIP for A/B test.**
- `PR_F3b_make_v23` -- Translation QA. **SKIP for A/B test.**
- `F4 redteam` -- Cultural/comprehension review + PhraseBank prompt. **SKIP for A/B test.**
- `G governance` -- Governance log. **SKIP for A/B test.**

### Supporting sheets (reference material):
- `Quality stds` -- 160 quality rules across 10 domains. **Fold the most critical ones into both pipelines' validation layers.** The sequential pipeline uses these as red team criteria. The REPL pipeline should reference them in catcher validators.
- `Software_Libraries_v23` -- Tool mappings (pyDMNrules, pyxform, Z3, etc.). **Use to determine which validators to implement.**
- `WriteOnce_Code_v23` -- 5 reusable scripts. **Reference for orchestrator design.**
- `orchestrator_pseudo` -- 108 lines of pseudocode for the DAG runner. **Primary reference for building the sequential orchestrator.**
- `YAML DAG Schema for orch` -- Full pipeline definition. **Primary reference for stage dependencies and contracts.**
- `ChangeLog_v24` -- Version notes. **Informational only.**

---

## 5. KEY ARCHITECTURAL TRADEOFFS (FROM CHAT HISTORY)

These are the arguments that have been made in the email thread with Professor Levine. Both sides have merit. The A/B test resolves them empirically.

### 5.1 Context visibility vs isolation

**Sequential argument (Levine):** Isolation makes diagnosis easier. If B6 fails, you know it's B6's prompt or B6's inputs. You don't have to untangle 200K tokens of REPL history to find the bug.

**REPL argument (Atharva):** Full context means the model can self-correct across phase boundaries. If predicate extraction reveals a missing variable, the model fixes it immediately. In sequential, that fix requires restarting from the upstream stage with a fresh prompt that has no memory of why the fix was needed.

**Resolution:** The A/B test measures which produces better outputs. Both arguments are plausible. Evidence decides.

### 5.2 Per-phase red team vs end-to-end catchers

**Sequential:** Red team at every phase catches errors early. An error in B1 is caught by B1's red team before B2 ever sees it.

**REPL:** Catchers run at every emit_artifact checkpoint (same logical locations). The difference is that catchers are frozen validators with fixed criteria, while the sequential red team is an LLM call that can be creative about what it checks. Frozen catchers have zero degrees of freedom (good for reproducibility). LLM red teams have high degrees of freedom (good for catching novel issues, bad for reproducibility).

**Resolution:** The A/B test measures whether per-phase LLM red teaming catches more issues than frozen catchers, and whether the additional variance from LLM red teams hurts or helps reliability.

### 5.3 Prompt count and degrees of freedom

**Sequential:** ~45 prompts. Each is a tunable variable. When the system fails, you can edit one prompt without touching the others. But: when you edit one prompt, it can contaminate downstream prompts' inputs. You cannot hold the other 44 constant while observing the effect of the one you changed.

**REPL:** 1 system prompt + catcher suite. When you edit the system prompt, you know exactly what changed: one document. But: you can't isolate "just fix the predicate extraction" without risking changes to other phases.

**Resolution:** The A/B test measures end-to-end output quality. If the REPL produces better DMNs with 1 prompt than sequential does with 45, the additional tunability of 45 prompts is a liability, not an asset.

### 5.4 The "can't go back" problem

**Sequential:** Once the supply list passes its red team gate, it is frozen. If Phase 6 discovers the supply list is wrong, the orchestrator must restart from Phase 2. The pseudocode handles this (`PROPAGATE_FIX_NEEDED`), but it means re-running upstream stages from scratch with new prompts that have no memory of the original reasoning.

**REPL:** The model edits the supply list variable in the REPL, re-emits the artifact, catchers re-validate, and it continues. No re-prompting. The original reasoning is still in context.

**Resolution:** If backward fixes are common (likely in healthcare extraction), the REPL has a structural advantage. If backward fixes are rare, the difference doesn't matter. The A/B test will reveal how often backward fixes are needed by counting how many times each pipeline restarts/re-emits.

---

## 6. INFRASTRUCTURE REQUIREMENTS

### 6.1 Render upgrade

**IMPORTANT:** Before running the A/B test, upgrade Render from 2GB to 4GB RAM. This makes the test blazing fast and allows both pipelines to run in parallel with no issues. The sequential pipeline makes many concurrent API calls, and the REPL pipeline runs a long continuous session with accumulated state. 4GB gives both plenty of headroom to run simultaneously.

Specifically:
- Upgrade RAM from 2GB to 4GB on Render
- Ensure request timeout is >= 15 minutes (REPL sessions can run 5-10 minutes)
- Disable auto-sleep/scale-to-zero during test runs

### 6.2 Model Parity and Parallelization

**Model parity is non-negotiable.** Both pipelines must use the exact same model family and model type for all LLM calls. If the REPL uses `claude-sonnet-4-20250514`, the sequential pipeline uses `claude-sonnet-4-20250514` for maker, red team, AND repair calls. No mixing models between pipelines. No using a cheaper model for red team in one pipeline but not the other. The comparison is architecture vs architecture, not model vs model.

**Parallelization strategy:**

The two pipelines run in parallel WITH EACH OTHER. Launch both simultaneously. They are completely independent: different API sessions, different artifact directories, no shared state.

```
[Sequential Run 1] ──────────────────────────> [done]
[Sequential Run 2] ──────────────────────────> [done]
[Sequential Run 3] ──────────────────────────> [done]
[REPL Run 1]       ──────────────────────────> [done]
[REPL Run 2]       ──────────────────────────> [done]
[REPL Run 3]       ──────────────────────────> [done]
                                                  ↓
                                          [upload patients]
                                                  ↓
                                          [gate harness]
```

All 6 runs can launch at the same time if the Render instance and API rate limits allow it. At minimum, run one sequential and one REPL in parallel.

**Within each pipeline, the constraints are different:**

- **Sequential pipeline is inherently serial within a run.** B2 depends on B1's output. B6 depends on B4's output. You cannot parallelize the stages within a single sequential run. However, you CAN and SHOULD make each individual API call as fast as possible: stream responses, use the fastest available API endpoint, minimize prompt overhead, don't add artificial delays between stages. The sequential pipeline's wall-clock time is the sum of all its API calls. Make each one fast.

- **REPL pipeline is a single continuous session.** It's one long API call with tool use. It can parallelize internally via `llm_query_batched` (e.g., extracting all modules in parallel), but the session itself is one thread.

**Speed matters for the comparison.** One of the metrics in the gate harness report is wall-clock time per run. If the sequential pipeline takes 12 minutes and the REPL takes 4 minutes to produce equivalent-quality DMNs, that's decision-relevant information.

### 6.3 Database

Both pipelines persist artifacts to Neon Postgres. The schema should include:
- `run_id` (UUID): unique per pipeline run
- `pipeline_type`: "sequential" or "repl"
- `artifact_name`: from the alignment table
- `artifact_value`: JSON
- `emitted_at`: timestamp
- `run_number`: 1, 2, or 3 (for test-retest)
- `api_tokens_used`: total tokens for this artifact
- `wall_clock_ms`: time to produce this artifact

### 6.4 Cost budget

One clean A/B test: one sequential run, one REPL run, same input, compare outputs.

| Component | Estimated cost |
|---|---|
| PDF-to-JSON conversion (Unstructured API or local `unstructured` library) | $0 if local, ~$1-2 if using Unstructured hosted API |
| Sequential pipeline (one run: ~15 chained API calls with red team + repair loops) | ~$3-8 |
| REPL pipeline (one run: one long session with sub-calls) | ~$5-10 |
| Gate harness (deterministic computation, no API calls) | $0 |
| **Total for one A/B comparison** | **~$10-20** |

If we later want to measure test-retest reliability (run each pipeline 3x and compare within-pipeline variance), that's 3x the pipeline cost, bringing total to ~$30-60. But the first milestone is one clean head-to-head comparison.

### 6.5 API keys

Both pipelines use the same BYOK pattern:
```
ANTHROPIC_API_KEY=sk-...
GOOGLE_API_KEY=...  # optional, for Gemini comparison later
```

### 6.6 Input Pipeline

The input is NOT the raw PDF. Both pipelines receive the same pre-processed JSON.

**Step 0 (before either pipeline runs):** Convert the WHO 2012 PDF to structured JSON using an extraction library (e.g., `unstructured`, `pymupdf`, or equivalent). This JSON becomes the `guide` variable. It should contain sections, page references, text content, and any table structures the library can extract.

```
PDF -> json_converter.py -> guide.json -> [both pipelines receive guide.json]
```

This is critical for the A/B test: if each pipeline did its own PDF parsing, differences in PDF extraction would confound the architecture comparison. Both pipelines must see byte-identical input.

The REPL pipeline already expects a `guide` JSON variable. The sequential pipeline needs to receive the same JSON as the grounding context in each maker prompt.

Path: `inputs/who_2012_guide.json` (generated once, used by both pipelines)

### 6.7 Guide JSON Management

Once a PDF has been parsed to JSON, it is stored in the database (hashed by content) and can be reused without re-parsing. The UI should support two flows:

1. **Upload new PDF:** User uploads a PDF, the converter produces a JSON, the JSON is content-hashed and stored in Neon. This becomes a selectable guide for future runs.

2. **Select existing guide:** User picks from a list of previously parsed guides. No re-parsing needed. The harness loads the stored JSON by hash.

**Duplicate handling:** If a user uploads the same PDF again, the system versions it with a timestamp rather than deduplicating. The reason: we may change the JSON extraction logic (different `unstructured` settings, better table parsing, etc.) and want to test whether a new extraction produces different pipeline outputs from the old extraction on the same PDF. Each version is stored as a separate entry:

```
who_2012_guide_2026-04-09T14:30:00.json  (unstructured v0.15, default settings)
who_2012_guide_2026-04-12T09:00:00.json  (unstructured v0.16, table extraction enabled)
```

Both are selectable. The gate harness can compare pipeline outputs across guide versions to isolate whether a difference is caused by the extraction architecture or by the JSON conversion step.

**Database schema for guides:**

| Column | Type | Description |
|---|---|---|
| `guide_id` | UUID | Primary key |
| `source_pdf_name` | text | Original filename |
| `content_hash` | text | SHA-256 of the JSON content |
| `converter_version` | text | Version of the PDF-to-JSON converter used |
| `converter_settings` | jsonb | Settings/config used for conversion |
| `created_at` | timestamp | When this version was created |
| `guide_json` | jsonb | The actual parsed content |

Each pipeline run references a `guide_id`, so the full provenance chain is: PDF -> converter version -> guide JSON -> pipeline type -> artifacts -> DMN.

---

## 7. WHAT WE DON'T HAVE YET (AND WHAT TO DO ABOUT IT)

### 7.1 Clinician-validated patient sets

We don't have them. The strong base is 2-3 guides with 500-1000 clinically approved patients per guide. This requires clinical effort from the medical team. We've asked Professor Levine for a starting set.

**For now:** Use PatientBot/GraderBot to create a synthetic patient set, or wait for a clinician-approved set from Levine's team. Either way, the patient set is created externally and uploaded to the harness as a CSV. The harness does not generate patients.

**The chicken-and-egg problem:** If generating a comprehensive patient set for each manual requires the same clinical expertise as making the DMN by hand, the pipeline doesn't save time. The patient set must be either (a) small enough to create quickly, (b) reusable across manuals, or (c) algorithmically generated from the DMN structure. Option (c) has the limitation that you're testing one stochastic system with a deterministic derivative of its own output, but it still catches non-determinism (the primary failure mode we're measuring).

**For level (i) without clinician input:** We can use the existing PatientBot/GraderBot infrastructure to generate and grade synthetic patients algorithmically from the DMN structure. This is sufficient for measuring test-retest reliability. If a clinician-approved version of the patient set becomes available, it replaces the bot-generated one as the primary test set.

### 7.2 Z3/SMT solver integration

The REPL pipeline calls `z3_check()` for exhaustiveness proofs. This is referenced but may not be fully implemented. Check `backend/validators/` for the current state. If not implemented, stub it out for the A/B test and implement after.

### 7.3 pyDMNrules / DMN linter

The sequential pipeline references `pyDMNrules` and a DMN linter for validation. Check if these are available as pip packages. If not, use the structural validators from the REPL catcher suite for both pipelines.

---

## 8. RUNNING THE TEST

### The flow is three distinct stages, not one script:

```
STAGE 1: Prepare input (run once)
  PDF -> json_converter.py -> guide.json

STAGE 2: Run both pipelines (run 3x each)
  guide.json -> sequential pipeline -> artifacts + DMN
  guide.json -> REPL pipeline -> artifacts + DMN
  (these can run in parallel, they are independent)

STAGE 3: Test (run AFTER pipelines complete, AFTER patient set uploaded)
  Upload patient_set.csv or patient_set.xlsx
  -> gate harness runs patients through all 6 DMN outputs
  -> comparison report
```

**IMPORTANT:** The patient set is NOT an input to the pipelines. It is uploaded AFTER both pipelines have finished and produced their DMN artifacts. The pipelines never see the patient set. The patient set is the measurement instrument, not part of the treatment.

If no clinician-validated patient set is available yet, use PatientBot/GraderBot output as the input CSV. The preferred path is: pipelines finish, we send DMN outputs to the clinical team, they review and provide a patient set, we upload it and run the comparison.

### Step 1: Scaffold
```
claude code  # with this file + xlsx + system_prompt.md as context
# It builds: sequential/, repl/, harness/, converter/
```

### Step 2: Convert PDF to JSON
```
python converter/pdf_to_json.py inputs/who_2012.pdf -o inputs/who_2012_guide.json
```

### Step 3: Upgrade Render
```
# Manual step: upgrade Render from 2GB to 4GB RAM
# Set timeout >= 15min
# Disable auto-sleep during test runs
```

### Step 4: Configure
```
cp .env.example .env
# Add API keys
# Set GUIDE_JSON_PATH=inputs/who_2012_guide.json
```

### Step 5: Run pipelines
```
python harness/run_arena.py --runs 3 --output results/
```

This runs:
1. Sequential pipeline 3 times on guide.json
2. REPL pipeline 3 times on guide.json (parallel with sequential)
3. Saves all intermediate artifacts + final DMNs to results/

### Step 6: Upload patient set and test
```
# Upload clinician-validated or PatientBot-generated patient set
python harness/test_patients.py results/ --patients patient_set.csv
```

### Step 7: Report
```
python harness/report.py results/
```

Outputs:
- Test-retest reliability per pipeline (do 3 runs of the same pipeline agree?)
- Cross-pipeline comparison (do the two pipelines agree?)
- Per-patient clinical outcome diffs
- Cost and timing comparison

---

## 9. SUCCESS CRITERIA

The A/B test has a pre-registered decision mechanism:

1. **If REPL wins or ties on all three metrics** (test-retest reliability, clinical equivalence rate, validator pass rate): use REPL for the CHT deliverable.

2. **If sequential wins on all three:** use sequential.

3. **If split:** use whichever has higher clinical equivalence rate (patient outcomes), since that's the metric that matters for patient safety.

4. **If both fail** (neither achieves >90% test-retest reliability): the problem is harder than either architecture can solve alone, and we need to revisit the approach entirely.

**"Win"** on test-retest = higher percentage of synthetic patients producing identical clinical outcomes across 3 runs of the same pipeline.

**"Win"** on clinical equivalence = lower number of patients producing different clinical outcomes when comparing the best run from each pipeline.

**"Win"** on validator pass rate = fewer critical issues flagged by the shared validator suite on the final artifact.

---

## APPENDIX A: CURRENT REPL SYSTEM PROMPT

The full system prompt is in `system_prompt.md` (separate file). Key sections:

1. Role: Knowledge Engineer
2. Role: DMN Architect  
3. Identification Frame (the model is the treatment variable in a reliability study)
4. REPL Environment Instructions (guide, llm_query, validate, z3_check, emit_artifact)
5. Variable Naming Codebook (equip_, supply_, q_, ex_, v_, lab_, etc.)
6. Predicate Convention (boolean-only, fail-safe, thresholds)
7. Logic Integrity Standards
8. Safe Endpoint Standards
9. Missingness Model
10. Anti-Gravity Data Standards (information only flows from manual to output)
11. Queue Management (Activator COLLECT, Router FIRST)
12. DMN Subset (FIRST, COLLECT, UNIQUE only; boolean inputs only)
13. Required Extraction Strategy (7 phases, 7 checkpoints)
14. Safety Footer

## APPENDIX B: SEQUENTIAL PIPELINE STAGE MAP

From `Automating_CHT_v25.xlsx`:

```
Phase A: Manual Prep [SKIP FOR A/B TEST]
  A2_make -> A3_make -> Human sign-off

Phase B: Extraction [CORE -- BUILD THIS]
  B1_make (facts + workflow)
    -> B2_make (context vars + complaint mappings)
    -> B3_make (consolidation policies)
    -> B4_make (enrich FactSheet)
    -> B5b_make (boolean literals)
    -> B6_make (rule decomposition + DMN tables)
    -> B7 (integrative spec)
    -> B8_redteam (data model integrity audit)
    -> B9_make (DMN-to-XLSForm compiler)

Phase C: XLSForm [DOWNSTREAM -- SKIP FOR INITIAL A/B]
  C1_make -> C2_make

Phase D: Testing [USE FOR GATE HARNESS DESIGN]
  D1b_make (synthetic patients) -> D1c_make (robot execution) -> D4_make (e2e equivalence)

Phase E: Human Feedback [SKIP]
Phase F: Phrase Bank + Translation [INCLUDE F1 ONLY]
Phase G: Governance [SKIP]
```

## APPENDIX C: QUALITY STANDARDS SUMMARY

From the `Quality stds` sheet, the most critical rules for the A/B test:

**Decision Logic (must-pass for both pipelines):**
- MECE decision tables (mutually exclusive, collectively exhaustive)
- Path termination (every path ends at refer/treat/observe/no_diagnosis)
- Endpoint reachability (each endpoint reachable by at least one input)
- Safe 'no diagnosis' endpoint exists
- No forward references (rules depend only on previously established observations)
- Danger-sign monotonicity (worse findings never downgrade care)
- Deterministic behavior (identical inputs always produce identical outputs)
- Else/default rows present in every table

**Observations (must-pass):**
- Observation schema (id, label, type, units, help_text)
- Re-measurement rules for implausible values
- Inconclusive handling with defined fallback

**Supplies (must-pass):**
- Stock-out backup for every required supply
- Bidirectional observation-supply link

**Integration (must-pass):**
- Safety checks (e.g., no oral meds if unconscious)
- Cross-module interaction check
- Pre-referral care consolidation
- Medicine prioritization for largest dose
- Follow-up merge (earliest dominates)

**Testing (gate harness implements these):**
- Golden test per rule row
- Multiple diagnosis cases
- Edge cases on all boundaries
- Consistency (same input -> same output)
- Coverage metric (% of rules exercised by tests)
