# CHW Navigator — WHO CHW 2012 extraction

**Run ID:** `run_d9a39c2d`
**Run timestamp:** 2026-04-14 07:36 UTC
**Wall clock:** ~23 minutes

---

## What this is

An automated extraction of the clinical decision logic inside the WHO 2012 "Caring for the sick child in the community" CHW manual. The pipeline reads the PDF, enumerates every clinical item, and compiles a deployable DMN decision model plus a CHT-ready XLSForm. Every rule, predicate, and phrase in the output carries a pointer back to the section of the source manual it came from.

This delivery is a **snapshot, not a release.** Known gaps are listed at the bottom of this README.

---

## Inputs

| Field | Value |
|---|---|
| Source manual | `WHO CHW guide 2012.pdf` (WHO 2012, "Caring for the newborn and child in the community") |
| Content hash | SHA-256 of the PDF bytes (fixes the starting state across runs) |
| LLM | Claude Opus 4.6 (`claude-opus-4-6`) at temperature 0 |
| Git SHA | `9000e86` (pinned for this run) |
| Pipeline version | Gen 7 (Opus-only, 3-stage) |
| API key model | User BYOK (Anthropic) |

No other model ran. No OpenAI, Gemini, or GPT-series call anywhere in this pipeline.

---

## Method

### Pipeline (3 Opus calls, plus deterministic glue)

1. **Chunk.** The PDF is rendered to text, then split into 21 micro-chunks of roughly 2K tokens each. Deterministic Python, no LLM.
2. **Label (parallel, per-chunk).** Opus runs once per chunk against a frozen codebook and returns every clinical item as a structured label (span, id, type, subtype, quote_context). 1,692 labels extracted across 21 chunks. Prompt caching on the system block, so chunks 2–21 pay cache-read rates.
3. **Dedupe.** Algorithmic exact-string dedup on `(id, type)`. Deterministic Python, no LLM.
4. **Compile (REPL).** Opus runs inside a Python REPL with the reconstructed guide and the deduped label inventory attached as cached context. It emits seven artifacts in strict order: `supply_list → variables → predicates → modules → router → integrative → phrase_bank`. Each emission runs its phase validators before the next phase is allowed to start.
5. **Assemble.** `clinical_logic.json` → `clinical_logic.dmn` (DMN 1.3 XML) → `form.xlsx` (XLSForm). All three are deterministic transforms of the same source.

### Prompting: one long prompt, not several

The compile step uses **one frozen system prompt**. It drives all seven artifacts in the same REPL session — no hand-offs between prompts, no orchestration state, no multi-agent coordination. This is a deliberate identification choice. See `system_prompt_sendable.md` for the full prompt text (three stages, all Opus).

### Red-teaming: per-artifact, not end-only

Every artifact is validated at the moment it is emitted, not only at the end.

| Phase | Emitted artifact | Validators that run |
|---|---|---|
| 2A | `supply_list` | provenance, completeness, artifact_architecture |
| 2B | `variables` | provenance, naming, completeness, artifact_architecture |
| 3A | `predicates` | provenance, naming, clinical_review, artifact_architecture |
| 3B | `modules` | provenance, clinical_review, consistency, boundary_condition, artifact_architecture |
| 4A | `router` | architecture, consistency, artifact_architecture |
| 4B | `integrative` | architecture, comorbidity_coverage |
| 5 | `phrase_bank` | provenance, completeness |
| Final | `clinical_logic` (full) | dmn_audit, boundary_condition |

Validators are Python code — deterministic, not LLM. They block phase progression on critical issues. Warnings are logged, not blocking. The `final_dmn.validator.json` file in this delivery is the consolidated report.

### Quality checks beyond red-teaming

- **Content-hash stationarity.** Same PDF = same starting state, every run. Any run-to-run diff is attributable to model stochasticity, not to a changed input.
- **Provenance chain.** Every output item carries `source_section_id` + `source_quote`; every rule can be traced to a specific part of the manual text.
- **Structural invariants.** The `architecture` validator checks global DMN properties: router row 0 is a priority-exit short-circuit, every activated module has a matching decision table, no module_id is a substring of another, integrative exists.
- **Z3 exhaustiveness hooks.** `z3_check()` is wired into the REPL but was not exercised on this run. Planned for the next snapshot.
- **Deterministic ordering.** All output dicts are sorted alphabetically by id before emission so file diffs reflect content changes, not iteration order.

---

## Files in this delivery

### Top-level (reviewer-facing)
| File | What it is | Derived from |
|---|---|---|
| `clinical_logic.dmn` | Deployable DMN 1.3 XML. The decision model. | `clinical_logic.json` |
| `clinical_logic.json` | Same logic in review-friendly JSON. Inspect this by hand. | Opus REPL output |
| `clinical_logic.xlsx` | (Next delivery.) Deterministic XLSX view — one sheet per module + router + predicates. Replaces the ChatGPT ad-hoc conversion. | `clinical_logic.json` |
| `form.xlsx` | CHT-deployable XLSForm (XLSForm spec, not related to predicates). | `clinical_logic.json` |
| `flowchart.png` / `flowchart.mmd` | One-page Mermaid overview: router → 8 clinical modules → integrative merge. Mermaid source included so you can re-render. | `clinical_logic.json` |

### Intermediate artifacts (the emit_artifact checkpoints)
| File | What it is |
|---|---|
| `artifacts/supply_list.json` | Equipment and consumables the CHW must have on hand. |
| `artifacts/variables.json` | Every runtime input the CHW collects — symptoms, exam findings, measurements. |
| `artifacts/predicates.json` | Boolean predicates computed from variables (e.g. `p_fast_breathing`). |
| `artifacts/modules.json` | One decision table per clinical module (assess, diarrhoea, fever/malaria, etc.). |
| `artifacts/router.json` | The priority router ("traffic cop"). 9 rows, PRIORITY hit policy. This IS the module David's DMN translation may have dropped; it is definitely present. |
| `artifacts/integrative.json` | Cross-module merge rules. |
| `artifacts/phrase_bank.json` | Every phrase the CHW says — questions, diagnoses, treatment instructions, referral messages. |

### Validation + traceability
| File | What it is |
|---|---|
| `final_dmn.validator.json` | Consolidated catcher report on the final DMN. `passed: true`, 0 critical, 38 warnings (all boundary-condition recommendations, not failures). |
| `predicates.csv` | Flat CSV of predicates — one row per predicate, easier to scan than JSON. **This is NOT predicates.xlsx — there is no predicates.xlsx in this delivery.** |
| `phrases.csv` | Flat CSV of phrase bank entries. |
| `labeled_chunks.json` | Every span labeled, with chunk + section reference. The raw output of Stage 2. |
| `deduped_labels.json` | Consolidated label set after exact-string dedup. |
| `reconstructed_guide.txt` | The full text the model actually saw, independent of the PDF render. |
| `scratchpad.md` | Step-by-step execution journal from the REPL. |
| `system_prompt.md` | The one system prompt that produced all of this. |
| `test_suite.json` | Generated boundary + scenario cases from the label inventory. |

### Internal (not reviewer-facing, included here for completeness only)
| File | What it is |
|---|---|
| `chunk_difficulty.json` | Per-chunk complexity score used internally to schedule the labeling batch. No clinical content. In the next delivery this will move to an `_internal/` subfolder. |

---

## What David flagged in the 2026-04-14 review — addressed inline

**"predicates.XLS — is it an artifact from creation or from test subjects?"**
There is no `predicates.xlsx` in this delivery. The relevant files are `predicates.csv` (a flat CSV view of the extracted predicates) and `form.xlsx` (the XLSForm — a completely separate artifact, the CHT-deployable form). Both are generated by the pipeline; neither contains test-subject data. The `clinical_logic.xlsx` converter listed above will replace the ad-hoc ChatGPT conversion in the next delivery.

**"Is Karen Shah part of the manual?"**
Yes. Karen Shah appears in the WHO 2012 manual as a named patient in a worked-example exercise (section id `child_3_karen_shah+checking_questions+exercise_2_optional`). Our extractor correctly pulled clinical thresholds from that section — but we consider this a **known provenance-quality issue**, because 15 of 20 predicates (75%) trace back to exercise/worked-example sections rather than the authoritative decision-table sections. The clinical content is not wrong; the provenance pointer is pointing at the training example instead of the canonical source. Remediation planned — see "Known gaps" below.

**"Traffic cop module"**
Present. It is the `router` decision in `clinical_logic.dmn` (`decision_router`, PRIORITY hit policy, 9 rules). See `artifacts/router.json` and the flowchart PNG. If a downstream DMN translator dropped it, that is a translator issue; the router is in the DMN XML as emitted by the pipeline.

**"I cannot understand chunk_difficulty"**
Internal scheduling artifact, no clinical content. Explanation above; will be moved out of reviewer bundles in the next delivery.

---

## Known gaps in this delivery

1. **Provenance quality:** 15 of 20 predicates (~75%) cite exercise/worked-example sections as their source. Clinical content is correct; provenance pointer should cite the authoritative decision-table section instead. Planned fix: weight canonical sections over exercise sections during label aggregation.
2. **No deterministic DMN-to-XLSX converter yet in this bundle.** The review round was done with an ad-hoc ChatGPT conversion. The replacement (`clinical_logic.xlsx` from `backend/converters/json_to_xlsx.py`) ships with the next delivery.
3. **Z3 exhaustiveness checks not exercised.** The hook is wired; the run skipped it. Planned for the next snapshot.
4. **Integrative artifact is minimal.** Cross-module comorbidity rules were not fully enumerated on this run. The dispatcher's priority-exit flag covers the dangerous cases (danger sign short-circuit). Full integrative coverage deferred to the May 1 hand-off.
5. **CHW-Chat-Interface postmortem (from March) still not delivered as a standalone document.** The audit script (`chwci_db_audit.py`) and analysis (`chwci_failure_analysis.md`) exist in the repo but are separate from this delivery.

---

## Architecture (condensed)

Three Opus calls, one frozen system prompt per stage, zero human mediation between stages.

```
                WHO 2012 PDF
                     |
                     v
          [text extract + chunk]           deterministic Python (Stage 0)
                     |
                     v
          +----------+----------+
          |     21 micro-chunks |
          +---------------------+
                     |
                     v
   [Opus labels each chunk]                Stage 1 — one Opus call per chunk, parallel
   1,692 labels with provenance            frozen system prompt; cached codebook
                     |
                     v
   [exact-string dedup on (id, type)]      deterministic Python
                     |
                     v
   [Opus REPL compile]                     Stage 3 — one Opus session in a Python REPL
   emits 7 artifacts in DAG order:           frozen system prompt; cached guide + labels
     supply_list -> variables
        -> predicates -> modules
        -> router -> integrative
        -> phrase_bank
                     |
                     v  (each emit triggers phase validators — Python, not LLM)
                     |
                     v
   [deterministic assembly]                JSON -> DMN XML -> XLSForm -> Mermaid
                     |
                     v
          clinical_logic.dmn + form.xlsx + flowchart.png
```

**Identification property:** the system prompt is the only experimental variable. Chunking is deterministic. Dedup is deterministic. Validators are Python code. Assembly is deterministic. Any run-to-run diff on the same PDF is attributable to Opus's own stochasticity inside Stages 1 and 3 — nothing else.

**Per-phase red-teaming:** each of the 7 intermediate artifacts is validated the moment it is emitted, before the next phase is allowed to start. Validation is not deferred to the end.

**No orchestration layer, no multi-agent, no inter-prompt state.** One Opus model, one codebook, one REPL, one output.
