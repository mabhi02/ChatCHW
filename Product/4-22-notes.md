# 4-22 notes

Working file. Atharva will add more. Captured from the 4-21 session so it doesn't drop on the next compaction.

---

## 1. Per-artifact provenance (raised first)

### Problem
Across multiple runs we can't tell which `supply_list.json` (or any of the other 6 core artifacts) we're holding. Comparing run 2's supply_list to run 5's supply_list requires byte-diffing files. There is no stable identifier per artifact.

### Scope
Applies to **all artifacts**, not just the 7 core. That covers:
- 7 core (Stage 3 emits): `supply_list, variables, predicates, modules, router, integrative, phrase_bank`
- 3 derived: `data_flow, referential_integrity, clinical_logic`
- 1 trajectory: `stage3_repl`
- 4 non-JSON: `clinical_logic.dmn, form.xlsx, flowchart.png, flowchart.md`

### Design choices (open — Atharva to confirm)

**Choice 1: inline `_provenance` block vs sidecar `.provenance.json` file.**
- Inline breaks list-shaped artifacts (`variables.json` is `list[267]`, can't add a top-level dict key without breaking every reader).
- Sidecar is non-invasive but doubles file count. Drift risk mitigated by recording `content_sha256` inside the sidecar — any verifier catches mismatches.
- **Recommendation: sidecar.**

**Choice 2: pure content hash vs Merkle-style hash.**
- Pure: hash covers only the artifact's own bytes. Lets you say "two runs collapsed to byte-identical supply_list."
- Merkle: hash includes parent hashes. Lets you say "this supply_list was derived from this exact pipeline state."
- **Recommendation: pure `content_sha256` + `parents[]` list of parent SHAs.** Two runs with same content_sha256 but different parents is itself a useful signal (model converged to same answer from different inputs).

**Choice 3: backfill Run 2 + forward for Run 3 vs forward-only.**
- Backfill is a one-shot script that walks `backend/output/run_8970cf65/` and emits sidecars for what's on disk.
- **Recommendation: backfill + forward.**

### Strawman provenance shape

```json
{
  "artifact_kind": "supply_list",
  "content_sha256": "a3f1...",
  "run_id": "run_8970cf65",
  "created_at": "2026-04-15T22:28:29Z",
  "pipeline_git_sha": "9000e86",
  "model": "claude-opus-4-6",
  "source_manual": {
    "name": "Caring for the sick child in the community",
    "publisher": "WHO, Geneva, 2012",
    "pdf_sha256": "1d2708940727...",
    "page_count": 140
  },
  "parents": [
    {"kind": "deduped_labels", "content_sha256": "..."},
    {"kind": "reconstructed_guide", "content_sha256": "..."}
  ],
  "schema_version": 1
}
```

### Implementation cost
~80 lines: new `backend/gen7/provenance.py` (`compute_artifact_id`, `write_with_provenance`, `verify_sidecar`), edits to ~7 write sites in `pipeline.py`, one backfill script for Run 2. ~30 min, no LLM-pipeline behavior change.

---

## 2. Cross-AI verification gate (this is the new directive)

### Goal
An artifact is not "final" until a **different AI** has read the prompts and phase outputs, double-checked the artifact, and agreed. Divergence is logged. Only when the verifier AI agrees does the supply_list (or any artifact) get marked final.

### Why this is not a multi-agent violation
The k=1 thesis applies to **generation**: the system prompt is the experimental variable, generation must stay single-prompt. This is a **verification gate**, structurally identical to the existing catchers/classification gates already in the architecture. Catchers are instruments — adding more verifier gates is degree-of-freedom-zero as long as the verifier output only gates "final/not-final" status, never edits content.

### Quality standard the verifier checks against
- **xlsx** (XLSForm): conforms to XLSForm spec, no broken references, all required columns present, choice lists complete
- **prompts** (the 3-stage system_prompt.md bundle): match what the pipeline actually sends to Opus
- **phases** (Stage 1A, 1B, 3 outputs as captured in `labeled_chunks.json` and `stage3_repl.json`): internally consistent, no codebook violations, no orphan references that the auto-reconciler papered over without flagging

### Architecture

```
[ Opus produces artifact ] 
        |
        v
[ Verifier AI reads: prompts + phase outputs + artifact ]
        |
        v
[ Verifier AI emits: { agree: bool, divergences: [...] } ]
        |
        +--> agree=true  --> mark artifact as "final" (provenance.status = "verified")
        |
        +--> agree=false --> log divergence to artifacts/verification_log.json
                              do NOT mark as final
                              human reviews divergence list
```

### Verifier model choice
- Must be **a different model family** from the generator (Opus). Per earlier session: variance argument is GPT/Gemini producing different output is what makes the verifier independent. Sonnet vs Opus is same family, lower variance, weaker independence claim.
- Candidates: `gpt-5.4` (OpenAI keys are auth-OK but currently `insufficient_quota` — need top-up), `gemini-2.x`, `claude-opus-4-6` is excluded (same as generator), `claude-sonnet-4-6` is permissible-but-weaker.
- Pre-flight: top up OpenAI billing OR enable a Gemini key, OR fall back to Sonnet with the caveat noted.

### Divergence log shape

```json
{
  "artifact_kind": "supply_list",
  "artifact_content_sha256": "a3f1...",
  "verifier_model": "gpt-5.4",
  "verifier_run_at": "2026-04-22T10:15:00Z",
  "agree": false,
  "divergences": [
    {
      "type": "missing_item",
      "severity": "warn",
      "detail": "Verifier identified 'thermometer' in Stage 1A labels but supply_list omits it",
      "evidence_chunk": 7,
      "evidence_quote": "use a thermometer to measure axillary temperature"
    },
    {
      "type": "spurious_item",
      "severity": "info",
      "detail": "supply_list contains 'equip_muac_strap' but verifier traced no labels referencing it"
    }
  ]
}
```

### Final-status rule
An artifact's provenance gains `status: "verified"` only when:
1. Verifier AI ran against latest `content_sha256`
2. `agree: true` returned
3. Divergence list is empty OR all divergences are `severity: info`

If any `severity: warn` or `error` divergence remains, status stays `"unverified"` and the run's README must list the unverified artifacts in a callout.

### Open design questions for tomorrow
- Does the verifier see the **full prompts** or just the artifact + source manual? Full prompts make the check stronger but the verifier becomes prompt-aware (could collude on shared blind spots). Source-only is more independent but the verifier doesn't know what the generator was constrained to do.
- Is divergence per-artifact or per-item-within-artifact? Per-item is finer-grained but explodes the log.
- Does the verifier also double-check the **xlsx**, **dmn**, **mermaid**? Or only the JSON artifacts? xlsx is mechanically derived from `clinical_logic.json` — if the JSON is verified, the xlsx inherits via the deterministic converter. So verify the JSONs; trust the converters.
- What happens if the verifier itself disagrees with a re-run (verifier is non-deterministic)? Run verifier 3x and require 2/3 majority? Single-shot at temperature 0?

---

## 3. Loose ends from 4-21 session (for context)

- Run 2 (`run_8970cf65`): $14.18, 1390s wall, Opus mono, 21 chunks, 1675 raw labels → 1054 deduped. README + FAILURES.md split shipped in `CHW_Nav_Run_2_FUll.zip`.
- 7-way Sonnet experiment (4-18) showed 0.0% codebook mismatch and 0.58 Jaccard at $0.97 vs Opus mono's 5.3% / 0.41 / $1.79. **Not wired into `labeler.py` yet** — `OPUS_MODEL` is still the constant. Wiring is planned but not done.
- Both OpenAI keys (`OPENAI_API_KEY`, `OPENAI_API_KEY_ALT`) are auth-OK but `insufficient_quota` on completions. Top-up needed before either can run as the verifier in section 2.
- README identifies source as "WHO CHW guide 2012" but lacks full title ("Caring for the sick child in the community"), publisher, pdf_sha, page_count. Header table needs to be expanded — naturally falls out of the provenance work in section 1.

---

## 4. Ordering / priority rule for items inside artifacts

Source-of-truth rule for ordering items in any artifact (variables list, modules list, supply_list, rules within a module, predicate evaluation order, anything where the output has a sequence):

1. **Manual order wins.** If the source manual implies a priority by where things appear (chapter order, section order, list order within a section), use that order. The manual's chapter sequence *is* the priority signal — earlier chapters dominate later ones.
2. **If the manual gives an explicit indication, follow it.** "Ask about cough before fever" or "check breathing first, then temperature" beats both manual-order and alphabetical. These are explicit ordering instructions and override defaults. Capture them as `_ordering_hint` on the affected items, with `source_section_id` and `source_quote` so the priority is auditable.
3. **Fallback: alphabetical by id.** Only when (1) and (2) provide no signal — e.g., two variables introduced in the same paragraph with no priority indication — order alphabetically by id. Deterministic, reviewable, no model judgment.

### Why this matters for the pipeline
- **Stage 3 currently emits in arbitrary order** (whatever order Opus's REPL session happened to produce). That's not reproducible across runs and not aligned with the manual's clinical priority.
- **The XLSForm question order** is what the CHW actually sees in the field. Wrong order = wrong triage flow.
- **Predicate evaluation order matters** when predicates reference other predicates — manual order naturally encodes the clinical reasoning chain.

### Implementation notes
- Add an `_order_source` field to each item: one of `"manual"` (with `source_section_id`/`source_position`), `"explicit_hint"` (with `source_quote`), or `"alphabetical_fallback"`. Lets the verifier in section 2 check that ordering is justified, not arbitrary.
- The chunker already preserves `chunk_index` and `section_id`. Threading these through Stage 1 → Stage 3 lets the assembler sort by manual position deterministically.
- Stage 3 prompt should be amended to instruct Opus to preserve manual order in emitted lists, and to surface `_ordering_hint` whenever the source manual says X-before-Y. This is a Stage 3 prompt change, not a generation-architecture change.

### Open questions
- Does this apply within `modules` too? E.g., should the modules list itself be in manual chapter order (likely yes — chapter 3 cough module before chapter 5 fever module)?
- For `phrase_bank`, is alphabetical fine since phrases are referenced by id and have no clinical priority? Probably.
- For `supply_list`, manual order vs grouped-by-category? Manual order keeps it auditable; grouping is cosmetic. Default to manual order.

---

## 5. Readability of xlsx and flowchart deliverables

The current `form.xlsx` and `flowchart.png` are technically correct but unreadable as deliverables. They need to be **broken into parts** with explicit cross-references between parts.

### Problem (Run 2 baseline)
- `form.xlsx` is one monolithic XLSForm — long, dense, no visual breakpoints. A reviewer scrolling through it loses context after the first screen.
- `flowchart.png` is a single 98 KB image rendered from a 49 KB Mermaid source. Too many nodes and edges in one frame; arrows cross; nothing is legible at zoom-to-fit.

### Direction
**Break by clinical part, not by file size.** The natural seams are the modules (cough, fever, diarrhoea, etc.) — each is a self-contained decision flow with its own inputs, predicates, rules, and exits. Splitting at module boundaries makes each piece reviewable in isolation while preserving the routing between them.

### xlsx changes
- One **summary sheet** at the front: list of modules + page references + symptom they cover. Acts as the table of contents.
- One sheet per module (or one tab per module within the same workbook). Each sheet has its own `survey`, `choices`, and `settings` blocks scoped to that module.
- A **routing sheet** at the back: the priority-exit rules and inter-module transitions, with cross-references to the per-module sheets ("see sheet `mod_cough` row 14").
- Optional: a **glossary sheet** with the variable id → human-readable label mapping, sorted alphabetically.

### Flowchart changes
- **One flowchart per module**, not one mega-flowchart for the whole pipeline. Each module gets its own `flowchart_mod_<name>.png` rendered from its own `.mmd` source.
- An **index flowchart** at the top level (`flowchart_index.png`) showing only the module boxes and the routing between them — every box is a hyperlink-style reference like "→ see flowchart_mod_cough" rendered as the box label.
- Each per-module flowchart has explicit **entry and exit nodes** that name the next flowchart: "Enter from index", "Exit to flowchart_mod_fever (if fever co-present)", "Exit to flowchart_priority_exit (if danger sign)".
- Render each per-module flowchart at higher resolution since they're smaller — `mmdc -s 4 -w 1800` instead of `-s 3 -w 1400` for the index.

### Why this works structurally
The pipeline already produces a **module DAG** in `clinical_logic.json` — modules have `routes_to` and `triggered_by` fields that encode the inter-module edges. The breakup is mechanical: walk the DAG, render each module as its own subgraph, render the DAG itself as the index. No new clinical logic is being invented; the structure is being **revealed** instead of flattened.

### Implementation notes
- Add a new converter `backend/converters/mermaid_split.py` that takes `clinical_logic.json` and emits N module-scoped Mermaid sources + 1 index Mermaid source. Each gets rendered to PNG via the existing `mermaid_to_png.py` (already prefers local mmdc, so no external dependencies).
- The existing `clinical_logic.dmn` and `form.xlsx` converters need a `--split` flag: when set, emit per-module DMN snippets and per-module XLSForm tabs. Keep the monolithic versions as fallbacks for tools that need a single file.
- README needs an update: instead of pointing at `flowchart.png`, point at `flowchart_index.png` first and explain the per-module flowchart naming convention.

### Open questions
- For DMN: should we emit per-module DMN files too (`clinical_logic_mod_cough.dmn`), or only split the visualization layer? DMN engines typically consume one file — splitting would require an orchestrator. Probably keep DMN monolithic, split only xlsx and flowchart.
- For very small modules (< 5 nodes), is it worth a dedicated flowchart, or fold them into a "minor modules" composite? Probably dedicated — consistency over compactness for reviewer cognitive load.
- What about rule density inside a single large module (e.g., if `mod_cough` has 40 rules)? Sub-split by symptom severity tier? Defer; cross that bridge if a single module ends up unreadable post-split.

---

## 6. Findings from professor artifacts (3 PDFs read 4-22)

Three PDFs in `professor artifacts/`. Read via parallel subagents.

### What each PDF is

- **`CHW Navigator hand-off.pdf`** (6 pages, no images) — Project hand-off / scope brief. How to hand the pipeline to a CHT volunteer "AI squad" forming spring 2026. Lists deliverables, evidence-of-quality requirements, custom tools, intermediate artifacts, failure-message standards, security guarantees.
- **`Predicate table.pdf`** (7 pages, no images) — Prescriptive schema spec for the predicate-table artifact. Defines 10 required columns, per-column authoring guidance, a worked example (`p_fever`), 7 design rules, and a 4-stage authoring loop (LLM → red-team → repair → Z3).
- **`Verification-First Clinical Engineering.pdf`** (52 slides, 5 images including 2 Gemini-rendered queue diagrams) — Methodology deck. "Zero-Trust Pipeline" with 5-layer Verification Stack: adversarial extraction, formal verification, human sovereignty, traceability, differential testing.

### Cross-cutting message

The three documents are **complementary, not redundant**: hand-off = scope, Verification-First = methodology, Predicate table = worked-example artifact spec. Together they describe the *target* system. **Run 2 is far from that target.**

### How the existing 4-22 sections map to PDF positions

| 4-22 section | Hand-off | Predicate | Verification-First | Verdict |
|---|---|---|---|---|
| 1. Provenance sidecars | Not mentioned | Not addressed | Implicit, coarser-grained (container hash p.40 + per-snippet hash p.9) | 4-22 is finer-grained refinement; **add a container-hash level on top to align** |
| 2. Cross-AI verifier | Implicit, contradicts (wants edit-capable red-team) | Implicit, single-family + Z3 | **Central thesis** — but goes deeper: 2 parallel pipelines + 3-cycle adjudicator + MOH sign-off + Z3 | 4-22 design = MVP first layer; **k=1 tension on PDF's 3-cycle generator retry — must decide explicitly** |
| 3. Ordering rule | Not mentioned | Not mentioned | **Actively contradicts** — p.12 uses alphabetical as a real priority signal | **Hold the line; 4-22 is the right call** |
| 4. xlsx/flowchart breakup | Not mentioned | Not addressed | Implicit — every module already has its own DMN + flowchart + done-flag | 4-22 file-org = natural physical realization of PDF logical decomposition |

### New gaps the PDFs surface (not in current 4-22)

- **Predicate-table schema migration** — 4 missing columns (`units`, `missingness_rule` taxonomy, `allowed_input_domain`, `rounding/parsing_rule`); operator-grammar fix (`==`→`=`, uppercase `AND`/`OR`→lowercase, no quoted strings); dedup the `p_*` / `p_has_*` / `p_*_days` triplets
- **Z3 formal verification** — reachability, mutual-exclusivity, cycle-freeness, monotonicity invariants ("worsening symptom never reduces referral intensity")
- **MOH human-sovereignty gates** — three approval points: ambiguity resolution, logic review (plain-English + flowcharts, NOT code), translation sign-off; diff-highlighting on updates
- **Synthetic-patient differential testing** — boundary-triple generation (e.g., 37.4 / 37.5 / 37.6); A/B/C three-way equivalence (DMN ↔ flowchart ↔ XLSForm); mutation testing with 100% catch-rate requirement
- **"Source of each variable" intermediate artifact** — universal / module-local / conditionally-collected classification (Verification PDF p.17). This is exactly the orphan-variable fix from Run 2's FAILURES.md
- **ICD-10 / SNOMED data dictionary** — variable + diagnosis + drug code mapping (Hand-off p.5)
- **Translation pipeline** — two independent translator AIs + back-translator + reviewer + clinician-editable phrase book (Verification PDF p.28)
- **Stockout coverage enumeration** — explicit "ALERT: Manual does not cover this case" rows for each diagnosis × supply combination (Verification PDF p.19)
- **Two distinct traffic cops** — Cop-1 (queue-builder, multi-hit policy) + Cop-2 (next-module-picker, mutually-exclusive). Likely currently merged in Stage 3 output
- **Allowable FEEL subset spec** — explicit list of which FEEL operators are permitted in DMN expressions (Hand-off p.4 open question; Verification PDF p.36 "no FEEL arithmetic")
- **XLSForm security scans** — static check for `pulldata()` with remote URLs + custom-widget references; runtime traffic intercept (Hand-off p.6)
- **Manual-quality failure messages** — pipeline must distinguish "manual is vague / contradictory / lacks instructions for stockout" from "our software is inconsistent" (Hand-off p.5)

### Direct violations Run 2 has of PDF requirements

- 267 auto-registered ids = symbol-table consistency rule violated (Verification PDF p.35)
- 122 orphan variables = "source of each variable" artifact missing (Verification PDF p.17)
- Stage 1B JSON parse with silent fallback = "no hidden logic / explicit ELSE" rule violated (Verification PDF p.36)
- `predicates.json` field `fail_safe: 0` collapses 4-option missingness taxonomy into a single integer (Predicate PDF section 5.6)
- 17 auto-registered predicate stubs with empty `threshold_expression` = "every predicate must be computable without interpretation" violated (Predicate PDF section 7)
- Duplicate predicates (`p_chest_indrawing` vs `p_has_chest_indrawing`, `p_fever` vs `p_has_fever` vs `p_fever_days`, `p_muac_red`/`p_muac_yellow` vs `p_muac_colour`) = "no duplication" rule violated (Predicate PDF section 7)
- Operator grammar (`==`, uppercase `AND`/`OR`, quoted string literals) = restricted-grammar rule violated (Predicate PDF section 5.4)
- Section ids like `b_beauty+contents+a_story` instead of "WHO IMCI 2014 page 33" = clinical provenance rule violated (Predicate PDF section 5.9)

---

## 7. Concrete pipeline changes (the action plan)

Grouped by impact tier. Each item names the file(s) to edit, the change to make, and the source PDF / 4-22 section that requires it.

### Tier 0 — Hygiene fixes (nothing new, just close known gaps)

These are direct violations Run 2 has that don't require new architecture.

1. **Fix predicate operator grammar** (`backend/gen7/pipeline.py` Stage 3 prompt)
   - Change Stage 3 system prompt to emit `=` (not `==`), lowercase `and`/`or`/`not` (not `AND`/`OR`), no quoted string literals
   - Predicate PDF section 5.4
2. **Dedup predicate triplets** (`backend/validators/referential_integrity.py`)
   - Add a deterministic post-Stage-3 step that collapses `p_X` / `p_has_X` / `p_X_days` into one canonical entry
   - Predicate PDF section 7 ("no duplication")
3. **Add missing predicate columns** (`backend/gen7/pipeline.py` Stage 3 prompt + downstream consumers)
   - Add to Stage 3 prompt: emit `units`, `missingness_rule` (one of `FALSE if missing` / `TRIGGER_REFERRAL` / `BLOCK_DECISION` / `ALERT: NO RULE SPECIFIED`), `allowed_input_domain`, `rounding/parsing_rule` for every predicate
   - Predicate PDF sections 5.5-5.8
4. **Fix predicate provenance** (`backend/gen7/pipeline.py` Stage 3 prompt)
   - Replace internal section ids (`b_beauty+contents+a_story`) with citable form: "WHO 2012 page <N>". Requires threading PDF page numbers through chunker + labeler + Stage 3
   - Predicate PDF section 5.9; Hand-off PDF requirement for citable provenance

### Tier 1 — Provenance + xlsx/flowchart breakup (already in 4-22 sections 1 and 4)

These are pure deterministic changes; no LLM behavior change.

5. **Per-artifact provenance sidecars** (new file `backend/gen7/provenance.py`)
   - `compute_artifact_id(content_bytes) → sha256`, `write_with_provenance(path, data, kind, parents)`, `verify_sidecar(path)`
   - Edit ~7 write sites in `backend/gen7/pipeline.py` (lines 306, 309, 315, 878, 909, 915, 920, 937, 999, 1007)
   - Backfill script for `run_8970cf65/`
   - 4-22 section 1; Verification PDF p.40 alignment via container hash
6. **Container-level hash** (new field in main provenance manifest)
   - Single `container_sha256` rolled up over all per-artifact SHAs in DAG order. Goes in `manifest.json` at the run root
   - Verification PDF p.40 ("single container with a unique hash")
7. **xlsx breakup by module** (`backend/converters/xlsform.py` or wherever XLSForm is generated)
   - Add `--split` flag: emit TOC sheet + per-module sheets + routing sheet + glossary sheet
   - Keep monolithic version as fallback
   - 4-22 section 4
8. **Flowchart breakup by module** (new file `backend/converters/mermaid_split.py`)
   - Walk `clinical_logic.json` module DAG; emit one Mermaid source per module + one index Mermaid; render each via existing `mermaid_to_png.py`
   - Update README to point at `flowchart_index.png` first
   - 4-22 section 4

### Tier 2 — Ordering rule + missing intermediate artifacts (add to pipeline)

9. **Manual-order threading** (`backend/gen7/chunker.py` → `labeler.py` → `pipeline.py`)
   - Chunker already preserves `chunk_index` and `section_id`. Add `manual_position` (page number + offset). Thread through Stage 1 → Stage 3
   - Stage 3 prompt must emit `_order_source: "manual" | "explicit_hint" | "alphabetical_fallback"` for every item in every list-shaped artifact
   - Add deterministic post-step that asserts `_order_source` is set for every item
   - 4-22 section 3
10. **"Source of each variable" artifact** (new in `backend/gen7/data_flow.py`)
    - Already partially built. Extend to classify every variable as `universal` / `module_local` / `cross_module` / `derived` / `conditionally_collected` (the last is new — variables collected only when an upstream condition fires, e.g. "if temp missing → ex_temp_C")
    - This zeros out the 122 orphan-variable problem from Run 2
    - Verification PDF p.17
11. **ICD-10 / SNOMED data dictionary** (new artifact `artifacts/data_dictionary.json`)
    - For every variable + diagnosis + drug, emit code mapping. Stage 3 prompt must request these codes; if not present in source manual, mark as `code_source: "lookup_required"` for human review
    - Hand-off PDF p.5
12. **Two-traffic-cop split** (`backend/gen7/pipeline.py` Stage 3 prompt + downstream DMN converter)
    - Stage 3 must emit `router.json` as **two distinct DMN tables**: `cop1_queue_builder` (multi-hit) and `cop2_next_module` (mutually-exclusive)
    - DMN converter merges if needed for engine consumption; keep them separate at JSON layer for review
    - Verification PDF pp.24, 27, 29
13. **Stockout coverage enumeration** (new validator `backend/validators/stockout_coverage.py`)
    - For every diagnosis × supply pair, check that either (a) supply is available and rule covers it, or (b) explicit `ALERT: Manual does not cover stockout` row exists
    - Verification PDF p.19

### Tier 3 — Cross-AI verification gate (4-22 section 2)

14. **Verifier harness** (new directory `backend/verifier/`)
    - `verifier/runner.py` — takes artifact + relevant prompts/phase outputs; calls verifier model; returns `{agree, divergences[]}`
    - `verifier/divergence.py` — divergence schema + severity classification
    - `verifier/models.py` — model-family selection (NOT Opus or Sonnet — must be GPT or Gemini)
    - 4-22 section 2
15. **Verifier model dependency** (`.env` + billing)
    - **BLOCKING:** both OpenAI keys are `insufficient_quota`. Top up OR add a Gemini key. Without this, verifier can only run as Sonnet (same-family, weaker independence claim)
16. **Verification log** (new artifact `artifacts/verification_log.json`)
    - Per-artifact: `artifact_kind`, `content_sha256`, `verifier_model`, `agree`, `divergences[]`, `verifier_run_at`
    - Goes alongside provenance sidecars
17. **`status: "verified"` field** (extend provenance schema)
    - Provenance sidecar gains `status` field: `"unverified" | "verified" | "verification_failed"`
    - README must list any `unverified` or `verification_failed` artifacts in a callout
    - 4-22 section 2

### Tier 4 — Z3 formal verification (Verification PDF pp.38, 41)

18. **Z3 dependency** (`backend/requirements.txt`)
    - Add `z3-solver`
19. **Z3 properties checker** (new directory `backend/z3/`)
    - `z3/dmn_to_smt.py` — translate `clinical_logic.json` predicates + rules into SMT-LIB
    - `z3/properties.py` — encode required properties: every rule reachable, no two diagnoses simultaneously triggered for same patient unless `integrative` allows it, no cycles in module DAG, monotonicity (worsening symptom never reduces referral intensity)
    - `z3/synthetic_patients.py` — generate boundary-triple patients per predicate (e.g., for `temp_C >= 37.5` emit patients with `temp_C ∈ {37.4, 37.5, 37.6}`)
    - Verification PDF pp.38, 41
20. **A/B/C equivalence harness** (new file `backend/eval/equivalence.py`)
    - For each synthetic patient, run through (a) `clinical_logic.json` interpreter, (b) DMN engine on `clinical_logic.dmn`, (c) XLSForm on `form.xlsx`. Assert outputs are identical
    - 100% pass required to mark run as `equivalence_verified`
    - Verification PDF pp.42-43

### Tier 5 — Generation-side improvements (architectural decisions required)

21. **Wire 7-way Sonnet labeler** (`backend/gen7/labeler.py`)
    - Already validated experimentally (4-18). Cost drop ~50%, codebook mismatch 5.3% → 0%
    - Replace `OPUS_MODEL` with `SONNET_MODEL`, write 7 narrow-prompt builders, fire parallel asyncio, deterministic merge
    - **Decision required first**: do you preserve full raw labels with per-pass attribution (sidecar `*_dropped_labels.json`), or only stats? Recommendation: full raw with provenance (option 3 from earlier discussion)
22. **Two independent pipelines + consensus gate** (Verification PDF p.6)
    - The 7-way labeler is the natural place to split: 4 Sonnet passes + 3 GPT passes (or any model-family split). Require behavioral equivalence on the merged label set before passing to Stage 3
    - **Architectural decision required**: this stretches k=1 — two pipelines means two system prompts means k=2 at the labeling layer. Either accept k=2 here OR define a stricter rule ("k=1 within a model family; consensus across families is meta-verification")
23. **k=1 vs PDF's 3-cycle repair loop — pick one** (Verification PDF p.34, Hand-off PDF p.4 open question)
    - PDF wants generator to retry up to 3 times conditioned on red-team feedback. This violates the pure k=1 thesis (generation becomes multi-turn)
    - **Decision required**: either (a) reject the repair loop, keep verifier as pure read-only gate (4-22 section 2 today); (b) accept the repair loop but document the k=1 amendment ("k=1 per turn, ≤3 turns conditioned on adversarial feedback"); (c) split the difference — verifier emits read-only feedback, but a separate human curator decides whether to re-run
24. **MOH human-sovereignty gates** (new state machine in `backend/server.py` + frontend)
    - Three pause points: ambiguity-resolution (when manual is vague), logic-review (MOH approves plain-English logic), translation-sign-off (local staff approve wording)
    - Pipeline state must persist across MOH wait time; needs DB schema for `approval_state`
    - Updates must diff-highlight modified logic
    - Verification PDF p.37
25. **Translation pipeline** (new directory `backend/translation/`)
    - Two independent translators + back-translator + reviewer
    - Phrase-bank stays editable by clinicians/CHWs; software locks to approved phrases only
    - Verification PDF p.28

### Tier 6 — Reproducibility infrastructure

26. **Test-retest harness** (`backend/eval/reliability.py`)
    - Run pipeline N times on same manual, assert clinically-identical output (not byte-identical — same diagnoses/treatments per synthetic patient)
    - Verification PDF p.46
27. **Inter-rater harness** (`backend/eval/inter_rater.py`)
    - Run pipeline with different LLM arrangements (Opus, GPT, Gemini, Sonnet 7-way), assert clinically-identical output
    - Verification PDF p.46
28. **Mutation testing** (`backend/eval/mutation.py`)
    - Mutate predicates/rules deliberately; pipeline must catch 100% of clinical-output divergence
    - Verification PDF p.45

### What this changes about Run 3

If Tiers 0-2 ship before Run 3:
- Predicate table conforms to spec (10 columns, lowercase ops, no dups)
- Every artifact has provenance + container hash
- xlsx has TOC + per-module tabs; flowchart has index + per-module PNGs with cross-references
- Manual order is preserved end-to-end with `_order_source` tags
- Orphan-variable count goes to 0 (data-flow classification covers it)
- `predicates.json` cites "WHO 2012 page N", not internal section slugs
- ICD-10/SNOMED dictionary exists (with `lookup_required` flags for human review)

If Tier 3 ships:
- Every artifact gets a `verified` / `unverified` flag from a non-Opus verifier
- README has explicit callout if anything is unverified

If Tiers 4-6 ship:
- Z3 proves reachability + monotonicity + mutual-exclusivity
- A/B/C equivalence runs against 1000 synthetic patients
- Mutation testing catches 100% of injected flaws
- Test-retest + inter-rater confirm reliability

### Architectural decisions still open (require Atharva sign-off before Tier 3+)

- **D1**: Repair loop yes/no? (item 23) — answer drives whether verifier feedback is read-only or actionable
- **D2**: Two-pipeline consensus yes/no? (item 22) — answer drives whether 7-way Sonnet stays single-family or splits 4+3 across families
- **D3**: Verifier model — GPT (needs OpenAI top-up) or Gemini (needs new key)? (item 15)
- **D4**: Container hash scope — does it include the source PDF SHA + git SHA + all artifact SHAs, or only artifacts? (item 6)
- **D5**: Where does MOH approval persist — file artifact, DB row, or external system? (item 24)

---

## 8. (Atharva's space — add below)
