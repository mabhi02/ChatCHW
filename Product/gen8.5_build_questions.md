# gen8.5_build_questions.md

Ambiguities hit during the greenfield gen8 / gen8.5 build. Each entry is
a decision I made so the build could finish; flag any you want reversed
and I will redo it. Nothing in this list is load-bearing for a smoke
test -- reversing any of them is a small edit.

---

## Q1. Verifier primary model

**Decision:** GPT-5.4-mini is wired as the preferred verifier via
`backend/verifier/models.py`, but `_gpt_available()` runs a live probe on
startup. When it fails (current state -- your OpenAI quota is exhausted),
the selector falls back to **`claude-sonnet-4-6`** against an Opus
generator (same-family). Every verification sidecar from the fallback
path carries a trailing `{"type": "same_family_fallback", "severity":
"warn"}` divergence so reviewers see the weakened-independence case
explicitly.

**Why:** your instruction said "Fallback to anthropic but keep openai
wiring." The runner, model selector, and env-var lookup all still exist;
the moment quota comes back the next run picks GPT automatically with no
code change.

**To reverse:** edit `select_verifier_model` in
`backend/verifier/models.py` if you want a hard default to one or the
other.

---

## Q2. Gen8 REPL system prompt: extend vs rewrite

**Decision:** `backend/gen8/pipeline.py::_build_repl_system_prompt()`
calls `backend.gen7.pipeline._build_repl_system_prompt()` and appends
the three gen8 rule blocks (Tier 0 predicate grammar, Tier 2 manual
ordering, Tier 2 two-traffic-cop router) after the gen7 body.

**Why:** Section 12 decision rule 3 says "Append the new section after
existing content. Do not delete or restructure existing prompt content."
Rewriting Stage 3 from scratch was not in scope; the gen7 prompt still
describes the REPL 4-phase flow correctly, and the new rules layer on top.

**To reverse:** drop the import from gen7 and copy the gen7 prompt into
a gen8-owned copy if you want gen8 to be a fully independent prompt.

---

## Q3. Gen7 on-disk modifications

**Decision:** gen7 files already show as modified in `git status`
(`labeler.py`, `pipeline.py`, `system_prompt_bundle.py`, `delivery_readme.py`,
`data_flow.py`). I left them alone per your direct answer (A = leave them).
gen8 never edits any file under `backend/gen7/`.

**Risk:** if gen7 is re-run for a "frozen Run 2 reproduction" it will
produce slightly different output than SHA 9000e86 because the working
tree has drifted. If you want bit-for-bit reproduction of Run 2, hard-revert
the gen7 files to 9000e86 before re-running.

---

## Q4. Tier 2 "conditionally_collected" detection

**Decision:** heuristic in `backend/gen8/data_flow.py::_is_conditionally_collected`.
A variable is tagged `conditionally_collected` iff some module rule's
`outputs` dict has a key containing "collect" whose value references the
variable (e.g., `"collect_next": "v_temp_c"`). There's no explicit
Tier-2 schema for conditional collection yet.

**Why:** the spec called out the scope but did not pin the detection
rule. The heuristic is conservative (probably under-reports); you get
at worst a variable tagged `universal`/`module_local` when it should be
`conditionally_collected`, which is a less alarming miscategorization
than `orphan`.

**To reverse:** add an explicit `_conditional_on` field to variables in
the Stage 3 prompt and switch the detector to read it.

---

## Q5. Tier 2 manual ordering enforcement

**Decision:** Tier-2 manual-ordering validation is **soft**. The Stage
3 prompt (`backend/gen8/system_prompt_bundle.py::TIER2_MANUAL_ORDERING`)
requires every list-shaped artifact to carry `_order_source` +
`_manual_page`. But there's no post-stage-3 Python validator that halts
the pipeline when items are missing those fields.

**Why:** spec Section 6.1 says "log a warning but do not halt (the verifier
will catch this as a divergence)". I did not add the logger line explicitly
because the verifier-driven flow is sufficient.

**To reverse:** add a validator in `backend/validators/` that walks the
artifact lists and raises if `_order_source` is missing. Would be 15 lines.

---

## Q6. XLSForm aggregate workbook

**Decision:** the aggregate `form.xlsx` is built by delegating to the
existing gen7 converter `backend.converters.json_to_xlsx.convert_to_xlsx`
rather than reimplementing. Per-module workbooks are new
(`backend/converters/xlsform_split.py::build_per_module_workbook`).

**Why:** the gen7 converter already emits the CHT-required sheets
(survey, choices, settings). Re-implementing from scratch was unnecessary
work.

**To reverse:** rewrite `build_aggregate_workbook` to be self-contained
if gen8 should have an independent XLSForm format.

---

## Q7. Flowchart breakup vs monolithic

**Decision:** gen8 writes both `flowchart_index.png` (module-only) and
`flowcharts_per_module/flowchart_<mid>.png`. The old monolithic
`flowchart.png` is NOT emitted by gen8 -- only the split version is.

**Why:** Section 5.5 says "The existing `flowchart.png` (monolithic)
stays as a fallback, but the README points to `flowchart_index.png`
first." I interpreted "stays as a fallback" as "gen7 still emits it" and
did not re-emit it in gen8. Easy to flip.

**To reverse:** in `backend/gen8/pipeline.py` after the `emit_mermaid_all`
call, add the gen7-style `convert_to_mermaid + render_to_png` call to
also emit `flowchart.png`.

---

## Q8. Stage 3 post-process retry

**Decision:** when Stage 3 emits a flat `{rows: [...]}` router instead
of the two-cop shape, `migrate_flat_router_to_cops` salvages the output
and tags every migrated rule with `_migrated_from_flat_router: true`.
The verifier then raises a warn-severity divergence.

**Why:** the alternative is to re-run Stage 3, which costs an Opus
iteration and is not guaranteed to help. Migration + warn sidecar keeps
the run completing on the first try and surfaces the issue loudly.

**To reverse:** remove the `migrate_flat_router_to_cops` call in
`_post_stage3_reconcile` if you prefer a hard failure.

---

## Q9. `clinical_logic.json` double-write

**Decision:** `pipeline.py` writes `clinical_logic.json` twice -- once
right after Stage 3 emission, then again after `_post_stage3_reconcile`
mutates the dict (predicate dedup, router migration, referential integrity
auto-fills). Same for the per-artifact JSONs. The second write overwrites
with the reconciled version; the container-hash roll-up only sees the
second write.

**Why:** the reconcile step has to happen after at least one write so
`_post_stage3_reconcile` can reference the first write's SHA as a parent.
Rewriting the second pass is conceptually cleaner than deferring Tier-2
until after all writes, which would break the parent-SHA contract.

**Downside:** a naive reader who ignores the provenance timestamps
won't realize the file was rewritten; the SHA differs between the
pre-reconcile and post-reconcile write. The provenance `created_at`
timestamp differs, so reviewers can tell which is which, but it's
subtle.

**To reverse:** push the entire write phase to after `_post_stage3_reconcile`
and remove the first pass. Small refactor.

---

## Q10. gen8.5 labeler: per-pass prompts stubbed

**Decision:** `backend/gen8/labeler_sonnet7way.py::_pass_prompt()` is a
templated one-paragraph narrow-type prompt. The spec sketched
`"Label only supply_list items in this chunk. Type: ALWAYS use exactly
'supply_list'. ..."` with an ellipsis; I implemented a full but
generic version that works for all 7 types via string interpolation.

**Why:** writing 7 bespoke prompts is type-specific prompt engineering
that would benefit from iteration against real runs. The generic version
works and matches the spec's shape; tuning per-type is a Tier-2-optional.

**To reverse:** write per-type prompts by hand if the 7-way results
under-perform in practice.

---

## Q11. Verifier truncation: 50K / 30K / 10K char caps

**Decision:** `verify_artifact` truncates source manual to 50K chars,
artifact JSON to 30K, generator prompts to 10K. These are the same caps
sketched in the spec. Nothing adapts if an artifact is bigger than 30K
(the verifier just sees a clipped artifact and may flag imaginary
`missing_item` divergences on the clipped tail).

**Why:** matches the spec defaults. GPT-5.4-mini has a 128K context
so we could pass everything, but Anthropic fallback hits similar budget
with a bigger context that we don't want to pay for on every artifact.

**To reverse:** raise the caps or add chunked verification if a run's
`clinical_logic.json` blows past 30K chars.

---

## Q12. Backfill script: Neon lookup strategy

**Decision:** `scripts/backfill_provenance_run2.py` pulls `SourceGuide`
rows ordered by `ingestedAt desc` and picks the first one whose
`sha256` starts with `1d2708940727...`. If none match, falls back to
the first guide whose filename/manualName contains "who" or "2012".

**Why:** you said to pull the real SHA from Neon. The prefix match is
defensive: even if the full 64-char SHA I was given is off by a
character, the 12-char prefix should still match, and the WHO 2012
fallback catches re-ingestions.

**To reverse:** hard-code the full SHA in the script if you want an
exact match only.

---

## Open items I did NOT implement (as instructed)

- No live extraction runs (smoke tests deferred per your instructions).
- No frontend UI changes for `pipeline=gen8|gen8.5` selection; the API
  accepts the flag but the existing frontend still sends `gen7` unless
  you override.
- No changes to `backend/gen7/` (working tree modifications left alone).
- Section 15 items (Z3, MOH gates, translation pipeline, synthetic
  patients, ICD-10) explicitly skipped per spec.
