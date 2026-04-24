# CHW Navigator — engineering handoff

**Purpose:** Give the next owner or squad enough context to run, extend, or transfer the system without re-deriving intent from scattered docs.  
**Sources:** historical CHW handoff brief (Word), this repo’s `README.md`, `Product/` submodule docs and code, `Testing/` prototypes, `Medical/` collateral, and `Product/issues.txt` (internal audit synthesis — **triage before treating as current truth**).  
**Last updated:** 2026-04-23  

---

## Handoff overview

**CHW Navigator** is a verification-oriented pipeline: **MOH/WHO CHW manuals** → structured extraction → **boolean-style clinical logic** (`clinical_logic.json`) → deterministic **DMN 1.3**, **Mermaid**, **XLSForm**, and flat CSVs, with validators and (where wired) formal checks and eval harnesses.

The **Handoff** Word document frames a **community transfer** goal: align with **CHT’s volunteer “squad”** (spring 2026) that helps non-programmers build forms. Success for that audience is defined partly as:

- **Needs assessment** for squad members (employer, expertise, blockers, what “success” means).
- **Evidence** the approach matters: manual → **DMN / flowcharts clinicians can approve** → **deterministic** DMN → **XLSForm** usable inside CHT patterns.
- **Quality narrative:** red-teaming, formal tests, Z3-style properties, test-retest / inter-method reliability, synthetic patients for **equivalence** across DMN, Z3 view, and XLSForm (as the long-term research story).
- **Offerings to partners:** QA plan, theory + evidence, cost-effectiveness story, MD access, credits, platform assumptions, phased partnership (mentor → review → cheerlead).
- **Documentation expectations:** maintainer manual, user-facing MOH manual, video, synced prompts/quality standards, clear **intermediate artifact** specs and **failure modes** (repair loop non-convergence, DMN vs form divergence, vague manuals, CHW-facing errors).
- **Security posture (aspirational):** XLSForm should avoid network side effects; static checks for `pulldata()` / remote URLs / custom JS; process for commits and backups.

This `handoff.md` **does not** replace the Word doc for stakeholder narrative; it **bridges** that narrative to **what exists in git today**.

---

## Current architecture

### Umbrella repo (`ChatCHW`)

| Path | Role |
|------|------|
| `Guidelines/` | Policy PDFs (WHO CHW 2012, 2024 diarrhoea & pneumonia). |
| `Product/` | **Git submodule** → [`CHW_RLM`](https://github.com/atharvajpatel/CHW_RLM) — primary app and backend. |
| `Medical/` | Excel DMN teaching modules, verification deck (`Verification-First Clinical Engineering.pptx`). |
| `Testing/` | DMN runners, parsers, CHT harness experiments (`Gigi/`, `Angelina/`, `Aaron/`). |
| Root docs | `STATUS.md`, `RUNBOOK.md`, `ARTIFACTS.md`, `QUALITY_AND_VERIFICATION.md`, `PLATFORM_INTEGRATION.md`, `PROVENANCE_TRACE.md`. |
| `old/` | Legacy; **do not use** for current architecture. |

### Product submodule (authoritative implementation)

**Ingest time:** PDF → render → **Unstructured.io hi_res** → assemble **`guide_json`** → cache in **Neon** (SHA256 of PDF).  
**Extraction time (Gen 7 v2):** micro-chunk → **Opus labeling** (2 calls/chunk + dedup) → inject reconstructed guide + labels into REPL context → **REPL** (`rlm_runner.py`) phases (scan/plan, parallel Module Maker, Python dispatcher, flat artifacts) → **`clinical_logic.json`** → **deterministic converters** → `.dmn`, `form.xlsx`, `flowchart.md`, CSVs.

**Frontend:** Next.js app — upload, BYOK Anthropic key, SSE for extraction progress, cost tracker, downloads.

Canonical engineering write-up: **`Product/ARCHITECTURE.md`**. Staged research vocabulary (FACTSHEET, full sequential DAG, A/B gate economics): **`Product/PIPELINE.md`**. REPL/RLM rationale: **`Product/ORCHESTRATOR.md`**.

---

## What is done

- **End-to-end demo path:** ingest WHO-scale PDFs, run Gen 7 extraction, download artifacts (`Product/ARCHITECTURE.md` reference run `run_d9a39c2d`).
- **Ingestion pipeline** with Unstructured hi_res, assembler, Neon cache, vision post-processing (see `Product/backend/ingestion/`).
- **Gen 7:** chunker, labeler with dedup, pipeline wiring, cost tracking, session concurrency limits (**Redis**), SSE (`Product/backend/session_manager.py`, `Product/frontend/`).
- **Converters:** JSON → DMN XML, XLSForm, Mermaid, CSV (`Product/backend/converters/`).
- **Validators / catchers:** substantial JSON + DMN validation layer (`Product/backend/validators/`, `Product/backend/prompts/validators/`); hybrid audit documented in `Product/HYBRID_AUDIT_REPORT.md`.
- **Eval gate harness:** `Product/backend/eval/gate_harness.py` for comparative runs (documented in hybrid audit — **note:** `ARCHITECTURE.md` §13.12 still says “no eval harness”; treat the code as superseding that bullet until the doc is edited).
- **Automated tests:** on the order of **40** backend tests listed in `Product/ARCHITECTURE.md` §12.3 (chunker, labeler, pipeline stub, converters, session manager, rlm_runner).
- **Top-level docs:** `README.md` maps folders; `Product/` contains deep specs, arena notes, `Automating_CHT_v25.xlsx`, final test artifacts under `Product/4-14-final-test-artifacts/`.
- **Testing folder:** Gigi’s JSON runner README, Angelina DMN preprocessor + docs, Aaron CHT harness + orchestrator fixtures.

---

## What is in progress

- **CHT squad alignment** (from Word doc): needs assessment, defining “success” with volunteers, scoping what they want (manual→DMN vs DMN→form automation).
- **Evidence packaging:** narrative that combines red-team/repair, Z3, synthetic patients, and **equivalence** across representations — partially implemented in code, not fully packaged as repeatable “one button” CI for every manual.
- **PIPELINE.md build priorities** still list **Stage 2–3 “FACTSHEET” style** extraction as not built / partial relative to the full **Automating_CHT** spreadsheet DAG; Gen 7 is a **different slice** that goes from **guide_json + labels** straight to **clinical_logic**. Reconciling “spec pipeline” vs “shipped Gen 7” is ongoing product/architecture work.
- **Per `Product/ARCHITECTURE.md` §13:** placeholder chunk labeling waste, streaming UX, FINAL_VAR parse retries, per-module cost breakdown, determinism tests, frozen `test_suite.json` consumers, Module Maker partial-failure retries, optional Neon cache for label phases, **gold-standard eval JSON** vs reference run.

---

## Open risks

| Risk | Notes |
|------|--------|
| **Clinical / liability** | Outputs are research and tooling artifacts until **MOH-approved**; wrong referral or dosing logic is a patient-safety issue. Human gates in `CHW Navigator v1 (1).md` remain mandatory for real deployment. |
| **LLM variance** | Same guide can yield non-identical JSON across runs; need determinism / equivalence tests (§13.8) and clear “cosmetic vs semantic” diff policy. |
| **Cost & rate limits** | Full runs are API-heavy (see `Product/ARCHITECTURE.md`, `Product/PIPELINE.md` cost sections); Opus tier and caching behavior matter. |
| **Ingestion quality** | Placeholder OCR text on many pages (~50% of pages affected per §13.1) wastes labeling budget until pre-filter ships. |
| **Security / abuse (needs verification)** | `Product/issues.txt` flags **critical** concerns: model-generated code in REPL with **`environment="local"`** (potential RCE from malicious PDF / prompt injection), sync Anthropic calls blocking the event loop, discarded `asyncio.Task` references for background jobs, unbounded concurrency in vision path, body size / memory issues on upload. **Assume these are hypotheses until each is reproduced and fixed or dismissed on current `main`.** |
| **Doc drift** | Example: eval harness exists but §13.12 still reads “No eval harness.” Submodule may move faster than umbrella docs. |

---

## Known gaps

1. **Full PIPELINE stages 2–3** (PROVENANCE_JSON → FACTSHEET → BOOLEAN_LITERALS / RULE_CRITERIA as separate shipped stages) vs **Gen 7** shortcut — see `Product/PIPELINE.md` Build Priority table.
2. **Stage 6 measurement** per spreadsheet spec: deterministic DMN executor + D1b-style synthetic generator — not fully realized as production CI (`Product/PIPELINE.md`).
3. **Stage 5 completeness:** why-trace JSON, module queue in XLSForm, compiler patch log — partially built (`Product/PIPELINE.md`).
4. **Gold `clinical_logic`** for regression: proposed in `ARCHITECTURE.md` §13.12 but not described as committed in this workspace snapshot.
5. **Community onboarding assets** from Word doc: maintainer manual, synced prompt directory, video walkthrough, standardized failure messages — largely **outside** current repo depth.
6. Historical Word/PDF brief versions are not machine-diffable; this markdown should be updated when stakeholder narrative changes materially.

---

## Outstanding questions

From historical handoff notes and engineering docs (non-exhaustive):

- What does the **CHT AI squad** define as MVP: **manual→DMN**, **DMN→XLSForm**, or **full stack** with verification UI?
- Should every **Maker** prompt have a paired **Red team + Repair** prompt (orchestration policy)?
- What **FEEL subset** is allowed in predicate expressions for your target DMN runtime?
- **iOS vs Android** and **offline** constraints for any demo packaged for MOH?
- Who **owns** MD sign-off, translation review, and incident response after open-sourcing?
- How will **secrets** (BYOK vs platform keys), **Neon/Redis**, and **Render** (or other host) be provisioned for community forks?
- **Equivalence acceptance:** what threshold of clinical agreement (diagnosis / referral / treatment) across DMN vs XLSForm vs Z3-backed checks is acceptable for “ship”?

---

## Recommended next steps

1. **Triage `Product/issues.txt`** against current `main`: reproduce blockers, file tracked issues, fix or document false positives — especially REPL sandbox and upload limits.
2. **Reconcile docs:** update `Product/ARCHITECTURE.md` §13.12 to mention `gate_harness.py`; link this `handoff.md` from `README.md` if desired.
3. **Ship “happy path” for contributors:** single script or Makefile target: submodule init, `docker compose` or local backend+frontend, one small PDF test (`WHO_CHW_guide_2012_test.pdf`).
4. **Gold standard:** commit `backend/eval/gold/who_chw_guide.json` (or equivalent) and a pytest that diffs semantics, per §13.12 proposal.
5. **CHT squad package:** 1-page “what we offer,” link to `Medical/` verification deck + `Product/4-14-final-test-artifacts/`, list API costs and time from `ARCHITECTURE.md`.
6. **Implement quick wins from §13:** placeholder pre-filter (§13.1), FINAL_VAR retry (§13.6), Module Maker JSON retry (§13.10).
7. **Security review** before any multi-tenant or public deployment: REPL execution model, file upload path, outbound calls from generated forms.

---

## Key links

| Resource | Path or URL |
|----------|-------------|
| Submodule remote | `https://github.com/atharvajpatel/CHW_RLM.git` (see root `.gitmodules`) |
| Repo map | `README.md` |
| Gen 7 architecture | `Product/ARCHITECTURE.md` |
| Full pipeline vocabulary & gates | `Product/PIPELINE.md` |
| REPL / RLM design | `Product/ORCHESTRATOR.md` |
| Product philosophy & MOH workflow | `Product/CHW Navigator v1 (1).md` |
| Hybrid / gate accounting | `Product/HYBRID_AUDIT_REPORT.md` |
| Internal audit synthesis | `Product/issues.txt` |
| Gate harness code | `Product/backend/eval/gate_harness.py` |
| DMN JSON black-box runner | `Testing/Gigi/README.md` |
| CHT harness notes | `Testing/Aaron/cht-harness/notes.md` |
| Status board | `STATUS.md` |
| Artifact contracts | `ARTIFACTS.md` |
| Quality and verification model | `QUALITY_AND_VERIFICATION.md` |
| Platform integration checklist | `PLATFORM_INTEGRATION.md` |

---

## Handoff checklist

Use this when transferring to a new lead, vendor, or community squad.

- [ ] Clone **with submodules**: `git submodule update --init --recursive`
- [ ] Read `README.md`, then `Product/ARCHITECTURE.md` §1–2 and §12–13
- [ ] Obtain **Anthropic**, **OpenAI**, **Unstructured**, **Neon**, **Upstash** credentials; configure `Product/backend/.env` and `Product/frontend/.env.local` per Appendix B in `ARCHITECTURE.md`
- [ ] Run `cd Product/backend && pytest -v`
- [ ] Run one **ingestion + extraction** on `WHO_CHW_guide_2012_test.pdf` (or full guide if budget approved)
- [ ] Inspect outputs: `clinical_logic.json`, `form.xlsx`, `clinical_logic.dmn`, `flowchart.md`
- [ ] Review **`Product/issues.txt`** and either close items or migrate to GitHub Issues
- [ ] Confirm **who pays** API invoices and **who holds** MOH relationship for clinical sign-off
- [ ] Export or archive **Neon/Redis** data policy (PII, retention) if user uploads go beyond test PDFs
- [ ] Update **this file** and root docs (`STATUS.md`, `RUNBOOK.md`, `ARTIFACTS.md`) together when behavior changes

---

## Ownership transfer notes

- **Intent (from Word doc):** Move toward **CHT open-source community** collaboration via a **time-bounded squad**; early involvement as **mentors**, then **review**, then hands-off support; named contacts in the source doc include **Mariam**, **Bailey** (needs assessment), and a commitment that **Atharva** (and the author of “Promise me…”) remain reachable for questions — **update names and contact channels here when ownership changes.**
- **Code ownership:** Day-to-day implementation lives in the **`Product` submodule**; issues and PRs should target that repository for application changes. This **`ChatCHW`** repo can hold umbrella policy PDFs, handoff materials, medical teaching artifacts, and integration tests that span folders.
- **Binary / non-code assets:** `Medical/*.xlsx`, `Medical/*.pptx` and external program briefs — ensure the receiving org has **license and redistribution** rights for WHO PDFs and internal decks.
- **Legacy:** Do not route new work through `old/`; it exists for archaeology only.

---

*If you update key stakeholder narrative, append a one-line change log here (date + summary) so git history stays traceable.*
