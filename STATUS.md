# CHW Navigator — status board

This file is the umbrella repo's quick status view. For deep implementation details, `Product/` docs are the source of truth.

**Last updated:** 2026-04-24

---

## Scope and source of truth

- **Current runtime implementation:** `Product/ARCHITECTURE.md`
- **Current/target pipeline vocabulary:** `Product/PIPELINE.md`
- **RLM/REPL orchestration rationale:** `Product/ORCHESTRATOR.md`

When this file and `Product/` disagree, trust `Product/`.

---

## Shipped (current)

| Capability | Status | Owner path | Evidence |
|---|---|---|---|
| Manual ingestion (PDF -> structured guide) | Current | `Product/backend/ingestion/` | `Product/ARCHITECTURE.md` ingest flow |
| Gen7 extraction (`guide_json` -> `clinical_logic.json`) | Current | `Product/backend/gen7/`, `Product/backend/rlm_runner.py` | Reference runs in `Product/backend/output/` |
| Deterministic converters (DMN/XLSForm/Mermaid/CSV) | Current | `Product/backend/converters/` | Generated artifacts in output folders |
| Backend+frontend runtime app | Current | `Product/backend/`, `Product/frontend/` | `Product/ARCHITECTURE.md` |
| Core automated tests | Current | `Product/backend/tests/` | `pytest` suite |
| Cross-check harnesses | Current | `Testing/` | Gigi/Angelina/Aaron toolchains |

---

## In progress

| Capability | Status | Why it matters | Primary refs |
|---|---|---|---|
| Stable equivalence workflow vs medical reference DMNs | In progress | Real-world deployment confidence requires repeatable comparison criteria | `workflow.md`, `Testing/`, `Medical/` |
| Determinism and semantic diff checks | In progress | Distinguish harmless reorderings from behavioral drift | `Product/ARCHITECTURE.md` open issues |
| Pipeline docs harmonization (spec vs shipped) | In progress | Reduce confusion between long-horizon pipeline and Gen7 shipped path | `Product/PIPELINE.md`, `workflow.md` |

---

## Planned

| Capability | Status | Notes |
|---|---|---|
| Formal artifact contracts and owners | Planned | Introduced in `ARTIFACTS.md`; operationalize in CI over time |
| Quality gate matrix by risk type | Planned | Introduced in `QUALITY_AND_VERIFICATION.md`; link to concrete checks per release |
| Deployment integration conformance checklist | Planned | Introduced in `PLATFORM_INTEGRATION.md` |
| Contributor "single command" run experience | Planned | Captured in `RUNBOOK.md` as baseline manual runbook |

---

## Deprecated / out of scope

- `old/` is legacy and not part of current architecture.

