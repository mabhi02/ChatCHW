# CHW Navigator — ChatCHW

**ChatCHW** is the umbrella repository for **CHW Navigator**: a verification-oriented pipeline that turns **MOH/WHO CHW clinical manuals** into **deterministic decision logic** (DMN, JSON) and **deployment-ready artifacts** (XLSForm, Mermaid, CSVs). AI is used in the **build** phase; the **runtime** story targets familiar, offline-first forms—not generative advice at the bedside.

---

## Repository layout

```
ChatCHW/
├── README.md              ← You are here (map of the umbrella repo)
├── workflow.md            End-to-end information flow (manual → logic → tests)
├── value_added.md         Stakeholder value narrative (Markdown)
├── handoff.md             Engineering handoff, risks, checklist
├── Guidelines/            Source policy PDFs (WHO / national manuals)
├── Medical/               Clinical reference DMNs (spreadsheets) + verification deck
├── Product/               Git submodule → CHW_RLM (FastAPI + Next.js pipeline)
├── Testing/               Cross-check harnesses (DMN runner, CHT prototype, etc.)
└── old/                   Legacy code — do not use for current work
```

---

## Documentation at the repository root

| File | Use when you need… |
|------|-------------------|
| **`workflow.md`** | How a **manual** becomes **`guide_json`**, **`clinical_logic.json`**, **DMN/XLSForm**, then runs through **`Product/` tests** and **`Testing/`** comparisons to **medical-team** reference tables. |
| **`value_added.md`** | The **why**: CHW constraints, status quo vs Navigator, value propositions, references (Markdown restatement of the former Word “value added” brief). |
| **`handoff.md`** | **Ownership transfer**: architecture summary, done vs in-progress, risks (including audit notes), links into `Product/`, handoff checklist. |

---

## `Guidelines/`

Authoritative **clinical inputs** in PDF form (versioned here so deployments can swap manuals without forking the app submodule).

- Example: WHO CHW sick-child guidance and related diarrhoea/pneumonia policy PDFs.

---

## `Medical/`

**Human-facing clinical collateral** (not consumed directly by the Gen 7 compiler today):

| Content | Role |
|---------|------|
| **`SP26 Medical DMN Tables/*.xlsx`** | Reference / teaching **DMN-style modules** (e.g. cough/pneumonia, fever/malaria) for **comparison** with pipeline output. |
| **`Verification-First Clinical Engineering.pptx`** | Stakeholder deck on **verification-first** engineering. |

Use these in **workshops and QA**: same synthetic case → spreadsheet intent vs **Product** DMN / JSON runner / harness.

---

## `Product/` (submodule [`CHW_RLM`](https://github.com/atharvajpatel/CHW_RLM))

All **runnable** backend and frontend code, plus deep engineering specs.

```bash
git clone --recurse-submodules <repo-url>
# or: git submodule update --init --recursive
```

| Doc (in `Product/`) | Topic |
|---------------------|--------|
| **`ARCHITECTURE.md`** | Gen 7 v2: ingest vs extraction, labeling, REPL, SSE, sessions, reference run. |
| **`PIPELINE.md`** | Staged artifact vocabulary, quality gates, long-horizon research pipeline. |
| **`ORCHESTRATOR.md`** | RLM/REPL rationale and loop design. |
| **`CHW Navigator v1 (1).md`** | “Gold master” philosophy: MOH gates, red team, Z3, synthetic patients. |

**Code entry points:** `Product/backend/ingestion/` → `Product/backend/gen7/`, `rlm_runner.py` → `validators/`, `converters/` → `frontend/`. Eval: `Product/backend/eval/gate_harness.py`. Tests: `Product/backend/tests/`.

---

## `Testing/`

Prototypes that connect **product JSON** to **DMN tooling** and **CHT-style** runners (they do **not** replace `Product/backend/tests/`).

| Path | Focus |
|------|--------|
| **`Testing/Gigi/`** | `product_dmn_runner.py` on `clinical_logic.json` — see `Testing/Gigi/README.md`. |
| **`Testing/Angelina/dmn/`** | DMN XML parse/preprocess, simulated patients, `docs/` per-module schemas. |
| **`Testing/Aaron/cht-harness/`**, **`cht-api/`** | Orchestrator form harness (`notes.md`, fixtures). |

---

## `old/`

Deprecated frontend/backend sketches. **Out of scope** for current architecture; kept for archaeology only.

---

## Quick links

| Goal | Location |
|------|----------|
| Full artifact ladder & quality gates | `Product/PIPELINE.md` |
| Run and ship Gen 7 | `Product/ARCHITECTURE.md` |
| REPL / RLM design | `Product/ORCHESTRATOR.md` |
| Product philosophy & MOH workflow | `Product/CHW Navigator v1 (1).md` |
| End-to-end data flow in this repo | **`workflow.md`** |
| Stakeholder value story | **`value_added.md`** |
| Maintainer handoff | **`handoff.md`** |
| JSON logic vs sample patients | `Testing/Gigi/README.md` |
| CHT harness field order & conditionals | `Testing/Aaron/cht-harness/notes.md` |
| Hand-built DMN spreadsheets | `Medical/SP26 Medical DMN Tables/` |

---

## Source of truth

If umbrella docs and **`Product/`** disagree on behavior, **`Product/` on `main`** wins. This README is the **map**; submodule docs and code are the **spec** for the running system.
