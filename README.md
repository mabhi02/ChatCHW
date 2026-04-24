# CHW Navigator — ChatCHW repository

This repository is the **umbrella workspace** for **CHW Navigator**: a verification-oriented pipeline that turns **Ministry of Health (MOH) and WHO community health worker (CHW) clinical manuals** into **deterministic, testable decision logic** and **deployment-ready artifacts** (for example DMN 1.3 XML, Mermaid flowcharts, and CHT-compatible XLSForms). Clinical content stays anchored to the source guideline; engineering supplies extraction, validation, compilation, and equivalence checks—not ad hoc “vibes-based” coding.

The `old/` directory is **legacy / superseded** material and is intentionally **out of scope** for this overview.

---

## What CHW Navigator is (in one paragraph)

CHW Navigator is a **closed-loop build-and-verify** system: ingest a manual, extract structured facts and logic with LLM assistance under strict gates, compile to **boolean-only** decision representations, run **deterministic validators and formal checks** (including Z3-style properties where wired), stress-test with **synthetic patients**, and emit bundles suitable for **CHT, OpenSRP, CommCare, or ODK**. The product philosophy—human sovereignty at critical gates, adversarial QA, mathematical consistency checks, and large-scale simulation—is documented in `Product/CHW Navigator v1 (1).md` and aligned with the staged vocabulary and quality gates in `Product/PIPELINE.md`.

---

## How the top-level folders connect

| Folder | Role in CHW Navigator |
|--------|------------------------|
| **`Guidelines/`** | **Authoritative clinical inputs** in PDF form: WHO CHW guidance and related policy (e.g. sick-child diarrhoea and pneumonia). These are the kinds of sources the pipeline is designed to ingest and cite. |
| **`Product/`** | **Primary implementation** ([`CHW_RLM`](https://github.com/atharvajpatel/CHW_RLM) Git submodule): FastAPI backend, Next.js frontend, PDF → structured guide ingestion (Unstructured + vision), **Gen 7** labeling + REPL extraction, validators, converters to DMN/XLSForm/Mermaid/CSV, eval harness, and the main engineering docs (`ARCHITECTURE.md`, `PIPELINE.md`, `ORCHESTRATOR.md`, etc.). |
| **`Medical/`** | **Clinical curriculum and verification collateral** alongside code: spreadsheet DMN modules (e.g. cough/pneumonia, fever/malaria), consolidated “Navigator modules,” and a **verification-first clinical engineering** deck for teaching or stakeholder review. These bridge **classroom / MOH sign-off** artifacts with what the product pipeline must preserve (cutpoints, modules, traceability). |
| **`Testing/`** | **Black-box and integration-style experiments** that connect “product JSON” to **DMN parsing**, **patient preprocessing**, **CHT harness** runs, and **execution logs**—without replacing the main test suite inside `Product/backend/tests/`. |
| **`Project Overview/`** | **Program narrative** (charter, value proposition) in Word form for sponsors and partners. |

End-to-end story: **`Guidelines/` PDFs** (or manuals copied under `Product/`) → **`Product/`** ingestion and extraction → validated **`clinical_logic`** → deterministic **DMN / XLSForm** → **`Medical/`**-style module tables and decks for review → **`Testing/`** harnesses and runners to compare behaviors and logs.

---

## Guidelines (`Guidelines/`)

- `WHO CHW guide 2012.pdf` — canonical large manual example used across product docs and sample data.
- `WHO Guidelines diarrhea & pneumonia 2024.pdf` — newer policy PDF aligned with sick-child modules (diarrhoea, respiratory) that Navigator-style logic must track when national programs adopt updated WHO guidance.

These sit **outside** the submodule so policy PDFs can be versioned or swapped per deployment without forking application code.

---

## Product (`Product/` — submodule)

Clone with submodules:

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

**Start here for engineering detail:**

- **`ARCHITECTURE.md`** — Gen 7 v2 ingest vs extraction split, labeling, REPL, SSE, session limits, reference run figures.
- **`PIPELINE.md`** — Full staged vocabulary (PROVENANCE_JSON → FACTSHEET → DMN_JSON → deployment formats), quality gates, A/B (REPL vs sequential) framing, cost and infra notes.
- **`ORCHESTRATOR.md`** — RLM/REPL rationale, eight-step loop from manual JSON to validated logic to converters.
- **`CHW Navigator v1 (1).md`** — “Gold master” philosophy: layers (instructions, meaning, decision tables, executable, testing), MOH gates, red team/repair, Z3 and synthetic patients.

**Code entry points (high level):**

- Ingestion: `Product/backend/ingestion/`
- Extraction: `Product/backend/gen7/`, `Product/backend/rlm_runner.py`
- Validation: `Product/backend/validators/`, `Product/backend/prompts/validators/`
- Converters: `Product/backend/converters/`
- Gate / eval: `Product/backend/eval/gate_harness.py`
- Web UI: `Product/frontend/`

The submodule also carries **`Automating_CHT_v25.xlsx`**, arena specs, final test artifacts under `Product/4-14-final-test-artifacts/`, and duplicate WHO PDFs used by the product repo’s own workflows.

---

## Medical (`Medical/`)

| Asset | Purpose |
|-------|---------|
| `SP26 Medical DMN Tables/*.xlsx` | **Teaching / module authoring**: worked DMN-style tables for cough (pneumonia), fever (malaria), and a consolidated “Navigator modules” workbook—useful as **ground truth examples** when comparing pipeline output to clinician-built tables. |
| `Verification-First Clinical Engineering.pptx` | **Stakeholder deck** on verification-first engineering for clinical decision support—pairs with the formal gates described in `Product/PIPELINE.md` and the governance language in `Product/CHW Navigator v1 (1).md`. |

Binary formats are not diff-friendly; treat this folder as **human-facing clinical source collateral**, not runtime config.

---

## Testing (`Testing/`)

Prototypes and experiments **named by contributor**, all oriented around the same question: *does compiled logic behave the same way across representations and harnesses?*

| Path | Focus |
|------|--------|
| **`Testing/Gigi/`** | `product_dmn_runner.py` — runs **`clinical_logic.json`** through router + module rules with structured logs; see `Testing/Gigi/README.md` for CLI modes (`built_in`, `dmn_ready_json`, `angelina_output`). |
| **`Testing/Angelina/dmn/`** | DMN XML parsing, preprocessing, simulated patients, and **`docs/`** notes on input schemas per module (cough, diarrhoea, fever/malaria, danger signs). |
| **`Testing/Aaron/cht-harness/`** and **`Testing/Aaron/cht-api/`** | **CHT / ODK-style** harness: Node (`mocha`), orchestrator **XML/XLSX** fixtures, `notes.md` on answer ordering and conditional fields for the sick-child-style orchestrator form. |

Together, **`Testing/`** bridges **`Product/backend`** outputs to **DMN tooling** and **mobile form runners**; use it when validating equivalence narratives from `Product/PIPELINE.md` outside the main Python test tree.

---

## Project overview (`Project Overview/`)

- `CHW Navigator Charter.docx` and `Value added of CHW Navigator.docx` — executive framing for **why** the system exists and **what** economic or programmatic value it targets, complementing the technical specs under `Product/`.

---

## What is intentionally excluded

- **`old/`** — Deprecated frontend/backend and integration sketches. It is **not** part of the current architecture described in `Product/ARCHITECTURE.md` and is omitted from integration instructions here.

---

## Quick navigation

| I want to… | Go to… |
|------------|--------|
| Understand the full artifact ladder and gates | `Product/PIPELINE.md` |
| Onboard to the running system (Gen 7) | `Product/ARCHITECTURE.md` |
| Understand REPL / RLM design choices | `Product/ORCHESTRATOR.md` |
| Read the product philosophy and MOH workflow | `Product/CHW Navigator v1 (1).md` |
| Run JSON logic against sample patients | `Testing/Gigi/README.md` |
| Wire CHT harness tests | `Testing/Aaron/cht-harness/notes.md` |
| Compare with hand-built DMN spreadsheets | `Medical/SP26 Medical DMN Tables/` |

If documentation and code disagree, **trust the submodule `Product/` on `main`** for the shipped behavior and keep this README as the **map**, not the spec.
