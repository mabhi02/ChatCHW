# Medical-team workbook normalization (`Testing/Gigi/medical_team`)

Transforms the four clinical Excel workbooks in `Testing/Gigi/` into **machine-readable JSON bundles** under `normalized_rules/`.  
This step does **not** execute rules—it only preserves structure for a future `medical_team_runner.py` and comparisons with product-team logs.

## Workbooks consumed (inputs)

Reads these files alongside this folder (`../` from here):

| File | Artifact |
|------|----------|
| `Navigator Modules.xlsx` | Consolidated navigator (opening, orchestration stubs, diarrhea tables, dosing) |
| `Diarrhea Module.xlsx` | Stand-alone diarrhea sheet |
| `Fever (Malaria) Module.xlsx` | Diagnosis / treatment / integrative tables |
| `Pneumonia (Cough) Module.xlsx` | Diagnosis / treatment tables |

## Outputs

| JSON | Source workbook |
|------|-----------------|
| `normalized_rules/navigator.json` | `Navigator Modules.xlsx` |
| `normalized_rules/diarrhea.json` | `Diarrhea Module.xlsx` |
| `normalized_rules/fever_malaria.json` | `Fever (Malaria) Module.xlsx` |
| `normalized_rules/pneumonia_cough.json` | `Pneumonia (Cough) Module.xlsx` |

Each workbook file is emitted as `{ "source_file", "normalized_at_iso", "modules": [ ... ] }`.  
Inside `modules`, each workbook tab/sheet produces one logical module (`module_id`). **Opening Module** emits one module with `"sections"` (two header blocks merged for traceability). **Diarrhea dosing** uses `"kind": "embedded_dosing_rule_blocks"` because the sheet nests multiple `"Rule"` micro-tables in column **B**.

## How to run

```bash
cd Testing/Gigi/medical_team
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python normalize_medical_modules.py
```

## Configuration

`sheet_config.json` holds per-sheet options:

- **`mode`**: `"rule_table"` (default grids), `"opening_sections"`, `"embedded_rule_blocks"`, or inferred defaults.
- **`header_row`** / **`first_data_row`** for classic rule rows.
- **`stop_col_a_contains`**: substring list for column **A** to stop ingestion (captures `Notes:` / `Comments`).
- Duplicate Excel headers (`ref_feeding_program` duplicated) become `__dup_*` suffixes in JSON keys.

## Classification (conditions vs outputs)

Heuristics (`_role_for_header`):

- Typical **conditions**: columns starting `q_`, many `ex_`, MUAC predicates `lab_muac_*`, demographics `demo_*`.
- Typical **outputs**: `tx_`, `rx_`, `adv_`, referrals `ref_health_*`, integrative flags `need_*`, diagnosis-derived `dx_*` on diagnosis sheets only.
- **Treatment** sheets treat `dx_*` as **inputs** preconditions (`treatment_dx_as_condition`).
- **Traffic cop grids**: route label (`Start?`) → `metadata.route_scenario`, last **`Results`** column → `outputs`.
- **Opening module**: branching columns (`YES` / `NO` / `'DON'T KNOW'` / …) preserved verbatim as `outputs`; prompt text under `Question`/`Observation`.

**Blank cells**: omitted entirely from both `conditions` and `outputs` (don’t-know / wildcard intent is ambiguous in Excel—we keep only populated cells).

## Sheets explicitly ignored within a tab

- **Thin / Swollen navigator sheets**: appendix blocks after `Traffic cop#` grids (marketing-style navigation) skipped once column **A** matches `traffic cop`.
- **`Notes:`/`Comments` rows**: stop ingestion for Fever/Pneumonia/Diarrhea navigator diagnosis tables.
- Rows that are visually empty across all modeled columns are skipped.

## Follow-up manual review hints

Open questions for medical owners:

1. Cells mixing guidance + citations (`YES (pp. 66)`), `ANY`, `AGE-THRESHOLD`, etc.—kept verbatim; interpreters must not parse medically yet.
2. Some spreadsheets embed **pseudo-rules** (e.g., fast-breathing definition rows) identical to diagnoses—flag later if runners should classify them separately.
3. Diarrhea dosing embedder guesses `Age_between_*` predicates as **conditions**; other dosing knobs default to **outputs**—easy to revise in mapping if needed.

## Relationship to product-team runner

Untouched artifacts under `Testing/Gigi/` (`clinical_logic.json`, `product_dmn_runner.py`, …).  
Medical JSON here is staged for a future parallel runner emitting the same coarse log envelopes.
