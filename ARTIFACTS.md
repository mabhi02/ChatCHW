# CHW Navigator — artifact contracts

This file defines umbrella-level artifact contracts (producer, consumer, checks).
Deep schemas and implementation details live in `Product/`.

---

## Contract table

| Artifact | Produced by | Consumed by | Validation / checks | Canonical location |
|---|---|---|---|---|
| Manual PDF | `Guidelines/` or uploaded guide | Ingestion pipeline | File ingestion + parsing checks | `Guidelines/`, `Product/` uploads |
| `guide_json` | Ingestion (`Product/backend/ingestion/`) | Gen7 chunking/labeling/extraction | Ingestion integrity checks | Runtime/cache (see Product docs) |
| Labeled chunks / deduped labels | Gen7 labeler | Gen7 REPL extraction | Label parsing and dedup checks | Run outputs under `Product/backend/output/` |
| `clinical_logic.json` | Gen7 extraction | Converters, tests, harnesses | Validators + optional formal checks | Run outputs under `Product/backend/output/` |
| `clinical_logic.dmn` | Deterministic DMN converter | DMN tools, comparison workflows | Converter checks + downstream parse | Run outputs |
| `form.xlsx` | Deterministic XLSForm converter | CHT-style harnesses / deploy workflows | Converter checks + harness execution | Run outputs |
| `flowchart.md` | Mermaid converter | Human review, workshop QA | Visual sanity check | Run outputs |
| `predicates.csv` / `phrases.csv` | CSV converter | Audits/localization/review | Schema / downstream usage checks | Run outputs |
| Medical reference DMN tables (`*.xlsx`) | Medical team | Comparison workflow | Human review + scenario comparison | `Medical/SP26 Medical DMN Tables/` |

---

## Ownership model

- **Pipeline runtime truth:** `Product/`
- **Umbrella flow and cross-team process:** root docs (`workflow.md`, `QUALITY_AND_VERIFICATION.md`, `PLATFORM_INTEGRATION.md`)
- **Reference clinical intent artifacts:** `Medical/`
- **Cross-check harnesses:** `Testing/`

---

## Change policy

When artifact shape changes:

1. Update implementation in `Product/`.
2. Update schema/contract docs in `Product/` first.
3. Update this umbrella contract table (producer/consumer/check columns).
4. Update `workflow.md` if flow edges changed.

