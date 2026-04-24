# CHW Navigator — provenance trace (example)

A practical trace template showing how one clinical recommendation should be explainable across artifacts.

Detailed schema fields and implementation behavior are defined in `Product/`.

---

## Trace path

1. **Manual statement** (source text / table in guideline PDF)
2. **Structured guide representation** (`guide_json` section/block)
3. **Extracted logic artifact** (`clinical_logic.json` predicate/module/router row)
4. **Compiled DMN row** (`clinical_logic.dmn`)
5. **Compiled form expression** (`form.xlsx` calculate/relevant path)
6. **Harness/test outcome** (`Testing/` execution logs)

---

## Trace template

Use this table for any key rule (diagnosis, referral, treatment, follow-up):

| Layer | Evidence | Location |
|---|---|---|
| Manual quote | Exact text/table excerpt | `Guidelines/...pdf` + review notes |
| Structured element | Section/block id and parsed text | `guide_json` / ingestion artifacts |
| Logic rule | Predicate/module/routing condition | `clinical_logic.json` |
| DMN mapping | Decision table + rule row | `clinical_logic.dmn` |
| Form mapping | XLSForm calculate/relevant expression | `form.xlsx` |
| Runtime check | Expected vs observed output | `Testing/...` logs |

---

## Usage

- Use for incident analysis.
- Use for clinician sign-off walkthroughs.
- Use to justify behavior changes after guideline updates.

