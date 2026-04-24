# CHW Navigator — platform integration guide (deployment-facing)

This document describes integration expectations for real-world deployment targets.
Implementation specifics belong to `Product/` and target platform docs.

---

## Integration principle

Compile once from reviewed logic artifacts, then deploy deterministic assets into existing CHW ecosystems (for example CHT-compatible XLSForm flows), preserving offline behavior and auditability.

---

## Supported integration modes (conceptual)

| Mode | Artifact | Typical consumer |
|---|---|---|
| Decision-model review | `clinical_logic.dmn` | Clinical reviewers, DMN tooling |
| Mobile form deployment | `form.xlsx` | CHT/ODK-like form runtimes |
| Visual review | `flowchart.md` | Clinical and implementation review workshops |
| Audit/export | CSV extracts | QA, localization, governance |

---

## Deployment checklist

- [ ] Verify deterministic outputs from the same reviewed `clinical_logic.json`.
- [ ] Confirm offline-safe behavior for the target workflow.
- [ ] Validate required platform constraints (field types, calculations, ordering).
- [ ] Run comparison scenarios through target harness and JSON/DMN baselines.
- [ ] Record known deviations and acceptance rationale.
- [ ] Confirm no unintended network side effects in generated forms/workflows.

---

## Integration test expectations

Minimum for each release candidate:

1. One end-to-end extraction on representative guide input.
2. Converter outputs generated and opened successfully (DMN/XLSForm/Mermaid).
3. At least one harness execution in `Testing/` recorded.
4. Medical reference comparison performed on agreed synthetic scenarios.
5. Release note captures behavior deltas and unresolved risks.

---

## Primary references

- `workflow.md` (umbrella information flow)
- `QUALITY_AND_VERIFICATION.md` (quality model)
- `Product/ARCHITECTURE.md` (current implementation behavior)
- `Testing/` (execution harnesses)

