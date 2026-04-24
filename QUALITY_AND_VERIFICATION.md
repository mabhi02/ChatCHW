# CHW Navigator — quality, verification, and red-teaming

High-level quality model for real-world deployment readiness.
Detailed validators/prompts/checkers are maintained in `Product/`.

---

## Quality model (layers)

1. **Clinical source governance**
   - Manuals/addenda reviewed by clinicians.
   - Ambiguities (stockouts, thresholds, missing paths) are surfaced and resolved.

2. **Extraction quality controls**
   - Structured ingestion checks.
   - Labeling and extraction consistency checks.

3. **Artifact-level validation**
   - Deterministic validators for structural integrity.
   - LLM-assisted catchers/red-team style checks where configured.

4. **Deterministic conversion checks**
   - JSON -> DMN/XLSForm/Mermaid converters must preserve logic intent.

5. **Behavioral equivalence checks**
   - Synthetic scenarios and harness runs across JSON/DMN/XLSForm paths.
   - Comparison against medical team reference tables.

6. **Release readiness**
   - Risk acceptance, traceability, and documented sign-off path.

---

## Risk-to-check matrix

| Risk type | Primary controls | Where implemented |
|---|---|---|
| Missing clinical paths (e.g. stockout) | Gap analysis + red-team checks + clinician review | `Product/` prompts/validators + review process |
| Inconsistent logic between formats | Deterministic converters + cross-harness comparisons | `Product/backend/converters/`, `Testing/` |
| Silent regressions after updates | Test suite + scenario replay + diff process | `Product/backend/tests/`, `workflow.md` |
| Hallucination at point of care | No generative runtime inference at bedside | Deployment model described in `value_added.md` |
| Ambiguous deployment behavior | Platform integration checklist and constraints | `PLATFORM_INTEGRATION.md` |

---

## Red-team and verification expectations

- Treat red-team output as triage evidence, not auto-truth.
- Require reproducible test case for critical findings.
- Keep a closed loop:
  - finding -> fix -> test -> documented disposition.
- Ensure each release documents:
  - what was tested,
  - what remains unverified,
  - who accepted residual risk.

---

## Source-of-truth links

- `Product/PIPELINE.md` (quality gates and staged architecture)
- `Product/ARCHITECTURE.md` (current implementation/testing details)
- `Product/HYBRID_AUDIT_REPORT.md` (hybrid validation context)
- `Product/backend/validators/` and `Product/backend/prompts/validators/`

