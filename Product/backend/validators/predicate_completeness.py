"""Detect stub predicates and emit a structured report.

Stub predicates are auto-registered placeholders inserted by
`referential_integrity.audit_and_reconcile` when Stage 3 references a
predicate id (in some module rule) but never *defines* it. The reconciler
inserts an entry just so downstream conversion does not crash; the
predicate has no `formal_definition` / `inputs_used` / `units` /
`missingness_rule` / `provenance`.

These are landmines for clinical safety: when a DMN rule fires
`p_can_drink`, the engine has no way to compute it. We therefore:

  1. Detect every stub at run time
  2. Emit `predicate_completeness.json` listing them
  3. Surface each stub as a `severity: warn` divergence the verifier
     worklist picks up automatically.

The Predicate Table spec (Section 7) explicitly forbids hidden logic:
"All thresholds must appear here. No hidden logic outside the predicate
table." Stubs are exactly that hidden logic.
"""

from __future__ import annotations


REQUIRED_FIELDS_FOR_DEFINITION: tuple[str, ...] = (
    "formal_definition",
    "inputs_used",
    "units",
    "missingness_rule",
    "provenance",
)


def _is_stub(p: dict) -> bool:
    """Return True iff `p` looks like an auto-registered placeholder."""
    if p.get("_auto_registered"):
        # Reconciler-inserted placeholders are stubs unless somebody
        # filled in a definition by hand later.
        if not (p.get("formal_definition") or p.get("threshold_expression")):
            return True
    # Non-reconciler stubs: predicate without a usable formal_definition AND
    # without inputs_used. Treat as stub regardless of provenance.
    if not (p.get("formal_definition") or p.get("threshold_expression")):
        if not (p.get("inputs_used") or p.get("source_vars")):
            return True
    return False


def detect_stubs(predicates: list[dict]) -> list[dict]:
    """Return the list of predicates that look like undefined stubs."""
    stubs: list[dict] = []
    for p in predicates:
        if not isinstance(p, dict):
            continue
        if _is_stub(p):
            missing = [
                f for f in REQUIRED_FIELDS_FOR_DEFINITION
                if not p.get(f)
            ]
            stubs.append({
                "id": p.get("id", "?"),
                "human_label": p.get("human_label") or p.get("description_clinical") or "",
                "missing_fields": missing,
                "auto_registered": bool(p.get("_auto_registered")),
                "source_quote": p.get("source_quote", ""),
            })
    return stubs


def build_report(predicates: list[dict]) -> dict:
    """Build the `predicate_completeness.json` payload."""
    stubs = detect_stubs(predicates)
    return {
        "summary": {
            "total_predicates": len(predicates),
            "stub_count": len(stubs),
            "stub_pct": round(100.0 * len(stubs) / max(1, len(predicates)), 1),
        },
        "stubs": stubs,
    }


def to_divergences(predicates: list[dict]) -> list[dict]:
    """Render stubs as verification divergences for the worklist.

    Each stub becomes one `severity: warn` divergence so the reviewer
    sees them ranked alongside the verifier's other findings.
    """
    out: list[dict] = []
    for s in detect_stubs(predicates):
        out.append({
            "type": "stub_predicate",
            "severity": "warn",
            "detail": (
                f"Predicate {s['id']!r} is a stub: missing "
                f"{', '.join(s['missing_fields'])}. "
                "Stage 3 referenced this predicate in a rule but never defined it; "
                "the auto-reconciler inserted a placeholder. "
                "Per Predicate Table spec section 7, all thresholds must appear "
                "in the predicate table -- no hidden logic."
            ),
            "evidence": {
                "id": s["id"],
                "missing_fields": s["missing_fields"],
                "auto_registered": s["auto_registered"],
            },
        })
    return out
