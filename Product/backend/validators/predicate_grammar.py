"""Tier-0 predicate grammar validator + triplet deduper.

Enforces the gen8 strict predicate schema:
  - id starts with `p_`
  - operators in `formal_definition` limited to `> < >= <= = and or not ( )`
  - banned tokens: `==`, uppercase `AND/OR/NOT`, quoted string literals
  - required fields: id, description_clinical, inputs_used, formal_definition,
    units, missingness_rule, allowed_input_domain, rounding_parsing_rule,
    provenance
  - `missingness_rule` in {FALSE_IF_MISSING, TRIGGER_REFERRAL, BLOCK_DECISION,
    ALERT_NO_RULE_SPECIFIED}

Also collapses `p_X` / `p_has_X` / `p_X_days` triplets to a single canonical
predicate (shortest id wins).
"""

from __future__ import annotations

import re

BANNED_TOKENS: tuple[str, ...] = ("==", " AND ", " OR ", " NOT ", '"', "'")
REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "description_clinical",
    "inputs_used",
    "formal_definition",
    "units",
    "missingness_rule",
    "allowed_input_domain",
    "rounding_parsing_rule",
    "provenance",
)
VALID_MISSINGNESS: frozenset[str] = frozenset({
    "FALSE_IF_MISSING",
    "TRIGGER_REFERRAL",
    "BLOCK_DECISION",
    "ALERT_NO_RULE_SPECIFIED",
})

# Canonical operator set. We allow lowercase `and/or/not` as keywords plus
# the comparison operators. Anything matching the banned list still fails
# even if it passes this permissive regex.
_FORMAL_ALLOWED = re.compile(r"^[a-zA-Z0-9_\s\(\)\.\,\[\]\-\+\*/><=]+$")

_PROVENANCE_RE = re.compile(r"^(WHO|UNICEF|MOH|PIH)\s+\d{4}\s+page\s+\d+", re.IGNORECASE)


def validate_predicate(p: dict) -> list[str]:
    """Return a list of validation error strings. Empty = valid."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in p:
            errors.append(f"missing required field: {field}")
            continue
        val = p.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            # `units` is allowed to be "(boolean)" so non-empty is enforced
            errors.append(f"empty required field: {field}")

    pid = str(p.get("id") or "")
    if pid and not pid.startswith("p_"):
        errors.append(f"id must start with 'p_': {pid!r}")

    expr = str(p.get("formal_definition") or "")
    if expr:
        for tok in BANNED_TOKENS:
            if tok in expr:
                errors.append(f"banned token in formal_definition: {tok!r}")
        if not _FORMAL_ALLOWED.match(expr):
            errors.append(f"formal_definition contains disallowed characters: {expr!r}")

    miss = p.get("missingness_rule")
    if miss not in VALID_MISSINGNESS:
        errors.append(f"invalid missingness_rule: {miss!r}")

    prov = str(p.get("provenance") or "")
    if prov and not _PROVENANCE_RE.match(prov):
        errors.append(f"provenance must match '<PUBLISHER> <YEAR> page <N>': {prov!r}")

    return errors


def _concept_of(pid: str) -> str:
    """Normalize a predicate id down to its bare concept for dedup.

    `p_has_fever` -> `fever`
    `p_fever_days` -> `fever`
    `p_fever` -> `fever`
    """
    concept = pid
    for prefix in ("p_has_", "p_"):
        if concept.startswith(prefix):
            concept = concept[len(prefix):]
            break
    for suffix in ("_days", "_count", "_value"):
        if concept.endswith(suffix):
            concept = concept[: -len(suffix)]
            break
    return concept


def dedup_predicate_triplets(predicates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse p_X / p_has_X / p_X_days triplets to one canonical entry.

    Returns (kept, dropped). `dropped` entries carry a `_dedup_reason` field
    referencing the canonical id so the worklist can explain what happened.
    """
    by_concept: dict[str, list[dict]] = {}
    missing_id: list[dict] = []
    for p in predicates:
        pid = str(p.get("id") or "")
        if not pid:
            entry = dict(p)
            entry["_dedup_reason"] = "missing_id"
            missing_id.append(entry)
            continue
        by_concept.setdefault(_concept_of(pid), []).append(p)

    kept: list[dict] = []
    dropped: list[dict] = list(missing_id)
    for concept, group in by_concept.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Prefer shortest id (most canonical). Stable tie-breaker on raw id.
        group.sort(key=lambda q: (len(str(q.get("id", ""))), str(q.get("id", ""))))
        kept.append(group[0])
        for d in group[1:]:
            out = dict(d)
            out["_dedup_reason"] = f"duplicate of {group[0].get('id')!r} (concept={concept})"
            dropped.append(out)
    return kept, dropped


def validate_all(predicates: list[dict]) -> dict:
    """Run `validate_predicate` over the whole list and return a summary dict."""
    per_pred: list[dict] = []
    for p in predicates:
        errs = validate_predicate(p)
        per_pred.append({"id": p.get("id", "?"), "errors": errs})
    total_errors = sum(len(x["errors"]) for x in per_pred)
    return {
        "total": len(predicates),
        "with_errors": sum(1 for x in per_pred if x["errors"]),
        "total_errors": total_errors,
        "per_predicate": per_pred,
    }
