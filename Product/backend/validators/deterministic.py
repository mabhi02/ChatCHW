"""Deterministic phase validators — pure Python, no LLM calls.

These replace three of the eight LLM-based catchers with deterministic
Python checks. The remaining catchers (clinical_review, completeness,
comorbidity_coverage, dmn_audit, boundary_condition) stay LLM-based but
run with temperature=0 and majority vote across N=3 calls in
phases.py::_call_catcher_majority.

Why deterministic where possible: LLM catchers are stochastic even at
temperature=0. For structural invariants that can be expressed as code,
deterministic checks eliminate that variance entirely. The April 17 gate
run cannot afford catcher-level variance to be confused with system-prompt
variance — the whole point of the gate is to measure system prompt effects
in isolation.

Each function returns the same dict shape as _call_catcher in phases.py:

    {
        "passed": bool,
        "critical_issues": [str, ...],
        "warnings": [str, ...],
    }

The functions are pure (no side effects, no I/O). They are sync, not async,
because they don't make API calls.

Mapping of which LLM catchers each function replaces:

    check_provenance         <- provenance_json.txt
    check_consistency        <- consistency_json.txt
    check_module_architecture <- module_architecture_json.txt

The remaining LLM catchers continue to live in backend/prompts/validators/
and are invoked via _call_catcher_majority in phases.py.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Deterministic check 1: provenance
# ---------------------------------------------------------------------------


def check_provenance(artifact: Any, guide_json: Optional[dict] = None) -> dict:
    """Walk the artifact and verify every entry that should have provenance has it.

    A "provenance-bearing entry" is any dict that contains at least one of:
        - source_section_id
        - source_quote
        - page_ref

    For each such entry, this check requires:
        - At least one of {source_section_id, page_ref} is present and non-empty
          (CRITICAL: blocks emit_artifact)
        - source_quote is present and non-empty
          (WARNING: surfaces but does not block)

    The walk is recursive over dicts and lists, so the artifact can be
    nested arbitrarily deep. The path string in error messages uses JSONPath-
    like notation so the model can locate the offending entry quickly.
    """
    critical_issues: list[str] = []
    warnings: list[str] = []

    def walk(obj: Any, path: str = "$") -> None:
        if isinstance(obj, dict):
            has_prov_marker = any(
                k in obj for k in ("source_section_id", "source_quote", "page_ref")
            )
            if has_prov_marker:
                section_id = obj.get("source_section_id") or obj.get("page_ref")
                quote = obj.get("source_quote")
                if not section_id or (isinstance(section_id, str) and not section_id.strip()):
                    critical_issues.append(
                        f"{path}: provenance entry missing source_section_id or page_ref"
                    )
                if not quote or (isinstance(quote, str) and not quote.strip()):
                    warnings.append(
                        f"{path}: provenance entry missing source_quote"
                    )
            for k, v in obj.items():
                walk(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(artifact)
    return {
        "passed": len(critical_issues) == 0,
        "critical_issues": critical_issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Deterministic check 2: consistency
# ---------------------------------------------------------------------------


def check_consistency(artifact: Any, guide_json: Optional[dict] = None) -> dict:
    """Check structural consistency: duplicate IDs and substring collisions.

    Specifically:
        - For each of {module_id, predicate_id, rule_id}: collect all values
          anywhere in the artifact and flag duplicates as CRITICAL.
        - For the same ID sets: flag substring collisions as WARNINGS.
          A substring collision is when one ID is contained in another
          (e.g., "mod_fever" is contained in "mod_fever_malaria"), which
          breaks delimiter-based set membership checks unless the runtime
          uses pipe delimiters or similar.

    The check is conservative: it does not infer the IDs from context;
    it only looks for explicit fields named module_id, predicate_id, rule_id.
    Catches the failure modes the consistency_json catcher prompt was checking
    for, without depending on an LLM call.
    """
    critical_issues: list[str] = []
    warnings: list[str] = []

    def collect_ids(obj: Any, key_name: str) -> list[str]:
        ids: list[str] = []
        if isinstance(obj, dict):
            v = obj.get(key_name)
            if isinstance(v, str) and v.strip():
                ids.append(v)
            for child in obj.values():
                ids.extend(collect_ids(child, key_name))
        elif isinstance(obj, list):
            for child in obj:
                ids.extend(collect_ids(child, key_name))
        return ids

    for key_name in ("module_id", "predicate_id", "rule_id"):
        ids = collect_ids(artifact, key_name)
        if not ids:
            continue
        # Duplicate check
        seen: set[str] = set()
        for i in ids:
            if i in seen:
                critical_issues.append(f"Duplicate {key_name}: {i!r}")
            seen.add(i)
        # Substring collision check (only on unique IDs)
        unique_sorted = sorted(seen, key=len)
        for i, a in enumerate(unique_sorted):
            for b in unique_sorted[i + 1 :]:
                if a != b and a in b:
                    warnings.append(
                        f"{key_name} substring collision: {a!r} is a prefix/substring of "
                        f"{b!r} (may break delimiter-unsafe set membership checks)"
                    )

    return {
        "passed": len(critical_issues) == 0,
        "critical_issues": critical_issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Deterministic check 3: module architecture (Activator/Router/Integrative)
# ---------------------------------------------------------------------------


def check_module_architecture(artifact: Any, guide_json: Optional[dict] = None) -> dict:
    """Check Activator/Router/Module/Integrative structural invariants.

    Specifically:
        - If the artifact (or a nested dict) is an Activator, hit_policy must
          be COLLECT.
        - If the artifact (or a nested dict) is a Router, hit_policy must be
          FIRST.
        - For a Router with rules, the first rule (row 0) should reference
          the pipeline's canonical composite alert predicate
          (`p_danger_sign_present` by convention, regardless of what the
          source manual calls its high-priority criteria). This is a
          WARNING because some catcher output formats allow variance in
          how the predicate is textually named.
        - Every module_id referenced in the activator's outputs must have a
          corresponding rule in the router (CRITICAL).

    The check inspects nested dicts so it works whether the artifact is a
    full clinical_logic dict or a single artifact (router-only, integrative-
    only, etc.).
    """
    critical_issues: list[str] = []
    warnings: list[str] = []

    if not isinstance(artifact, dict):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [
                f"Artifact is not a dict ({type(artifact).__name__}); "
                f"skipping architecture check"
            ],
        }

    # Resolve activator: either at top-level key "activator" OR the artifact
    # itself if it has hit_policy == COLLECT
    activator: Optional[dict] = None
    if isinstance(artifact.get("activator"), dict):
        activator = artifact["activator"]
    elif artifact.get("hit_policy") == "COLLECT":
        activator = artifact

    if activator is not None:
        hp = activator.get("hit_policy")
        if hp and hp != "COLLECT":
            critical_issues.append(
                f"Activator must have hit_policy=COLLECT, found {hp!r}"
            )
        elif not hp:
            warnings.append("Activator missing explicit hit_policy field")

    # Resolve router: either at top-level key "router" OR the artifact itself
    router: Optional[dict] = None
    if isinstance(artifact.get("router"), dict):
        router = artifact["router"]
    elif artifact.get("hit_policy") == "FIRST" and "rules" in artifact:
        router = artifact

    if router is not None:
        hp = router.get("hit_policy")
        if hp and hp != "FIRST":
            critical_issues.append(
                f"Router must have hit_policy=FIRST, found {hp!r}"
            )
        elif not hp:
            warnings.append("Router missing explicit hit_policy field")

        # Row 0 emergency-short-circuit check
        rules = router.get("rules") or []
        if rules:
            first_rule = rules[0]
            if isinstance(first_rule, dict):
                first_str = (
                    str(first_rule.get("condition", ""))
                    + " "
                    + str(first_rule.get("inputs", ""))
                ).lower()
                emergency_markers = (
                    "danger",
                    "emergency",
                    "p_danger",
                    "short_circuit",
                    "urgent_referral",
                )
                if not any(marker in first_str for marker in emergency_markers):
                    warnings.append(
                        "Router row 0 may not be the emergency short-circuit; "
                        "expected to reference a danger-sign predicate "
                        "(p_danger_sign_present, p_emergency, etc.)"
                    )

        # module_id reachability: every module referenced in the activator
        # should have a router entry
        if activator is not None:
            activator_modules: set[str] = set()
            for rule in activator.get("rules") or []:
                if isinstance(rule, dict) and rule.get("module_id"):
                    activator_modules.add(rule["module_id"])
            router_modules: set[str] = set()
            for rule in rules:
                if isinstance(rule, dict):
                    next_mod = rule.get("next_module") or rule.get("module_id")
                    if next_mod:
                        router_modules.add(next_mod)
            unreachable = activator_modules - router_modules
            for mod in sorted(unreachable):
                critical_issues.append(
                    f"Module {mod!r} is in the Activator but has no Router rule "
                    f"(unreachable from the routing layer)"
                )

    return {
        "passed": len(critical_issues) == 0,
        "critical_issues": critical_issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Public dispatch table
# ---------------------------------------------------------------------------


# Maps catcher-name -> deterministic-function. Used by phases.py to
# substitute these implementations for the LLM catchers of the same name.
DETERMINISTIC_CATCHERS: dict[str, Any] = {
    "provenance_json": check_provenance,
    "consistency_json": check_consistency,
    "module_architecture_json": check_module_architecture,
}
