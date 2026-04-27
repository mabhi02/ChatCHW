"""Deterministic architecture / structural checks (gate generation 2, 2026-04-11).

This module hosts the structural rules that USED to live inside LLM
completeness catchers (gen 1). Moved out because they conflated three
different things in one blocking gate:

    1. Recall ("did we cover everything in the PDF?")
    2. Style ("fail_safe=0 is wrong for q_-driven predicates")
    3. Architecture ("router row 0 must reference the composite alert
       predicate that the pipeline canonically names p_danger_sign_present,
       regardless of what the manual itself calls its high-priority
       criteria")

Only (1) is what the user cares about as a PASS criterion. (2) and (3)
are useful feedback but should not block a run that has 100% recall.

Gen 1 LLM catchers also had high variance on (2) and (3) — Haiku would
flag different things on different runs, and sometimes hallucinate rules
that weren't in the frozen prompt (citing named external protocols that
weren't in the guide). Deterministic Python eliminates that variance
entirely.

Each function below:
  - Takes the ARTIFACT dict as input (and sometimes peer artifacts)
  - Returns a dict with the same shape catcher functions return:
      {"passed": bool, "critical_issues": [...], "warnings": [...]}
  - Is a pure function: no I/O, no LLM calls, no mutation of inputs
  - Is fast: O(n) in artifact size, runs in <10ms on a full-WHO artifact

Integration: phases.py calls these alongside the (now recall-only) LLM
catchers via the `run_*_architecture_checks` functions. Results are
aggregated with a SPECIAL rule: architecture criticals are demoted to
warnings UNLESS they're in the `HARD_ARCHITECTURE_RULES` set, which is
the small list of genuinely-blocking structural invariants (e.g. router
row 0 MUST reference the `p_danger_sign_present` composite alert
predicate, otherwise the DMN converter generates a compiling-but-
clinically-unsafe output).

Everything else becomes a warning. Runs don't fail on soft architecture
issues — the user sees the warnings in the validator sidecar and can
decide whether to fix them.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Hard architecture rules that MUST block a run (keep this list small)
# ---------------------------------------------------------------------------
# These are the invariants where a violation produces a clinically-unsafe
# or broken artifact that downstream converters cannot render correctly.
# Everything NOT in this set becomes a warning.
HARD_ARCHITECTURE_RULES: frozenset[str] = frozenset({
    # Router row 0 must reference the danger-sign predicate. Without this,
    # DMN conversion produces a router that doesn't short-circuit on
    # emergencies, which is clinically unsafe.
    "router_row0_no_danger_sign",
    # Every module must have at least one rule. A module with zero rules
    # cannot produce any output — dead module, DMN converter will emit
    # an empty decision table that silently never fires.
    "module_empty_rules",
    # Activator hit_policy must be COLLECT. FIRST/UNIQUE would drop
    # all-but-one active modules, causing comorbidity handling to silently
    # discard conditions the patient actually has.
    "activator_wrong_hit_policy",
    # Router hit_policy must be FIRST. COLLECT would fire every matching
    # row, which is wrong for a priority-ordered router.
    "router_wrong_hit_policy",
})


def _ok() -> dict:
    """Empty-pass result (no issues)."""
    return {"passed": True, "critical_issues": [], "warnings": []}


def _result(
    criticals: list[str],
    warnings: list[str],
) -> dict:
    """Build a standard catcher result, auto-demoting non-hard criticals."""
    hard_criticals = [
        c for c in criticals
        if any(rule_id in c for rule_id in HARD_ARCHITECTURE_RULES)
    ]
    soft_criticals = [c for c in criticals if c not in hard_criticals]
    return {
        "passed": len(hard_criticals) == 0,
        "critical_issues": hard_criticals,
        "warnings": warnings + soft_criticals,
    }


# ---------------------------------------------------------------------------
# Supply list architecture checks
# ---------------------------------------------------------------------------


def check_supply_list_architecture(supply_list: Any, guide_json: Any = None) -> dict:
    """Structural checks on the supply_list artifact.

    Rules:
      - Each entry must have `id`, `kind`, `display_name`
      - `id` must start with `equip_` for kind="equipment", `supply_` for kind="consumable"
      - `kind` must be "equipment" or "consumable"

    All rules are SOFT — violations are warnings, not criticals. Downstream
    converters tolerate missing supply items by generating an empty inventory
    sheet.
    """
    if not isinstance(supply_list, (list, dict)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"supply_list is not a list or dict ({type(supply_list).__name__}); skipping architecture check"],
        }

    # Normalize dict-keyed-by-id to list
    if isinstance(supply_list, dict):
        items = [{"id": k, **v} if isinstance(v, dict) else {"id": k, "_raw": v} for k, v in supply_list.items()]
    else:
        items = supply_list

    warnings: list[str] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            warnings.append(f"[arch/supply_list] entry[{i}] is not a dict: {type(entry).__name__}")
            continue
        item_id = entry.get("id") or f"<entry[{i}] no id>"
        kind = entry.get("kind")
        if not entry.get("id"):
            warnings.append(f"[arch/supply_list] entry[{i}] missing `id` field")
        if not entry.get("display_name"):
            warnings.append(f"[arch/supply_list] {item_id}: missing `display_name`")
        if kind not in ("equipment", "consumable", None):
            warnings.append(f"[arch/supply_list] {item_id}: `kind` should be 'equipment' or 'consumable', got '{kind}'")
        if kind == "equipment" and isinstance(item_id, str) and not item_id.startswith("equip_"):
            warnings.append(f"[arch/supply_list] {item_id}: equipment id should start with 'equip_'")
        if kind == "consumable" and isinstance(item_id, str) and not item_id.startswith("supply_"):
            warnings.append(f"[arch/supply_list] {item_id}: consumable id should start with 'supply_'")

    return _result(criticals=[], warnings=warnings)


# ---------------------------------------------------------------------------
# Variables architecture checks
# ---------------------------------------------------------------------------


_VARIABLE_PREFIXES = ("q_", "ex_", "v_", "lab_", "img_", "hx_", "demo_")


def check_variables_architecture(variables: Any, guide_json: Any = None) -> dict:
    """Structural checks on the variables artifact.

    All rules are SOFT:
      - Each entry must have `id`, `display_name`
      - `id` must start with a recognized prefix (q_/ex_/v_/lab_/img_/hx_/demo_)
    """
    if not isinstance(variables, (list, dict)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"variables is not a list or dict ({type(variables).__name__}); skipping architecture check"],
        }

    if isinstance(variables, dict):
        # Handle both {items: [...]} and {id: {...}} shapes
        if "items" in variables and isinstance(variables["items"], list):
            items = variables["items"]
        else:
            items = [{"id": k, **v} if isinstance(v, dict) else {"id": k} for k, v in variables.items()]
    else:
        items = variables

    warnings: list[str] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            warnings.append(f"[arch/variables] entry[{i}] is not a dict: {type(entry).__name__}")
            continue
        var_id = entry.get("id") or f"<entry[{i}] no id>"
        if not entry.get("id"):
            warnings.append(f"[arch/variables] entry[{i}] missing `id` field")
            continue
        if not entry.get("display_name"):
            warnings.append(f"[arch/variables] {var_id}: missing `display_name`")
        if isinstance(var_id, str) and not any(var_id.startswith(p) for p in _VARIABLE_PREFIXES):
            warnings.append(
                f"[arch/variables] {var_id}: id should start with one of "
                f"{', '.join(_VARIABLE_PREFIXES)}"
            )

    return _result(criticals=[], warnings=warnings)


# ---------------------------------------------------------------------------
# Predicates architecture checks (the one that caught Run #8 hardest)
# ---------------------------------------------------------------------------


def check_predicates_architecture(predicates: Any, guide_json: Any = None) -> dict:
    """Structural checks on the predicates artifact.

    The OLD gen-1 LLM catcher flagged these as CRITICAL and blocked runs.
    Now they are all SOFT WARNINGS. The model can still pass a run with
    non-ideal fail_safe values — the runtime behavior is still correct as
    long as the threshold_expression is right.

    Rules:
      - Each entry must have `predicate_id`, `threshold_expression`, `fail_safe`
      - `predicate_id` must start with `p_`
      - `fail_safe` must be 0 or 1
      - Convention warning: q_-driven predicates usually have fail_safe=0
        (assume no symptom when not asked); v_/ex_/lab_/img_-driven predicates
        usually have fail_safe=1 (assume danger when measurement missing)
      - Composite danger-sign predicates (p_danger_sign_*) should have fail_safe=1
    """
    if not isinstance(predicates, (list, dict)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"predicates is not a list or dict ({type(predicates).__name__}); skipping"],
        }

    if isinstance(predicates, dict):
        items = [{"predicate_id": k, **v} if isinstance(v, dict) else {"predicate_id": k} for k, v in predicates.items()]
    else:
        items = predicates

    warnings: list[str] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            warnings.append(f"[arch/predicates] entry[{i}] is not a dict")
            continue
        pid = entry.get("predicate_id") or entry.get("id") or f"<entry[{i}] no id>"
        if not entry.get("predicate_id") and not entry.get("id"):
            warnings.append(f"[arch/predicates] entry[{i}] missing `predicate_id`")
            continue
        if isinstance(pid, str) and not pid.startswith("p_"):
            warnings.append(f"[arch/predicates] {pid}: id should start with 'p_'")
        if not entry.get("threshold_expression"):
            warnings.append(f"[arch/predicates] {pid}: missing `threshold_expression`")
        fail_safe = entry.get("fail_safe")
        if fail_safe is None:
            warnings.append(f"[arch/predicates] {pid}: missing `fail_safe` (should be 0 or 1)")
        elif fail_safe not in (0, 1):
            warnings.append(f"[arch/predicates] {pid}: `fail_safe` should be 0 or 1, got {fail_safe!r}")

        # Soft convention warnings on fail_safe values
        source_vars = entry.get("source_vars") or []
        if not isinstance(source_vars, list):
            source_vars = []
        if isinstance(pid, str) and "danger_sign" in pid and fail_safe == 0:
            warnings.append(
                f"[arch/predicates] {pid}: composite danger-sign predicates "
                f"should default fail_safe=1 (assume danger when missing)"
            )
        # q_-driven predicates usually default 0
        if (
            fail_safe == 1
            and source_vars
            and all(isinstance(v, str) and v.startswith("q_") for v in source_vars)
        ):
            warnings.append(
                f"[arch/predicates] {pid}: q_-driven predicates usually "
                f"default fail_safe=0 (assume no symptom when not asked), "
                f"not 1. Verify this is intentional."
            )
        # v_/ex_/lab_/img_-driven predicates usually default 1
        if (
            fail_safe == 0
            and source_vars
            and all(
                isinstance(v, str) and any(v.startswith(p) for p in ("v_", "ex_", "lab_", "img_"))
                for v in source_vars
            )
            and isinstance(pid, str)
            and "danger_sign" not in pid
        ):
            warnings.append(
                f"[arch/predicates] {pid}: measurement-driven predicates "
                f"usually default fail_safe=1 (assume danger when missing), "
                f"not 0. Verify this is intentional."
            )

    return _result(criticals=[], warnings=warnings)


# ---------------------------------------------------------------------------
# Modules architecture checks
# ---------------------------------------------------------------------------


def check_modules_architecture(modules: Any, guide_json: Any = None) -> dict:
    """Structural checks on the modules artifact.

    HARD rules (block the run):
      - Each module must have at least one rule (empty modules are dead code)

    SOFT rules (warnings):
      - Each module must have a hit_policy
      - Each rule must have inputs + outputs
      - input_columns and output_columns must be non-empty
    """
    if not isinstance(modules, (list, dict)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"modules is not a list or dict ({type(modules).__name__}); skipping"],
        }

    if isinstance(modules, dict):
        # {module_id: {...}} shape
        items = [(k, v) for k, v in modules.items()]
    else:
        items = [(m.get("module_id") if isinstance(m, dict) else f"<m[{i}]>", m) for i, m in enumerate(modules)]

    criticals: list[str] = []
    warnings: list[str] = []
    for mid, m in items:
        if not isinstance(m, dict):
            warnings.append(f"[arch/modules] {mid}: module is not a dict ({type(m).__name__})")
            continue
        rules = m.get("rules")
        if not rules or (isinstance(rules, list) and len(rules) == 0):
            criticals.append(f"[arch/modules] module_empty_rules: {mid} has zero rules")
        if not m.get("hit_policy"):
            warnings.append(f"[arch/modules] {mid}: missing `hit_policy`")
        if not m.get("input_columns"):
            warnings.append(f"[arch/modules] {mid}: missing or empty `input_columns`")
        if not m.get("output_columns"):
            warnings.append(f"[arch/modules] {mid}: missing or empty `output_columns`")
        if isinstance(rules, list):
            for ri, r in enumerate(rules):
                if not isinstance(r, dict):
                    warnings.append(f"[arch/modules] {mid} rule[{ri}]: not a dict")
                    continue
                if not r.get("inputs"):
                    warnings.append(f"[arch/modules] {mid} rule[{ri}]: missing `inputs`")
                if not r.get("outputs"):
                    warnings.append(f"[arch/modules] {mid} rule[{ri}]: missing `outputs`")

    return _result(criticals=criticals, warnings=warnings)


# ---------------------------------------------------------------------------
# Router architecture checks (gets most of gen-1's hard rules)
# ---------------------------------------------------------------------------


def check_router_architecture(router: Any, modules: Any = None) -> dict:
    """Structural checks on the router artifact.

    HARD rules (block the run):
      - Row 0 MUST reference a danger-sign predicate
      - hit_policy MUST be "FIRST"

    SOFT rules (warnings):
      - Last row should be a fallback (all inputs "-")
      - No duplicate priority values
      - Actions should reference module_ids that exist in modules artifact
      - Each row should have a condition and an action
    """
    # Router may be at top level OR wrapped as {"router": {...}, "activator": {...}}
    if isinstance(router, dict) and "router" in router and isinstance(router["router"], dict):
        router = router["router"]

    if not isinstance(router, dict):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"router is not a dict ({type(router).__name__}); skipping"],
        }

    criticals: list[str] = []
    warnings: list[str] = []

    hit_policy = router.get("hit_policy", "").upper()
    if hit_policy and hit_policy != "FIRST":
        criticals.append(
            f"[arch/router] router_wrong_hit_policy: router hit_policy must be FIRST, got '{hit_policy}'"
        )
    elif not hit_policy:
        warnings.append("[arch/router] router missing `hit_policy` (should be FIRST)")

    rules = router.get("rules")
    if not isinstance(rules, list) or not rules:
        warnings.append("[arch/router] router has no rules list")
        return _result(criticals=criticals, warnings=warnings)

    # Row 0 check — HARD rule
    first_row = rules[0] if rules else None
    if isinstance(first_row, dict):
        condition = str(first_row.get("condition", "")).lower()
        inputs = first_row.get("inputs", []) or []
        first_row_text = condition + " " + " ".join(str(i) for i in inputs) if isinstance(inputs, list) else condition
        if "danger_sign" not in first_row_text and "danger sign" not in first_row_text:
            criticals.append(
                f"[arch/router] router_row0_no_danger_sign: row 0 must reference "
                f"p_danger_sign_present or a danger-sign predicate. Got condition: {condition!r}"
            )

    # Last row fallback check — SOFT
    last_row = rules[-1] if rules else None
    if isinstance(last_row, dict):
        last_inputs = last_row.get("inputs", [])
        if not (isinstance(last_inputs, list) and all(i == "-" or i is None for i in last_inputs)):
            last_condition = last_row.get("condition", "").lower() if isinstance(last_row.get("condition"), str) else ""
            if "default" not in last_condition and "fallback" not in last_condition and "always" not in last_condition:
                warnings.append(
                    f"[arch/router] last row should be a fallback "
                    f"(all inputs '-' OR condition='always'/'default')"
                )

    # Duplicate priority check — SOFT
    priorities = [r.get("priority") for r in rules if isinstance(r, dict) and r.get("priority") is not None]
    if len(priorities) != len(set(priorities)):
        warnings.append("[arch/router] duplicate priority values found")

    # Action reference check — SOFT
    if modules is not None:
        known_module_ids: set[str] = set()
        if isinstance(modules, dict):
            known_module_ids = set(modules.keys())
        elif isinstance(modules, list):
            known_module_ids = {m.get("module_id", "") for m in modules if isinstance(m, dict)}
        for i, r in enumerate(rules):
            if not isinstance(r, dict):
                continue
            action = r.get("next_module") or r.get("action") or r.get("module_id")
            if isinstance(action, str) and action and action not in known_module_ids and "integrative" not in action.lower() and "done" not in action.lower() and "refer" not in action.lower():
                warnings.append(
                    f"[arch/router] row {i}: action '{action}' does not match any module_id in modules artifact"
                )

    # Condition + action presence — SOFT
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            continue
        if not r.get("condition") and not r.get("inputs"):
            warnings.append(f"[arch/router] row {i}: missing both `condition` and `inputs`")
        if not r.get("action") and not r.get("next_module") and not r.get("module_id"):
            warnings.append(f"[arch/router] row {i}: missing `action`/`next_module`")

    return _result(criticals=criticals, warnings=warnings)


# ---------------------------------------------------------------------------
# Activator architecture checks (often embedded in router or top-level)
# ---------------------------------------------------------------------------


def check_activator_architecture(activator: Any) -> dict:
    """Structural checks on the activator artifact.

    HARD rules:
      - hit_policy MUST be "COLLECT"

    SOFT:
      - Must have at least one rule
      - Rules must have inputs + outputs
    """
    if activator is None:
        return _ok()
    if isinstance(activator, dict) and "activator" in activator:
        activator = activator["activator"]
    if not isinstance(activator, dict):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"activator is not a dict ({type(activator).__name__}); skipping"],
        }

    criticals: list[str] = []
    warnings: list[str] = []

    hit_policy = activator.get("hit_policy", "").upper()
    if hit_policy and hit_policy != "COLLECT":
        criticals.append(
            f"[arch/activator] activator_wrong_hit_policy: activator hit_policy "
            f"must be COLLECT, got '{hit_policy}'"
        )
    elif not hit_policy:
        warnings.append("[arch/activator] activator missing `hit_policy` (should be COLLECT)")

    rules = activator.get("rules", [])
    if not isinstance(rules, list) or not rules:
        warnings.append("[arch/activator] activator has no rules")

    return _result(criticals=criticals, warnings=warnings)


# ---------------------------------------------------------------------------
# Integrative architecture checks
# ---------------------------------------------------------------------------


def check_integrative_architecture(integrative: Any) -> dict:
    """Structural checks on the integrative artifact.

    Gen-1 required that four specific categories be present (referral
    priority, treatment dedup, follow-up merge, safety). In practice this
    is very hard to detect deterministically because the categories are
    encoded as natural-language rule descriptions. We demote these to
    warnings and only check shape.

    SOFT rules:
      - Must be a dict with a `rules` list, OR a list of rule dicts
      - Each rule has a `description` or equivalent
    """
    if integrative is None:
        return _ok()
    if not isinstance(integrative, (dict, list)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"integrative is not a dict or list ({type(integrative).__name__}); skipping"],
        }

    warnings: list[str] = []
    if isinstance(integrative, dict):
        rules = integrative.get("rules", [])
    else:
        rules = integrative

    if not isinstance(rules, list):
        warnings.append("[arch/integrative] rules is not a list")
        return _result(criticals=[], warnings=warnings)

    if len(rules) == 0:
        warnings.append(
            "[arch/integrative] integrative has zero rules — comorbidity handling "
            "is structural, verify this is intentional"
        )

    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            warnings.append(f"[arch/integrative] rule[{i}] is not a dict")
            continue
        if not r.get("description") and not r.get("rule") and not r.get("name"):
            warnings.append(f"[arch/integrative] rule[{i}] missing human-readable description")

    return _result(criticals=[], warnings=warnings)


# ---------------------------------------------------------------------------
# Phrase bank architecture checks
# ---------------------------------------------------------------------------


_PHRASE_CATEGORIES = {"question", "diagnosis", "treatment", "advice", "referral", "follow_up", "procedure", "examination"}


def check_phrase_bank_architecture(phrase_bank: Any) -> dict:
    """Structural checks on the phrase_bank artifact.

    SOFT rules:
      - Each entry has `phrase_id` (or id), `text` (or english_text), `category`
      - `category` should be one of the known set
    """
    if not isinstance(phrase_bank, (list, dict)):
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [f"phrase_bank is not a list or dict ({type(phrase_bank).__name__}); skipping"],
        }

    if isinstance(phrase_bank, dict):
        if "phrases" in phrase_bank and isinstance(phrase_bank["phrases"], list):
            items = phrase_bank["phrases"]
        else:
            items = [{"phrase_id": k, **v} if isinstance(v, dict) else {"phrase_id": k} for k, v in phrase_bank.items()]
    else:
        items = phrase_bank

    warnings: list[str] = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            warnings.append(f"[arch/phrase_bank] entry[{i}] is not a dict")
            continue
        pid = entry.get("phrase_id") or entry.get("message_id") or entry.get("id") or f"<entry[{i}]>"
        if not entry.get("text") and not entry.get("english_text") and not entry.get("message"):
            warnings.append(f"[arch/phrase_bank] {pid}: missing `text` / `english_text`")
        category = entry.get("category")
        if category and category not in _PHRASE_CATEGORIES:
            warnings.append(
                f"[arch/phrase_bank] {pid}: category '{category}' not in "
                f"{sorted(_PHRASE_CATEGORIES)}"
            )

    return _result(criticals=[], warnings=warnings)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
# phases.py imports this to run all architecture checks at the end of
# each phase's validator bundle. Results are merged into the validator
# sidecar JSON under the "architecture" key.

ARCHITECTURE_CHECKS: dict[str, Any] = {
    "supply_list": check_supply_list_architecture,
    "variables": check_variables_architecture,
    "predicates": check_predicates_architecture,
    "modules": check_modules_architecture,
    "router": check_router_architecture,
    "activator": check_activator_architecture,
    "integrative": check_integrative_architecture,
    "phrase_bank": check_phrase_bank_architecture,
}
