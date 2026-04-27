from typing import Any

from backend.validators.architecture import validate_architecture
from backend.validators.completeness import validate_completeness
from backend.validators.clinical import validate_clinical
from backend.validators.naming import validate_naming


def _normalize_logic(logic: dict) -> dict:
    """Filter non-dict items from known list fields and guard against lists
    being strings.

    Gen 2.4 hotfix (2026-04-12): the first gen 2.4 run crashed all three
    structural validators with `'str' object has no attribute 'get'` because
    the model emitted a clinical_logic whose `modules`, `predicates`, or
    `phrase_bank` array contained a string token instead of a dict object
    (likely a stray id placeholder like `"p_fever"` inserted where a full
    predicate dict was expected). The validators iterate these lists and
    call `.get()` on every element; a string element crashes them.

    Before this normalization step, the outer try/except in run_all_validators
    caught the AttributeError but produced a generic "crashed on unexpected
    shape" error message that gave the model no actionable signal to fix the
    problem. After normalization, the bad elements are silently dropped so
    the validators run on the clean subset and emit specific errors about
    missing required fields — which the model CAN act on.

    This is defensive filtering, not repair. The model still has a broken
    artifact; we just ensure the validator pass produces useful output
    instead of a crash message.
    """
    normalized = dict(logic)

    for field in ("modules", "predicates", "phrase_bank"):
        items = normalized.get(field)
        if isinstance(items, list):
            normalized[field] = [x for x in items if isinstance(x, dict)]
        elif items is not None:
            # Field exists but is not a list at all — replace with empty list.
            normalized[field] = []

    # activator.rules is also iterated in naming.py — guard it too.
    activator = normalized.get("activator")
    if isinstance(activator, dict):
        acts = dict(activator)
        rules = acts.get("rules")
        if isinstance(rules, list):
            acts["rules"] = [r for r in rules if isinstance(r, dict)]
        elif rules is not None:
            acts["rules"] = []
        normalized["activator"] = acts
    elif activator is not None and not isinstance(activator, dict):
        normalized["activator"] = {}

    return normalized


def run_all_validators(logic: Any) -> dict:
    """Run all structural validators on the clinical logic JSON.

    Defensive: if `logic` is not a dict, short-circuit to a single type-guard
    error rather than crashing inside `logic.get(...)` calls in the sub-
    validators. This mirrors the guards at the REPL entry points in
    rlm_runner.py so the crash path is defined in exactly one way.

    Gen 2.4: additionally normalizes `logic` via `_normalize_logic` to filter
    non-dict items from known list fields (modules, predicates, phrase_bank,
    activator.rules). This prevents the sub-validators from crashing on
    string-typed list elements — instead they run on the clean subset and
    emit specific errors the model can act on.

    Returns:
        {
            "passed": bool,
            "error_count": int,
            "errors": [{"validator": str, "message": str, "severity": str}, ...]
        }
    """
    if not isinstance(logic, dict):
        return {
            "passed": False,
            "error_count": 1,
            "errors": [{
                "validator": "type_guard",
                "message": (
                    f"run_all_validators expects a dict clinical_logic, got "
                    f"{type(logic).__name__}"
                ),
                "severity": "error",
            }],
        }

    # Gen 2.4 hotfix: sanitize the logic BEFORE handing it to sub-validators.
    # Each validator will see a shape that only contains dict elements in the
    # list-typed fields it iterates.
    sanitized = _normalize_logic(logic)

    errors: list[dict] = []

    for name, validator_fn in [
        ("architecture", validate_architecture),
        ("completeness", validate_completeness),
        ("clinical", validate_clinical),
        ("naming", validate_naming),
    ]:
        try:
            results = validator_fn(sanitized)
        except (AttributeError, TypeError) as exc:
            # A sub-validator hit an unexpected shape even after normalization
            # (e.g. a nested dict had a string where a list was expected).
            # Record as a single error and keep going so the other validators
            # still run -- partial results are more useful than a full crash.
            # AttributeError: `.get()` on a non-dict.
            # TypeError: iteration on a non-iterable, or subscripting a
            # string with a str key.
            results = [{
                "message": (
                    f"{name} validator crashed on unexpected nested shape: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "severity": "error",
            }]
        for err in results:
            errors.append({"validator": name, **err})

    return {
        "passed": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }
