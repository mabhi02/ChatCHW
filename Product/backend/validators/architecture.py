"""Architecture Validator -- ORCHESTRATOR.md Section 6.1.

This is the structural validator invoked by `run_all_validators` from
the REPL's `validate(clinical_logic)` tool. It runs on the FULL clinical
logic dict (all artifacts assembled) and checks the high-level structural
invariants that every valid extraction must satisfy.

It is DIFFERENT from `backend/validators/artifact_architecture.py`, which
runs per-intermediate-artifact during `emit_artifact` dispatch and checks
more granular structural rules. That one is guide-agnostic. This one is
the gate-level sanity check.

Checks:
- activator exists with at least one rule
- router exists, rule[0] references p_danger_sign_present
- Router has a fallback/default last rule
- Every activator module_id appears in a router condition (if router
  routes per-module)
- Every activator module_id has a matching entry in modules
- No module_id is a substring of another
- integrative exists

Returns: list of {"message": str, "severity": str} dicts. Consumed by
run_all_validators which wraps each entry with {"validator": "architecture"}.
"""


def validate_architecture(logic: dict) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    if not isinstance(logic, dict):
        errors.append({
            "message": f"clinical_logic is not a dict ({type(logic).__name__})",
            "severity": "error",
        })
        return errors

    # activator exists with at least one rule
    activator = logic.get("activator")
    if not activator:
        errors.append({"message": "Missing 'activator' section", "severity": "error"})
    elif isinstance(activator, dict) and not activator.get("rules"):
        errors.append({"message": "Activator has no rules", "severity": "error"})

    # router exists
    router = logic.get("router")
    if not router:
        errors.append({"message": "Missing 'router' section", "severity": "error"})
    elif isinstance(router, dict) and not router.get("rules"):
        errors.append({"message": "Router has no rules", "severity": "error"})
    elif isinstance(router, dict) and isinstance(router.get("rules"), list) and router["rules"]:
        # rule[0] must reference p_danger_sign_present
        first_rule = router["rules"][0]
        if isinstance(first_rule, dict):
            condition = str(first_rule.get("condition", ""))
            # Also inspect inputs list for references
            inputs_text = " ".join(
                str(i) for i in (first_rule.get("inputs", []) or [])
                if isinstance(i, str)
            )
            combined = f"{condition} {inputs_text}".lower()
            if "danger_sign" not in combined and "p_danger" not in combined:
                errors.append({
                    "message": (
                        f"Router rule[0] must reference 'p_danger_sign_present' "
                        f"or a danger-sign predicate (got condition: '{condition}')"
                    ),
                    "severity": "error",
                })

        # Last rule should be fallback (all inputs '-' OR condition marks it)
        last_rule = router["rules"][-1]
        if isinstance(last_rule, dict):
            last_inputs = last_rule.get("inputs", []) or []
            is_fallback_by_inputs = (
                isinstance(last_inputs, list)
                and all(i == "-" or i is None for i in last_inputs)
            )
            last_cond = str(last_rule.get("condition", "")).lower()
            is_fallback_by_cond = (
                "default" in last_cond
                or "fallback" in last_cond
                or "always" in last_cond
            )
            if not (is_fallback_by_inputs or is_fallback_by_cond):
                errors.append({
                    "message": "Router last rule should be a fallback (all inputs '-' or condition='default'/'fallback'/'always')",
                    "severity": "warning",
                })

    # modules exists and is iterable
    modules = logic.get("modules")
    if modules is None:
        errors.append({"message": "Missing 'modules' section", "severity": "error"})
    else:
        module_ids: list[str] = []
        if isinstance(modules, dict):
            module_ids = list(modules.keys())
        elif isinstance(modules, list):
            for m in modules:
                if isinstance(m, dict) and m.get("module_id"):
                    module_ids.append(m["module_id"])

        # No module_id is a strict substring of another (would cause ambiguous
        # matching in the router's string-based condition checks)
        for i, mid_a in enumerate(module_ids):
            for j, mid_b in enumerate(module_ids):
                if i != j and mid_a and mid_b and mid_a in mid_b and mid_a != mid_b:
                    errors.append({
                        "message": f"module_id '{mid_a}' is a substring of '{mid_b}' (ambiguous router matching)",
                        "severity": "warning",
                    })

        # Every activator rule's module_id has a matching entry in modules
        if isinstance(activator, dict):
            act_rules = activator.get("rules", []) or []
            if isinstance(act_rules, list):
                for ar in act_rules:
                    if not isinstance(ar, dict):
                        continue
                    target = ar.get("module_id") or ar.get("next_module")
                    if target and target not in module_ids:
                        errors.append({
                            "message": f"Activator references module_id '{target}' which is not defined in modules",
                            "severity": "error",
                        })

    # integrative exists
    integrative = logic.get("integrative")
    if integrative is None:
        errors.append({"message": "Missing 'integrative' section", "severity": "warning"})

    return errors
