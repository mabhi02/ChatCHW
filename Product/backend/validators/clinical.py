"""Clinical Validator -- ORCHESTRATOR.md Section 6.3.

Checks:
- Every module's last rule has all inputs = "-" (default/else row)
- Rule input array lengths match input_columns length
- Rule output array lengths match output_columns length
- All rule inputs are strictly "true", "false", or "-"
- No raw numeric thresholds appear in rule inputs
- Every predicate fail_safe is 0 or 1
- Fail-safe sanity: predicates with v_ source vars should have fail_safe=1
"""

import re

from backend.schema import VALID_RULE_INPUTS, EQUIPMENT_PREFIXES


# Pattern to detect raw numeric values in rule inputs
NUMERIC_PATTERN = re.compile(r"^[<>=!]+\s*\d+\.?\d*$|^\d+\.?\d*$")


def validate_clinical(logic: dict) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    modules = logic.get("modules", [])
    predicates = logic.get("predicates", [])

    for module in modules:
        module_id = module.get("module_id", "unknown")
        input_cols = module.get("input_columns", [])
        output_cols = module.get("output_columns", [])
        rules = module.get("rules", [])

        if not rules:
            errors.append({
                "message": f"Module '{module_id}' has no rules",
                "severity": "error",
            })
            continue

        # Last rule must be default (all inputs "-")
        last_rule = rules[-1]
        last_inputs = last_rule.get("inputs", [])
        if not all(inp == "-" for inp in last_inputs):
            errors.append({
                "message": f"Module '{module_id}' last rule is not a default (all inputs should be '-')",
                "severity": "error",
            })

        for idx, rule in enumerate(rules):
            rule_label = f"Module '{module_id}' rule[{idx}]"
            inputs = rule.get("inputs", [])
            outputs = rule.get("outputs", [])

            # Input array length matches input_columns
            if len(inputs) != len(input_cols):
                errors.append({
                    "message": f"{rule_label}: input count ({len(inputs)}) != input_columns count ({len(input_cols)})",
                    "severity": "error",
                })

            # Output array length matches output_columns
            if len(outputs) != len(output_cols):
                errors.append({
                    "message": f"{rule_label}: output count ({len(outputs)}) != output_columns count ({len(output_cols)})",
                    "severity": "error",
                })

            # All rule inputs must be "true", "false", or "-"
            for i, inp in enumerate(inputs):
                if inp not in VALID_RULE_INPUTS:
                    errors.append({
                        "message": f"{rule_label}: input[{i}] = '{inp}' is not a valid boolean input (must be 'true', 'false', or '-')",
                        "severity": "error",
                    })

                # No raw numeric thresholds in rule inputs
                if NUMERIC_PATTERN.match(str(inp)):
                    errors.append({
                        "message": f"{rule_label}: input[{i}] = '{inp}' appears to be a numeric threshold; thresholds must go in predicates",
                        "severity": "error",
                    })

            # Provenance required
            if not rule.get("provenance"):
                errors.append({
                    "message": f"{rule_label}: missing provenance",
                    "severity": "warning",
                })

    # Every predicate fail_safe is 0 or 1
    for pred in predicates:
        pid = pred.get("predicate_id", "unknown")
        fail_safe = pred.get("fail_safe")

        if fail_safe not in (0, 1):
            errors.append({
                "message": f"Predicate '{pid}': fail_safe must be 0 or 1 (got: {fail_safe})",
                "severity": "error",
            })

        # Fail-safe sanity: predicates with v_ or lab_ source vars should have fail_safe=1
        source_vars = pred.get("source_vars", [])
        has_equipment_var = any(
            any(sv.startswith(prefix) for prefix in EQUIPMENT_PREFIXES)
            for sv in source_vars
        )
        if has_equipment_var and fail_safe == 0:
            errors.append({
                "message": f"Predicate '{pid}': has equipment-dependent source vars ({source_vars}) but fail_safe=0; should be 1 (conservative)",
                "severity": "warning",
            })

    return errors
