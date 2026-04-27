"""Naming Validator -- ORCHESTRATOR.md Section 6.4 + CHW Navigator codebook.

Checks:
- predicate_id matches ^p_[a-z][a-z0-9_]*$
- module_id matches ^mod_[a-z][a-z0-9_]*$
- message_id matches ^m_[a-z][a-z0-9_]*$
- source_vars match ^(q|ex|v|lab|demo|hx|img|sys|prev|calc)_[a-z][a-z0-9_]*$
- output columns match ^(dx|tx|ref|adv|fu|m|sev|out|rx|proc|need|err|s)_[a-z][a-z0-9_]*$
"""

import re

# Naming patterns from ORCHESTRATOR.md + CHW Navigator codebook
PATTERNS = {
    "predicate_id": re.compile(r"^p_[a-z][a-z0-9_]*$"),
    "module_id": re.compile(r"^mod_[a-z][a-z0-9_]*$"),
    "message_id": re.compile(r"^m_[a-z][a-z0-9_]*$"),
    "source_var": re.compile(
        r"^(q|ex|v|lab|demo|hx|img|sys|prev|calc)_[a-z][a-z0-9_]*$"
    ),
    "input_column": re.compile(
        r"^(q|ex|v|lab|demo|hx|img|sys|prev|calc|p|s|need)_[a-z][a-z0-9_]*$"
    ),
    "output_column": re.compile(
        r"^(dx|tx|ref|adv|fu|m|sev|out|rx|proc|need|err|s)_[a-z][a-z0-9_]*$"
    ),
}


def validate_naming(logic: dict) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    # Validate predicate_ids
    for pred in logic.get("predicates", []):
        pid = pred.get("predicate_id", "")
        if pid and not PATTERNS["predicate_id"].match(pid):
            errors.append({
                "message": f"Predicate ID '{pid}' does not match naming convention (^p_[a-z][a-z0-9_]*$)",
                "severity": "error",
            })

        # Validate source_vars
        for sv in pred.get("source_vars", []):
            if sv and not PATTERNS["source_var"].match(sv):
                errors.append({
                    "message": f"Predicate '{pid}' source_var '{sv}' does not match naming convention",
                    "severity": "error",
                })

    # Validate module_ids
    for module in logic.get("modules", []):
        mid = module.get("module_id", "")
        if mid and not PATTERNS["module_id"].match(mid):
            errors.append({
                "message": f"Module ID '{mid}' does not match naming convention (^mod_[a-z][a-z0-9_]*$)",
                "severity": "error",
            })

        # Validate input_columns
        for col in module.get("input_columns", []):
            if col and not PATTERNS["input_column"].match(col):
                errors.append({
                    "message": f"Module '{mid}' input column '{col}' does not match naming convention",
                    "severity": "error",
                })

        # Validate output_columns
        for col in module.get("output_columns", []):
            if col and not PATTERNS["output_column"].match(col):
                errors.append({
                    "message": f"Module '{mid}' output column '{col}' does not match naming convention",
                    "severity": "error",
                })

    # Validate message_ids
    for phrase in logic.get("phrase_bank", []):
        mid = phrase.get("message_id", "")
        if mid and not PATTERNS["message_id"].match(mid):
            errors.append({
                "message": f"Message ID '{mid}' does not match naming convention (^m_[a-z][a-z0-9_]*$)",
                "severity": "error",
            })

    # Validate activator module_ids
    activator = logic.get("activator", {})
    for rule in activator.get("rules", []):
        mid = rule.get("module_id", "")
        if mid and not PATTERNS["module_id"].match(mid):
            errors.append({
                "message": f"Activator module_id '{mid}' does not match naming convention",
                "severity": "error",
            })

    return errors
