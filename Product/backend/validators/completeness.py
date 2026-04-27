"""Completeness Validator -- ORCHESTRATOR.md Section 6.2.

Checks:
- Every p_ variable in any module's input_columns exists in predicates
- Every m_ value in any module's outputs exists in phrase_bank
- p_danger_sign_present exists in predicates
- Every predicate has non-empty source_vars, threshold_expression, page_ref
- Every phrase entry has non-empty english_text, page_ref
"""


def validate_completeness(logic: dict) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    predicates = logic.get("predicates", [])
    phrase_bank = logic.get("phrase_bank", [])
    modules = logic.get("modules", [])

    predicate_ids = {p["predicate_id"] for p in predicates if "predicate_id" in p}
    message_ids = {m["message_id"] for m in phrase_bank if "message_id" in m}

    # Every p_ variable in any module's input_columns exists in predicates
    for module in modules:
        module_id = module.get("module_id", "unknown")
        for col in module.get("input_columns", []):
            if col.startswith("p_") and col not in predicate_ids:
                errors.append({
                    "message": f"Module '{module_id}' references predicate '{col}' not found in predicates array",
                    "severity": "error",
                })

    # Every m_ value in any module's outputs exists in phrase_bank
    for module in modules:
        module_id = module.get("module_id", "unknown")
        for rule in module.get("rules", []):
            for output_val in rule.get("outputs", []):
                # Message IDs can be comma-separated in output cells
                for token in str(output_val).split(","):
                    token = token.strip()
                    if token.startswith("m_") and token not in message_ids:
                        errors.append({
                            "message": f"Module '{module_id}' references message '{token}' not found in phrase_bank",
                            "severity": "error",
                        })

    # p_danger_sign_present must exist
    if "p_danger_sign_present" not in predicate_ids:
        errors.append({
            "message": "Required predicate 'p_danger_sign_present' not found in predicates",
            "severity": "error",
        })

    # Every predicate has non-empty required fields
    for pred in predicates:
        pid = pred.get("predicate_id", "unknown")
        if not pred.get("source_vars"):
            errors.append({
                "message": f"Predicate '{pid}' has empty source_vars",
                "severity": "error",
            })
        if not pred.get("threshold_expression"):
            errors.append({
                "message": f"Predicate '{pid}' has empty threshold_expression",
                "severity": "error",
            })
        if not pred.get("page_ref"):
            errors.append({
                "message": f"Predicate '{pid}' has empty page_ref",
                "severity": "error",
            })

    # Every phrase entry has non-empty required fields
    for phrase in phrase_bank:
        mid = phrase.get("message_id", "unknown")
        if not phrase.get("english_text"):
            errors.append({
                "message": f"Phrase '{mid}' has empty english_text",
                "severity": "error",
            })
        if not phrase.get("page_ref"):
            errors.append({
                "message": f"Phrase '{mid}' has empty page_ref",
                "severity": "error",
            })

    return errors
