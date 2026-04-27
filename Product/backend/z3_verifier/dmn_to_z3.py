"""Compile JSON decision tables to Z3 constraints.

This module handles the translation from the clinical_logic JSON format
to Z3 BoolRef expressions for formal verification.
"""

from z3 import Bool, BoolRef, BoolVal, And, Not

from backend.z3_verifier.evergreen_tests import DMNRule, DMNTable, DMNPackage


def create_z3_variables(logic: dict) -> dict[str, BoolRef]:
    """Create Z3 Bool variables for all unique input columns."""
    all_vars: set[str] = set()
    for module in logic.get("modules", []):
        all_vars.update(module.get("input_columns", []))
    return {name: Bool(name) for name in sorted(all_vars)}


def compile_rule_antecedent(
    inputs: list[str],
    input_cols: list[str],
    z3_vars: dict[str, BoolRef],
) -> BoolRef:
    """Compile a single rule's input pattern into a Z3 BoolRef.

    "true"  -> var
    "false" -> Not(var)
    "-"     -> no constraint (don't care)
    """
    conjuncts: list[BoolRef] = []
    for idx, inp in enumerate(inputs):
        if idx >= len(input_cols):
            break
        var = z3_vars.get(input_cols[idx])
        if var is None:
            continue
        if inp == "true":
            conjuncts.append(var)
        elif inp == "false":
            conjuncts.append(Not(var))

    if not conjuncts:
        return BoolVal(True)
    return And(*conjuncts) if len(conjuncts) > 1 else conjuncts[0]


def compile_module_table(
    module: dict,
    z3_vars: dict[str, BoolRef],
) -> DMNTable:
    """Compile a single module dict into a DMNTable."""
    module_id = module.get("module_id", "unknown")
    hit_policy = module.get("hit_policy", "FIRST")
    input_cols = module.get("input_columns", [])
    output_cols = module.get("output_columns", [])

    rules: list[DMNRule] = []
    for idx, rule in enumerate(module.get("rules", [])):
        antecedent = compile_rule_antecedent(
            rule.get("inputs", []), input_cols, z3_vars
        )
        outputs = {}
        for oi, ov in enumerate(rule.get("outputs", [])):
            if oi < len(output_cols):
                outputs[output_cols[oi]] = ov

        rules.append(DMNRule(
            rule_id=f"{module_id}_r{idx}",
            when=antecedent,
            outputs=outputs,
        ))

    return DMNTable(
        table_id=module_id,
        hit_policy=hit_policy,
        rules=rules,
        kind="CLASSIFIER",
    )


def compile_logic_to_package(
    logic: dict,
    domain: BoolRef | None = None,
) -> DMNPackage:
    """Full compilation: JSON clinical_logic -> DMNPackage.

    Args:
        logic: The clinical_logic JSON dict.
        domain: Optional domain constraint. Defaults to True (unconstrained).
    """
    z3_vars = create_z3_variables(logic)
    tables = [
        compile_module_table(m, z3_vars)
        for m in logic.get("modules", [])
    ]
    return DMNPackage(
        domain=domain or BoolVal(True),
        tables=tables,
    )
