"""Z3-generated synthetic patients.

For each rule in each table, Z3 produces a concrete patient (boolean assignment)
that triggers that rule. These are boundary-correct synthetic patients that can
feed the external test harness without manual authoring.
"""

from z3 import BoolRef, BoolVal, And, Not, Or, Solver, sat

from backend.z3_verifier.evergreen_tests import DMNPackage


def generate_synthetic_patients(pkg: DMNPackage) -> list[dict]:
    """Generate one synthetic patient per rule per table.

    For FIRST/PRIORITY tables, the patient must trigger that specific rule
    (satisfying the rule's antecedent while NOT satisfying any earlier rule).

    Returns list of:
    {
        "table_id": str,
        "rule_id": str,
        "patient": {"var_name": True/False, ...},
        "hit_policy": str,
    }
    """
    patients: list[dict] = []

    for table in pkg.tables:
        for idx, rule in enumerate(table.rules):
            constraints: list[BoolRef] = [pkg.domain]

            if table.hit_policy.upper() in {"FIRST", "PRIORITY"} and idx > 0:
                # For FIRST: must fire this rule specifically (not earlier ones)
                prev_antecedents = Or(*[table.rules[j].when for j in range(idx)])
                constraints.append(Not(prev_antecedents))

            constraints.append(rule.when)

            s = Solver()
            for c in constraints:
                s.add(c)

            if s.check() == sat:
                model = s.model()
                patient: dict[str, bool] = {}
                for d in model.decls():
                    val = model[d]
                    patient[str(d.name())] = str(val) == "True"

                patients.append({
                    "table_id": table.table_id,
                    "rule_id": rule.rule_id,
                    "patient": patient,
                    "hit_policy": table.hit_policy,
                })

    return patients
