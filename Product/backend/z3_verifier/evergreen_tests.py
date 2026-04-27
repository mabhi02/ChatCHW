"""Evergreen DMN Verification Tests using Z3.

Adapted from CHW Navigator v1 (lines 4270-5020).
Checks: domain SAT, reachability, exhaustiveness, MECE, shadowing, router cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

from z3 import (
    Bool,
    BoolRef,
    BoolVal,
    Solver,
    sat,
    unsat,
    And,
    Or,
    Not,
    ModelRef,
    is_true,
)


# ---------------------
# Data structures
# ---------------------


@dataclass(frozen=True)
class DMNRule:
    """A single DMN rule with Z3 antecedent."""

    rule_id: str
    when: BoolRef
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DMNTable:
    """A DMN decision table."""

    table_id: str
    hit_policy: str  # COLLECT, UNIQUE, FIRST, PRIORITY
    rules: list[DMNRule]
    kind: str = "GENERIC"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DMNPackage:
    """The full DMN package to verify."""

    domain: BoolRef
    tables: list[DMNTable]
    router_table_ids: set[str] = field(default_factory=set)


@dataclass
class TestResult:
    test_id: str
    table_id: str | None
    passed: bool
    message: str
    witness: dict[str, Any] | None = None


@dataclass
class VerificationReport:
    results: list[TestResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]

    def add_result(
        self,
        test_id: str,
        table_id: str | None,
        passed: bool,
        message: str,
        witness: dict[str, Any] | None = None,
    ) -> None:
        self.results.append(TestResult(test_id, table_id, passed, message, witness))

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "checks": [
                {
                    "testId": r.test_id,
                    "tableId": r.table_id,
                    "passed": r.passed,
                    "message": r.message,
                    "witness": r.witness,
                }
                for r in self.results
            ],
            "warnings": self.warnings,
        }


# ---------------------
# Z3 helpers
# ---------------------


def _solve(constraints: list[BoolRef]) -> tuple[bool, ModelRef | None]:
    """Solve constraints. Returns (is_sat, model_if_sat)."""
    s = Solver()
    for c in constraints:
        s.add(c)
    res = s.check()
    if res == sat:
        return True, s.model()
    return False, None


def _model_to_dict(model: ModelRef) -> dict[str, Any]:
    """Convert Z3 model to dict."""
    out: dict[str, Any] = {}
    for d in model.decls():
        out[str(d.name())] = str(model[d])
    return out


# ---------------------
# Evergreen tests
# ---------------------


def test_domain_sat(pkg: DMNPackage, report: VerificationReport) -> None:
    """T00: DOMAIN must be SAT (otherwise everything is vacuous)."""
    ok, _ = _solve([pkg.domain])
    report.add_result(
        "T00_DOMAIN_SAT",
        None,
        ok,
        "DOMAIN is satisfiable" if ok else "DOMAIN is UNSAT (vacuous spec)",
    )


def test_rule_reachability(pkg: DMNPackage, report: VerificationReport) -> None:
    """T02: For each rule, DOMAIN AND rule.when must be SAT."""
    for t in pkg.tables:
        for r in t.rules:
            ok, m = _solve([pkg.domain, r.when])
            report.add_result(
                "T02_RULE_REACHABLE",
                t.table_id,
                ok,
                f"Rule {r.rule_id} is reachable"
                if ok
                else f"Rule {r.rule_id} is unreachable (dead code)",
                _model_to_dict(m) if m and not ok else None,
            )


def test_table_exhaustive(pkg: DMNPackage, report: VerificationReport) -> None:
    """T03: For each table, DOMAIN AND NOT(any rule matches) must be UNSAT."""
    for t in pkg.tables:
        if not t.rules:
            report.add_result(
                "T03_TABLE_EXHAUSTIVE",
                t.table_id,
                False,
                "Table has zero rules",
            )
            continue

        any_match = Or(*[r.when for r in t.rules])
        ok, m = _solve([pkg.domain, Not(any_match)])
        # If SAT, we found an uncovered input state
        is_exhaustive = not ok
        report.add_result(
            "T03_TABLE_EXHAUSTIVE",
            t.table_id,
            is_exhaustive,
            f"Table {t.table_id} is exhaustive"
            if is_exhaustive
            else f"Table {t.table_id} is NOT exhaustive: found uncovered input",
            _model_to_dict(m) if m and ok else None,
        )


def test_unique_mece(pkg: DMNPackage, report: VerificationReport) -> None:
    """For UNIQUE tables: mutual exclusivity + exhaustiveness."""
    for t in pkg.tables:
        if t.hit_policy.upper() != "UNIQUE":
            continue

        # Mutual exclusivity: for all pairs, DOMAIN AND Ai AND Aj must be UNSAT
        for r1, r2 in combinations(t.rules, 2):
            ok, m = _solve([pkg.domain, r1.when, r2.when])
            report.add_result(
                "T_UNIQUE_ME",
                t.table_id,
                not ok,
                f"Rules {r1.rule_id} and {r2.rule_id} are mutually exclusive"
                if not ok
                else f"Rules {r1.rule_id} and {r2.rule_id} can both fire (violates UNIQUE)",
                _model_to_dict(m) if m and ok else None,
            )

        # Exhaustiveness
        if t.rules:
            any_match = Or(*[r.when for r in t.rules])
            ok, m = _solve([pkg.domain, Not(any_match)])
            report.add_result(
                "T_UNIQUE_EXHAUSTIVE",
                t.table_id,
                not ok,
                f"UNIQUE table {t.table_id} is exhaustive"
                if not ok
                else f"UNIQUE table {t.table_id} has gaps",
                _model_to_dict(m) if m and ok else None,
            )


def test_first_priority_shadowing(
    pkg: DMNPackage, report: VerificationReport
) -> None:
    """For FIRST/PRIORITY tables: check shadowing and ordered exhaustiveness."""
    for t in pkg.tables:
        if t.hit_policy.upper() not in {"FIRST", "PRIORITY"}:
            continue

        if not t.rules:
            continue

        fires: list[BoolRef] = []
        prev_or: BoolRef | None = None

        for idx, r in enumerate(t.rules):
            if idx == 0:
                fire = r.when
                prev_or = r.when
            else:
                fire = And(r.when, Not(prev_or))
                prev_or = Or(prev_or, r.when)
            fires.append(fire)

            # Each fires_k must be SAT (not completely shadowed)
            ok, _ = _solve([pkg.domain, fire])
            report.add_result(
                "T_FIRST_SHADOWING",
                t.table_id,
                ok,
                f"Rule {r.rule_id} can fire under FIRST ordering"
                if ok
                else f"Rule {r.rule_id} is completely shadowed by earlier rules",
            )

        # Ordered exhaustiveness
        ok, m = _solve([pkg.domain, Not(Or(*fires))])
        report.add_result(
            "T_FIRST_EXHAUSTIVE",
            t.table_id,
            not ok,
            f"FIRST table {t.table_id} is exhaustive under ordering"
            if not ok
            else f"FIRST table {t.table_id} has gap under ordering",
            _model_to_dict(m) if m and ok else None,
        )


# ---------------------
# Router cycle detection
# ---------------------


def build_router_graph(pkg: DMNPackage) -> dict[str, set[str]]:
    """Build directed graph from router rules (structural, not SMT)."""
    g: dict[str, set[str]] = {}
    for t in pkg.tables:
        if t.table_id not in pkg.router_table_ids:
            continue
        for r in t.rules:
            frm = r.outputs.get("from_module")
            nxt = r.outputs.get("next_module")
            if frm and nxt:
                g.setdefault(frm, set()).add(nxt)
    return g


def find_cycles(g: dict[str, set[str]]) -> list[list[str]]:
    """DFS cycle detection."""
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(u: str) -> None:
        visiting.add(u)
        stack.append(u)
        for v in g.get(u, set()):
            if v in visiting:
                i = stack.index(v)
                cycles.append(stack[i:] + [v])
            elif v not in visited:
                dfs(v)
        stack.pop()
        visiting.remove(u)
        visited.add(u)

    for node in list(g.keys()):
        if node not in visited:
            dfs(node)
    return cycles


# ---------------------
# Compile clinical_logic JSON to DMNPackage
# ---------------------


def compile_to_dmn_package(logic: dict) -> DMNPackage:
    """Compile clinical_logic JSON into a DMNPackage for Z3 verification.

    Creates Z3 Bool variables for each predicate and builds rule antecedents
    from the boolean input patterns in the decision tables.
    """
    # Collect all unique input column names across modules
    all_input_vars: set[str] = set()
    for module in logic.get("modules", []):
        all_input_vars.update(module.get("input_columns", []))

    # Create Z3 Bool variables
    z3_vars: dict[str, BoolRef] = {name: Bool(name) for name in all_input_vars}

    # Domain: allow any combination (weak domain; could be strengthened)
    domain: BoolRef = BoolVal(True)

    tables: list[DMNTable] = []

    for module in logic.get("modules", []):
        module_id = module.get("module_id", "unknown")
        hit_policy = module.get("hit_policy", "FIRST")
        input_cols = module.get("input_columns", [])
        rules: list[DMNRule] = []

        for idx, rule in enumerate(module.get("rules", [])):
            inputs = rule.get("inputs", [])
            # Build Z3 antecedent from boolean input pattern
            conjuncts: list[BoolRef] = []
            for col_idx, inp in enumerate(inputs):
                if col_idx >= len(input_cols):
                    break
                var_name = input_cols[col_idx]
                var = z3_vars.get(var_name)
                if var is None:
                    continue
                if inp == "true":
                    conjuncts.append(var)
                elif inp == "false":
                    conjuncts.append(Not(var))
                # "-" means don't care; no constraint added

            if conjuncts:
                antecedent = And(*conjuncts) if len(conjuncts) > 1 else conjuncts[0]
            else:
                # Default rule (all "-"): always true
                antecedent = BoolVal(True)

            outputs = {}
            output_cols = module.get("output_columns", [])
            for oi, ov in enumerate(rule.get("outputs", [])):
                if oi < len(output_cols):
                    outputs[output_cols[oi]] = ov

            rules.append(DMNRule(
                rule_id=f"{module_id}_r{idx}",
                when=antecedent,
                outputs=outputs,
            ))

        tables.append(DMNTable(
            table_id=module_id,
            hit_policy=hit_policy,
            rules=rules,
            kind="CLASSIFIER",
        ))

    return DMNPackage(domain=domain, tables=tables)


# ---------------------
# Main entry point
# ---------------------


def verify_package(pkg: DMNPackage) -> VerificationReport:
    """Run the full evergreen verification suite."""
    report = VerificationReport()

    test_domain_sat(pkg, report)
    # If domain is UNSAT, stop early
    if report.results and not report.results[0].passed:
        return report

    test_rule_reachability(pkg, report)
    test_table_exhaustive(pkg, report)
    test_unique_mece(pkg, report)
    test_first_priority_shadowing(pkg, report)

    # Router cycle detection (structural)
    if pkg.router_table_ids:
        g = build_router_graph(pkg)
        cycles = find_cycles(g)
        if cycles:
            report.add_warning(f"Router graph has potential cycles: {cycles}")

    return report


def verify_clinical_logic(logic) -> dict:
    """High-level entry point: compile JSON to DMNPackage and verify.

    Returns dict compatible with the z3_check() REPL function:
    {"all_passed": bool, "checks": [...], "warnings": [...]}

    Defensive: if `logic` is not a dict, short-circuit with an empty report
    and a single warning. The REPL's z3_check() already guards this case,
    but the library-level guard means direct callers of verify_clinical_logic
    are also safe.
    """
    if not isinstance(logic, dict):
        return {
            "all_passed": False,
            "checks": [],
            "warnings": [
                f"verify_clinical_logic expects a dict, got {type(logic).__name__}"
            ],
        }
    try:
        pkg = compile_to_dmn_package(logic)
    except (AttributeError, TypeError, KeyError) as exc:
        # Something inside logic was an unexpected shape (e.g. modules was
        # a string, or a rule was not a dict). Report it instead of crashing.
        return {
            "all_passed": False,
            "checks": [],
            "warnings": [f"compile_to_dmn_package crashed on shape: {exc}"],
        }
    report = verify_package(pkg)
    return report.to_dict()
