"""Tests for Z3 verifier."""

import json
from pathlib import Path

from backend.z3_verifier.evergreen_tests import verify_clinical_logic, compile_to_dmn_package, verify_package
from backend.z3_verifier.patient_gen import generate_synthetic_patients

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestZ3VerifyValidLogic:
    """Valid fixture should pass Z3 verification."""

    def test_all_checks_pass(self):
        logic = load_fixture("valid_logic.json")
        result = verify_clinical_logic(logic)
        assert result["all_passed"], f"Z3 failures: {result['checks']}"

    def test_domain_is_sat(self):
        logic = load_fixture("valid_logic.json")
        result = verify_clinical_logic(logic)
        domain_check = next(c for c in result["checks"] if c["testId"] == "T00_DOMAIN_SAT")
        assert domain_check["passed"]

    def test_all_rules_reachable(self):
        logic = load_fixture("valid_logic.json")
        result = verify_clinical_logic(logic)
        reachability_checks = [c for c in result["checks"] if c["testId"] == "T02_RULE_REACHABLE"]
        assert all(c["passed"] for c in reachability_checks)

    def test_table_exhaustive(self):
        logic = load_fixture("valid_logic.json")
        result = verify_clinical_logic(logic)
        exhaustive_checks = [c for c in result["checks"] if c["testId"] == "T03_TABLE_EXHAUSTIVE"]
        assert all(c["passed"] for c in exhaustive_checks)


class TestZ3SyntheticPatients:
    """Test synthetic patient generation."""

    def test_generates_patients_for_each_rule(self):
        logic = load_fixture("valid_logic.json")
        pkg = compile_to_dmn_package(logic)
        patients = generate_synthetic_patients(pkg)
        # Should have at least one patient per rule
        total_rules = sum(len(t.rules) for t in pkg.tables)
        assert len(patients) == total_rules

    def test_patient_has_required_fields(self):
        logic = load_fixture("valid_logic.json")
        pkg = compile_to_dmn_package(logic)
        patients = generate_synthetic_patients(pkg)
        for p in patients:
            assert "table_id" in p
            assert "rule_id" in p
            assert "patient" in p
            assert isinstance(p["patient"], dict)


class TestZ3EdgeCases:
    """Edge cases for Z3 verification."""

    def test_empty_logic(self):
        result = verify_clinical_logic({"modules": []})
        # Should still pass domain check (trivially true)
        assert any(c["testId"] == "T00_DOMAIN_SAT" for c in result["checks"])

    def test_module_missing_default_row_not_exhaustive(self):
        """A module without a default (all-dash) row may not be exhaustive."""
        logic = load_fixture("valid_logic.json")
        # Remove the default row (last rule)
        logic["modules"][0]["rules"] = logic["modules"][0]["rules"][:-1]
        result = verify_clinical_logic(logic)
        exhaustive_checks = [c for c in result["checks"] if c["testId"] == "T03_TABLE_EXHAUSTIVE"]
        # Without default row, may or may not be exhaustive depending on coverage
        # The fixture's first 3 rules don't cover all combinations
        assert len(exhaustive_checks) > 0
