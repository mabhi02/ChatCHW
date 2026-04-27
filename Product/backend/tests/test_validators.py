"""Tests for all 4 validators."""

import json
from pathlib import Path

from backend.validators import run_all_validators
from backend.validators.architecture import validate_architecture
from backend.validators.completeness import validate_completeness
from backend.validators.clinical import validate_clinical
from backend.validators.naming import validate_naming

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


class TestValidLogic:
    """Valid fixture should pass all validators."""

    def test_all_validators_pass(self):
        logic = load_fixture("valid_logic.json")
        result = run_all_validators(logic)
        # Filter out warnings; only check errors
        real_errors = [e for e in result["errors"] if e.get("severity") == "error"]
        assert len(real_errors) == 0, f"Unexpected errors: {real_errors}"

    def test_architecture_passes(self):
        logic = load_fixture("valid_logic.json")
        errors = validate_architecture(logic)
        real_errors = [e for e in errors if e.get("severity") == "error"]
        assert len(real_errors) == 0

    def test_completeness_passes(self):
        logic = load_fixture("valid_logic.json")
        errors = validate_completeness(logic)
        real_errors = [e for e in errors if e.get("severity") == "error"]
        assert len(real_errors) == 0

    def test_clinical_passes(self):
        logic = load_fixture("valid_logic.json")
        errors = validate_clinical(logic)
        real_errors = [e for e in errors if e.get("severity") == "error"]
        assert len(real_errors) == 0

    def test_naming_passes(self):
        logic = load_fixture("valid_logic.json")
        errors = validate_naming(logic)
        real_errors = [e for e in errors if e.get("severity") == "error"]
        assert len(real_errors) == 0


class TestInvalidLogic:
    """Invalid fixture should catch multiple errors."""

    def test_catches_errors(self):
        logic = load_fixture("invalid_logic.json")
        result = run_all_validators(logic)
        assert not result["passed"]
        assert result["error_count"] > 0

    def test_architecture_catches_missing_activator_rules(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_architecture(logic)
        messages = [e["message"] for e in errors]
        assert any("no rules" in m.lower() or "no modules" in m.lower() for m in messages)

    def test_architecture_catches_bad_router_first_rule(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_architecture(logic)
        messages = [e["message"] for e in errors]
        assert any("p_danger_sign_present" in m for m in messages)

    def test_clinical_catches_numeric_threshold(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_clinical(logic)
        messages = [e["message"] for e in errors]
        assert any("numeric" in m.lower() or "not a valid boolean" in m.lower() for m in messages)

    def test_clinical_catches_mismatched_array_length(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_clinical(logic)
        messages = [e["message"] for e in errors]
        assert any("count" in m.lower() for m in messages)

    def test_naming_catches_bad_predicate_id(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_naming(logic)
        messages = [e["message"] for e in errors]
        assert any("BadPredName" in m for m in messages)

    def test_naming_catches_bad_module_id(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_naming(logic)
        messages = [e["message"] for e in errors]
        assert any("ModCough" in m for m in messages)

    def test_completeness_catches_missing_danger_sign(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_completeness(logic)
        messages = [e["message"] for e in errors]
        assert any("p_danger_sign_present" in m for m in messages)

    def test_clinical_catches_bad_fail_safe(self):
        logic = load_fixture("invalid_logic.json")
        errors = validate_clinical(logic)
        messages = [e["message"] for e in errors]
        assert any("fail_safe" in m for m in messages)


class TestEdgeCases:
    """Edge cases and missing fields."""

    def test_empty_dict(self):
        result = run_all_validators({})
        assert not result["passed"]

    def test_missing_integrative(self):
        logic = load_fixture("valid_logic.json")
        del logic["integrative"]
        errors = validate_architecture(logic)
        messages = [e["message"] for e in errors]
        assert any("integrative" in m.lower() for m in messages)

    def test_module_without_default_row(self):
        logic = load_fixture("valid_logic.json")
        # Remove the default (last) rule
        logic["modules"][0]["rules"] = logic["modules"][0]["rules"][:-1]
        errors = validate_clinical(logic)
        messages = [e["message"] for e in errors]
        assert any("default" in m.lower() for m in messages)
