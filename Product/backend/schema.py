"""JSON schema definitions for the clinical_logic output.

Defines the structure that the RLM REPL session must produce.
Based on ORCHESTRATOR.md Section 4.
"""

from typing import Any

# Valid input cell values for boolean-only DMN
VALID_RULE_INPUTS = {"true", "false", "-"}

# Variable prefix taxonomy (pipeline convention; domain-agnostic)
VALID_PREFIXES = {
    # User-entered
    "q_",       # question / self-reported
    "ex_",      # exam / clinician observation
    "v_",       # vital / measurement (device, count, timed)
    "lab_",     # lab / point-of-care test
    "img_",     # imaging (rare)
    "hx_",      # baseline history / chronic conditions
    "demo_",    # demographics
    # System / computed
    "sys_",     # system / platform context
    "prev_",    # prior encounter state
    "calc_",    # calculated numeric
    "p_",       # predicate flag (boolean)
    "s_",       # score (risk/triage)
    "dx_",      # diagnosis / classification
    "sev_",     # severity label
    "need_",    # workflow need (activator outputs)
    "tx_",      # treatment intent
    "rx_",      # medication execution
    "proc_",    # procedure (non-drug)
    "ref_",     # referral destination
    "adv_",     # advice / counselling
    "fu_",      # follow-up scheduling
    "out_",     # disposition
    "err_",     # data quality / validation
    "m_",       # message ID (phrase bank link)
    "mod_",     # module ID
}

# Prefixes that indicate equipment-dependent measurement (fail_safe should be 1)
EQUIPMENT_PREFIXES = {"v_", "lab_", "img_"}

# Prefixes that indicate self-reported input (fail_safe should be 0)
CAREGIVER_PREFIXES = {"q_"}  # name kept for backwards-compat; also covers hx_-style history

# Computed-only prefixes (must never appear as user-entered fields)
COMPUTED_ONLY_PREFIXES = {
    "sys_", "prev_", "calc_", "p_", "s_", "dx_", "sev_",
    "need_", "tx_", "rx_", "proc_", "ref_", "adv_", "fu_",
    "out_", "err_",
}

# Phrase bank categories
VALID_PHRASE_CATEGORIES = {
    "question", "diagnosis", "treatment", "advice",
    "referral", "followup", "instruction",
}

# Hit policies
VALID_HIT_POLICIES = {"FIRST", "COLLECT", "UNIQUE", "PRIORITY"}


def get_expected_top_level_keys() -> set[str]:
    """Return the expected top-level keys in the clinical_logic JSON."""
    return {"modules", "activator", "router", "integrative", "predicates", "phrase_bank"}


def validate_schema_shape(logic: dict) -> list[dict[str, str]]:
    """Validate that the top-level structure is correct.

    Returns a list of error dicts: [{"message": str, "severity": str}]
    """
    errors: list[dict[str, str]] = []
    expected = get_expected_top_level_keys()
    actual = set(logic.keys())

    missing = expected - actual
    for key in sorted(missing):
        errors.append({
            "message": f"Missing required top-level key: '{key}'",
            "severity": "error",
        })

    return errors


def make_fixture_valid() -> dict[str, Any]:
    """Generate a minimal valid clinical_logic fixture for testing.

    Generic placeholder data only — no disease names, drug names, or
    manual-specific vocabulary. This function exists as dead code today
    (no callers) but is preserved as a schema-shape reference for any
    future test that wants a baseline valid structure.
    """
    return {
        "modules": [
            {
                "module_id": "mod_topic_a",
                "display_name": "Topic A Assessment",
                "hit_policy": "FIRST",
                "input_columns": ["p_alert_criterion_a", "p_moderate_criterion_a"],
                "output_columns": ["dx_classification", "tx_treatment", "ref_hospital_urgent", "m_message_ids"],
                "rules": [
                    {
                        "inputs": ["true", "-"],
                        "outputs": ["classification_severe", "tx_urgent", "true", "m_dx_severe_topic_a"],
                        "provenance": {"page": "p.1", "quote": "placeholder quote 1"},
                    },
                    {
                        "inputs": ["false", "true"],
                        "outputs": ["classification_moderate", "tx_standard", "false", "m_dx_moderate_topic_a"],
                        "provenance": {"page": "p.2", "quote": "placeholder quote 2"},
                    },
                    {
                        "inputs": ["-", "-"],
                        "outputs": ["classification_low", "tx_watchful_waiting", "false", "m_dx_low_topic_a"],
                        "provenance": {"page": "p.3", "quote": "placeholder quote 3"},
                    },
                ],
            },
        ],
        "activator": {
            "input_columns": ["q_symptom_a", "q_symptom_b"],
            "rules": [
                {"inputs": ["true", "-"], "module_id": "mod_topic_a"},
                {"inputs": ["-", "true"], "module_id": "mod_topic_b"},
            ],
        },
        "router": {
            "rules": [
                {"condition": "p_danger_sign_present == true", "next_module": "mod_urgent_escalation", "priority": 0},
                {"condition": "mod_topic_a in required AND mod_topic_a not in completed", "next_module": "mod_topic_a", "priority": 1},
                {"condition": "always", "next_module": "mod_integrative", "priority": 99},
            ],
        },
        "integrative": {
            "description": "Combine module outputs: highest referral wins, additive treatments, shortest follow-up",
            "rules": [],
        },
        "predicates": [
            {
                "predicate_id": "p_danger_sign_present",
                "source_vars": ["q_symptom_alert_1", "q_symptom_alert_2", "ex_finding_alert"],
                "threshold_expression": "q_symptom_alert_1 OR q_symptom_alert_2 OR ex_finding_alert",
                "human_label": "Any alert / red-flag criterion present (composite)",
                "fail_safe": 1,
                "page_ref": "p.1",
            },
            {
                "predicate_id": "p_alert_criterion_a",
                "source_vars": ["ex_finding_alert"],
                "threshold_expression": "ex_finding_alert == true",
                "human_label": "Alert criterion for topic A",
                "fail_safe": 0,
                "page_ref": "p.1",
            },
            {
                "predicate_id": "p_moderate_criterion_a",
                "source_vars": ["v_measurement_a", "demo_age_group"],
                "threshold_expression": "v_measurement_a >= threshold_for_age_group",
                "human_label": "Moderate criterion for topic A",
                "fail_safe": 1,
                "page_ref": "p.2",
            },
        ],
        "phrase_bank": [
            {
                "message_id": "m_dx_severe_topic_a",
                "category": "diagnosis",
                "english_text": "Severe topic-A classification. Initiate urgent treatment and refer.",
                "placeholder_vars": [],
                "page_ref": "p.1",
            },
            {
                "message_id": "m_dx_moderate_topic_a",
                "category": "diagnosis",
                "english_text": "Moderate topic-A classification. Give standard treatment for {days} days.",
                "placeholder_vars": ["days"],
                "page_ref": "p.2",
            },
            {
                "message_id": "m_dx_low_topic_a",
                "category": "diagnosis",
                "english_text": "Low severity. Advise home care.",
                "placeholder_vars": [],
                "page_ref": "p.3",
            },
        ],
    }
