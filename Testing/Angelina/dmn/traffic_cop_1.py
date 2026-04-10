"""
Traffic Cop 1 — Queue Initialization - 
Angelina Wang

Reads the patient's symptom flags and builds the complete ordered list of
clinical modules the patient needs to go through, before any modules are run.

Input:  symptom flags dict (from patient record + Data Converter output)
Output: ordered module queue (list of strings)
"""

# Mapping from symptom flag -> module name, in the order they should appear
SYMPTOM_MODULE_MAP = [
    ("has_fever",    "fever_module"),
    ("has_cough",    "cough_module"),
    ("has_diarrhea", "diarrhea_module"),
    # Add additional symptom->module mappings here as Medical team delivers them
]


def build_module_queue(symptom_flags: dict) -> list:
    """
    Builds the ordered module queue for a patient based on their symptom flags.

    Args:
        symptom_flags (dict): Boolean flags from the patient record and/or Data Converter output. 
        {
            "has_fever": True,
            "has_cough": True,
            "has_diarrhea": False
        }

    Returns:
        list: Ordered queue of module names to execute. 
        [
            "opening_module",
            "fever_module",
            "cough_module",
            "integrative_module"
        ]
    """
    queue = []
    
    queue.append("opening_module")

    for flag, module in SYMPTOM_MODULE_MAP:
        if symptom_flags.get(flag, False):
            queue.append(module)

    queue.append("integrative_module")

    return queue


# Test, remove before final integration
if __name__ == "__main__":
    test_cases = [
        # Fever + cough
        {"has_fever": True, "has_cough": True, "has_diarrhea": False},
        # Diarrhea only
        {"has_fever": False, "has_cough": False, "has_diarrhea": True},
        # No symptoms
        {"has_fever": False, "has_cough": False, "has_diarrhea": False},
        # All symptoms
        {"has_fever": True, "has_cough": True, "has_diarrhea": True},
    ]

    for flags in test_cases:
        result = build_module_queue(flags)
        print(f"Input:  {flags}")
        print(f"Output: {result}\n")