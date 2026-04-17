"""
Traffic Cop 1 — Flag-Based Module Dispatcher
Angelina Wang

Reads the patient's symptom flags and done flags to determine
the next clinical module to run.

Replaces the older queue-based approach per professor's updated
orchestration direction. Instead of building a full queue upfront,
the dispatcher is called after each module completes and returns
the next module to run based on:
  - has_<symptom>  : whether the patient has this symptom
  - <module>_done  : whether this module has already run

Input:  patient flags dict (from dmn_preprocessor.py output)
Output: next module to run (str), or None if all done
"""

# Priority order — dispatcher checks modules in this order
MODULE_PRIORITY = [
    ("has_cough",    "cough_mod_done",    "cough_module"),
    ("has_fever",    "fever_mod_done",    "fever_module"),
    ("has_diarrhea", "diarrhea_mod_done", "diarrhea_module"),
    # Add additional symptom->module mappings here as medical team delivers them
]


def get_next_module(patient_flags: dict) -> str | None:
    """
    Returns the next module to run for this patient based on symptom
    and done flags.

    Args:
        patient_flags (dict): Patient flags from dmn_preprocessor output.
            Expected keys:
                has_cough        (bool)
                has_fever        (bool)
                has_diarrhea     (bool)
                cough_mod_done   (bool)
                fever_mod_done   (bool)
                diarrhea_mod_done (bool)

    Returns:
        str:  Name of the next module to run.
        None: All relevant modules have been run — proceed to aggregate.

    Example:
        flags = {
            "has_cough": True,    "cough_mod_done": False,
            "has_fever": True,    "fever_mod_done": True,
            "has_diarrhea": False, "diarrhea_mod_done": False,
        }
        get_next_module(flags)  # -> "cough_module"
    """
    for has_flag, done_flag, module_name in MODULE_PRIORITY:
        if patient_flags.get(has_flag, False) and not patient_flags.get(done_flag, False):
            return module_name

    return None  # all relevant modules done — ready for aggregate_final


if __name__ == "__main__":
    test_cases = [
        # Cough + fever, nothing done yet
        {
            "has_cough": True,  "cough_mod_done": False,
            "has_fever": True,  "fever_mod_done": False,
            "has_diarrhea": False, "diarrhea_mod_done": False,
        },
        # Cough done, fever still pending
        {
            "has_cough": True,  "cough_mod_done": True,
            "has_fever": True,  "fever_mod_done": False,
            "has_diarrhea": False, "diarrhea_mod_done": False,
        },
        # Diarrhea only, nothing done
        {
            "has_cough": False, "cough_mod_done": False,
            "has_fever": False, "fever_mod_done": False,
            "has_diarrhea": True, "diarrhea_mod_done": False,
        },
        # All done
        {
            "has_cough": True,  "cough_mod_done": True,
            "has_fever": True,  "fever_mod_done": True,
            "has_diarrhea": True, "diarrhea_mod_done": True,
        },
        # No symptoms
        {
            "has_cough": False, "cough_mod_done": False,
            "has_fever": False, "fever_mod_done": False,
            "has_diarrhea": False, "diarrhea_mod_done": False,
        },
    ]

    for flags in test_cases:
        result = get_next_module(flags)
        print(f"Input:  {flags}")
        print(f"Next module: {result}\n")