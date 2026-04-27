"""
dmn_preprocessor.py
───────────────────
Converts one patient record from output.json into a DMN-ready input dictionary
aligned to the product team's variable naming convention (variables.json).

Angelina Wang
"""

from __future__ import annotations
import json
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# DMN Input Schema
# Field names match product team's variables.json prefixes:
#   demo_ = demographics
#   q_    = CHW questions
#   ex_   = exam findings
#   hx_   = history / context
#   lab_  = lab results
#   v_    = measured values
#   mod_  = module done flags
# Full documentation: docs/dmn_input_schema_note.md
# Field mapping:      docs/dmn_mapping_note.md
# ──────────────────────────────────────────────────────────────────────────────
DMN_INPUT_SCHEMA = {
    # ── Demographics ─────────────────────────────────────────────────────────
    "demo_child_age_months":              int,   # age * 12

    # ── Cough / Respiratory ──────────────────────────────────────────────────
    "q_has_cough":                        bool,  # complaint == "Cough"
    "q_cough_duration_days":              int,   # duration in days; 0 if no cough
    "v_respiratory_rate_per_min":         int,   # raw RR value; 0 if absent
    "ex_chest_indrawing":                 bool,  # exam_findings["Chest indrawing"]

    # ── Diarrhoea ────────────────────────────────────────────────────────────
    "q_has_diarrhoea":                    bool,  # complaint == "Diarrhea"
    "q_diarrhoea_duration_days":          int,   # duration in days; 0 if no diarrhoea
    "q_blood_in_stool":                   bool,  # chw_questions["Blood in stool"]

    # ── Fever / Malaria ──────────────────────────────────────────────────────
    "q_has_fever":                        bool,  # temperature >= 37.5°C
    "q_fever_duration_days":              int,   # duration in days; 0 if no fever
    "hx_malaria_area":                    bool,  # context flag — default False
    "lab_rdt_malaria_result":             str,   # "positive" | "negative" | "not_done"

    # ── Nutrition ────────────────────────────────────────────────────────────
    "v_muac_strap_colour":                str,   # "red" | "yellow" | "green"
    "ex_swelling_both_feet":              bool,  # context flag — default False

    # ── Danger signs ─────────────────────────────────────────────────────────
    "q_convulsions":                      bool,  # not in dataset — default False
    "q_vomits_everything":                bool,  # not in dataset — default False
    "q_not_able_to_drink_or_feed":        bool,  # chw_questions["Unable to drink"]
    "ex_unusually_sleepy_or_unconscious": bool,  # chw_questions["Lethargic"]
    "is_priority_exit":                   bool,  # default False — set by engine

    # ── Router mod_done flags (product team naming from variables.json) ──────
    "mod_assess_done":                    bool,  # default False
    "mod_danger_sign_check_done":         bool,  # default False
    "mod_diarrhoea_treatment_done":       bool,  # default False
    "mod_fast_breathing_treatment_done":  bool,  # default False
    "mod_fever_malaria_treatment_done":   bool,  # default False
    "mod_malnutrition_screening_done":    bool,  # default False
}


# ──────────────────────────────────────────────────────────────────────────────
# Raw-field helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_respiratory_rate(rr_raw: str) -> Optional[int]:
    """Extract integer from strings like '61/min'. Returns None if absent."""
    if not rr_raw:
        return None
    match = re.match(r"(\d+)", str(rr_raw).strip())
    return int(match.group(1)) if match else None


def _parse_temperature(temp_raw: str) -> Optional[float]:
    """Extract float from strings like '39.2°C'. Returns None if absent."""
    if not temp_raw:
        return None
    match = re.search(r"([\d.]+)", str(temp_raw).strip())
    return float(match.group(1)) if match else None


def _duration_in_days(duration: dict) -> int:
    """
    Normalise duration dict -> integer days.
    Handles units: days, weeks, months (approximate).
    """
    value = int(duration.get("value", 0))
    unit  = str(duration.get("unit", "days")).lower()
    if unit in ("week", "weeks"):
        return value * 7
    elif unit in ("month", "months"):
        return value * 30
    else:
        return value


def _yes(value) -> bool:
    """Coerce a CHW answer to bool. Handles 'Yes'/'No', True/False, etc."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "true", "1")


# ──────────────────────────────────────────────────────────────────────────────
# Core preprocessor
# ──────────────────────────────────────────────────────────────────────────────
def to_dmn_input(patient: dict, context: Optional[dict] = None) -> dict:
    """
    Convert one patient record from output.json to a flat DMN input dict
    using the product team's variable naming convention (variables.json).

    Args:
        patient : One element from the output.json list.
        context : Optional deployment/visit context for fields not in output.json.
                  Supported keys:
                    malaria_area          (bool, default False)
                    rdt_result            (str,  default "not_done")
                    muac_result           (str,  default "green")
                    swelling_of_both_feet (bool, default False)

    Returns:
        dict: Flat DMN-ready patient dictionary matching product team field names.
    """
    ctx = context or {}

    complaint     = str(patient.get("complaint", "")).strip()
    age           = int(patient.get("age", 0))
    duration      = patient.get("duration", {"value": 0, "unit": "days"})
    chw           = patient.get("chw_questions", {})
    exam          = patient.get("exam_findings", {})

    duration_days = _duration_in_days(duration)

    # ── Cough / Respiratory ──────────────────────────────────────────────────
    cough_present  = complaint == "Cough"
    rr_raw         = exam.get("Respiratory rate", "")
    rr_value       = _parse_respiratory_rate(rr_raw)
    chest_indrawing = _yes(exam.get("Chest indrawing", ""))

    # ── Diarrhoea ────────────────────────────────────────────────────────────
    has_diarrhoea  = complaint == "Diarrhea"
    if "No blood in stool" in chw:
        blood_in_stool = False
    else:
        blood_in_stool = _yes(chw.get("Blood in stool", ""))

    # ── Fever ────────────────────────────────────────────────────────────────
    temp_raw       = exam.get("Temperature", "")
    temp_value     = _parse_temperature(temp_raw)
    has_fever      = (temp_value is not None) and (temp_value >= 37.5)

    # ── Danger signs ─────────────────────────────────────────────────────────
    unable_to_drink = _yes(chw.get("Unable to drink", ""))
    lethargic       = _yes(chw.get("Lethargic", chw.get("Lethargic/unconscious", "")))

    # ── Context / deployment fields (not in output.json) ────────────────────
    malaria_area          = bool(ctx.get("malaria_area", False))
    rdt_result            = str(ctx.get("rdt_result", "not_done"))
    muac_colour           = str(ctx.get("muac_result", "green"))
    swelling_of_both_feet = bool(ctx.get("swelling_of_both_feet", False))

    # ── Assemble ─────────────────────────────────────────────────────────────
    return {
        # Demographics
        "demo_child_age_months":              age * 12,

        # Cough / Respiratory
        "q_has_cough":                        cough_present,
        "q_cough_duration_days":              duration_days if cough_present else 0,
        "v_respiratory_rate_per_min":         rr_value if rr_value is not None else 0,
        "ex_chest_indrawing":                 chest_indrawing,

        # Diarrhoea
        "q_has_diarrhoea":                    has_diarrhoea,
        "q_diarrhoea_duration_days":          duration_days if has_diarrhoea else 0,
        "q_blood_in_stool":                   blood_in_stool,

        # Fever / Malaria
        "q_has_fever":                        has_fever,
        "q_fever_duration_days":              duration_days if has_fever else 0,
        "hx_malaria_area":                    malaria_area,
        "lab_rdt_malaria_result":             rdt_result,

        # Nutrition
        "v_muac_strap_colour":                muac_colour,
        "ex_swelling_both_feet":              swelling_of_both_feet,

        # Danger signs
        "q_convulsions":                      False,   # not in output.json
        "q_vomits_everything":                False,   # not in output.json
        "q_not_able_to_drink_or_feed":        unable_to_drink,
        "ex_unusually_sleepy_or_unconscious": lethargic,
        "is_priority_exit":                   False,   # set by engine

        # Router mod_done flags — all False on entry, engine updates after each run
        "mod_assess_done":                    False,
        "mod_danger_sign_check_done":         False,
        "mod_diarrhoea_treatment_done":       False,
        "mod_fast_breathing_treatment_done":  False,
        "mod_fever_malaria_treatment_done":   False,
        "mod_malnutrition_screening_done":    False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch helper
# ──────────────────────────────────────────────────────────────────────────────
def batch_preprocess(
    path: str,
    context: Optional[dict] = None,
) -> list[dict]:
    """
    Load output.json and return a list of DMN-ready dicts, one per patient.
    Each dict also carries "_row_number" for traceability.
    """
    with open(path, encoding="utf-8") as f:
        patients = json.load(f)

    results = []
    for p in patients:
        dmn = to_dmn_input(p, context=context)
        dmn["_row_number"] = p.get("_row_number")
        results.append(dmn)
    return results


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "output.json"

    with open(path, encoding="utf-8") as f:
        patients = json.load(f)

    # Example 1: row 3 — Severe pneumonia child (chest indrawing, fast RR)
    p1 = next(p for p in patients if p["_row_number"] == 3)
    # Example 2: row 5 — Diarrhoea
    p2 = next(p for p in patients if p["_row_number"] == 5)
    # Example 3: row 9 — Pneumonia fast breathing
    p3 = next(p for p in patients if p["_row_number"] == 9)

    examples = []
    for patient in [p1, p2, p3]:
        dmn = to_dmn_input(patient)
        dmn["_row_number"]       = patient["_row_number"]
        dmn["_source_complaint"] = patient["complaint"]
        dmn["_source_diagnosis"] = patient["diagnosis"]
        dmn["_source_age"]       = patient["age"]
        examples.append(dmn)

    # Save
    out_path = "dmn_example_outputs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} examples to {out_path}")

    # Also print for inspection
    for ex in examples:
        print(f"\nRow {ex['_row_number']} — {ex['_source_diagnosis']}")
        for k, v in ex.items():
            if not k.startswith("_"):
                print(f"  {k:<35} {v}")