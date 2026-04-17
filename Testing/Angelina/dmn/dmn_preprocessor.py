"""
dmn_preprocessor.py
───────────────────
Patient-to-DMN Preprocessor
Angelina Wang

Converts one patient record from output.json into a DMN-ready input
dictionary that can be passed directly to the DMN execution engine.

DMN-side equivalent of cht_translator.py.

Usage (single patient):
    from dmn_preprocessor import to_dmn_input
    dmn_dict = to_dmn_input(patient, context={"malaria_area": True})

Usage (batch):
    from dmn_preprocessor import batch_preprocess
    dmn_patients = batch_preprocess("output.json")
"""

from __future__ import annotations
import json
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# DMN Input Schema
# Full field documentation: docs/dmn_input_schema_note.md
# Field mapping from output.json: docs/dmn_mapping_note.md
# ──────────────────────────────────────────────────────────────────────────────
DMN_INPUT_SCHEMA = {

    "age_months":                int,   # age * 12
    # ── Module A & E (Cough / Respiratory) ──────────────────────────────────
    "cough_present":             bool,  # complaint == "Cough"
    "cough_duration_days":       int,   # duration in days; 0 if no cough
    "fast_breathing_present":    bool,  # RR above WHO age-adjusted threshold
    "chest_indrawing_present":   bool,  # exam_findings["Chest indrawing"]
    "cough_with_fast_breathing": bool,  # cough_present AND fast_breathing_present
    "chest_indrawing":           bool,  # alias for module E (same source as above)
    # ── Module B (Diarrhoea) ─────────────────────────────────────────────────
    "has_diarrhoea":             bool,  # complaint == "Diarrhea"
    "diarrhoea_duration_days":   int,   # duration in days; 0 if no diarrhoea
    "blood_in_stool":            bool,  # chw_questions["Blood in stool"]
    # ── Module C (Fever / Malaria) ───────────────────────────────────────────
    "hot_with_fever":            bool,  # temperature >= 37.5°C
    "fever_duration_days":       int,   # duration in days; 0 if no fever
    "malaria_area":              bool,  # context flag — default False
    "rdt_available":             bool,  # context flag — default False
    "rdt_result":                str,   # "positive" | "negative" | "not_done"
    # ── Module D (Nutrition) ─────────────────────────────────────────────────
    "muac_result":               str,   # "red" | "yellow" | "green" — default "green"
    "swelling_of_both_feet":     bool,  # context flag — default False

    # has_<module> — whether this module should run for this patient
    "has_cough":                 bool,  # alias of cough_present for dispatcher
    "has_diarrhea":              bool,  # alias of has_diarrhoea for dispatcher
    "has_fever":                 bool,  # alias of hot_with_fever for dispatcher

    # <module>_done — set False on entry, updated by engine after each module runs
    "cough_mod_done":            bool,  # default False
    "diarrhea_mod_done":         bool,  # default False
    "fever_mod_done":            bool,  # default False
    "nutrition_mod_done":        bool,  # default False
}


# ──────────────────────────────────────────────────────────────────────────────
# WHO IMCI fast-breathing thresholds (breaths/min)
# ──────────────────────────────────────────────────────────────────────────────
def _fast_breathing_threshold(age_years: int) -> int:
    """Return the WHO IMCI fast-breathing threshold for the given age in years."""
    age_months = age_years * 12  # approximate; fine for integer-year ages
    if age_months < 2:
        return 60
    elif age_months < 12:
        return 50
    elif age_years < 5:
        return 40
    else:
        return 30


# ──────────────────────────────────────────────────────────────────────────────
# Raw-field helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_respiratory_rate(rr_raw: str) -> Optional[int]:
    """Extract integer from strings like '61/min'.  Returns None if absent."""
    if not rr_raw:
        return None
    match = re.match(r"(\d+)", str(rr_raw).strip())
    return int(match.group(1)) if match else None


def _parse_temperature(temp_raw: str) -> Optional[float]:
    """Extract float from strings like '39.2°C'.  Returns None if absent."""
    if not temp_raw:
        return None
    match = re.search(r"([\d.]+)", str(temp_raw).strip())
    return float(match.group(1)) if match else None


def _duration_in_days(duration: dict) -> int:
    """
    Normalise duration dict → integer days.
    Handles units: days, weeks, months (approximate).
    """
    value = int(duration.get("value", 0))
    unit  = str(duration.get("unit", "days")).lower()
    if unit in ("week", "weeks"):
        return value * 7
    elif unit in ("month", "months"):
        return value * 30
    else:  # days (or unknown — treat as days)
        return value


def _yes(value) -> bool:
    """Coerce a CHW answer to bool.  Handles 'Yes'/'No', True/False, etc."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "true", "1")


# ──────────────────────────────────────────────────────────────────────────────
# Core preprocessor
# ──────────────────────────────────────────────────────────────────────────────
def to_dmn_input(patient: dict, context: Optional[dict] = None) -> dict:
    """
    Convert one patient record from output.json to a flat DMN input dict.

    Args:
        patient : One element from the output.json list.
        context : Optional deployment/visit context to supply fields that are
                  not present in output.json.  Supported keys:
                    malaria_area       (bool, default False)
                    rdt_available      (bool, default False)
                    rdt_result         (str,  default "not_done")
                    muac_result        (str,  default "green")
                    swelling_of_both_feet (bool, default False)

    Returns:
        dict: Flat DMN-ready patient dictionary.
    """
    ctx = context or {}

    complaint      = str(patient.get("complaint", "")).strip()
    age            = int(patient.get("age", 0))
    duration       = patient.get("duration", {"value": 0, "unit": "days"})
    chw            = patient.get("chw_questions", {})
    exam           = patient.get("exam_findings", {})

    duration_days  = _duration_in_days(duration)

    # ── Cough / Respiratory ──────────────────────────────────────────────────
    cough_present  = complaint == "Cough"

    rr_raw         = exam.get("Respiratory rate", "")
    rr_value       = _parse_respiratory_rate(rr_raw)
    threshold      = _fast_breathing_threshold(age)
    fast_breathing = (rr_value is not None) and (rr_value >= threshold)

    chest_indrawing_raw     = exam.get("Chest indrawing", "")
    chest_indrawing_bool    = _yes(chest_indrawing_raw)

    cough_with_fast_breathing = cough_present and fast_breathing

    # ── Diarrhoea ────────────────────────────────────────────────────────────
    has_diarrhoea  = complaint == "Diarrhea"

    blood_raw      = chw.get("Blood in stool", chw.get("No blood in stool", ""))
    # "No blood in stool" key means blood=False; handle both keys
    if "No blood in stool" in chw:
        blood_in_stool = False
    else:
        blood_in_stool = _yes(blood_raw)

    # ── Fever ────────────────────────────────────────────────────────────────
    temp_raw       = exam.get("Temperature", "")
    temp_value     = _parse_temperature(temp_raw)
    hot_with_fever = (temp_value is not None) and (temp_value >= 37.5)

    # ── Context / deployment fields (not in output.json) ────────────────────
    malaria_area          = bool(ctx.get("malaria_area", False))
    rdt_available         = bool(ctx.get("rdt_available", False))
    rdt_result            = str(ctx.get("rdt_result", "not_done"))
    muac_result           = str(ctx.get("muac_result", "green"))
    swelling_of_both_feet = bool(ctx.get("swelling_of_both_feet", False))

    # ── Assemble ─────────────────────────────────────────────────────────────
    return {

        # Demographics 
        "age_months":                age * 12,
        
        # Module A & E — Cough / Respiratory
        "cough_present":             cough_present,
        "cough_duration_days":       duration_days if cough_present else 0,
        "fast_breathing_present":    fast_breathing,
        "chest_indrawing_present":   chest_indrawing_bool,
        "cough_with_fast_breathing": cough_with_fast_breathing,
        "chest_indrawing":           chest_indrawing_bool,   # module E alias

        # Module B — Diarrhoea
        "has_diarrhoea":             has_diarrhoea,
        "diarrhoea_duration_days":   duration_days if has_diarrhoea else 0,
        "blood_in_stool":            blood_in_stool,

        # Module C — Fever / Malaria
        "hot_with_fever":            hot_with_fever,
        "fever_duration_days":       duration_days if hot_with_fever else 0,
        "malaria_area":              malaria_area,
        "rdt_available":             rdt_available,
        "rdt_result":                rdt_result,

        # Module D — Nutrition (context-supplied)
        "muac_result":               muac_result,
        "swelling_of_both_feet":     swelling_of_both_feet,

        # Dispatcher flags (professor's new orchestration model)
        "has_cough":                 cough_present,
        "has_diarrhea":              has_diarrhoea,
        "has_fever":                 hot_with_fever,
 
        # module_done flags — initialized False, updated by engine after each run
        "cough_mod_done":            False,
        "diarrhea_mod_done":         False,
        "fever_mod_done":            False,
        "nutrition_mod_done":        False,
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

    examples = []
    for patient in [p1, p2]:
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
                print(f"  {k:<30} {v}")