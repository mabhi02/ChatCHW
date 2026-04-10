"""
cht_translator.py
-----------------
Translates one patient record from parsed_rows.json format
into the JSON format CHT's API expects.
"""

import re
import time


def to_cht_format(patient: dict) -> dict:
    """
    Convert a parsed_rows.json patient record to CHT API format.

    Args:
        patient: one patient record from parsed_rows.json

    Returns:
        dict in CHT API format
    """
    chw      = patient.get("chw_questions", {})
    exam     = patient.get("exam_findings", {})
    duration = patient.get("duration", {})
    complaint = patient.get("complaint", "")

    # age (years) → age_months
    age_months = int(patient.get("age", 0)) * 12

    # duration → days
    duration_days = None
    if isinstance(duration, dict):
        value = duration.get("value", 0)
        unit  = duration.get("unit", "days")
        duration_days = value * 7 if "week" in unit else value

    # exam_findings["Respiratory rate"] → breaths_per_minute (strip "/min")
    rr_raw = exam.get("Respiratory rate", "")
    rr_match = re.search(r"(\d+)", str(rr_raw))
    breaths_per_minute = int(rr_match.group(1)) if rr_match else None

    # helper: look up yes/no, default to "no"
    def yn(source: dict, key: str) -> str:
        val = source.get(key, "")
        return "yes" if str(val).strip().lower() == "yes" else "no"

    result = {
        "_meta": {
            "form": "orchestrator",
            "reported_date": int(time.time() * 1000)
        },

        # demographics
        "age_months": age_months,
        "sex":        str(patient.get("sex", "")).lower(),

        # danger signs — default "no" if missing
        "convulsions_present":                yn(chw, "Convulsions"),
        "unusually_sleepy_or_unconscious":    yn(chw, "Lethargic"),
        "not_able_to_drink_or_feed_anything": yn(chw, "Unable to drink"),
        "vomits_everything":                  yn(chw, "Vomits everything"),
        "chest_indrawing":                    yn(exam, "Chest indrawing"),
        "swelling_of_both_feet":              yn(chw, "Swelling of both feet"),
        "muac_color":                         patient.get("muac_color", "green"),

        # cough
        "cough_present":       "yes" if complaint == "Cough" else "no",
        "cough_duration_days": duration_days if complaint == "Cough" else None,
        "breaths_per_minute":  breaths_per_minute,

        # diarrhoea
        "diarrhoea_present":       "yes" if complaint == "Diarrhea" else "no",
        "diarrhoea_duration_days": duration_days if complaint == "Diarrhea" else None,
        "blood_in_stool":          yn(chw, "Blood in stool"),

        # fever
        "fever_present":       "yes" if complaint == "Fever" else "no",
        "fever_duration_days": duration_days if complaint == "Fever" else None,
    }

    # remove None values
    return {k: v for k, v in result.items() if v is not None}


if __name__ == "__main__":
    import json

    with open("output.json") as f:
        records = json.load(f)

    # Test first 3 records
    for i, patient in enumerate(records[:3]):
        print(f"=== Patient {i+1} (row {patient.get('_row_number')}) ===")
        print("Input:", patient.get("age_group"), patient.get("age"), "y,",
              patient.get("sex"), patient.get("complaint"), patient.get("duration"))
        result = to_cht_format(patient)
        print("Output:", json.dumps(result, indent=2))
        print()