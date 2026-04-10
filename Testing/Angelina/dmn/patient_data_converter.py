"""
Parse raw simulated patient rows and convert continuous values
into the boolean variables the DMN decision tables expect.

Input format (from simulated_chw_patients.xlsx):
  age_years     : int  (years — NOTE: DMN uses age_months)
  complaint     : "Cough" / "Diarrhea" / "Sore throat"
  duration      : "5 days" / "3 weeks" / "4 weeks"
  chw_questions : free-text e.g. "Cough: Yes; Difficulty breathing: Yes; Lethargic: Yes"
  exam_findings : free-text e.g. "Respiratory rate: 52/min; Temperature: 38.6°C; Chest indrawing: Yes"

Continuous → boolean conversions
---------------------------------
  age_years × 12            → age_months
  temperature_c >= 38.0     → hot_with_fever
      ^ NEED MEDICAL TEAM TO DECIDE EXACT CUTOFF
  RR (respiratory rate) > 50  (age ≤ 12 mo)   → fast_breathing_present
  RR > 40  (age > 12 mo)   → fast_breathing_present
      ^ NEED MEDICAL TEAM
  duration → days           → cough/diarrhoea/fever_duration_days
      (DMN FEEL does the >= 14 chronic check itself)

Missing data (in DMN but not found in patient data) 
-----------------
  muac_result          — not in dataset (no MUAC measurement)
  swelling_of_both_feet — not in dataset
  malaria_area          — not in dataset (geographic context)
  rdt_available/result  — not in dataset
  convulsions           — not captured in narrative

Usage
-----
    import pandas as pd
    from patient_to_facts import parse_row, parse_dataframe, explain_row

    df = pd.read_excel("simulated_chw_patients.xlsx", sheet_name="simulated_chw_patients.csv", header=None)
    df.columns = ["age_group","grader_row","age_years","sex","complaint",
                  "duration","chw_questions","exam_findings","distractor","rubric_ref","diagnosis"]
    df = df.iloc[1:].reset_index(drop=True)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    facts_list = parse_dataframe(df)          # list of dicts, one per row
    facts      = parse_row(df.iloc[0].to_dict())  # single row
"""

from __future__ import annotations
import re
from typing import Optional

"""
Once given cutoffs from medical team, we can update the boundaries here.
"""

FEVER_TEMP_THRESHOLD_C: float = 38.0
FAST_BREATHING_YOUNG_MONTHS: int = 12
FAST_BREATHING_YOUNG_RR: int = 50
FAST_BREATHING_OLDER_RR: int = 40
CHILD_AGE_YEARS: int = 18
CHRONIC_DURATION_DAYS: int = 14

DAYS_PER_WEEK: int = 7


# Parsing helpers

def _parse_kv(text: str) -> dict[str, str]:
    """
    Parse semicolon-separated 'Key: Value' narrative string.
    'Cough: Yes; Difficulty breathing: Yes' → {'Cough': 'Yes', 'Difficulty breathing': 'Yes'}
    """
    result = {}
    for part in str(text).split(";"):
        part = part.strip()
        if ":" in part:
            key, _, val = part.partition(":")
            result[key.strip()] = val.strip()
    return result


def _yn(val: Optional[str]) -> Optional[bool]:
    """'Yes' → True, 'No' → False, missing → None."""
    if val is None:
        return None
    v = str(val).strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _parse_duration_days(s: str) -> Optional[int]:
    """'5 days' → 5,  '3 weeks' → 21,  unrecognised → None."""
    m = re.match(r"(\d+)\s+(day|week)s?", str(s).strip().lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    return n if unit == "day" else n * DAYS_PER_WEEK


def _extract_temperature(exam_kv: dict) -> Optional[float]:
    s = exam_kv.get("Temperature", "")
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def _extract_respiratory_rate(exam_kv: dict) -> Optional[int]:
    s = exam_kv.get("Respiratory rate", "")
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None



# Main conversion logic

def _fast_breathing(age_months: float, rr: Optional[int]) -> Optional[bool]:
    """
    Apply age-banded fast-breathing threshold.
    Mirrors XLSForm fast_breathing_calc formula exactly.
    Returns None when RR is absent.
    """
    if rr is None:
        return None
    if age_months <= FAST_BREATHING_YOUNG_MONTHS:
        return rr > FAST_BREATHING_YOUNG_RR      # PLACEHOLDER threshold
    else:
        return rr > FAST_BREATHING_OLDER_RR       # PLACEHOLDER threshold


def parse_row(row: dict) -> dict:
    """
    Convert one raw patient row to a DMN facts dict.

    Continuous values converted:
      age_years × 12          → age_months
      temperature_c ≥ 37.5    → hot_with_fever        [PLACEHOLDER]
      RR vs age band          → fast_breathing_present [PLACEHOLDER]
      duration string         → *_duration_days (int, DMN does the >= 14 check itself)

    Text/narrative parsed:
      chw_questions keys: Cough, Difficulty breathing, Blood in stool,
                          Lethargic, Lethargic/unconscious, Unable to drink
      exam_findings keys: Respiratory rate, Temperature, Chest indrawing

    Returns only non-None values (DMN wildcards cover gaps).
    """
    complaint = str(row.get("complaint", "")).strip()
    age_years  = float(row.get("age_years") or 0)
    age_months = age_years * 12

    chw  = _parse_kv(row.get("chw_questions", "") or "")
    exam = _parse_kv(row.get("exam_findings",  "") or "")

    duration_days   = _parse_duration_days(row.get("duration", "") or "")
    temperature_c   = _extract_temperature(exam)
    respiratory_rate = _extract_respiratory_rate(exam)

    hot_with_fever = (
        temperature_c >= FEVER_TEMP_THRESHOLD_C    # *** PLACEHOLDER
        if temperature_c is not None else None
    )

    is_cough_case       = complaint == "Cough"
    is_diarrhoea_case   = complaint == "Diarrhea"

    # Fast breathing only relevant for cough/respiratory presentations
    fast_breathing = _fast_breathing(age_months, respiratory_rate) if is_cough_case else None

    cough_present = is_cough_case or (_yn(chw.get("Cough")) is True)

    # Chest indrawing: only appears in exam_findings when present;
    # absence in a cough case means False, absent for non-cough means unknown.
    chest_indrawing_raw = _yn(exam.get("Chest indrawing"))
    chest_indrawing_val = (
        chest_indrawing_raw if chest_indrawing_raw is not None
        else (False if is_cough_case else None)
    )

    cough_with_fast_breathing = (
        bool(cough_present and fast_breathing)
        if (is_cough_case and fast_breathing is not None)
        else None
    )

    blood_in_stool = _yn(chw.get("Blood in stool")) if is_diarrhoea_case else None

    lethargic    = _yn(chw.get("Lethargic")) or _yn(chw.get("Lethargic/unconscious"))
    unable_drink = _yn(chw.get("Unable to drink"))

    # ---- fever duration: only meaningful when fever is present ----
    fever_duration_days = duration_days if hot_with_fever else None

    raw: dict = {
        # Demographics
        "age_months":               age_months,

        # Module A — Cough / Respiratory
        "cough_present":            cough_present if is_cough_case else False,
        "cough_duration_days":      duration_days if is_cough_case else None,
        "fast_breathing_present":   fast_breathing,
        "chest_indrawing_present":  chest_indrawing_val,   # Module A name

        # Module B — Diarrhoea
        "has_diarrhoea":            is_diarrhoea_case,
        "diarrhoea_duration_days":  duration_days if is_diarrhoea_case else None,
        "blood_in_stool":           blood_in_stool,

        # Module C — Fever
        "hot_with_fever":           hot_with_fever,
        "fever_duration_days":      fever_duration_days,
        # malaria_area / rdt_available / rdt_result — not in dataset

        # Module E — Pneumonia (overlapping names with Module A)
        "cough_with_fast_breathing": cough_with_fast_breathing,
        "chest_indrawing":           chest_indrawing_val,  # Module E name

        # Danger signs (partial — dataset doesn't capture all)
        "unusually_sleepy_or_unconscious":    lethargic,
        "not_able_to_drink_or_feed_anything": unable_drink,
    }

    return {k: v for k, v in raw.items() if v is not None}


def parse_dataframe(df) -> list[dict]:
    """Convert a full DataFrame to a list of DMN fact dicts."""
    return [parse_row(row.to_dict()) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def explain_row(row: dict) -> str:
    """
    Return a human-readable derivation summary for one row.
    Shows which PLACEHOLDER thresholds were applied and what they produced.
    """
    age_years  = float(row.get("age_years") or 0)
    age_months = age_years * 12
    complaint  = str(row.get("complaint", ""))
    duration   = str(row.get("duration", ""))
    chw        = _parse_kv(row.get("chw_questions", "") or "")
    exam       = _parse_kv(row.get("exam_findings",  "") or "")
    duration_days = _parse_duration_days(duration)

    temp = _extract_temperature(exam)
    rr   = _extract_respiratory_rate(exam)

    lines = [
        f"Patient  : {age_years}y ({age_months:.0f} mo)  |  {complaint}  |  {duration}  →  {duration_days} days",
        f"Raw data : temp={temp}°C  RR={rr} bpm  chest_indrawing={'Yes' if exam.get('Chest indrawing') else 'No'}",
        "Derived  :",
    ]

    if temp is not None:
        result = temp >= FEVER_TEMP_THRESHOLD_C
        lines.append(
            f"  hot_with_fever           = {result}"
            f"  ← {temp}°C {'≥' if result else '<'} {FEVER_TEMP_THRESHOLD_C}°C (WHO fever threshold)"
        )

    if rr is not None and complaint == "Cough":
        if age_months <= FAST_BREATHING_YOUNG_MONTHS:
            threshold, band = FAST_BREATHING_YOUNG_RR, f"≤{FAST_BREATHING_YOUNG_MONTHS} mo"
        else:
            threshold, band = FAST_BREATHING_OLDER_RR, f">{FAST_BREATHING_YOUNG_MONTHS} mo"
        result = rr > threshold
        lines.append(
            f"  fast_breathing_present   = {result}"
            f"  ← {rr} bpm {'>' if result else '≤'} {threshold} bpm WHO IMCI ({band})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import pandas as pd

    df = pd.read_excel("simulated_chw_patients.xlsx", sheet_name="simulated_chw_patients.csv", header=None)
    df.columns = ["age_group","grader_row","age_years","sex","complaint",
                  "duration","chw_questions","exam_findings",
                  "distractor","rubric_ref","diagnosis"]
    df = df.iloc[1:].reset_index(drop=True)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= 5:
            break
        print(explain_row(row.to_dict()))
        facts = parse_row(row.to_dict())
        print("DMN facts:", json.dumps(facts, indent=2))
        print()
        