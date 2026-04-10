# DMN Input Schema Note

**Module:** dmnParser  
**File:** dmn_preprocessor.py  
**Last updated:** 2026-04-10

---

## What This Is

This note defines the fields the DMN engine expects as input for each
clinical decision module. All fields are produced by `dmn_preprocessor.py`
from a raw patient record in `output.json`.

---

## Module A — Cough / Respiratory

| Field | Type | Description |
|---|---|---|
| `cough_present` | bool | True if complaint == "Cough" |
| `cough_duration_days` | int | Duration in days; 0 if no cough |
| `fast_breathing_present` | bool | RR above WHO age-adjusted threshold |
| `chest_indrawing_present` | bool | Chest indrawing observed in exam |

## Module B — Diarrhoea

| Field | Type | Description |
|---|---|---|
| `has_diarrhoea` | bool | True if complaint == "Diarrhea" |
| `diarrhoea_duration_days` | int | Duration in days; 0 if no diarrhoea |
| `blood_in_stool` | bool | True if CHW reported blood in stool |

## Module C — Fever / Malaria

| Field | Type | Description |
|---|---|---|
| `hot_with_fever` | bool | True if temperature >= 37.5°C |
| `fever_duration_days` | int | Duration in days; 0 if no fever |
| `malaria_area` | bool | Deployment context flag — default False |
| `rdt_available` | bool | Deployment context flag — default False |
| `rdt_result` | str | "positive" / "negative" / "not_done" |

## Module D — Nutrition

| Field | Type | Description |
|---|---|---|
| `muac_result` | str | "red" / "yellow" / "green" — default "green" |
| `swelling_of_both_feet` | bool | Default False |

## Module E — Cough + Fast Breathing

| Field | Type | Description |
|---|---|---|
| `cough_with_fast_breathing` | bool | True if cough_present AND fast_breathing_present |
| `chest_indrawing` | bool | Same source as chest_indrawing_present — alias for module E |

---

## Notes

- Module E's `chest_indrawing` and Module A's `chest_indrawing_present`
  are the same underlying fact, two key names in the rules. Both are
  populated from `exam_findings["Chest indrawing"]`.
- `malaria_area`, `rdt_available`, `rdt_result`, `muac_result`, and
  `swelling_of_both_feet` are not present in `output.json`. They are
  injected via the optional `context` dict in `to_dmn_input()`.
- See `dmn_mapping_note.md` for the full field-by-field source mapping.