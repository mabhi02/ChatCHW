# DMN Input Schema Note
**File:** dmn_preprocessor.py  
**Author:** Angelina Wang  
**Updated:** 2026-04-10  
**Aligned to:** product team `variables.json`

Defines the fields `dmn_preprocessor.py` produces from a raw patient record in `output.json`.
Field names match the product team's variable naming convention.
For source mappings see `dmn_mapping_note.md`. For assumptions see `dmn_assumptions_note.md`.

---

## Demographics

| Field | Type | Description |
|---|---|---|
| `demo_child_age_months` | int | age * 12 |

## Cough / Respiratory

| Field | Type | Description |
|---|---|---|
| `q_has_cough` | bool | True if complaint == "Cough" |
| `q_cough_duration_days` | int | Duration in days; 0 if no cough |
| `v_respiratory_rate_per_min` | int | Raw RR value; 0 if absent |
| `ex_chest_indrawing` | bool | True if chest indrawing observed |

## Diarrhoea

| Field | Type | Description |
|---|---|---|
| `q_has_diarrhoea` | bool | True if complaint == "Diarrhea" |
| `q_diarrhoea_duration_days` | int | Duration in days; 0 if no diarrhoea |
| `q_blood_in_stool` | bool | True if CHW reported blood in stool |

## Fever / Malaria

| Field | Type | Description |
|---|---|---|
| `q_has_fever` | bool | True if temperature >= 37.5°C |
| `q_fever_duration_days` | int | Duration in days; 0 if no fever |
| `hx_malaria_area` | bool | Context flag — default False |
| `lab_rdt_malaria_result` | str | "positive" / "negative" / "not_done" |

## Nutrition

| Field | Type | Description |
|---|---|---|
| `v_muac_strap_colour` | str | "red" / "yellow" / "green" — default "green" |
| `ex_swelling_both_feet` | bool | Default False |

## Danger Signs

| Field | Type | Description |
|---|---|---|
| `q_convulsions` | bool | Default False — not in dataset |
| `q_vomits_everything` | bool | Default False — not in dataset |
| `q_not_able_to_drink_or_feed` | bool | From chw_questions["Unable to drink"] |
| `ex_unusually_sleepy_or_unconscious` | bool | From chw_questions["Lethargic"] |
| `is_priority_exit` | bool | Default False — set by engine on danger sign |

## Router Flags

| Field | Type | Description |
|---|---|---|
| `mod_assess_done` | bool | Default False — updated by engine |
| `mod_danger_sign_check_done` | bool | Default False — updated by engine |
| `mod_diarrhoea_treatment_done` | bool | Default False — updated by engine |
| `mod_fast_breathing_treatment_done` | bool | Default False — updated by engine |
| `mod_fever_malaria_treatment_done` | bool | Default False — updated by engine |
| `mod_malnutrition_screening_done` | bool | Default False — updated by engine |