# DMN Field Mapping Note
**Project:** Patient-to-DMN Preprocessor  
**Author:** Angelina Wang  
**Updated:** 2026-04-10  
**Aligned to:** product team `variables.json` naming convention

Maps every field in `DMN_INPUT_SCHEMA` to its source in `output.json`
and whether it is raw or derived.

---

## Mapping Table

| DMN Field (product team name) | Source in output.json | Raw / Derived | Notes |
|---|---|---|---|
| `demo_child_age_months` | `age` | derived | `age * 12` |
| `q_has_cough` | `complaint` | derived | `complaint == "Cough"` |
| `q_cough_duration_days` | `duration.value` + `duration.unit` | derived | converted to int days; 0 if no cough |
| `v_respiratory_rate_per_min` | `exam_findings["Respiratory rate"]` | derived | parsed int from `"61/min"`; 0 if absent |
| `ex_chest_indrawing` | `exam_findings["Chest indrawing"]` | derived | `True` if value is `"Yes"` |
| `q_has_diarrhoea` | `complaint` | derived | `complaint == "Diarrhea"` |
| `q_diarrhoea_duration_days` | `duration.value` + `duration.unit` | derived | converted to int days; 0 if no diarrhoea |
| `q_blood_in_stool` | `chw_questions["Blood in stool"]` | derived | `True` if `"Yes"`; `False` if key is `"No blood in stool"` |
| `q_has_fever` | `exam_findings["Temperature"]` | derived | parsed float; `True` if `>= 37.5°C` |
| `q_fever_duration_days` | `duration.value` + `duration.unit` | derived | converted to int days; 0 if no fever |
| `hx_malaria_area` | — not in output.json — | context default | `False` unless passed via `context` dict |
| `lab_rdt_malaria_result` | — not in output.json — | context default | `"not_done"` unless passed via `context` dict |
| `v_muac_strap_colour` | — not in output.json — | context default | `"green"` unless passed via `context` dict |
| `ex_swelling_both_feet` | — not in output.json — | context default | `False` unless passed via `context` dict |
| `q_convulsions` | — not in output.json — | context default | `False` — not captured in dataset |
| `q_vomits_everything` | — not in output.json — | context default | `False` — not captured in dataset |
| `q_not_able_to_drink_or_feed` | `chw_questions["Unable to drink"]` | derived | `True` if value is `"Yes"` |
| `ex_unusually_sleepy_or_unconscious` | `chw_questions["Lethargic"]` or `chw_questions["Lethargic/unconscious"]` | derived | `True` if either key is `"Yes"` |
| `is_priority_exit` | — computed by engine — | default `False` | engine sets to `True` if any danger sign found |
| `mod_assess_done` | — system flag — | default `False` | engine updates after module runs |
| `mod_danger_sign_check_done` | — system flag — | default `False` | engine updates after module runs |
| `mod_diarrhoea_treatment_done` | — system flag — | default `False` | engine updates after module runs |
| `mod_fast_breathing_treatment_done` | — system flag — | default `False` | engine updates after module runs |
| `mod_fever_malaria_treatment_done` | — system flag — | default `False` | engine updates after module runs |
| `mod_malnutrition_screening_done` | — system flag — | default `False` | engine updates after module runs |

---

## Fast-Breathing Threshold

The preprocessor outputs `v_respiratory_rate_per_min` as a raw integer.
The product team computes fast-breathing themselves via `predicates.json`:

| Age band | Threshold |
|---|---|
| 2–11 months | >= 50 breaths/min |
| 12–59 months | >= 40 breaths/min |

Note: product team predicates only cover children up to 5 years (`demo_child_age_months < 60`).
Our dataset includes adults — fast-breathing threshold for adults is not defined in `predicates.json`.

---

## Context Fields

Six fields have no counterpart in `output.json` and are injected via the optional
`context` dict in `to_dmn_input()`:

```python
to_dmn_input(patient, context={
    "malaria_area": True,
    "rdt_result": "positive",
    "muac_result": "yellow",
    "swelling_of_both_feet": False
})
```