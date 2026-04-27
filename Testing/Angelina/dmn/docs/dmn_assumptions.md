# DMN Preprocessor — Assumptions & Additional Fields
**Author:** Angelina Wang  
**Last Updated:** 2026-04-10

---

## Ambiguous Mappings

| Field | Assumption | Action Needed |
|---|---|---|
| `q_has_fever` | Derived from temperature >= 37.5°C. Product team treats this as a CHW-reported boolean, not a measurement. | Medical team to confirm cutoff or clarify source |
| `ex_unusually_sleepy_or_unconscious` | Mapped from `chw_questions["Lethargic"]` or `chw_questions["Lethargic/unconscious"]` — whichever is present | Medical team to confirm these are equivalent |
| `v_respiratory_rate_per_min` | Output as raw integer. Product team computes fast-breathing threshold internally via `predicates.json` | No action needed unless adult threshold is required |

---

## Fields Defaulted to False / Not Done

These fields are not in `output.json` and default to safe values. Inject via `context` dict to test specific scenarios.

| Field | Default | Reason absent |
|---|---|---|
| `hx_malaria_area` | `False` | Geographic context, not in dataset |
| `lab_rdt_malaria_result` | `"not_done"` | Point-of-care result, not in dataset |
| `v_muac_strap_colour` | `"green"` | Not measured in dataset |
| `ex_swelling_both_feet` | `False` | Not measured in dataset |
| `q_convulsions` | `False` | Not captured in dataset |
| `q_vomits_everything` | `False` | Not captured in dataset |

---

## Additional Product Team Fields Beyond Emett's Sample Schema

Fields in `variables.json` that are not in Emett's sample schema but are included in the preprocessor:

| Field | Why Added |
|---|---|
| `q_convulsions` | Danger sign — part of `p_any_danger_sign` predicate |
| `q_vomits_everything` | Danger sign — part of `p_any_danger_sign` predicate |
| `q_not_able_to_drink_or_feed` | Danger sign — part of `p_any_danger_sign` predicate |
| `ex_unusually_sleepy_or_unconscious` | Danger sign — part of `p_any_danger_sign` predicate |
| `v_respiratory_rate_per_min` | Required by `p_fast_breathing_threshold` predicate |
| `is_priority_exit` | Router short-circuit flag — set by engine on danger sign |
| `mod_*_done` flags | Required by `router.json` to track module completion |

Fields in `variables.json` that are **not** in the preprocessor because they have no source in `output.json` and are not derivable:

| Field | Reason Excluded |
|---|---|
| `demo_caregiver_name` | Not in dataset |
| `demo_child_name` | Not in dataset |
| `demo_child_sex` | Not in dataset |
| `q_difficulty_drinking_or_feeding` | Not in dataset — distinct from `q_not_able_to_drink_or_feed` |
| `q_vomiting` | Not in dataset — distinct from `q_vomits_everything` |
| `sys_encounter_start` | System flag set by engine, not by preprocessor |

---

## Scope Note

Product team predicates are scoped to children under 5 years (`demo_child_age_months < 60`). Our dataset includes adults and elders. The preprocessor populates all fields regardless of age — behavior for out-of-scope patients depends on how the engine handles unmatched predicates.