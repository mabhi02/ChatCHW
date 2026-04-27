"""WHO IMCI canonical danger sign coverage.

The integrated WHO IMCI / community-IMCI (cIMCI) protocol enumerates
specific danger signs that ANY CHW algorithm must cover. Stage 3 may
miss a subset because the source manual phrases them differently or
because the LLM compressed them into broader categories. This validator
gives clinicians a structured list of which canonical signs are
captured (in any of: predicates, variables, phrases, module rules) and
which are missing.

Coverage is checked by token match across:
  - predicate ids and `description_clinical`
  - variable ids and `display_name`
  - phrase ids and text
  - module ids
We accept multiple synonyms per sign (e.g. "lethargic" / "unusually
sleepy" / "drowsy" / "unconscious" all map to the
"unusually_sleepy_or_unconscious" canonical sign).
"""

from __future__ import annotations

import json


# Each entry: (canonical_id, [synonym tokens], severity_if_missing)
# Severity levels (per the IMCI protocol):
#   "error" -- omitting this sign would cause a missed referral / risk death
#   "warn"  -- structural concern: sign is present in IMCI but the system may
#              under-detect it
CANONICAL_DANGER_SIGNS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("convulsions",
     ("convulsion", "seizure", "fitting"), "error"),
    ("unusually_sleepy_or_unconscious",
     ("unusually_sleepy", "unusually sleepy", "unconscious",
      "lethargic", "drowsy", "obtunded"), "error"),
    ("not_able_to_drink_or_feed",
     ("not_able_to_drink", "not_able_to_feed", "unable to drink",
      "unable to feed", "cannot drink", "cannot feed",
      "not able to drink or feed"), "error"),
    ("vomits_everything",
     ("vomits_everything", "vomiting everything", "vomits all", "vomiting all"),
     "error"),
    ("chest_indrawing",
     ("chest_indrawing", "chest indrawing", "lower chest wall indrawing",
      "subcostal retraction", "retractions"), "error"),
    ("stridor",
     ("stridor", "noisy breathing in calm child", "inspiratory stridor"),
     "error"),
    ("severe_pallor",
     ("severe_pallor", "severe pallor", "very pale palms",
      "palmar pallor", "extreme paleness"), "error"),
    ("severe_dehydration_signs",
     ("sunken_eyes", "skin pinch goes back very slowly",
      "skin pinch", "very thirsty", "lethargic and not able to drink"),
     "warn"),
    ("muac_red",
     ("muac_red", "muac red", "muac < 11.5", "muac<11.5"),
     "error"),
    ("oedema_both_feet",
     ("oedema_both_feet", "swelling both feet", "bilateral pitting edema",
      "bilateral oedema"), "error"),
    ("blood_in_stool",
     ("blood_in_stool", "blood in stool", "bloody diarrhoea", "dysentery"),
     "warn"),
    ("fever_7_days_or_more",
     ("fever_7_days_or_more", "fever for 7 days", "fever 7 or more days",
      "prolonged fever"),
     "warn"),
    ("cough_14_days_or_more",
     ("cough_14_days_or_more", "cough 14 days", "chronic cough",
      "cough for 14 or more days"),
     "warn"),
)


def _flat_searchable_text(clinical_logic: dict) -> str:
    """Build one big lowercase haystack out of the artifact dict."""
    parts: list[str] = []

    for p in clinical_logic.get("predicates", []) or []:
        parts.append(str(p.get("id", "")))
        parts.append(str(p.get("description_clinical", "")))
        parts.append(str(p.get("human_label", "")))
        parts.append(str(p.get("formal_definition", "")))

    for v in clinical_logic.get("variables", []) or []:
        if isinstance(v, dict):
            parts.append(str(v.get("id", "")))
            parts.append(str(v.get("display_name", "")))

    phrases = clinical_logic.get("phrase_bank", [])
    if isinstance(phrases, list):
        for ph in phrases:
            if isinstance(ph, dict):
                parts.append(str(ph.get("id", "")))
                parts.append(str(ph.get("text", "")))
                parts.append(str(ph.get("display_name", "")))

    modules = clinical_logic.get("modules", {})
    if isinstance(modules, dict):
        for mid, m in modules.items():
            parts.append(str(mid))
            if isinstance(m, dict):
                parts.append(json.dumps(m.get("rules", []), default=str))

    return "\n".join(parts).lower()


def check(clinical_logic: dict) -> dict:
    """Return per-sign coverage."""
    haystack = _flat_searchable_text(clinical_logic)
    coverage: list[dict] = []
    missing_critical: list[str] = []
    for canonical_id, synonyms, severity in CANONICAL_DANGER_SIGNS:
        hit_synonym = next(
            (s for s in synonyms if s.lower() in haystack),
            None,
        )
        covered = hit_synonym is not None
        coverage.append({
            "canonical_id": canonical_id,
            "covered": covered,
            "severity_if_missing": severity,
            "matched_synonym": hit_synonym,
            "synonyms_tried": list(synonyms),
        })
        if not covered and severity == "error":
            missing_critical.append(canonical_id)

    return {
        "summary": {
            "total_canonical_signs": len(CANONICAL_DANGER_SIGNS),
            "covered": sum(1 for c in coverage if c["covered"]),
            "missing": sum(1 for c in coverage if not c["covered"]),
            "missing_critical": missing_critical,
        },
        "per_sign": coverage,
    }


def to_divergences(report: dict) -> list[dict]:
    """Render coverage gaps as verification divergences."""
    out: list[dict] = []
    for entry in report.get("per_sign", []):
        if entry["covered"]:
            continue
        out.append({
            "type": "missing_imci_danger_sign",
            "severity": entry["severity_if_missing"],
            "detail": (
                f"Canonical IMCI danger sign {entry['canonical_id']!r} is "
                f"not detected in any artifact. Expected one of: "
                f"{entry['synonyms_tried'][:5]}. WHO IMCI requires this sign "
                f"to be screened for every sick child visit."
            ),
            "evidence": {
                "canonical_id": entry["canonical_id"],
                "synonyms_tried": entry["synonyms_tried"],
            },
        })
    return out
