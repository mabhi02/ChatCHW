"""Convert clinical_logic JSON to flat CSVs for human review.

Renders predicates and phrase_bank arrays as CSV files.

Gen 7 schema: predicates use 'id' not 'predicate_id', phrase_bank uses
'id' not 'message_id', 'text' not 'english_text', 'source_section_id'
not 'page_ref', and 'placeholder_vars' is removed.
Backwards-compatible: falls back to old keys when new ones absent.
"""

import csv
import io


def _iter_entries(logic, key: str):
    """Yield dict entries for a top-level list-typed field.

    Handles three shapes the model sometimes produces:
      1. List of dicts (expected) -- yielded as-is
      2. Dict keyed by id (legacy) -- yielded as values with the id merged in
      3. Anything else -- yielded as an empty iterable (skip)

    Also skips any non-dict entries inside the list so one bad row doesn't
    crash the whole CSV.
    """
    if not isinstance(logic, dict):
        return
    value = logic.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
    elif isinstance(value, dict):
        # Legacy dict-keyed-by-id shape; yield the entries
        for item_id, item in value.items():
            if isinstance(item, dict):
                merged = dict(item)
                # Inject the id under a common field name if missing
                # Gen 7 uses "id"; also set legacy keys for compat
                if "id" not in merged:
                    merged["id"] = item_id
                if key == "predicates" and "predicate_id" not in merged:
                    merged["predicate_id"] = item_id
                elif key == "phrase_bank" and "message_id" not in merged:
                    merged["message_id"] = item_id
                yield merged


def convert_predicates_to_csv(logic) -> str:
    """Convert predicates array to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "predicate_id",
        "human_label",
        "source_vars",
        "threshold_expression",
        "fail_safe",
        "source_section_id",
    ])

    for pred in _iter_entries(logic, "predicates"):
        source_vars = pred.get("source_vars", [])
        if not isinstance(source_vars, list):
            source_vars = []
        # Gen 7 uses "id"; fall back to old "predicate_id"
        pid = pred.get("id") or pred.get("predicate_id", "")
        # Gen 7 uses "source_section_id"; fall back to old "page_ref"
        section = pred.get("source_section_id") or pred.get("page_ref", "")
        writer.writerow([
            pid,
            pred.get("human_label", ""),
            "; ".join(str(v) for v in source_vars),
            pred.get("threshold_expression", ""),
            pred.get("fail_safe", ""),
            section,
        ])

    return output.getvalue()


def convert_phrases_to_csv(logic) -> str:
    """Convert phrase_bank array to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "message_id",
        "category",
        "text",
        "module_context",
        "source_section_id",
    ])

    for phrase in _iter_entries(logic, "phrase_bank"):
        # Gen 7 uses "id"; fall back to old "message_id"
        mid = phrase.get("id") or phrase.get("message_id", "")
        # Gen 7 uses "text"; fall back to old "english_text"
        text = phrase.get("text") or phrase.get("english_text", "")
        # Gen 7 uses "module_context"; placeholder_vars removed
        module_ctx = phrase.get("module_context", "")
        # Gen 7 uses "source_section_id"; fall back to old "page_ref"
        section = phrase.get("source_section_id") or phrase.get("page_ref", "")
        writer.writerow([
            mid,
            phrase.get("category", ""),
            text,
            module_ctx,
            section,
        ])

    return output.getvalue()


def convert_to_csv(logic) -> dict[str, str]:
    """Convert clinical_logic to both CSV files.

    Returns dict with keys "predicates" and "phrases", values are CSV strings.
    Tolerates non-dict logic input by returning empty CSVs (header only).
    """
    return {
        "predicates": convert_predicates_to_csv(logic),
        "phrases": convert_phrases_to_csv(logic),
    }
