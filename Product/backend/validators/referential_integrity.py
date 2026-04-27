"""Referential-integrity audit + auto-reconciliation for clinical_logic.

Runs AFTER the Opus REPL emits the full clinical_logic dict. Checks that
every id referenced anywhere (in modules, router, predicates, integrative,
phrase_bank) is declared in the corresponding registry. Missing ids are
either auto-registered with an `_auto_registered: true` flag (so the DMN
stays runnable) or reported as critical issues.

This catches the Phase-A / Phase-B registry drift observed on the
2026-04-14 run, where 77% of referenced variables and 90% of referenced
predicates were absent from the registries.

All deterministic Python. No LLM calls.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Variable-prefix convention. Must stay in sync with the naming codebook.
_VARIABLE_PREFIXES = (
    "q_", "ex_", "v_", "lab_", "img_", "hx_", "demo_", "sys_", "prev_",
)
_PREDICATE_PREFIXES = ("p_",)
_MODULE_STATE_PREFIXES = ("mod_",)  # e.g. mod_fever_done, is_priority_exit
_PHRASE_PREFIXES = ("m_", "adv_", "tx_", "rx_", "ref_", "fu_")
_SUPPLY_PREFIXES = ("supply_", "equip_")


def _as_list(obj: Any) -> list[dict]:
    """Normalize list-or-dict-of-dicts to list of dicts."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            if isinstance(v, dict):
                out.append({**v, "_normalized_id": k})
        return out
    return []


def _ids_from(items: list[dict], id_keys: tuple[str, ...] = ("id", "module_id", "predicate_id")) -> set[str]:
    """Collect the set of declared ids in a registry."""
    out: set[str] = set()
    for item in items:
        for key in id_keys:
            v = item.get(key)
            if v:
                out.add(str(v))
                break
        else:
            v = item.get("_normalized_id")
            if v:
                out.add(str(v))
    return out


def _extract_tokens(text: str) -> set[str]:
    """Pull prefixed tokens out of a condition string, action string, etc.

    A token is a lowercase identifier that starts with one of the known
    prefixes (q_, ex_, p_, mod_, supply_, etc.).
    """
    if not text:
        return set()
    all_prefixes = (
        _VARIABLE_PREFIXES + _PREDICATE_PREFIXES + _MODULE_STATE_PREFIXES
        + _PHRASE_PREFIXES + _SUPPLY_PREFIXES
    )
    pat = r"\b(?:" + "|".join(re.escape(p) for p in all_prefixes) + r")[a-z0-9_]+\b"
    return set(re.findall(pat, text))


def _classify_missing(missing_id: str) -> str:
    """Which registry should a missing id be added to?"""
    if missing_id.startswith(_PREDICATE_PREFIXES):
        return "predicates"
    if missing_id.startswith(_SUPPLY_PREFIXES):
        return "supply_list"
    if missing_id.startswith(_PHRASE_PREFIXES):
        return "phrase_bank"
    # Default to variables (covers q_, ex_, v_, lab_, hx_, demo_, sys_, prev_,
    # and any mod_*_done / is_priority_exit style flags).
    return "variables"


def audit(clinical_logic: dict) -> dict:
    """Walk clinical_logic and report every missing reference.

    Returns a dict with keys:
      - passed: bool
      - stats: per-category total / declared / referenced / missing counts
      - missing_by_registry: {registry_name: [missing_ids]}
      - references: which consumers referenced each missing id
    """
    if not isinstance(clinical_logic, dict):
        return {"passed": False, "error": "clinical_logic is not a dict"}

    variables   = _as_list(clinical_logic.get("variables", []))
    predicates  = _as_list(clinical_logic.get("predicates", []))
    supply_list = _as_list(clinical_logic.get("supply_list", []))
    phrase_bank = _as_list(clinical_logic.get("phrase_bank", []))
    modules_raw = clinical_logic.get("modules", {})
    modules     = modules_raw if isinstance(modules_raw, dict) else {m.get("module_id", f"_{i}"): m for i, m in enumerate(_as_list(modules_raw))}
    router      = clinical_logic.get("router", {}) or {}
    integrative = clinical_logic.get("integrative", {}) or {}

    declared_vars     = _ids_from(variables)
    declared_preds    = _ids_from(predicates)
    declared_supplies = _ids_from(supply_list)
    declared_phrases  = _ids_from(phrase_bank)
    declared_modules  = set(modules.keys())

    # Module state flags (mod_X_done, is_priority_exit) count as declared
    # variables by convention; modules emit them as outputs.
    implicit_vars = {f"{mid}_done" for mid in declared_modules}
    implicit_vars.add("is_priority_exit")
    implicit_vars.add("sys_encounter_start")
    declared_vars_plus_implicit = declared_vars | implicit_vars

    # -- Collect every reference and who referenced it --
    # refs[id] = list of human-readable citations
    refs: dict[str, list[str]] = {}

    def _note(token: str, where: str) -> None:
        refs.setdefault(token, []).append(where)

    # Module inputs / outputs / rules
    for mid, m in modules.items():
        for inp in (m.get("inputs") or []):
            _note(inp, f"modules.{mid}.inputs")
        for out in (m.get("outputs") or []):
            _note(out, f"modules.{mid}.outputs")
        for ri, rule in enumerate(m.get("rules") or []):
            if isinstance(rule, dict):
                cond_tokens = _extract_tokens(rule.get("condition", "") or "")
                action_tokens = _extract_tokens(rule.get("action", "") or "")
                for t in cond_tokens:
                    _note(t, f"modules.{mid}.rules[{ri}].condition")
                for t in action_tokens:
                    _note(t, f"modules.{mid}.rules[{ri}].action")
                for out_k, out_v in (rule.get("outputs") or {}).items():
                    _note(out_k, f"modules.{mid}.rules[{ri}].outputs")

    # Router rows
    for ri, row in enumerate(router.get("rows") or router.get("rules") or []):
        if isinstance(row, dict):
            for t in _extract_tokens(row.get("condition", "") or ""):
                _note(t, f"router.rows[{ri}].condition")
            om = row.get("output_module") or row.get("next_module") or row.get("module_id")
            if om and om != "PRIORITY_EXIT":
                _note(om, f"router.rows[{ri}].output_module")

    # Predicate source_vars + threshold_expression
    for p in predicates:
        pid = p.get("id") or p.get("predicate_id") or "_"
        for sv in (p.get("source_vars") or []):
            _note(sv, f"predicates.{pid}.source_vars")
        for t in _extract_tokens(p.get("threshold_expression", "") or ""):
            _note(t, f"predicates.{pid}.threshold_expression")
        # Compound-predicate composition (if present)
        for comp in (p.get("composition", {}).get("terms", []) if isinstance(p.get("composition"), dict) else []):
            if isinstance(comp, str):
                _note(comp, f"predicates.{pid}.composition")

    # Integrative rules
    for ri, row in enumerate(integrative.get("rules") or []):
        if isinstance(row, dict):
            for t in _extract_tokens(row.get("condition", "") or ""):
                _note(t, f"integrative.rules[{ri}].condition")
            for t in _extract_tokens(row.get("action", "") or ""):
                _note(t, f"integrative.rules[{ri}].action")

    # -- Classify references --
    referenced_vars:     set[str] = set()
    referenced_preds:    set[str] = set()
    referenced_modules:  set[str] = set()
    referenced_phrases:  set[str] = set()
    referenced_supplies: set[str] = set()
    for token in refs.keys():
        if token.startswith(_PREDICATE_PREFIXES):
            referenced_preds.add(token)
        elif token.startswith(_MODULE_STATE_PREFIXES):
            # Module tokens can be module ids OR mod_X_done flags
            if token.endswith("_done"):
                referenced_vars.add(token)  # implicit var
            else:
                referenced_modules.add(token)
        elif token.startswith(_SUPPLY_PREFIXES):
            referenced_supplies.add(token)
        elif token.startswith(_PHRASE_PREFIXES):
            referenced_phrases.add(token)
        elif token.startswith(_VARIABLE_PREFIXES):
            referenced_vars.add(token)
        else:
            # Unknown prefix: flag as variable by default
            referenced_vars.add(token)

    # -- Missing sets --
    missing_vars     = referenced_vars     - declared_vars_plus_implicit - {"true", "false", "null"}
    missing_preds    = referenced_preds    - declared_preds
    missing_modules  = referenced_modules  - declared_modules
    missing_phrases  = referenced_phrases  - declared_phrases
    missing_supplies = referenced_supplies - declared_supplies

    # -- Unused entries (declared but never referenced) --
    unused_phrases  = declared_phrases  - referenced_phrases
    unused_supplies = declared_supplies - referenced_supplies
    unused_preds    = declared_preds    - referenced_preds
    unused_vars     = declared_vars     - referenced_vars

    stats = {
        "variables": {
            "declared": len(declared_vars),
            "referenced": len(referenced_vars),
            "missing": len(missing_vars),
            "unused": len(unused_vars),
        },
        "predicates": {
            "declared": len(declared_preds),
            "referenced": len(referenced_preds),
            "missing": len(missing_preds),
            "unused": len(unused_preds),
        },
        "modules": {
            "declared": len(declared_modules),
            "referenced": len(referenced_modules),
            "missing": len(missing_modules),
        },
        "phrase_bank": {
            "declared": len(declared_phrases),
            "referenced": len(referenced_phrases),
            "missing": len(missing_phrases),
            "unused": len(unused_phrases),
        },
        "supply_list": {
            "declared": len(declared_supplies),
            "referenced": len(referenced_supplies),
            "missing": len(missing_supplies),
            "unused": len(unused_supplies),
        },
    }

    missing_by_registry = {
        "variables":   sorted(missing_vars),
        "predicates":  sorted(missing_preds),
        "modules":     sorted(missing_modules),
        "phrase_bank": sorted(missing_phrases),
        "supply_list": sorted(missing_supplies),
    }
    unused_by_registry = {
        "phrase_bank": sorted(unused_phrases),
        "supply_list": sorted(unused_supplies),
        "predicates":  sorted(unused_preds),
        "variables":   sorted(unused_vars),
    }

    references = {k: sorted(set(v)) for k, v in refs.items()}

    passed = (
        not missing_vars
        and not missing_preds
        and not missing_modules
        and not missing_phrases
        and not missing_supplies
    )

    return {
        "passed": passed,
        "stats": stats,
        "missing_by_registry": missing_by_registry,
        "unused_by_registry": unused_by_registry,
        "references": references,
    }


def reconcile(clinical_logic: dict, audit_report: dict) -> tuple[dict, dict]:
    """Auto-register every missing id so the DMN stays runnable.

    Each auto-registered entry carries `_auto_registered: true` so a
    reviewer can tell it came from static analysis, not from label
    extraction. No clinical content is invented: we only add a skeleton
    entry with the id + prefix-derived metadata.

    Returns (patched_clinical_logic, reconciliation_report).
    """
    patched = {k: v for k, v in clinical_logic.items()}
    reconciliation = {"auto_registered": {r: [] for r in ("variables", "predicates", "supply_list", "phrase_bank", "modules")}}

    missing = audit_report.get("missing_by_registry", {})

    # -- variables --
    if missing.get("variables"):
        current = patched.get("variables") or []
        if isinstance(current, dict):
            current_list = [{**v, "id": k} for k, v in current.items()]
            to_dict = True
        else:
            current_list = list(current)
            to_dict = False

        for var_id in missing["variables"]:
            prefix = next((p for p in _VARIABLE_PREFIXES if var_id.startswith(p)), None)
            # mod_X_done flags get special treatment
            if var_id.endswith("_done") and var_id.startswith("mod_"):
                kind = "module_state"
                data_type = "boolean"
            elif var_id == "is_priority_exit":
                kind = "module_state"
                data_type = "boolean"
            elif prefix in ("v_", "lab_", "img_"):
                kind = "measurement"
                data_type = "number"
            elif prefix in ("q_", "ex_"):
                kind = "observation"
                data_type = "boolean"
            elif prefix == "demo_":
                kind = "demographics"
                data_type = "string" if "id" in var_id or "name" in var_id or "village" in var_id else "number"
            else:
                kind = "unknown"
                data_type = "string"

            entry = {
                "id": var_id,
                "prefix": prefix or "",
                "display_name": var_id.replace("_", " ").capitalize(),
                "data_type": data_type,
                "kind": kind,
                "source_quote": "(auto-registered from referential-integrity pass; manual review required)",
                "source_section_id": "",
                "_auto_registered": True,
            }
            current_list.append(entry)
            reconciliation["auto_registered"]["variables"].append(var_id)

        patched["variables"] = (
            {e["id"]: {k: v for k, v in e.items() if k != "id"} for e in current_list}
            if to_dict else current_list
        )

    # -- predicates --
    if missing.get("predicates"):
        current = patched.get("predicates") or []
        if isinstance(current, dict):
            current_list = [{**v, "id": k} for k, v in current.items()]
            to_dict = True
        else:
            current_list = list(current)
            to_dict = False

        for pid in missing["predicates"]:
            entry = {
                "id": pid,
                "predicate_id": pid,
                "human_label": pid.replace("p_", "", 1).replace("_", " ").capitalize(),
                "source_vars": [],
                "threshold_expression": "",
                "fail_safe": 1 if "danger" in pid or "severe" in pid or "refer" in pid else 0,
                "source_quote": "(auto-registered from referential-integrity pass; manual review required)",
                "source_section_id": "",
                "_auto_registered": True,
            }
            current_list.append(entry)
            reconciliation["auto_registered"]["predicates"].append(pid)

        patched["predicates"] = (
            {e["id"]: {k: v for k, v in e.items() if k != "id"} for e in current_list}
            if to_dict else current_list
        )

    # -- supply_list --
    if missing.get("supply_list"):
        current_list = list(patched.get("supply_list") or [])
        for sid in missing["supply_list"]:
            kind = "equipment" if sid.startswith("equip_") else "consumable"
            entry = {
                "id": sid,
                "kind": kind,
                "display_name": sid.replace("supply_", "").replace("equip_", "").replace("_", " ").capitalize(),
                "used_by": [],
                "source_quote": "(auto-registered from referential-integrity pass; manual review required)",
                "source_section_id": "",
                "_auto_registered": True,
            }
            current_list.append(entry)
            reconciliation["auto_registered"]["supply_list"].append(sid)
        patched["supply_list"] = current_list

    # -- phrase_bank --
    if missing.get("phrase_bank"):
        current_list = list(patched.get("phrase_bank") or [])
        for pbid in missing["phrase_bank"]:
            entry = {
                "id": pbid,
                "phrase_id": pbid,
                "category": "auto",
                "text": f"({pbid}: auto-registered; manual review required)",
                "module_context": "",
                "source_quote": "",
                "source_section_id": "",
                "_auto_registered": True,
            }
            current_list.append(entry)
            reconciliation["auto_registered"]["phrase_bank"].append(pbid)
        patched["phrase_bank"] = current_list

    # -- modules --
    # Missing modules in router.output_module means the router points at a
    # module that wasn't emitted. That is usually a clinical content bug,
    # not something we can auto-fill. Flag but do not auto-register.
    reconciliation["missing_modules_unhandled"] = sorted(missing.get("modules", []))

    return patched, reconciliation


def audit_and_reconcile(clinical_logic: dict) -> dict:
    """Convenience wrapper: audit + (if gaps) reconcile.

    Returns a dict suitable to persist as `referential_integrity.json`:
      - audit_before: the pre-reconciliation audit report
      - reconciliation: what was auto-added
      - patched: the repaired clinical_logic
      - audit_after: the post-reconciliation audit (should have missing=0
        for the four fillable registries)
    """
    before = audit(clinical_logic)
    patched, reconciliation = reconcile(clinical_logic, before)
    after = audit(patched)
    return {
        "audit_before": before,
        "reconciliation": reconciliation,
        "patched": patched,
        "audit_after": after,
    }
