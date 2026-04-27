"""Per-variable scope + lifecycle derivation.

Builds `data_flow.json` from an already-emitted clinical_logic dict:
 - For each variable: which modules collect it vs use it, which predicates
   derive from it, its collection scope (universal / module-scoped /
   conditional), and its dependency on supply_list items.
 - For each predicate: which variables feed it, which modules consume it.

This is the artifact David asked for: "show provenance and use of
variables.json: what CHW collects or enters vs. derived logic. List the
modules where each variable is used or collected, to avoid double-
collecting a known variable."

All deterministic Python. Runs after referential_integrity auto-reconcile
so the registries are complete before we derive the flow graph.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


_USER_ENTERED_PREFIXES = ("q_", "ex_", "v_", "lab_", "img_", "hx_", "demo_")
_SYSTEM_PREFIXES       = ("sys_", "prev_")
_DERIVED_PREFIXES      = ("p_", "calc_", "s_", "dx_", "sev_", "need_",
                          "tx_", "rx_", "proc_", "ref_", "adv_", "fu_",
                          "out_", "err_", "m_")
_MODULE_STATE          = ("mod_",)


def _as_list(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [{**v, "_normalized_id": k} for k, v in obj.items() if isinstance(v, dict)]
    return []


def _id_of(item: dict) -> str:
    return str(
        item.get("id")
        or item.get("predicate_id")
        or item.get("module_id")
        or item.get("_normalized_id")
        or ""
    )


def _extract_tokens(text: str) -> set[str]:
    """Pull prefixed identifiers out of a condition/action string."""
    if not text:
        return set()
    return set(re.findall(r"\b[a-z][a-z0-9_]*\b", text))


def _classify_source(var_id: str) -> str:
    """Where does this variable come from?"""
    if var_id.endswith("_done") and var_id.startswith("mod_"):
        return "module_state"
    if var_id == "is_priority_exit":
        return "module_state"
    if var_id.startswith(_USER_ENTERED_PREFIXES):
        prefix = next((p for p in _USER_ENTERED_PREFIXES if var_id.startswith(p)), "")
        return {
            "q_":    "ask_chw",
            "ex_":   "observe_chw",
            "v_":    "measure_chw",
            "lab_":  "test_chw",
            "img_":  "image_chw",
            "hx_":   "history",
            "demo_": "demographics",
        }.get(prefix, "user_entered")
    if var_id.startswith(_SYSTEM_PREFIXES):
        return "system"
    if var_id.startswith(_DERIVED_PREFIXES):
        return "derived"
    return "unknown"


def _find_collectors(var_id: str, modules: dict) -> list[str]:
    """Which modules collect this variable (present in their inputs)?"""
    out = []
    for mid, m in modules.items():
        inputs = set(m.get("inputs") or [])
        if var_id in inputs:
            out.append(mid)
    return sorted(out)


def _find_users(var_id: str, modules: dict, predicates: list[dict], router: dict) -> dict:
    """Where is this variable READ (vs collected)?"""
    used_by_predicates = []
    for p in predicates:
        sources = set(p.get("source_vars") or [])
        tokens = _extract_tokens(p.get("threshold_expression", ""))
        if var_id in sources or var_id in tokens:
            used_by_predicates.append(_id_of(p))

    used_by_modules_in_rules = []
    for mid, m in modules.items():
        rule_hits = 0
        for rule in (m.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            cond = rule.get("condition", "") or ""
            if var_id in _extract_tokens(cond):
                rule_hits += 1
        if rule_hits:
            used_by_modules_in_rules.append({"module_id": mid, "rule_count": rule_hits})

    used_by_router = []
    for ri, row in enumerate(router.get("rows") or router.get("rules") or []):
        if isinstance(row, dict):
            cond = row.get("condition", "") or ""
            if var_id in _extract_tokens(cond):
                used_by_router.append({"row_index": ri, "priority": row.get("priority")})

    return {
        "used_by_predicates":         sorted(used_by_predicates),
        "used_by_modules_in_rules":   used_by_modules_in_rules,
        "used_by_router":             used_by_router,
    }


def _derive_scope(var_id: str, collectors: list[str], modules: dict) -> str:
    """Classify collection scope:
      - universal:   collected in the startup (priority=1) module; visible everywhere.
      - module:<id>: collected in exactly one non-startup module.
      - cross_module: collected in multiple modules (conditional / check-before-ask).
      - derived:     never "collected": computed from other inputs.
      - orphan:      no module collects it (should not happen after reconcile).
    """
    if _classify_source(var_id) in ("derived", "module_state", "system"):
        return "derived"
    if not collectors:
        return "orphan"

    startup_modules = {
        mid for mid, m in modules.items()
        if m.get("priority") == 1 or "assess" in mid or "identify" in mid or "demographics" in mid
    }
    if any(c in startup_modules for c in collectors):
        return "universal"
    if len(collectors) == 1:
        return f"module:{collectors[0]}"
    return "cross_module"


def _derive_supply_dependency(var_id: str, variables: list[dict], supply_list: list[dict]) -> list[str]:
    """What supply_list ids does this variable depend on (via declared depends_on_supply)?"""
    for v in variables:
        if _id_of(v) == var_id:
            return list(v.get("depends_on_supply") or [])
    return []


def derive(clinical_logic: dict) -> dict:
    """Build the full data_flow artifact."""
    variables   = _as_list(clinical_logic.get("variables", []))
    predicates  = _as_list(clinical_logic.get("predicates", []))
    supply_list = _as_list(clinical_logic.get("supply_list", []))
    modules_raw = clinical_logic.get("modules", {})
    modules     = modules_raw if isinstance(modules_raw, dict) else {
        _id_of(m): m for m in _as_list(modules_raw)
    }
    router      = clinical_logic.get("router", {}) or {}

    var_entries: list[dict] = []
    for v in variables:
        vid = _id_of(v)
        if not vid:
            continue
        collectors = _find_collectors(vid, modules)
        uses = _find_users(vid, modules, predicates, router)
        scope = _derive_scope(vid, collectors, modules)
        supplies = _derive_supply_dependency(vid, variables, supply_list)
        source = _classify_source(vid)

        var_entries.append({
            "id": vid,
            "display_name": v.get("display_name", ""),
            "source": source,
            "data_type": v.get("data_type", ""),
            "collection_scope": scope,
            "collected_in_modules": collectors,
            "used_by_predicates": uses["used_by_predicates"],
            "used_by_modules_in_rules": uses["used_by_modules_in_rules"],
            "used_by_router": uses["used_by_router"],
            "depends_on_supply": supplies,
            "source_section_id": v.get("source_section_id", ""),
            "source_quote": v.get("source_quote", ""),
            "_auto_registered": v.get("_auto_registered", False),
        })

    pred_entries: list[dict] = []
    for p in predicates:
        pid = _id_of(p)
        if not pid:
            continue
        source_vars = list(p.get("source_vars") or [])
        used_in_modules = []
        for mid, m in modules.items():
            inputs = set(m.get("inputs") or [])
            if pid in inputs:
                used_in_modules.append(mid)
            else:
                for rule in (m.get("rules") or []):
                    if isinstance(rule, dict) and pid in _extract_tokens(rule.get("condition", "") or ""):
                        if mid not in used_in_modules:
                            used_in_modules.append(mid)
                        break
        pred_entries.append({
            "id": pid,
            "human_label": p.get("human_label", ""),
            "threshold_expression": p.get("threshold_expression", ""),
            "source_vars": source_vars,
            "fail_safe": p.get("fail_safe", 0),
            "used_by_modules": sorted(used_in_modules),
            "source_section_id": p.get("source_section_id", ""),
            "source_quote": p.get("source_quote", ""),
            "_auto_registered": p.get("_auto_registered", False),
        })

    # Summary counts
    scope_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for ve in var_entries:
        scope_counts[ve["collection_scope"]] = scope_counts.get(ve["collection_scope"], 0) + 1
        source_counts[ve["source"]] = source_counts.get(ve["source"], 0) + 1

    return {
        "summary": {
            "total_variables": len(var_entries),
            "total_predicates": len(pred_entries),
            "scope_distribution": scope_counts,
            "source_distribution": source_counts,
            "orphan_variables": [v["id"] for v in var_entries if v["collection_scope"] == "orphan"],
        },
        "variables": var_entries,
        "predicates": pred_entries,
    }
