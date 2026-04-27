"""Extended per-variable scope + lifecycle derivation for gen8.

Supersedes `backend/gen7/data_flow.py`. Adds the five gen8 scopes called
out in the spec, any of which can also be tagged `orphan`:

  * `universal`              -- collected in the startup module; visible everywhere.
  * `module_local`           -- collected by exactly one non-startup module.
  * `cross_module`           -- collected by module A, used by module B.
  * `derived`                -- computed from other inputs (prefix p_/calc_/...).
  * `conditionally_collected` -- collected only when an upstream variable/flag fires.
  * `orphan`                 -- referenced somewhere but no module collects it.

A clean gen8 run must produce `summary.orphan == 0`. Orphan variables
become `severity: error` divergences.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


_USER_ENTERED_PREFIXES = ("q_", "ex_", "v_", "lab_", "img_", "hx_", "demo_")
_SYSTEM_PREFIXES = ("sys_", "prev_")
_DERIVED_PREFIXES = (
    "p_", "calc_", "s_", "dx_", "sev_", "need_", "tx_", "rx_",
    "proc_", "ref_", "adv_", "fu_", "out_", "err_", "m_",
)


def _as_list(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [{**v, "_normalized_id": k} for k, v in obj.items() if isinstance(v, dict)]
    return []


def _id_of(item: dict) -> str:
    return str(item.get("id") or item.get("predicate_id") or item.get("module_id")
               or item.get("_normalized_id") or "")


def _extract_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b[a-z][a-z0-9_]*\b", text))


def _classify_source(var_id: str) -> str:
    if var_id.endswith("_done") and var_id.startswith("mod_"):
        return "module_state"
    if var_id == "is_priority_exit":
        return "module_state"
    if var_id.startswith(_USER_ENTERED_PREFIXES):
        prefix = next((p for p in _USER_ENTERED_PREFIXES if var_id.startswith(p)), "")
        return {
            "q_": "ask_chw",
            "ex_": "observe_chw",
            "v_": "measure_chw",
            "lab_": "test_chw",
            "img_": "image_chw",
            "hx_": "history",
            "demo_": "demographics",
        }.get(prefix, "user_entered")
    if var_id.startswith(_SYSTEM_PREFIXES):
        return "system"
    if var_id.startswith(_DERIVED_PREFIXES):
        return "derived"
    return "unknown"


def _find_collectors(var_id: str, modules: dict) -> list[str]:
    out: list[str] = []
    for mid, m in modules.items():
        if not isinstance(m, dict):
            continue
        inputs = set(m.get("inputs") or [])
        if var_id in inputs:
            out.append(mid)
    return sorted(out)


def _is_conditionally_collected(var_id: str, modules: dict) -> bool:
    """Detect `only asked when <upstream>=...` patterns in module rules.

    Heuristic: a variable is conditionally collected iff some module's
    rule adds it as an input contingent on another variable.
    """
    for m in modules.values():
        if not isinstance(m, dict):
            continue
        for rule in (m.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            outputs = rule.get("outputs") or {}
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    if "collect" in str(k).lower() and var_id in _extract_tokens(str(v)):
                        return True
    return False


def _find_users(var_id: str, modules: dict, predicates: list[dict], router: dict) -> dict:
    used_by_predicates: list[str] = []
    for p in predicates:
        sources = set(p.get("inputs_used") or p.get("source_vars") or [])
        tokens = _extract_tokens(p.get("formal_definition", p.get("threshold_expression", "")))
        if var_id in sources or var_id in tokens:
            used_by_predicates.append(_id_of(p))

    used_by_modules: list[dict] = []
    for mid, m in modules.items():
        if not isinstance(m, dict):
            continue
        hits = 0
        for rule in (m.get("rules") or []):
            if isinstance(rule, dict):
                cond = rule.get("condition", "") or ""
                if var_id in _extract_tokens(cond):
                    hits += 1
        if hits:
            used_by_modules.append({"module_id": mid, "rule_count": hits})

    used_by_router: list[dict] = []
    # Two-cop shape
    for cop_key in ("cop1_queue_builder", "cop2_next_module"):
        cop = router.get(cop_key) or {}
        for ri, rule in enumerate(cop.get("rules") or []):
            if isinstance(rule, dict):
                cond_str = str(rule.get("conditions") or rule.get("condition") or "")
                if var_id in _extract_tokens(cond_str):
                    used_by_router.append({"cop": cop_key, "row_index": ri})
    # Fallback to legacy flat rows
    for ri, row in enumerate(router.get("rows") or []):
        if isinstance(row, dict):
            cond = row.get("condition", "") or ""
            if var_id in _extract_tokens(cond):
                used_by_router.append({"cop": "flat", "row_index": ri, "priority": row.get("priority")})

    return {
        "used_by_predicates": sorted(used_by_predicates),
        "used_by_modules_in_rules": used_by_modules,
        "used_by_router": used_by_router,
    }


def _derive_scope(var_id: str, collectors: list[str], modules: dict,
                  conditionally_collected: bool) -> str:
    """gen8 scope classifier."""
    if _classify_source(var_id) in ("derived", "module_state", "system"):
        return "derived"
    if conditionally_collected:
        return "conditionally_collected"
    if not collectors:
        return "orphan"

    startup_modules = {
        mid for mid, m in modules.items()
        if isinstance(m, dict) and (
            m.get("priority") == 1
            or "startup" in mid or "intro" in mid or "assess" in mid
            or "identify" in mid or "demographics" in mid
        )
    }
    if any(c in startup_modules for c in collectors):
        return "universal"
    if len(collectors) == 1:
        return "module_local"
    return "cross_module"


def derive(clinical_logic: dict) -> dict:
    variables = _as_list(clinical_logic.get("variables", []))
    predicates = _as_list(clinical_logic.get("predicates", []))
    supply_list = _as_list(clinical_logic.get("supply_list", []))
    modules_raw = clinical_logic.get("modules", {})
    modules = modules_raw if isinstance(modules_raw, dict) else {
        _id_of(m): m for m in _as_list(modules_raw)
    }
    router = clinical_logic.get("router", {}) or {}

    by_variable_id: dict[str, dict] = {}
    summary_scope: dict[str, int] = {
        "universal": 0, "module_local": 0, "cross_module": 0,
        "derived": 0, "conditionally_collected": 0, "orphan": 0,
    }

    for v in variables:
        vid = _id_of(v)
        if not vid:
            continue
        collectors = _find_collectors(vid, modules)
        cond_collected = _is_conditionally_collected(vid, modules)
        uses = _find_users(vid, modules, predicates, router)
        scope = _derive_scope(vid, collectors, modules, cond_collected)
        summary_scope[scope] = summary_scope.get(scope, 0) + 1

        by_variable_id[vid] = {
            "id": vid,
            "display_name": v.get("display_name", ""),
            "source": _classify_source(vid),
            "scope": scope,
            "collected_by": collectors,
            "used_by_modules": [u["module_id"] for u in uses["used_by_modules_in_rules"]],
            "used_by_predicates": uses["used_by_predicates"],
            "used_by_router": uses["used_by_router"],
            "ordering": {
                "_manual_page": v.get("_manual_page"),
                "_order_source": v.get("_order_source"),
            },
            "depends_on_supply": list(v.get("depends_on_supply") or []),
        }

    pred_entries: list[dict] = []
    for p in predicates:
        pid = _id_of(p)
        if not pid:
            continue
        inputs_used = list(p.get("inputs_used") or p.get("source_vars") or [])
        used_in_modules: list[str] = []
        for mid, m in modules.items():
            if not isinstance(m, dict):
                continue
            inputs = set(m.get("inputs") or [])
            if pid in inputs:
                used_in_modules.append(mid)
            else:
                for rule in (m.get("rules") or []):
                    if isinstance(rule, dict) and pid in _extract_tokens(rule.get("condition", "") or ""):
                        used_in_modules.append(mid)
                        break
        pred_entries.append({
            "id": pid,
            "description_clinical": p.get("description_clinical", p.get("human_label", "")),
            "formal_definition": p.get("formal_definition", p.get("threshold_expression", "")),
            "inputs_used": inputs_used,
            "units": p.get("units", ""),
            "missingness_rule": p.get("missingness_rule", ""),
            "allowed_input_domain": p.get("allowed_input_domain", ""),
            "rounding_parsing_rule": p.get("rounding_parsing_rule", ""),
            "provenance": p.get("provenance", ""),
            "used_by_modules": sorted(set(used_in_modules)),
        })

    return {
        "by_variable_id": by_variable_id,
        "predicates": pred_entries,
        "summary": {
            **summary_scope,
            "total_variables": len(by_variable_id),
            "total_predicates": len(pred_entries),
            "orphan_variables": [
                vid for vid, v in by_variable_id.items() if v["scope"] == "orphan"
            ],
        },
    }
