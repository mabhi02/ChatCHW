"""Tier-2 stockout coverage check.

For every (module, rule) that depends on a supply/equipment item, confirm
the module also has an `is-that-supply-unavailable` branch. If it does
not, emit a gap row: the manual either didn't cover stockout for that
item, or the extraction missed the fallback rule. Either way, the
integrator needs to either add an `ALERT: manual does not cover ...` row
to the DMN or surface the gap to MoH review.

Gaps are written to `artifacts/stockout_coverage.json` and fed to the
verifier as known issues (so every row turns into at least one
`severity: warn` divergence on the associated artifact).
"""

from __future__ import annotations

import re


def _as_list(obj) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [{**v, "_normalized_id": k} for k, v in obj.items() if isinstance(v, dict)]
    return []


def _id_of(item: dict) -> str:
    return str(item.get("id") or item.get("module_id") or item.get("_normalized_id") or "")


_SUPPLY_REF_RE = re.compile(r"\b(?:supply_|equip_)[a-z0-9_]+\b")


def _supplies_referenced_by_rule(rule: dict) -> set[str]:
    refs: set[str] = set()
    # Treatment supplies listed explicitly
    for k in ("treatment_supplies", "required_supplies", "uses_supplies"):
        val = rule.get(k)
        if isinstance(val, list):
            refs.update(str(s) for s in val if isinstance(s, str))
    # Supplies mentioned in outputs / conditions text
    for k in ("condition", "action"):
        val = rule.get(k)
        if isinstance(val, str):
            refs.update(_SUPPLY_REF_RE.findall(val))
    outputs = rule.get("outputs", {})
    if isinstance(outputs, dict):
        for v in outputs.values():
            if isinstance(v, str):
                refs.update(re.findall(_SUPPLY_REF_RE, v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        refs.update(re.findall(_SUPPLY_REF_RE, item))
    return refs


def _module_has_stockout_fallback(module: dict, supply_ref: str) -> bool:
    """Heuristic: is there any rule in this module whose condition mentions
    `<supply>_available = false` / `_unavailable = true` / stockout?
    """
    needle_names = (
        f"{supply_ref}_available",
        f"{supply_ref}_unavailable",
        f"{supply_ref}_stockout",
        f"{supply_ref}_out_of_stock",
    )
    for rule in (module.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        cond = str(rule.get("condition", "") or "").lower()
        if any(n in cond for n in needle_names):
            return True
        # Also honor structured "conditions" dicts (two-cop shape)
        conds = rule.get("conditions")
        if isinstance(conds, dict):
            for key, val in conds.items():
                key_l = str(key).lower()
                if key_l in needle_names:
                    return True
                if any(n in key_l for n in needle_names) and val in (True, "true", "unavailable"):
                    return True
    return False


def check_stockout_coverage(clinical_logic: dict) -> list[dict]:
    """Return a list of gap rows, one per (module, rule, supply) missing a fallback."""
    gaps: list[dict] = []
    supplies = {_id_of(s) for s in _as_list(clinical_logic.get("supply_list", []))}

    modules_raw = clinical_logic.get("modules", {})
    modules = modules_raw if isinstance(modules_raw, dict) else {
        _id_of(m): m for m in _as_list(modules_raw)
    }

    for mid, module in modules.items():
        if not isinstance(module, dict):
            continue
        for rule in (module.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", rule.get("rule_id", "")))
            refs = _supplies_referenced_by_rule(rule)
            for supply_ref in sorted(refs):
                if supply_ref not in supplies:
                    continue
                if not _module_has_stockout_fallback(module, supply_ref):
                    gaps.append({
                        "module": mid,
                        "rule": rid,
                        "supply": supply_ref,
                        "alert_message": (
                            f"ALERT: Manual does not cover stockout for {supply_ref} "
                            f"in {mid}/{rid}"
                        ),
                    })
    return gaps


def build_report(clinical_logic: dict) -> dict:
    gaps = check_stockout_coverage(clinical_logic)
    return {
        "summary": {
            "total_gaps": len(gaps),
            "gaps_by_module": _count_by(gaps, "module"),
            "gaps_by_supply": _count_by(gaps, "supply"),
        },
        "gaps": gaps,
    }


def _count_by(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        k = str(r.get(key, "?"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
