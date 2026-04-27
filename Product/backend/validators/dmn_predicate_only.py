"""Predicate-only DMN check (Predicate Table spec section 7).

The spec is explicit: 'DMN must only reference p_* and c_* variables.'
Raw `v_*` / `q_*` / `ex_*` / `lab_*` / `demo_*` references inside DMN
input expressions mean Stage 3 inlined raw inputs into rule conditions
rather than wrapping them in named predicates.

This validator scans `clinical_logic.dmn` (XML) and the in-memory
`clinical_logic.modules` rule conditions and flags any non-predicate
variable that appears in a rule. Each finding becomes one
`severity: warn` divergence; the worklist sorts by frequency so the
most common offenders surface first.
"""

from __future__ import annotations

import re
from pathlib import Path


_ALLOWED_PREFIXES: tuple[str, ...] = ("p_", "c_", "calc_")
# Variables we treat as "raw inputs" -- if they appear in a DMN/module
# rule condition, the Stage 3 compiler is leaking them past the predicate
# table.
_RAW_PREFIXES: tuple[str, ...] = (
    "q_", "ex_", "v_", "lab_", "img_", "hx_", "demo_", "sys_",
)
_TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")


def _extract_tokens_from_dmn(dmn_text: str) -> set[str]:
    """Pull every prefixed identifier out of `<input>` expressions in the DMN."""
    tokens: set[str] = set()
    for expr in re.findall(r'<input[^>]*expression="([^"]*)"', dmn_text):
        tokens.update(_TOKEN_RE.findall(expr))
    # Also capture FEEL `inputExpression` or text-content `<text>...</text>`
    for expr in re.findall(r'<text>([^<]*)</text>', dmn_text):
        tokens.update(_TOKEN_RE.findall(expr))
    return {t for t in tokens if t.startswith(_RAW_PREFIXES)}


def _extract_tokens_from_modules(modules: dict | list) -> set[str]:
    """Pull raw-prefix tokens out of every module rule condition."""
    if isinstance(modules, dict):
        iterable = modules.values()
    elif isinstance(modules, list):
        iterable = modules
    else:
        return set()

    tokens: set[str] = set()
    for m in iterable:
        if not isinstance(m, dict):
            continue
        for rule in (m.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            cond = rule.get("condition")
            if isinstance(cond, str):
                tokens.update(_TOKEN_RE.findall(cond))
            conds = rule.get("conditions")
            if isinstance(conds, dict):
                # gen8 two-cop: conditions are {key: value} or {expr: "..."}
                if "expr" in conds and isinstance(conds["expr"], str):
                    tokens.update(_TOKEN_RE.findall(conds["expr"]))
                for k in conds.keys():
                    if isinstance(k, str):
                        tokens.update(_TOKEN_RE.findall(k))
    return {t for t in tokens if t.startswith(_RAW_PREFIXES)}


def check(
    clinical_logic: dict,
    dmn_path: Path | None = None,
) -> dict:
    """Run the predicate-only check.

    Inputs:
      clinical_logic: the in-memory artifact dict
      dmn_path:       optional path to `clinical_logic.dmn`. When omitted,
                      only module rules are scanned.
    Returns a report with a per-token count + a list of suggestion stubs
    for each unwrapped raw variable.
    """
    raw_in_modules = _extract_tokens_from_modules(clinical_logic.get("modules", {}))
    raw_in_dmn: set[str] = set()
    if dmn_path is not None and dmn_path.exists():
        raw_in_dmn = _extract_tokens_from_dmn(
            dmn_path.read_text(encoding="utf-8", errors="replace")
        )

    # Cross-reference against the predicate table to see which raw refs
    # already have a wrapping predicate the model COULD have used.
    pred_inputs: dict[str, set[str]] = {}
    for p in (clinical_logic.get("predicates") or []):
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        for v in (p.get("inputs_used") or p.get("source_vars") or []):
            pred_inputs.setdefault(str(v), set()).add(pid)

    findings: list[dict] = []
    for token in sorted(raw_in_modules | raw_in_dmn):
        wraps = sorted(pred_inputs.get(token, set()))
        findings.append({
            "raw_variable": token,
            "found_in_modules": token in raw_in_modules,
            "found_in_dmn": token in raw_in_dmn,
            "wrapping_predicates_available": wraps,
            "remediation": (
                f"Replace direct references to {token!r} in rule conditions "
                f"with one of the wrapping predicates: {wraps}"
                if wraps else
                f"No predicate wraps {token!r}. Either add one to the predicate "
                f"table or remove the rule reference."
            ),
        })

    return {
        "summary": {
            "raw_var_count_in_modules": len(raw_in_modules),
            "raw_var_count_in_dmn": len(raw_in_dmn),
            "total_unique_raw_refs": len(raw_in_modules | raw_in_dmn),
            "spec_section": "Predicate Table spec section 7: DMN must only reference p_* and c_*",
        },
        "findings": findings,
    }


def to_divergences(report: dict) -> list[dict]:
    """Render predicate-only-DMN findings as verification divergences."""
    out: list[dict] = []
    for f in report.get("findings", []):
        sev = "warn"
        out.append({
            "type": "raw_variable_in_dmn",
            "severity": sev,
            "detail": (
                f"Raw variable {f['raw_variable']!r} is referenced directly in "
                f"{'DMN ' if f['found_in_dmn'] else ''}"
                f"{'modules ' if f['found_in_modules'] else ''}"
                f"-- spec section 7 requires predicate wrapping. "
                f"{f['remediation']}"
            ),
            "evidence": {
                "raw_variable": f["raw_variable"],
                "wrapping_predicates_available": f["wrapping_predicates_available"],
            },
        })
    return out
