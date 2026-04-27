"""Convert clinical_logic JSON to a labeled Mermaid flowchart.

The goal is a readable, self-contained diagram that a reviewer can read
without cross-referencing the JSON. Each node carries:
  - The predicate's human-readable label (not just the variable name)
  - Provenance (page reference)
  - The full classification/treatment/referral triple, not just dx

Structural choices:
  - Flowchart (graph TD) with subgraphs per clinical module
  - Start -> Activator (screening) -> Router (priority) -> Module -> Integrative -> Done
  - Emergency path highlighted red, fallback paths highlighted yellow
  - All labels wrapped in quotes so special characters render safely

The converter tolerates missing/partial shapes. Each section is guarded
so a malformed module doesn't destroy the whole diagram.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Label helpers -- Mermaid treats `()[]{}|<>:#"` as syntactically meaningful
# inside node labels. Wrapping the label in a quoted string (`["text"]`)
# lets most characters through, but we still need to escape the double
# quote itself and strip control characters.
# ---------------------------------------------------------------------------


def _esc(text: str | None, max_len: int = 80) -> str:
    """Safely escape a string for use as a Mermaid node label.

    Wraps in quotes; callers should emit `NODE["{_esc(text)}"]` (no extra
    quotes). Replaces problem characters that still break even inside
    quoted labels (double quote, newlines, angle brackets, hash, pipe).
    """
    if text is None:
        return ""
    s = str(text)
    # Collapse whitespace and truncate
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    # Characters that break Mermaid labels even inside quotes
    s = (
        s.replace('"', "'")
        .replace("|", "/")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("#", "№")
    )
    return s


def _safe_id(ident: str | None, prefix: str = "n") -> str:
    """Produce a Mermaid-safe node id (alphanumeric + underscore only)."""
    if not ident:
        return f"{prefix}_unknown"
    out = []
    for ch in str(ident):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_") or "unknown"
    # Mermaid IDs cannot start with a digit
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _normalize_list(value) -> list:
    """Coerce a section to a list of dicts.

    The model sometimes returns dict-keyed-by-id shapes. Handle both.
    """
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, dict):
                out.append({**v, "_normalized_id": k})
        return out
    return []


def _predicate_label_map(predicates: list) -> dict[str, str]:
    """Build {predicate_id: human_label + source_section_id} lookup."""
    m: dict[str, str] = {}
    for p in predicates:
        # Gen 7 uses "id"; fall back to old "predicate_id"
        pid = p.get("id") or p.get("predicate_id") or p.get("_normalized_id")
        if not pid:
            continue
        label = p.get("human_label") or p.get("label") or pid
        # Gen 7 uses source_section_id; fall back to old page_ref
        page = p.get("source_section_id") or p.get("page_ref") or ""
        m[pid] = f"{label} [{page}]" if page else label
    return m


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert_to_mermaid(logic) -> str:
    """Convert clinical_logic JSON to a labeled Mermaid flowchart string."""
    # Defensive top-level guard -- fall back to an empty dict so we still
    # emit a valid (minimal) flowchart instead of crashing on logic.get().
    if not isinstance(logic, dict):
        logic = {}
    lines: list[str] = ["graph TD"]

    # Style classes -- distinct colors per node category
    lines.append("    classDef start fill:#6b7280,stroke:#374151,color:#fff")
    lines.append("    classDef activator fill:#8b5cf6,stroke:#6d28d9,color:#fff")
    lines.append("    classDef router fill:#0891b2,stroke:#0e7490,color:#fff")
    lines.append("    classDef module fill:#4488ff,stroke:#2266cc,color:#fff")
    lines.append("    classDef decision fill:#fef3c7,stroke:#92400e,color:#000")
    lines.append("    classDef rule fill:#e0f2fe,stroke:#0369a1,color:#000")
    lines.append("    classDef default_rule fill:#fef9c3,stroke:#ca8a04,color:#000")
    lines.append("    classDef emergency fill:#dc2626,stroke:#7f1d1d,color:#fff,stroke-width:3px")
    lines.append("    classDef integrative fill:#16a34a,stroke:#14532d,color:#fff")
    lines.append("    classDef done fill:#059669,stroke:#064e3b,color:#fff")
    lines.append("")

    # Build predicate lookup so rules can show human labels
    predicates = _normalize_list(logic.get("predicates", []))
    pred_labels = _predicate_label_map(predicates)

    # Entry
    lines.append('    START(["Patient arrives"]):::start')
    lines.append("")

    # --- Activator / Router ---
    # Gen 7 uses "router" with "rows"; legacy used "activator" with "rules".
    # Screens all presenting complaints and flags modules that should run.
    activator = logic.get("router") or logic.get("activator") or {}
    if not isinstance(activator, dict):
        activator = {}
    act_rows = activator.get("rows") or activator.get("rules") or []
    if act_rows:
        act_cols = activator.get("input_columns") or activator.get("inputs") or []
        hit_pol = activator.get("hit_policy", "COLLECT").upper()
        lines.append(f'    subgraph ACT["Router ({hit_pol})"]')
        lines.append('        ACT_DECIDE{{"Screen complaints"}}:::activator')
        for r_idx, rule in enumerate(act_rows):
            # Gen 7: output_module; legacy: module_id or next_module
            mid = rule.get("output_module") or rule.get("module_id") or rule.get("next_module") or f"mod_{r_idx}"
            mid_safe = _safe_id(mid, "mod")
            # Gen 7 has condition string; legacy has positional inputs
            condition_str = rule.get("condition", "")
            positional_inputs = rule.get("inputs") if isinstance(rule.get("inputs"), list) else None
            if positional_inputs is not None:
                conds = []
                for i, inp in enumerate(positional_inputs):
                    if inp == "true" and i < len(act_cols):
                        col = act_cols[i]
                        conds.append(pred_labels.get(col, col))
                cond_text = _esc(" & ".join(conds) if conds else "any", 60)
            elif condition_str:
                cond_text = _esc(condition_str, 60)
            else:
                cond_text = "any"
            lines.append(f'        ACT_DECIDE -->|"{cond_text}"| ACT_FLAG_{mid_safe}["Required: {_esc(mid)}"]:::rule')
        lines.append("    end")
        lines.append("    START --> ACT_DECIDE")
        lines.append("")

    # --- Router (priority routing) ---
    # The "activator" variable above already holds the router/activator dict.
    # Here we render a separate priority-routing subgraph if the router has rows.
    # Gen 7 uses router.rows with output_module; legacy used router.rules with next_module.
    router = logic.get("router") or {}
    if not isinstance(router, dict):
        router = {}
    router_rows_list = router.get("rows") or router.get("rules") or []
    if isinstance(router, dict) and router_rows_list:
        hit_pol = router.get("hit_policy", "FIRST").upper()
        lines.append(f'    subgraph RTR["Router ({hit_pol})"]')
        lines.append('        ROUTER{{"Priority route"}}:::router')
        for idx, rule in enumerate(router_rows_list):
            condition = (rule.get("condition") or "").lower()
            # Gen 7: output_module; legacy: next_module or module_id
            next_mod = rule.get("output_module") or rule.get("next_module") or rule.get("module_id") or f"step_{idx}"
            next_safe = _safe_id(next_mod, "r")
            priority = rule.get("priority", idx)
            cond_label = _esc(rule.get("condition") or "default", 50)
            style = ":::rule"
            edge_label = f"P{priority}: {cond_label}"
            if "danger" in condition or "emergency" in condition or "urgent" in condition:
                style = ":::emergency"
                edge_label = f"[!] P{priority}: {cond_label}"
            elif "always" in condition or priority == 99 or priority == 999:
                style = ":::integrative"
            lines.append(f'        ROUTER -->|"{_esc(edge_label, 60)}"| RT_{next_safe}["{_esc(next_mod)}"]{style}')
        lines.append("    end")
        # Wire activator to router (or START if no activator)
        if act_rows:
            lines.append("    ACT_DECIDE --> ROUTER")
        else:
            lines.append("    START --> ROUTER")
        lines.append("")

    # --- Modules ---
    # Each clinical module becomes a subgraph of its rule decisions.
    modules = _normalize_list(logic.get("modules", []))
    for module in modules:
        mid = module.get("module_id") or module.get("_normalized_id") or "mod_unknown"
        mid_safe = _safe_id(mid, "mod")
        display = module.get("display_name") or mid
        hit_policy = module.get("hit_policy", "FIRST")
        # Gen 7 uses "inputs"/"outputs"; fall back to old "input_columns"/"output_columns"
        input_cols = module.get("inputs") or module.get("input_columns") or []

        lines.append(f'    subgraph SG_{mid_safe}["{_esc(display)} ({hit_policy})"]')
        # Entry node for the module
        lines.append(f'        {mid_safe}_ENTRY(["Enter: {_esc(display, 40)}"]):::module')

        rules = module.get("rules", []) or []
        prev = f"{mid_safe}_ENTRY"
        for r_idx, rule in enumerate(rules):
            inputs = rule.get("inputs", [])
            outputs = rule.get("outputs", [])
            provenance = rule.get("provenance") or {}

            # Gen 7 rules use "condition"/"action" strings rather than
            # positional input/output arrays. Handle both.
            rule_condition = rule.get("condition", "")
            rule_action = rule.get("action", "")

            # Build human-readable condition from inputs + predicate labels
            is_default = all((inp == "-" or inp is None) for inp in inputs) if inputs else False
            if not inputs and rule_condition:
                # Gen 7 style: condition is a string expression
                is_default = False
                cond_label = rule_condition
            elif is_default:
                cond_label = "default (else)"
            else:
                conds = []
                for i, inp in enumerate(inputs):
                    if inp == "-" or inp is None:
                        continue
                    if i < len(input_cols):
                        col = input_cols[i]
                        col_label = pred_labels.get(col, col)
                        if inp == "true":
                            conds.append(col_label)
                        elif inp == "false":
                            conds.append(f"NOT {col_label}")
                        else:
                            conds.append(f"{col_label}={inp}")
                cond_label = " AND ".join(conds) if conds else "any"

            # Build result label: dx / tx / referral / message refs
            out_cols = module.get("outputs") or module.get("output_columns") or []
            result_parts: list[str] = []
            if outputs:
                for i, val in enumerate(outputs):
                    if val in (None, "", "-"):
                        continue
                    label = out_cols[i] if i < len(out_cols) else f"out{i}"
                    # Highlight urgent referral
                    if "ref" in label and "urgent" in label:
                        if str(val).lower() == "true":
                            result_parts.append("[!] URGENT REFER")
                        continue
                    result_parts.append(f"{label}: {val}")
            elif rule_action:
                # Gen 7 style: action is a string like "dx = 'X'; ref = true"
                result_parts.append(rule_action)
            result_text = "; ".join(result_parts) if result_parts else "(no outputs)"

            # Provenance footer: Gen 7 uses source_quote/source_section_id;
            # legacy uses provenance dict with page/quote.
            page = rule.get("source_section_id") or provenance.get("page", "")
            quote = rule.get("source_quote") or provenance.get("quote", "")
            desc = rule.get("description", "")
            if page or quote:
                result_text += f"\\n[{_esc(page, 30)}]" if page else ""
                if quote:
                    result_text += f" '{_esc(quote, 40)}'"
            elif desc:
                result_text += f"\\n{_esc(desc, 50)}"

            rule_node = f"{mid_safe}_R{r_idx}"
            style = ":::default_rule" if is_default else ":::rule"
            # Check for urgent referral -> red styling
            is_urgent = False
            if outputs:
                # Gen 7 schema: outputs is a dict {output_name: value, ...}
                # Legacy schema: outputs is a list positionally matched to out_cols
                if isinstance(outputs, dict):
                    is_urgent = any(
                        "ref" in k and "urgent" in k
                        and str(v).lower() == "true"
                        for k, v in outputs.items()
                    )
                else:
                    is_urgent = any(
                        "ref" in (out_cols[i] if i < len(out_cols) else "")
                        and "urgent" in (out_cols[i] if i < len(out_cols) else "")
                        and str(outputs[i]).lower() == "true"
                        for i in range(len(outputs))
                    )
            elif rule_action:
                is_urgent = "ref_urgent" in rule_action and "true" in rule_action.lower()
            if is_urgent:
                style = ":::emergency"

            lines.append(f'        {rule_node}["{_esc(result_text, 140)}"]{style}')
            lines.append(f'        {prev} -->|"{_esc(cond_label, 70)}"| {rule_node}')

        lines.append("    end")

        # Wire router entry into this module if the router routes to it
        _rr = (router.get("rows") or router.get("rules") or []) if isinstance(router, dict) else []
        if any(
            (r.get("output_module") == mid or r.get("next_module") == mid or r.get("module_id") == mid)
            for r in _rr
        ):
            lines.append(f"    ROUTER -.->|if routed| {mid_safe}_ENTRY")
        lines.append("")

    # --- Integrative ---
    integrative = logic.get("integrative") or {}
    lines.append('    subgraph INT["Integrative (merge outputs)"]')
    lines.append('        INT_MERGE["Merge per-module outputs\\n• highest referral wins\\n• treatments additive (unless contraindicated)\\n• shortest follow-up"]:::integrative')
    if isinstance(integrative, dict):
        int_rules = integrative.get("rules", []) or []
        for idx, rule in enumerate(int_rules[:5]):  # cap at 5 to keep diagram readable
            desc = (
                rule.get("description")
                or rule.get("rule")
                or rule.get("name")
                or f"rule {idx}"
            )
            lines.append(f'        INT_RULE_{idx}["{_esc(desc, 80)}"]:::integrative')
            lines.append(f'        INT_MERGE --> INT_RULE_{idx}')
    lines.append("    end")
    lines.append("")

    # Exit
    lines.append('    DONE(["Care plan complete"]):::done')

    # Wire all modules to integrative, integrative to done
    for module in modules:
        mid = module.get("module_id") or module.get("_normalized_id") or "mod_unknown"
        mid_safe = _safe_id(mid, "mod")
        # Connect last rule of each module to integrative merge
        last_rule_idx = max(0, len(module.get("rules", []) or []) - 1)
        lines.append(f'    {mid_safe}_R{last_rule_idx} --> INT_MERGE')
    lines.append("    INT_MERGE --> DONE")

    return "\n".join(lines)
