"""Convert clinical_logic JSON to DMN 1.3 XML.

Maps JSON modules to DMN decision elements:
- Each module -> <decision> with <decisionTable>
- hitPolicy from module's hit_policy
- inputs -> <input> with typeRef="boolean"
- outputs -> <output>
- rules -> <rule> with <description> (source_quote + source_section_id)

Gen 7 schema: modules is dict-keyed-by-id, router replaces activator,
predicates/phrases use 'id' not 'predicate_id'/'message_id'.
Backwards-compatible: falls back to old keys when new ones absent.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"
DMNDI_NS = "https://www.omg.org/spec/DMN/20191111/DMNDI/"


def _bool_to_feel(value: str) -> str:
    """Convert our boolean input format to DMN FEEL expression."""
    if value == "true":
        return "true"
    elif value == "false":
        return "false"
    elif value == "-":
        return "-"
    return value


def convert_to_dmn(logic) -> str:
    """Convert clinical_logic JSON to DMN 1.3 XML string.

    Args:
        logic: The validated clinical_logic JSON dict.

    Returns:
        Pretty-printed DMN XML string.
    """
    # Defensive top-level guard. Callers (session_manager._generate_artifacts)
    # already normalize via _normalize_for_converters, which returns {} on
    # non-dict input, but a direct caller (tests, CLI tools) could still hand
    # us a string. Emit a minimal but valid DMN document so downstream file
    # writes don't crash.
    if not isinstance(logic, dict):
        logic = {}

    # Root definitions element
    definitions = ET.Element("definitions")
    definitions.set("xmlns", DMN_NS)
    definitions.set("xmlns:dmndi", DMNDI_NS)
    definitions.set("id", "chw_navigator_definitions")
    definitions.set("name", "CHW Navigator Clinical Logic")
    definitions.set("namespace", DMN_NS)

    # Input data elements for all predicates. Skip any predicate that isn't
    # a dict (stray string / None in a malformed list) so one bad row doesn't
    # sink the whole conversion.
    predicates = logic.get("predicates", [])
    if not isinstance(predicates, list):
        predicates = []
    for pred in predicates:
        if not isinstance(pred, dict):
            continue
        # Gen 7 uses "id"; fall back to old "predicate_id"
        pid = pred.get("id") or pred.get("predicate_id")
        if not pid:
            continue
        input_data = ET.SubElement(definitions, "inputData")
        input_data.set("id", pid)
        input_data.set("name", pred.get("human_label", pid))

        variable = ET.SubElement(input_data, "variable")
        variable.set("id", f"var_{pid}")
        variable.set("name", pid)
        variable.set("typeRef", "boolean")

    # Decision elements for each module.
    # Gen 7: modules is a dict keyed by module_id. Fall back to list.
    raw_modules = logic.get("modules", [])
    if isinstance(raw_modules, dict):
        modules_list = list(raw_modules.values())
    elif isinstance(raw_modules, list):
        modules_list = raw_modules
    else:
        modules_list = []

    for module in modules_list:
        if not isinstance(module, dict):
            continue
        module_id = module.get("module_id", "unknown")
        decision = ET.SubElement(definitions, "decision")
        decision.set("id", f"decision_{module_id}")
        decision.set("name", module.get("display_name", module_id))

        # Output variable
        out_var = ET.SubElement(decision, "variable")
        out_var.set("id", f"var_decision_{module_id}")
        out_var.set("name", f"{module_id}_output")
        out_var.set("typeRef", "string")

        # Gen 7 uses "inputs"/"outputs"; fall back to old "input_columns"/"output_columns"
        input_cols = module.get("inputs") or module.get("input_columns") or []
        output_cols = module.get("outputs") or module.get("output_columns") or []

        # Information requirements (inputs)
        for col in input_cols:
            info_req = ET.SubElement(decision, "informationRequirement")
            info_req.set("id", f"ir_{module_id}_{col}")
            req_input = ET.SubElement(info_req, "requiredInput")
            req_input.set("href", f"#{col}")

        # Decision table
        dt = ET.SubElement(decision, "decisionTable")
        dt.set("id", f"dt_{module_id}")
        dt.set("hitPolicy", module.get("hit_policy", "FIRST").upper())

        # Input columns
        for idx, col in enumerate(input_cols):
            inp = ET.SubElement(dt, "input")
            inp.set("id", f"input_{module_id}_{idx}")
            inp.set("label", col)
            inp_expr = ET.SubElement(inp, "inputExpression")
            inp_expr.set("id", f"ie_{module_id}_{idx}")
            inp_expr.set("typeRef", "boolean")
            text = ET.SubElement(inp_expr, "text")
            text.text = col

        # Output columns
        for idx, col in enumerate(output_cols):
            out = ET.SubElement(dt, "output")
            out.set("id", f"output_{module_id}_{idx}")
            out.set("label", col)
            out.set("name", col)
            out.set("typeRef", "string")

        # Rules
        for r_idx, rule in enumerate(module.get("rules", [])):
            rule_el = ET.SubElement(dt, "rule")
            rid = rule.get("rule_id", f"rule_{module_id}_{r_idx}")
            rule_el.set("id", rid)

            # Description: Gen 7 uses source_quote + source_section_id at
            # the predicate/phrase level; rules carry condition/action.
            # Fall back to old provenance dict if present.
            prov = rule.get("provenance") or {}
            source_quote = rule.get("source_quote") or prov.get("quote", "")
            source_section = rule.get("source_section_id") or prov.get("page", "")
            description_text = rule.get("description", "")
            # Build a description line from whatever is available
            desc_parts = []
            if source_section:
                desc_parts.append(f"[{source_section}]")
            if source_quote:
                desc_parts.append(source_quote)
            if description_text and not source_quote:
                desc_parts.append(description_text)
            # Gen 7 rule has "condition" and "action" as strings
            condition = rule.get("condition", "")
            action = rule.get("action", "")
            if condition and not desc_parts:
                desc_parts.append(f"IF {condition} THEN {action}")
            if desc_parts:
                desc = ET.SubElement(rule_el, "description")
                desc.text = " ".join(desc_parts)

            # Gen 7 rules use "condition"/"action" strings rather than
            # positional input/output arrays. If positional arrays exist
            # (old format), use them; otherwise emit condition as a single
            # inputEntry and action as a single outputEntry.
            positional_inputs = rule.get("inputs") if isinstance(rule.get("inputs"), list) else None
            positional_outputs = rule.get("outputs") if isinstance(rule.get("outputs"), list) else None

            if positional_inputs is not None:
                for i_idx, inp_val in enumerate(positional_inputs):
                    ie = ET.SubElement(rule_el, "inputEntry")
                    ie.set("id", f"ie_{module_id}_{r_idx}_{i_idx}")
                    text = ET.SubElement(ie, "text")
                    text.text = _bool_to_feel(inp_val)
            elif condition:
                ie = ET.SubElement(rule_el, "inputEntry")
                ie.set("id", f"ie_{module_id}_{r_idx}_0")
                text = ET.SubElement(ie, "text")
                text.text = condition

            if positional_outputs is not None:
                for o_idx, out_val in enumerate(positional_outputs):
                    oe = ET.SubElement(rule_el, "outputEntry")
                    oe.set("id", f"oe_{module_id}_{r_idx}_{o_idx}")
                    text = ET.SubElement(oe, "text")
                    text.text = f'"{out_val}"'
            elif action:
                oe = ET.SubElement(rule_el, "outputEntry")
                oe.set("id", f"oe_{module_id}_{r_idx}_0")
                text = ET.SubElement(oe, "text")
                text.text = f'"{action}"'

    # Router decision (Gen 7) / Activator decision (legacy).
    # Gen 7 uses "router" with {hit_policy, rows: [{priority, condition,
    # output_module, description}]}. Fall back to old "activator" shape.
    router = logic.get("router") or logic.get("activator") or {}
    if not isinstance(router, dict):
        router = {}

    # Determine rows: Gen 7 "rows", legacy "rules"
    router_rows = router.get("rows") or router.get("rules") or []
    if isinstance(router, dict) and router_rows:
        act_decision = ET.SubElement(definitions, "decision")
        act_decision.set("id", "decision_router")
        act_decision.set("name", "Router")

        act_dt = ET.SubElement(act_decision, "decisionTable")
        act_dt.set("id", "dt_router")
        hit_policy = router.get("hit_policy", "FIRST").upper()
        act_dt.set("hitPolicy", hit_policy)

        # Legacy activator had input_columns; Gen 7 router uses condition strings
        legacy_input_cols = router.get("input_columns") or router.get("inputs") or []
        for idx, col in enumerate(legacy_input_cols):
            inp = ET.SubElement(act_dt, "input")
            inp.set("id", f"input_router_{idx}")
            inp_expr = ET.SubElement(inp, "inputExpression")
            inp_expr.set("typeRef", "boolean")
            text = ET.SubElement(inp_expr, "text")
            text.text = col

        # If no legacy columns, add a single "condition" input for Gen 7
        if not legacy_input_cols:
            inp = ET.SubElement(act_dt, "input")
            inp.set("id", "input_router_condition")
            inp.set("label", "condition")
            inp_expr = ET.SubElement(inp, "inputExpression")
            inp_expr.set("typeRef", "string")
            text = ET.SubElement(inp_expr, "text")
            text.text = "condition"

        out = ET.SubElement(act_dt, "output")
        out.set("id", "output_router_module")
        out.set("label", "output_module")
        out.set("typeRef", "string")

        for r_idx, rule in enumerate(router_rows):
            rule_el = ET.SubElement(act_dt, "rule")
            rule_el.set("id", f"rule_router_{r_idx}")

            # Gen 7: condition is a string; legacy: positional inputs array
            positional = rule.get("inputs") if isinstance(rule.get("inputs"), list) else None
            if positional is not None:
                for i_idx, inp_val in enumerate(positional):
                    ie = ET.SubElement(rule_el, "inputEntry")
                    ie.set("id", f"ie_router_{r_idx}_{i_idx}")
                    text = ET.SubElement(ie, "text")
                    text.text = _bool_to_feel(inp_val)
            else:
                ie = ET.SubElement(rule_el, "inputEntry")
                ie.set("id", f"ie_router_{r_idx}_0")
                text = ET.SubElement(ie, "text")
                text.text = rule.get("condition", "-")

            oe = ET.SubElement(rule_el, "outputEntry")
            oe.set("id", f"oe_router_{r_idx}")
            text = ET.SubElement(oe, "text")
            # Gen 7 uses output_module; legacy uses module_id
            target = rule.get("output_module") or rule.get("module_id", "")
            text.text = f'"{target}"'

            # Add description if available
            desc_text = rule.get("description", "")
            if desc_text:
                desc_el = ET.SubElement(rule_el, "description")
                desc_el.text = desc_text

    # Pretty print
    rough = ET.tostring(definitions, encoding="unicode")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding=None)
