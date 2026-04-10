"""
Minimal DMN loader using lxml. Parses OMG DMN XML into a simple object model
compatible with the DMNParser (replaces the broken dmn_python PyPI package).
"""

import os
from typing import List, Any
from types import SimpleNamespace

from lxml import etree

DMN_NS = "https://www.omg.org/spec/DMN/20191111/MODEL/"
NS = {"dmn": DMN_NS}


def _text(el: Any) -> str:
    """Get text content of element, or empty string."""
    if el is None:
        return ""
    return (el.text or "").strip()


def _first(el: Any, path: str):
    """First matching child (path uses dmn: prefix)."""
    if el is None:
        return None
    return el.find(path, NS)


def _all(el: Any, path: str) -> List:
    """All matching children."""
    if el is None:
        return []
    return el.findall(path, NS)


def load_dmn(file_path: str) -> SimpleNamespace:
    """
    Load a DMN XML file and return a model with .decisions (list of decision objects).
    Each decision has .name and .decisionTable; the table has .inputs, .outputs, .rules, .hitPolicy.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DMN file not found: {file_path}")

    tree = etree.parse(file_path)
    root = tree.getroot()

    decisions = []
    for dec_el in _all(root, "dmn:decision"):
        name = dec_el.get("name") or dec_el.get("id") or "unnamed"
        dt_el = _first(dec_el, "dmn:decisionTable")
        if dt_el is None:
            decision = SimpleNamespace(name=name, decisionTable=None)
            decisions.append(decision)
            continue

        # Inputs: dmn:input with dmn:inputExpression text
        inputs = []
        for inp_el in _all(dt_el, "dmn:input"):
            expr_el = _first(inp_el, "dmn:inputExpression")
            text = _text(expr_el) if expr_el is not None else ""
            inputs.append(
                SimpleNamespace(
                    inputExpression=SimpleNamespace(text=text) if text else None,
                    label=inp_el.get("label"),
                )
            )

        # Outputs: dmn:output with name
        outputs = []
        for out_el in _all(dt_el, "dmn:output"):
            outputs.append(SimpleNamespace(name=out_el.get("name") or ""))

        # Rules: dmn:rule with dmn:inputEntry and dmn:outputEntry (dmn:text)
        rules = []
        for rule_el in _all(dt_el, "dmn:rule"):
            input_entries = [
                SimpleNamespace(text=_text(_first(e, "dmn:text")))
                for e in _all(rule_el, "dmn:inputEntry")
            ]
            output_entries = [
                SimpleNamespace(text=_text(_first(e, "dmn:text")))
                for e in _all(rule_el, "dmn:outputEntry")
            ]
            rules.append(
                SimpleNamespace(
                    inputEntries=input_entries,
                    outputEntries=output_entries,
                )
            )

        hit_policy = dt_el.get("hitPolicy") or "UNKNOWN"
        decision_table = SimpleNamespace(
            inputs=inputs,
            outputs=outputs,
            rules=rules,
            hitPolicy=hit_policy,
        )
        decision = SimpleNamespace(name=name, decisionTable=decision_table)
        decisions.append(decision)

    return SimpleNamespace(decisions=decisions)


class DMNImport:
    """Compatibility wrapper: same interface as dmn_python.DMNImport for DMNParser."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def importDMN(self) -> SimpleNamespace:
        return load_dmn(self.file_path)
