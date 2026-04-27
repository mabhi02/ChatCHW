"""Index flowchart + per-module flowcharts.

The index shows only module boxes and the routing edges between them;
each module links to its own flowchart PNG. Each per-module flowchart has
explicit Entry and Exit nodes naming its neighbours so a reviewer can
follow the graph without the index open.

Rendering reuses `backend.converters.mermaid_to_png.render_mermaid_to_png`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _as_list(obj) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        return [{**v, "_normalized_id": k} for k, v in obj.items() if isinstance(v, dict)]
    return []


def _id_of(item: dict) -> str:
    return str(item.get("id") or item.get("module_id") or item.get("_normalized_id") or "")


def _esc_label(s: str, max_len: int = 60) -> str:
    s = " ".join(str(s or "").split())
    if len(s) > max_len:
        s = s[: max_len - 1] + "..."
    return (s.replace('"', "'")
             .replace("|", "/")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("#", "No."))


def build_index_mermaid(clinical_logic: dict) -> str:
    """Top-level flowchart showing only modules + inter-module routes."""
    lines = ["flowchart TD"]
    modules_raw = clinical_logic.get("modules", {})
    modules = modules_raw if isinstance(modules_raw, dict) else {
        _id_of(m): m for m in _as_list(modules_raw)
    }
    lines.append('  Start(["Start"])')
    for mid, m in modules.items():
        if not mid:
            continue
        title = _esc_label(m.get("display_name") or m.get("title") or mid)
        lines.append(f'  {mid}["{title}"]')
    for mid, m in modules.items():
        if not mid:
            continue
        for route in (m.get("routes_to") or []):
            if isinstance(route, dict):
                tgt = str(route.get("target", "")).strip()
                cond = _esc_label(route.get("condition", ""), 40)
                if tgt and tgt in modules:
                    if cond:
                        lines.append(f"  {mid} -->|{cond}| {tgt}")
                    else:
                        lines.append(f"  {mid} --> {tgt}")
    return "\n".join(lines)


def build_module_mermaid(module: dict, all_modules: dict) -> str:
    """Per-module flowchart: decision nodes + explicit Entry/Exit bands."""
    mid = _id_of(module)
    lines = ["flowchart TD"]
    lines.append('  Entry(["Enter from index"])')
    first_rule_node = None
    for i, rule in enumerate(module.get("rules") or []):
        if not isinstance(rule, dict):
            continue
        rid = str(rule.get("id", rule.get("rule_id", f"r{i}")))
        cond = _esc_label(rule.get("condition", ""), 80)
        outputs = rule.get("outputs", {})
        out_summary = ", ".join(
            f"{k}={v}" for k, v in (outputs.items() if isinstance(outputs, dict) else [])
        )
        node_label = f"{rid}: {cond}"
        if out_summary:
            node_label += f" -> {_esc_label(out_summary, 60)}"
        lines.append(f'  {rid}{{"{node_label}"}}')
        if first_rule_node is None:
            first_rule_node = rid
    if first_rule_node:
        lines.append(f"  Entry --> {first_rule_node}")
    for route in (module.get("routes_to") or []):
        if isinstance(route, dict):
            tgt = str(route.get("target", "")).strip()
            cond = _esc_label(route.get("condition", ""), 40)
            if tgt:
                tgt_label = _esc_label(
                    (all_modules.get(tgt, {}) or {}).get("display_name", tgt)
                )
                exit_id = f"Exit_{tgt}"
                lines.append(f'  {exit_id}(["Exit to {tgt_label}<br>(if {cond})"])')
                if first_rule_node:
                    lines.append(f"  {first_rule_node} -->|{cond}| {exit_id}")
    return "\n".join(lines)


def emit_all(clinical_logic: dict, output_dir: Path) -> dict:
    """Write `flowchart_index.{md,png}` and `flowcharts_per_module/*`."""
    from backend.converters.mermaid_to_png import render_mermaid_to_png

    paths: dict[str, dict] = {}
    modules_raw = clinical_logic.get("modules", {})
    modules = modules_raw if isinstance(modules_raw, dict) else {
        _id_of(m): m for m in _as_list(modules_raw)
    }

    idx_md = output_dir / "flowchart_index.md"
    idx_src = build_index_mermaid(clinical_logic)
    idx_md.write_text(idx_src, encoding="utf-8")
    try:
        png_bytes, status = render_mermaid_to_png(idx_src)
        (output_dir / "flowchart_index.png").write_bytes(png_bytes)
        logger.info("mermaid_split: index rendered via %s", status)
    except Exception as exc:
        logger.warning("mermaid_split: index render failed: %s", exc)
    paths["index"] = {"md": idx_md, "png": output_dir / "flowchart_index.png"}

    modules_dir = output_dir / "flowcharts_per_module"
    modules_dir.mkdir(exist_ok=True)
    for mid, m in modules.items():
        if not isinstance(m, dict) or not mid:
            continue
        md_path = modules_dir / f"flowchart_{mid}.md"
        src = build_module_mermaid(m, modules)
        md_path.write_text(src, encoding="utf-8")
        try:
            png_bytes, status = render_mermaid_to_png(src)
            (modules_dir / f"flowchart_{mid}.png").write_bytes(png_bytes)
        except Exception as exc:
            logger.warning("mermaid_split: module %s render failed: %s", mid, exc)
        paths[f"mod_{mid}"] = {"md": md_path, "png": modules_dir / f"flowchart_{mid}.png"}

    return paths
