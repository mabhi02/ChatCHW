"""Divergence worklist builder.

Walks every `<artifact>.verification.json` under a run directory and
emits `divergence_worklist.json` at the run root. Rows are ranked by:
  1. severity (error > warn > info)
  2. frequency of the (artifact_kind, type) pair across the whole run
     (a divergence type that repeats across many artifacts gets higher
     priority than a one-off)

The worklist is the reviewer's starting point: errors first, then
patterns, then everything else. It does NOT edit the artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


_SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}


def build_worklist(
    output_dir: Path,
    extra_rows: list[dict] | None = None,
) -> Path:
    """Build the divergence worklist.

    `extra_rows` lets gen8 catcher modules inject divergences that did NOT
    come from the LLM verifier (stub predicates, raw-variable DMN refs,
    missing IMCI signs, etc). Each extra row must follow the same shape
    as a verifier divergence:
      {artifact_kind, type, severity, detail, evidence}
    plus an optional `artifact_path`.
    """
    rows: list[dict] = []
    for vpath in sorted(output_dir.rglob("*.verification.json")):
        try:
            v = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact_kind = v.get("artifact_kind", vpath.stem)
        artifact_path = str(vpath).removesuffix(".verification.json")
        for d in v.get("divergences", []) or []:
            rows.append({
                "artifact_kind": artifact_kind,
                "artifact_path": artifact_path,
                "type": d.get("type", ""),
                "severity": d.get("severity", "warn"),
                "detail": d.get("detail", ""),
                "evidence": d.get("evidence", {}),
            })

    # Catcher-injected divergences (stub predicates, raw vars in DMN, IMCI gaps).
    for row in (extra_rows or []):
        rows.append({
            "artifact_kind": row.get("artifact_kind", "catcher"),
            "artifact_path": row.get("artifact_path", ""),
            "type": row.get("type", ""),
            "severity": row.get("severity", "warn"),
            "detail": row.get("detail", ""),
            "evidence": row.get("evidence", {}),
        })

    freq = Counter((r["artifact_kind"], r["type"]) for r in rows)
    rows.sort(key=lambda r: (
        _SEVERITY_RANK.get(r["severity"], 99),
        -freq[(r["artifact_kind"], r["type"])],
        r["artifact_kind"],
        r["type"],
    ))

    summary = {
        "total_divergences": len(rows),
        "error_count": sum(1 for r in rows if r["severity"] == "error"),
        "warn_count": sum(1 for r in rows if r["severity"] == "warn"),
        "info_count": sum(1 for r in rows if r["severity"] == "info"),
        "top_patterns": [
            {"artifact_kind": k, "type": t, "count": c}
            for (k, t), c in freq.most_common(10)
        ],
    }

    out = output_dir / "divergence_worklist.json"
    out.write_text(
        json.dumps({"summary": summary, "ordered_worklist": rows}, indent=2),
        encoding="utf-8",
    )
    write_failures_md(output_dir, summary, rows)
    return out


def write_failures_md(output_dir: Path, summary: dict, rows: list[dict]) -> Path:
    """Render the worklist as a human-readable FAILURES.md.

    Emits only error- and warn-severity rows. Info rows live in the JSON
    worklist for completeness but the markdown is meant to be the
    reviewer's first stop, so we keep it short.
    """
    lines: list[str] = []
    lines.append("# Verifier Failures")
    lines.append("")
    lines.append("Auto-generated from `divergence_worklist.json`. Errors first, then warnings ranked by frequency.")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|---|---|")
    lines.append(f"| Errors | {summary.get('error_count', 0)} |")
    lines.append(f"| Warnings | {summary.get('warn_count', 0)} |")
    lines.append(f"| Info | {summary.get('info_count', 0)} |")
    lines.append("")

    actionable = [r for r in rows if r["severity"] in ("error", "warn")]
    if not actionable:
        lines.append("**No errors or warnings.** Verifier agreed with every artifact.")
    else:
        # Errors first
        errors = [r for r in actionable if r["severity"] == "error"]
        warns = [r for r in actionable if r["severity"] == "warn"]
        if errors:
            lines.append("## Errors (block release)")
            lines.append("")
            for r in errors:
                lines.append(f"- **{r['artifact_kind']} / {r['type']}**")
                lines.append(f"  - {r['detail']}")
                if r.get("evidence"):
                    lines.append(f"  - evidence: `{json.dumps(r['evidence'], default=str)[:300]}`")
            lines.append("")
        if warns:
            lines.append("## Warnings (review recommended)")
            lines.append("")
            for r in warns:
                lines.append(f"- **{r['artifact_kind']} / {r['type']}**")
                lines.append(f"  - {r['detail']}")
            lines.append("")

    out = output_dir / "FAILURES.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
