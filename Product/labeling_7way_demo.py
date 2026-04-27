"""7-way labeling demo: run on 1 chunk, produce final clean enriched JSON.

Demonstrates the full pipeline:
  1. 7 parallel Sonnet passes (one per artifact type)
  2. Merge all labels
  3. Reconcile: dedup exact (id,type), tiebreak intra-type naming conflicts,
     drop ungrounded spans
  4. Output 1 clean enriched JSON per chunk
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import anthropic
from dotenv import load_dotenv

load_dotenv()

SONNET_MODEL = "claude-sonnet-4-6"
REPO = Path(__file__).parent
RECONSTRUCTED = REPO / "backend/output/run_d9a39c2d/reconstructed_guide.txt"
CODEBOOK_FILE = REPO / "backend/system_prompt.py"
OUT = REPO / "labeling_7way_demo"
OUT.mkdir(exist_ok=True)


def load_codebook() -> str:
    src = CODEBOOK_FILE.read_text(encoding="utf-8")
    m = re.search(r'NAMING_CODEBOOK\s*=\s*"""(.*?)"""', src, re.DOTALL)
    return m.group(1).strip()


def pick_chunk() -> dict:
    """Pick chunk index 8 (dense clinical content, ~2K tokens)."""
    text = RECONSTRUCTED.read_text(encoding="utf-8")
    sections = re.split(r'\n(?=\[[^\]]+\])', text)
    target_chars = 8000
    chunks = []
    cur = ""
    for sec in sections:
        if len(cur) + len(sec) < target_chars:
            cur += "\n" + sec
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = sec
    if cur.strip(): chunks.append(cur.strip())
    idx = 8  # dense clinical chunk (c3 from experiments)
    return {"chunk_index": idx, "text": chunks[idx],
            "est_tokens": len(chunks[idx]) // 4,
            "section_id": "dense_clinical_chunk",
            "section_title": "Dense Clinical Chunk (index 8)"}


def parse_json_robust(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try: return json.loads(repaired)
    except json.JSONDecodeError: pass
    m = re.search(r'"labels"\s*:\s*\[', repaired)
    if m:
        start = m.end(); last = start; depth = 0; i = start
        while i < len(repaired):
            c = repaired[i]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: last = i + 1
            elif c == '"':
                i += 1
                while i < len(repaired) and repaired[i] != '"':
                    if repaired[i] == '\\': i += 1
                    i += 1
            i += 1
        if last > start:
            try: return json.loads(repaired[:last].rstrip().rstrip(",") + "]}")
            except Exception: pass
    return None


async def call_anthropic(client, system, user):
    resp = await client.messages.create(
        model=SONNET_MODEL, max_tokens=16384, temperature=0.0,
        system=[{"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text
    parsed = parse_json_robust(text)
    labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
    return {
        "labels": labels,
        "in_tok": resp.usage.input_tokens,
        "out_tok": resp.usage.output_tokens,
        "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
    }


# -----------------------------------------------------------------------
# 7 pass prompts — one per artifact type
# -----------------------------------------------------------------------

def _pass(focus, exact_type, types_desc, prefixes_desc, codebook):
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        f"Your job: read a chunk and find EVERY {focus}.\n\n"
        f"What to look for:\n{types_desc}\n\n"
        "DO NOT label anything outside this category.\n\n"
        f"Prefix conventions:\n{prefixes_desc}\n\n"
        f"NAMING CODEBOOK (reference):\n{codebook}\n\n"
        "For EVERY matching item produce:\n"
        "- span: exact verbatim text from the chunk\n"
        "- id: descriptive, content-derived, correct prefix\n"
        f"- type: ALWAYS use exactly \"{exact_type}\" (this exact string, no variations)\n"
        "- subtype: optional finer classification\n"
        "- quote_context: surrounding sentence\n\n"
        "RULES: label EVERYTHING matching. Over-label. lowercase_with_underscores.\n"
        "Numeric vars end with unit suffix (_per_min, _c, _mg, _days).\n\n"
        'OUTPUT: valid JSON only: {"labels": [...]}'
    )


def build_7_passes(codebook):
    return [
        ("supply_list", _pass(
            "physical item the CHW must possess (equipment, medication, test kit, consumable)",
            "supply_list",
            "  Physical items: medications, ORS packets, test kits, disposables (consumable)\n"
            "  and thermometer, MUAC strap, timer, durable tools (equipment)",
            "  supply_<item> = consumable\n  equip_<tool> = durable equipment",
            codebook)),
        ("variables", _pass(
            "runtime input the CHW collects, asks, observes, or measures",
            "variables",
            "  Inputs from patient assessment: questions to patient/caregiver,\n"
            "  exam findings (look, feel, listen), quantitative measurements,\n"
            "  lab/test results, history, demographics",
            "  q_ = question/symptom\n  ex_ = exam finding\n  v_ = measurement\n"
            "  lab_ = test result\n  hx_ = history\n  demo_ = demographics\n  img_ = imaging",
            codebook)),
        ("predicates", _pass(
            "boolean threshold or computed flag derived from variables",
            "predicates",
            "  Boolean conditions: age cutoffs, vital sign thresholds,\n"
            "  danger sign composites, test result interpretations",
            "  p_<flag> (e.g. p_fast_breathing_2_12mo, p_fever_present)",
            codebook)),
        ("modules", _pass(
            "clinical decision module or condition topic",
            "modules",
            "  Clinical topics/conditions: each illness, condition, or\n"
            "  assessment section = one module (diarrhoea, fever, pneumonia, etc.)",
            "  mod_<topic> (e.g. mod_diarrhoea, mod_pneumonia)",
            codebook)),
        ("phrase_bank", _pass(
            "message, instruction, advice, treatment, or referral the CHW communicates",
            "phrase_bank",
            "  CHW communication/actions: advice (fluids, feeding, bednet),\n"
            "  treatment instructions, medication dosing, referral messages,\n"
            "  follow-up scheduling, counselling phrases",
            "  m_ = message\n  adv_ = advice\n  tx_ = treatment\n"
            "  rx_ = medication dosing\n  ref_ = referral\n  fu_ = follow-up",
            codebook)),
        ("router", _pass(
            "routing rule, triage decision, or priority ordering",
            "router",
            "  Routing/triage decisions: danger sign short-circuits,\n"
            "  priority ordering between conditions, treat-at-home vs refer",
            "  router_<description> for all items",
            codebook)),
        ("integrative", _pass(
            "cross-module interaction, comorbidity rule, or combined care plan rule",
            "integrative",
            "  Rules about how modules interact: comorbidity handling,\n"
            "  combined referral policies, cross-condition treatment interactions",
            "  integrative_<description> for all items",
            codebook)),
    ]


# -----------------------------------------------------------------------
# Reconciliation: merge 7 pass outputs into 1 clean JSON
# -----------------------------------------------------------------------

PREFIX_OK = {
    "supply_list": ("supply_", "equip_"),
    "variables": ("q_", "ex_", "v_", "lab_", "hx_", "demo_", "img_", "sys_", "prev_"),
    "predicates": ("p_",),
    "modules": ("mod_",),
    "phrase_bank": ("m_", "adv_", "tx_", "rx_", "ref_", "fu_", "out_", "proc_", "dx_", "sev_"),
    "router": ("router", "rt_", "route_"),
    "integrative": ("integrative", "int_", "comorbid"),
}


def reconcile(labels: list[dict], chunk_text: str) -> dict:
    """Merge and clean labels from 7 passes into a single enriched structure.

    Returns:
      {
        "labels": [...],              # flat deduped list
        "by_type": {type: [...]},     # grouped by artifact type
        "cross_type_spans": [...],    # spans that appear in 2+ types (enrichment)
        "reconciliation_stats": {...} # what was done
      }
    """
    text_lower = chunk_text.lower()
    stats = {"raw": len(labels), "exact_dups_removed": 0,
             "ungrounded_removed": 0, "misprefixed_removed": 0,
             "naming_conflicts_resolved": 0}

    # Step 1: drop ungrounded spans (deterministic, no LLM)
    grounded = []
    for l in labels:
        span = (l.get("span", "") or "").strip()
        if not span:
            stats["ungrounded_removed"] += 1
            continue
        if span.lower() in text_lower:
            grounded.append(l)
        else:
            # fuzzy: check if >80% of words match
            words = span.lower().split()
            if words and sum(1 for w in words if w in text_lower) / len(words) >= 0.8:
                grounded.append(l)
            else:
                stats["ungrounded_removed"] += 1

    # Step 2: drop misprefixed labels
    prefix_ok = []
    for l in grounded:
        t = l.get("type", "")
        i = l.get("id", "")
        ok_prefixes = PREFIX_OK.get(t)
        if ok_prefixes and not any(i.startswith(p) for p in ok_prefixes):
            stats["misprefixed_removed"] += 1
        else:
            prefix_ok.append(l)

    # Step 3: exact (id, type) dedup — keep first seen, merge provenance
    seen = {}
    for l in prefix_ok:
        key = (l["id"], l["type"])
        if key not in seen:
            seen[key] = dict(l)
        else:
            stats["exact_dups_removed"] += 1
            # merge quote_context if different
            existing_qc = seen[key].get("quote_context", "")
            new_qc = l.get("quote_context", "")
            if new_qc and new_qc != existing_qc:
                seen[key]["quote_context"] = existing_qc

    # Step 4: intra-type naming conflict resolution
    # Group by (normalized_span, type) — if multiple IDs, keep shorter
    span_type_groups = defaultdict(list)
    for key, l in seen.items():
        norm_span = ' '.join((l.get("span", "") or "").lower().split())
        span_type_groups[(norm_span, l["type"])].append(key)

    drop_keys = set()
    for (span, typ), keys in span_type_groups.items():
        if len(keys) > 1:
            # keep the shortest ID (most canonical), drop others
            keys_sorted = sorted(keys, key=lambda k: len(k[0]))
            for k in keys_sorted[1:]:
                drop_keys.add(k)
                stats["naming_conflicts_resolved"] += 1

    deduped = [l for key, l in seen.items() if key not in drop_keys]

    # Step 5: sort by type then id for stable output
    type_order = ["supply_list", "variables", "predicates", "modules",
                  "phrase_bank", "router", "integrative"]
    type_rank = {t: i for i, t in enumerate(type_order)}
    deduped.sort(key=lambda l: (type_rank.get(l.get("type", ""), 99), l.get("id", "")))

    # Step 6: group by type
    by_type = defaultdict(list)
    for l in deduped:
        by_type[l["type"]].append(l)

    # Step 7: identify cross-type spans (enrichment metadata)
    span_types = defaultdict(set)
    for l in deduped:
        norm = ' '.join((l.get("span", "") or "").lower().split())
        span_types[norm].add(l["type"])
    cross_type = [{"span": s, "types": sorted(t)} for s, t in span_types.items() if len(t) > 1]
    cross_type.sort(key=lambda x: x["span"])

    stats["final"] = len(deduped)
    stats["cross_type_spans"] = len(cross_type)

    return {
        "labels": deduped,
        "by_type": {t: by_type[t] for t in type_order if t in by_type},
        "cross_type_spans": cross_type,
        "reconciliation_stats": stats,
    }


async def main():
    api_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_KEY not set")

    codebook = load_codebook()
    chunk = pick_chunk()
    print(f"Chunk: index={chunk['chunk_index']}, ~{chunk['est_tokens']} tokens")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    passes = build_7_passes(codebook)
    user_msg = f"CHUNK TO LABEL:\n{chunk['text']}"

    # ---- Phase 1: 7 parallel Sonnet calls ----
    print(f"\nRunning 7 parallel Sonnet passes...")
    t0 = time.time()
    tasks = [call_anthropic(client, sys_prompt, user_msg) for _, sys_prompt in passes]
    results = await asyncio.gather(*tasks)
    wall = time.time() - t0

    raw_labels = []
    total_cost = 0
    print(f"\n{'Pass':<20s} {'Labels':>7s} {'In tok':>8s} {'Out tok':>8s}")
    print("-" * 50)
    for (name, _), res in zip(passes, results):
        raw_labels.extend(res["labels"])
        c = (res["in_tok"] * 3 + res["cache_read"] * 0.3 + res["cache_write"] * 3.75 + res["out_tok"] * 15) / 1e6
        total_cost += c
        print(f"{name:<20s} {len(res['labels']):>7d} {res['in_tok']:>8d} {res['out_tok']:>8d}")
    print(f"{'TOTAL':<20s} {len(raw_labels):>7d}")
    print(f"\nWall clock: {wall:.1f}s, Cost: ${total_cost:.3f}")

    # ---- Phase 2: Reconcile ----
    print(f"\nReconciling {len(raw_labels)} raw labels...")
    enriched = reconcile(raw_labels, chunk["text"])
    stats = enriched["reconciliation_stats"]
    print(f"  Raw:                    {stats['raw']}")
    print(f"  Ungrounded removed:     {stats['ungrounded_removed']}")
    print(f"  Misprefixed removed:    {stats['misprefixed_removed']}")
    print(f"  Exact dups removed:     {stats['exact_dups_removed']}")
    print(f"  Naming conflicts fixed: {stats['naming_conflicts_resolved']}")
    print(f"  Cross-type spans:       {stats['cross_type_spans']}")
    print(f"  FINAL clean labels:     {stats['final']}")

    # ---- Phase 3: Build final enriched chunk JSON ----
    final_json = {
        "chunk_index": chunk["chunk_index"],
        "section_id": chunk["section_id"],
        "section_title": chunk["section_title"],
        "text": chunk["text"],
        "est_tokens": chunk["est_tokens"],
        "labeling": {
            "model": SONNET_MODEL,
            "method": "7-way-decomposed",
            "passes": 7,
            "wall_clock_s": round(wall, 1),
            "cost_usd": round(total_cost, 4),
        },
        "labels": enriched["labels"],
        "labels_by_type": enriched["by_type"],
        "cross_type_spans": enriched["cross_type_spans"],
        "reconciliation_stats": stats,
    }

    out_path = OUT / "chunk_enriched.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"\nFinal JSON: {out_path} ({os.path.getsize(out_path):,} bytes)")

    # ---- Phase 4: Validation ----
    print("\n=== VALIDATION ===")
    errors = []

    # All labels have required fields
    for l in enriched["labels"]:
        for field in ["span", "id", "type"]:
            if not l.get(field):
                errors.append(f"Missing {field} in label: {l}")

    # No duplicate (id, type) pairs
    id_types = [(l["id"], l["type"]) for l in enriched["labels"]]
    if len(id_types) != len(set(id_types)):
        from collections import Counter
        dups = [k for k, v in Counter(id_types).items() if v > 1]
        errors.append(f"Duplicate (id,type) pairs after dedup: {dups}")

    # All types are valid
    valid_types = {"supply_list", "variables", "predicates", "modules",
                   "phrase_bank", "router", "integrative"}
    for l in enriched["labels"]:
        if l["type"] not in valid_types:
            errors.append(f"Invalid type '{l['type']}' on label {l['id']}")

    # All prefixes correct
    for l in enriched["labels"]:
        ok = PREFIX_OK.get(l["type"])
        if ok and not any(l["id"].startswith(p) for p in ok):
            errors.append(f"Wrong prefix: {l['id']} for type {l['type']}")

    # All IDs lowercase_with_underscores
    for l in enriched["labels"]:
        if not re.match(r'^[a-z0-9_]+$', l["id"]):
            errors.append(f"Non-lowercase ID: {l['id']}")

    if errors:
        print(f"  FAILED: {len(errors)} errors")
        for e in errors[:10]:
            print(f"    - {e}")
    else:
        print("  PASSED: all validation checks")

    # ---- Show sample of final output ----
    print(f"\n=== FINAL JSON STRUCTURE ===")
    print(f"Top-level keys: {list(final_json.keys())}")
    print(f"\nlabels_by_type counts:")
    for t, items in final_json["labels_by_type"].items():
        print(f"  {t}: {len(items)}")
    print(f"\nSample labels (first 2 per type):")
    for t, items in final_json["labels_by_type"].items():
        print(f"\n  [{t}]")
        for l in items[:2]:
            print(f"    {l['id']:40s}  span=\"{l['span'][:50]}\"")

    print(f"\nCross-type spans (first 5):")
    for ct in enriched["cross_type_spans"][:5]:
        print(f"  \"{ct['span'][:60]}\" -> {ct['types']}")


if __name__ == "__main__":
    asyncio.run(main())
