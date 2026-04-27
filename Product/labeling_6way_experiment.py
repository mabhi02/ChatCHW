"""Labeling 6-way decomposition experiment.

Cell C3 = Sonnet 4.6, 6 parallel passes (one per artifact type, router+integrative merged).
Compared against C1 (Opus monolithic+QC) from the prior experiment's raw_results.json.

5 chunks x 3 runs. Measures recall vs C1, novel labels, codebook compliance,
within-cell Jaccard, cost, latency.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

OPUS_MODEL = "claude-opus-4-6"
SONNET_MODEL = "claude-sonnet-4-6"

REPO = Path(__file__).parent
RECONSTRUCTED = REPO / "backend/output/run_d9a39c2d/reconstructed_guide.txt"
CODEBOOK_FILE = REPO / "backend/system_prompt.py"
PRIOR_RESULTS = REPO / "labeling_ab_results/raw_results.json"
OUT = REPO / "labeling_6way_results"
OUT.mkdir(exist_ok=True)

PRICING = {
    OPUS_MODEL:   {"in": 5.00, "cache_read": 0.50, "cache_write": 6.25, "out": 25.00},
    SONNET_MODEL: {"in": 3.00, "cache_read": 0.30, "cache_write": 3.75, "out": 15.00},
}


def load_codebook() -> str:
    src = CODEBOOK_FILE.read_text(encoding="utf-8")
    m = re.search(r'NAMING_CODEBOOK\s*=\s*"""(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find NAMING_CODEBOOK in system_prompt.py")
    return m.group(1).strip()


def pick_chunks() -> list[dict]:
    """Same chunk selection as the 2-way experiment for apples-to-apples."""
    text = RECONSTRUCTED.read_text(encoding="utf-8")
    sections = re.split(r'\n(?=\[[^\]]+\])', text)
    target_chars = 8000
    chunks_all = []
    cur = ""
    for sec in sections:
        if len(cur) + len(sec) < target_chars:
            cur += "\n" + sec
        else:
            if cur.strip():
                chunks_all.append(cur.strip())
            cur = sec
    if cur.strip():
        chunks_all.append(cur.strip())
    n = len(chunks_all)
    indices = [int(i * (n - 1) / 4) for i in range(5)]
    selected = []
    for i, idx in enumerate(indices):
        selected.append({
            "chunk_id": f"c{i}",
            "source_index": idx,
            "text": chunks_all[idx],
            "n_chars": len(chunks_all[idx]),
            "est_tokens": len(chunks_all[idx]) // 4,
        })
    return selected


def parse_json_robust(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    m = re.search(r'"labels"\s*:\s*\[', repaired)
    if m:
        start = m.end()
        last_complete = start
        depth = 0
        i = start
        while i < len(repaired):
            c = repaired[i]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: last_complete = i + 1
            elif c == '"':
                i += 1
                while i < len(repaired) and repaired[i] != '"':
                    if repaired[i] == '\\': i += 1
                    i += 1
            i += 1
        if last_complete > start:
            try:
                return json.loads(repaired[:last_complete].rstrip().rstrip(",") + "]}")
            except Exception:
                pass
    return None


def cost_of(model: str, in_tok: int, cache_read: int, cache_write: int, out_tok: int) -> float:
    p = PRICING[model]
    return (in_tok * p["in"] + cache_read * p["cache_read"] + cache_write * p["cache_write"] + out_tok * p["out"]) / 1e6


async def call_anthropic(client, model: str, system: str, user: str, max_tokens: int = 16384) -> dict:
    t0 = time.time()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": user}],
    )
    dt = time.time() - t0
    text = resp.content[0].text
    parsed = parse_json_robust(text)
    labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
    return {
        "labels": labels, "raw_text": text, "elapsed_s": dt,
        "in_tok": in_tok, "out_tok": out_tok,
        "cache_read": cache_read, "cache_write": cache_write,
        "cost": cost_of(model, in_tok, cache_read, cache_write, out_tok),
    }


# -----------------------------------------------------------------------
# 6-way pass prompts
# -----------------------------------------------------------------------

def _pass_prompt(focus_name: str, types_desc: str, prefixes_desc: str, codebook: str) -> str:
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        f"Your job: read a chunk of clinical guide text and find EVERY {focus_name}.\n\n"
        f"Label items of these types ONLY:\n{types_desc}\n\n"
        f"DO NOT label anything outside these types. Other categories are handled by separate passes.\n\n"
        f"Prefix conventions:\n{prefixes_desc}\n\n"
        f"NAMING CODEBOOK (for reference):\n{codebook}\n\n"
        "For EVERY matching item, produce:\n"
        "- span: exact verbatim text from the chunk\n"
        "- id: descriptive, content-derived, follows prefix conventions\n"
        "- type: one of the types listed above\n"
        "- subtype: optional finer classification\n"
        "- quote_context: surrounding sentence\n\n"
        "RULES:\n"
        "1. Numeric variables MUST end with unit suffix: _per_min, _c, _mm, _mg, _kg, _days.\n"
        "2. Use lowercase_with_underscores only.\n"
        "3. Prefer over-labeling. Downstream dedup handles duplicates.\n"
        "4. Label EVERY matching item. Missing items is worse than extras.\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}'
    )


def build_pass_prompts(codebook: str) -> list[tuple[str, str]]:
    """Returns [(pass_name, system_prompt), ...] for the 6 passes."""
    return [
        ("supply_list", _pass_prompt(
            "physical item the CHW must possess",
            "  supply_list: physical items, equipment, medications, test kits.\n"
            "    - 'consumable' subtype: medications, ORS packets, test kits, disposables\n"
            "    - 'equipment' subtype: thermometer, MUAC strap, timer, durable tools",
            "  supply_<item_name> = consumable (e.g. supply_ors_packets, supply_amoxicillin_250mg)\n"
            "  equip_<tool_name> = durable equipment (e.g. equip_thermometer, equip_muac_strap)",
            codebook,
        )),
        ("variables", _pass_prompt(
            "runtime input the CHW collects, asks, observes, or measures during a visit",
            "  variables: runtime inputs from patient assessment.\n"
            "    - Questions asked to patient/caregiver\n"
            "    - Examination findings (look, feel, listen)\n"
            "    - Quantitative measurements (temperature, respiratory rate)\n"
            "    - Lab/test results\n"
            "    - History and demographics",
            "  q_<symptom> = self-reported symptom/history from patient\n"
            "  ex_<finding> = clinician observation/examination finding\n"
            "  v_<measurement>_<unit> = quantitative measurement (e.g. v_temp_c, v_breaths_per_min)\n"
            "  lab_<test> = point-of-care test result\n"
            "  hx_<condition> = baseline history / chronic condition\n"
            "  demo_<attribute> = demographics (age, sex)\n"
            "  img_<finding> = imaging finding (rare)",
            codebook,
        )),
        ("predicates", _pass_prompt(
            "boolean threshold or computed flag derived from variables",
            "  predicates: boolean conditions computed from one or more variables.\n"
            "    - Age-based cutoffs (e.g. age 2 months up to 12 months)\n"
            "    - Vital sign thresholds (e.g. breathing rate >= 50 bpm)\n"
            "    - Danger sign composites (any danger sign present)\n"
            "    - Test result interpretations (e.g. RDT positive)",
            "  p_<descriptive_flag> (e.g. p_fast_breathing_2_12mo, p_fever_present, p_danger_sign)\n"
            "  Each predicate implies a threshold expression over source variables.",
            codebook,
        )),
        ("modules", _pass_prompt(
            "clinical decision module or condition topic the manual covers",
            "  modules: distinct clinical decision topics/conditions.\n"
            "    - Each major illness, condition, or assessment section = one module\n"
            "    - Includes: diarrhoea, fever, pneumonia/fast breathing, malnutrition, danger signs, etc.\n"
            "    - A module is a unit of clinical reasoning, not a single fact",
            "  mod_<topic> (e.g. mod_diarrhoea, mod_pneumonia, mod_fever, mod_malnutrition)\n"
            "  Module IDs describe the clinical topic, never numbered.",
            codebook,
        )),
        ("phrase_bank", _pass_prompt(
            "message, instruction, advice, treatment, or referral the CHW communicates or performs",
            "  phrase_bank: things the CHW says, advises, treats, or does.\n"
            "    - Advice to caregiver (fluids, feeding, bednet, when to return)\n"
            "    - Treatment instructions (give ORS, give zinc, give amoxicillin)\n"
            "    - Medication dosing (drug name, dose, frequency, duration)\n"
            "    - Referral messages (urgently refer, write referral note)\n"
            "    - Follow-up scheduling\n"
            "    - Counselling phrases",
            "  m_<message> = generic message/phrase\n"
            "  adv_<topic> = advice / counselling text\n"
            "  tx_<treatment> = treatment instruction\n"
            "  rx_<drug>_<dose> = medication dosing instruction\n"
            "  ref_<destination> = referral message\n"
            "  fu_<schedule> = follow-up scheduling",
            codebook,
        )),
        ("router_integrative", _pass_prompt(
            "routing rule, triage decision, priority ordering, or cross-module interaction",
            "  router: routing/triage decisions.\n"
            "    - Danger sign short-circuits (if danger sign, refer immediately)\n"
            "    - Priority ordering between modules\n"
            "    - Decision flow: treat at home vs refer\n"
            "  integrative: comorbidity rules, cross-module interactions.\n"
            "    - Rules about how modules interact\n"
            "    - Combined care plans",
            "  For router items, use type='router'\n"
            "  For integrative items, use type='integrative'",
            codebook,
        )),
    ]


async def run_c3_sonnet_6way(client, codebook: str, chunk: dict, run_idx: int) -> dict:
    passes = build_pass_prompts(codebook)
    user = f"CHUNK TO LABEL:\n{chunk['text']}"

    t0 = time.time()
    tasks = [call_anthropic(client, SONNET_MODEL, sys_prompt, user) for _, sys_prompt in passes]
    results = await asyncio.gather(*tasks)
    wall = time.time() - t0

    merged = []
    call_details = []
    for (pass_name, _), res in zip(passes, results):
        merged.extend(res["labels"])
        call_details.append({"name": pass_name, **res})

    total_cost = sum(r["cost"] for r in results)
    return {
        "cell": "C3", "chunk_id": chunk["chunk_id"], "run": run_idx,
        "labels": merged,
        "calls": call_details,
        "per_pass_counts": {name: len(res["labels"]) for (name, _), res in zip(passes, results)},
        "total_cost": total_cost,
        "total_elapsed_s": wall,
    }


# -----------------------------------------------------------------------
# Scoring (same as 2-way experiment)
# -----------------------------------------------------------------------

PREFIX_OK = {
    "supply_list": ("supply_", "equip_"),
    "variables": ("q_", "ex_", "v_", "lab_", "hx_", "demo_", "img_", "sys_", "prev_"),
    "predicates": ("p_",),
    "modules": ("mod_",),
    "phrase_bank": ("m_", "adv_", "tx_", "rx_", "ref_", "fu_", "out_", "proc_"),
    "router": ("router", "rt_"),
    "integrative": ("integrative", "int_"),
}


def label_set(labels): return {(l.get("id", ""), l.get("type", "")) for l in labels if l.get("id")}


def codebook_mismatch_rate(labels):
    if not labels: return 0.0
    bad = sum(1 for l in labels if l.get("type") in PREFIX_OK and not any(l.get("id", "").startswith(p) for p in PREFIX_OK[l["type"]]))
    return bad / len(labels)


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def pairwise_jaccard(sets):
    if len(sets) < 2: return 1.0
    pairs = [jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i+1, len(sets))]
    return sum(pairs) / len(pairs)


def grounded_rate(labels, chunk_text):
    if not labels: return 0.0
    text_norm = chunk_text.lower()
    return sum(1 for l in labels if (l.get("span", "") or "").strip().lower() in text_norm) / len(labels)


def normalize(s): return ' '.join(s.lower().split())


def span_overlap(a, b):
    a_set, b_set = set(normalize(a).split()), set(normalize(b).split())
    if not a_set or not b_set: return 0
    return len(a_set & b_set) / min(len(a_set), len(b_set))


def fuzzy_span_recall(c1_labels_all, test_labels_all):
    """What fraction of C1's unique spans are semantically covered by test's spans?"""
    c1_spans = {normalize(l.get("span", "")) for r in c1_labels_all for l in r if normalize(l.get("span", ""))}
    test_spans = {normalize(l.get("span", "")) for r in test_labels_all for l in r if normalize(l.get("span", ""))}
    if not c1_spans: return 0.0, 0, 0
    exact = len(c1_spans & test_spans)
    fuzzy_matched = sum(1 for s1 in c1_spans if max((span_overlap(s1, s2) for s2 in test_spans), default=0) > 0.6)
    novel = len(test_spans - c1_spans)
    return fuzzy_matched / len(c1_spans), novel, len(c1_spans)


async def main():
    api_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_KEY not set in env")

    codebook = load_codebook()
    chunks = pick_chunks()
    print(f"Loaded {len(chunks)} chunks:")
    for c in chunks:
        print(f"  {c['chunk_id']}: source_idx={c['source_index']:3d}  ~{c['est_tokens']} tokens")

    # Load C1 baseline from prior experiment
    with open(PRIOR_RESULTS) as f:
        prior = json.load(f)
    c1_by_chunk = defaultdict(list)
    for r in prior["results"]:
        if r.get("cell") == "C1" and "labels" in r:
            c1_by_chunk[r["chunk_id"]].append(r)
    print(f"Loaded C1 baseline: {sum(len(v) for v in c1_by_chunk.values())} runs across {len(c1_by_chunk)} chunks")

    # Also load C2 (2-way) for three-way comparison
    c2_by_chunk = defaultdict(list)
    for r in prior["results"]:
        if r.get("cell") == "C2" and "labels" in r:
            c2_by_chunk[r["chunk_id"]].append(r)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    RUNS = 3
    c3_results = []

    for chunk in chunks:
        print(f"\n=== Chunk {chunk['chunk_id']} ({chunk['est_tokens']} tokens) ===")
        for run_i in range(RUNS):
            print(f"  run {run_i+1}/{RUNS}: ", end="", flush=True)
            try:
                r = await run_c3_sonnet_6way(client, codebook, chunk, run_i)
                counts = r["per_pass_counts"]
                parts = " ".join(f"{k}={v}" for k, v in counts.items())
                print(f"C3 [{len(r['labels'])} labels, ${r['total_cost']:.3f}] ({parts})")
                c3_results.append(r)
            except Exception as e:
                print(f"C3 FAILED: {e}")
                c3_results.append({"cell": "C3", "chunk_id": chunk["chunk_id"], "run": run_i, "error": str(e)})
            # Save incrementally
            with open(OUT / "raw_results_c3.json", "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "results": c3_results}, f, indent=2)

    # ------ Score: C1 vs C2 vs C3 ------
    print("\n\n=== SCORING: C1 (Opus mono) vs C2 (Sonnet 2-way) vs C3 (Sonnet 6-way) ===\n")

    c3_by_chunk = defaultdict(list)
    for r in c3_results:
        if "labels" in r:
            c3_by_chunk[r["chunk_id"]].append(r)

    header = (f"{'chunk':6s} {'C1_lab':7s} {'C2_lab':7s} {'C3_lab':7s} | "
              f"{'C1_jac':7s} {'C2_jac':7s} {'C3_jac':7s} | "
              f"{'C2_fzr':7s} {'C3_fzr':7s} | "
              f"{'C1_mis':7s} {'C2_mis':7s} {'C3_mis':7s} | "
              f"{'C1_gnd':7s} {'C2_gnd':7s} {'C3_gnd':7s} | "
              f"{'C1_$':7s} {'C2_$':7s} {'C3_$':7s}")
    print(header)
    print("-" * len(header))

    totals = {c: {"cost": 0, "elapsed": 0, "labels": 0, "jac": [], "mis": [], "gnd": [], "fzr": []}
              for c in ["C1", "C2", "C3"]}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        c1r = c1_by_chunk.get(cid, [])
        c2r = c2_by_chunk.get(cid, [])
        c3r = c3_by_chunk.get(cid, [])
        if not c1r or not c3r: continue

        def stats(runs, chunk_text):
            sets = [label_set(r["labels"]) for r in runs]
            jac = pairwise_jaccard(sets)
            avg_labels = sum(len(r["labels"]) for r in runs) / len(runs)
            avg_mis = sum(codebook_mismatch_rate(r["labels"]) for r in runs) / len(runs)
            avg_gnd = sum(grounded_rate(r["labels"], chunk_text) for r in runs) / len(runs)
            avg_cost = sum(r.get("total_cost", 0) for r in runs) / len(runs)
            avg_elapsed = sum(r.get("total_elapsed_s", 0) for r in runs) / len(runs)
            return avg_labels, jac, avg_mis, avg_gnd, avg_cost, avg_elapsed

        c1s = stats(c1r, chunk["text"])
        c2s = stats(c2r, chunk["text"]) if c2r else (0, 0, 0, 0, 0, 0)
        c3s = stats(c3r, chunk["text"])

        c2_fzr, _, _ = fuzzy_span_recall([r["labels"] for r in c1r], [r["labels"] for r in c2r]) if c2r else (0, 0, 0)
        c3_fzr, c3_novel, c1_span_count = fuzzy_span_recall([r["labels"] for r in c1r], [r["labels"] for r in c3r])

        print(f"{cid:6s} {c1s[0]:7.1f} {c2s[0]:7.1f} {c3s[0]:7.1f} | "
              f"{c1s[1]:7.3f} {c2s[1]:7.3f} {c3s[1]:7.3f} | "
              f"{c2_fzr:7.3f} {c3_fzr:7.3f} | "
              f"{c1s[2]*100:6.1f}% {c2s[2]*100:6.1f}% {c3s[2]*100:6.1f}% | "
              f"{c1s[3]*100:6.1f}% {c2s[3]*100:6.1f}% {c3s[3]*100:6.1f}% | "
              f"${c1s[4]:6.3f} ${c2s[4]:6.3f} ${c3s[4]:6.3f}")

        for cell, s, fzr in [("C1", c1s, 1.0), ("C2", c2s, c2_fzr), ("C3", c3s, c3_fzr)]:
            totals[cell]["labels"] += s[0]
            totals[cell]["jac"].append(s[1])
            totals[cell]["mis"].append(s[2])
            totals[cell]["gnd"].append(s[3])
            totals[cell]["cost"] += s[4]
            totals[cell]["elapsed"] += s[5]
            totals[cell]["fzr"].append(fzr)

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    print(f"\n{'':6s} {'C1_lab':7s} {'C2_lab':7s} {'C3_lab':7s} | "
          f"{'C1_jac':7s} {'C2_jac':7s} {'C3_jac':7s} | "
          f"{'C2_fzr':7s} {'C3_fzr':7s} | "
          f"{'C1_mis':7s} {'C2_mis':7s} {'C3_mis':7s} | "
          f"{'C1_gnd':7s} {'C2_gnd':7s} {'C3_gnd':7s} | "
          f"{'C1_$':7s} {'C2_$':7s} {'C3_$':7s}")
    print(f"{'TOTAL':6s} "
          f"{totals['C1']['labels']:7.0f} {totals['C2']['labels']:7.0f} {totals['C3']['labels']:7.0f} | "
          f"{avg(totals['C1']['jac']):7.3f} {avg(totals['C2']['jac']):7.3f} {avg(totals['C3']['jac']):7.3f} | "
          f"{avg(totals['C2']['fzr']):7.3f} {avg(totals['C3']['fzr']):7.3f} | "
          f"{avg(totals['C1']['mis'])*100:6.1f}% {avg(totals['C2']['mis'])*100:6.1f}% {avg(totals['C3']['mis'])*100:6.1f}% | "
          f"{avg(totals['C1']['gnd'])*100:6.1f}% {avg(totals['C2']['gnd'])*100:6.1f}% {avg(totals['C3']['gnd'])*100:6.1f}% | "
          f"${totals['C1']['cost']:6.2f} ${totals['C2']['cost']:6.2f} ${totals['C3']['cost']:6.2f}")

    # Save summary
    summary = {
        "C1": {k: (avg(v) if isinstance(v, list) else v) for k, v in totals["C1"].items()},
        "C2": {k: (avg(v) if isinstance(v, list) else v) for k, v in totals["C2"].items()},
        "C3": {k: (avg(v) if isinstance(v, list) else v) for k, v in totals["C3"].items()},
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUT}/")


if __name__ == "__main__":
    asyncio.run(main())
