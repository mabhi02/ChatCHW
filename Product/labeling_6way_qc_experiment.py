"""Labeling 6-way + QC pass experiment.

Cell C4 = Sonnet 4.6, 6 parallel passes + 1 Sonnet QC pass on merged labels.
Compared against C1 (Opus monolithic+QC) and C3 (Sonnet 6-way, no QC)
from prior experiments.

5 chunks x 3 runs.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

OPUS_MODEL = "claude-opus-4-6"
SONNET_MODEL = "claude-sonnet-4-6"

REPO = Path(__file__).parent
RECONSTRUCTED = REPO / "backend/output/run_d9a39c2d/reconstructed_guide.txt"
CODEBOOK_FILE = REPO / "backend/system_prompt.py"
PRIOR_2WAY = REPO / "labeling_ab_results/raw_results.json"
PRIOR_6WAY = REPO / "labeling_6way_results/raw_results_c3.json"
OUT = REPO / "labeling_6way_qc_results"
OUT.mkdir(exist_ok=True)

PRICING = {
    OPUS_MODEL:   {"in": 5.00, "cache_read": 0.50, "cache_write": 6.25, "out": 25.00},
    SONNET_MODEL: {"in": 3.00, "cache_read": 0.30, "cache_write": 3.75, "out": 15.00},
}


def load_codebook() -> str:
    src = CODEBOOK_FILE.read_text(encoding="utf-8")
    m = re.search(r'NAMING_CODEBOOK\s*=\s*"""(.*?)"""', src, re.DOTALL)
    return m.group(1).strip()


def pick_chunks() -> list[dict]:
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
    return [{"chunk_id": f"c{i}", "source_index": idx,
             "text": chunks_all[idx], "n_chars": len(chunks_all[idx]),
             "est_tokens": len(chunks_all[idx]) // 4} for i, idx in enumerate(indices)]


def parse_json_robust(text: str) -> Any:
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


def cost_of(model, in_tok, cache_read, cache_write, out_tok):
    p = PRICING[model]
    return (in_tok * p["in"] + cache_read * p["cache_read"] + cache_write * p["cache_write"] + out_tok * p["out"]) / 1e6


async def call_anthropic(client, model, system, user, max_tokens=16384):
    t0 = time.time()
    resp = await client.messages.create(
        model=model, max_tokens=max_tokens, temperature=0.0,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": user}],
    )
    dt = time.time() - t0
    text = resp.content[0].text
    parsed = parse_json_robust(text)
    labels = parsed.get("labels", []) if isinstance(parsed, dict) else []
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cr = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    cw = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
    return {"labels": labels, "elapsed_s": dt, "in_tok": in_tok, "out_tok": out_tok,
            "cache_read": cr, "cache_write": cw, "cost": cost_of(model, in_tok, cr, cw, out_tok)}


# -----------------------------------------------------------------------
# 6-way pass prompts (same as prior experiment)
# -----------------------------------------------------------------------

def _pass_prompt(focus, types_desc, prefixes_desc, codebook):
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        f"Your job: read a chunk of clinical guide text and find EVERY {focus}.\n\n"
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


def build_pass_prompts(codebook):
    return [
        ("supply_list", _pass_prompt(
            "physical item the CHW must possess",
            "  supply_list: physical items, equipment, medications, test kits.\n"
            "    - 'consumable' subtype: medications, ORS packets, test kits, disposables\n"
            "    - 'equipment' subtype: thermometer, MUAC strap, timer, durable tools",
            "  supply_<item_name> = consumable\n  equip_<tool_name> = durable equipment", codebook)),
        ("variables", _pass_prompt(
            "runtime input the CHW collects, asks, observes, or measures during a visit",
            "  variables: runtime inputs from patient assessment.\n"
            "    - Questions asked to patient/caregiver\n    - Examination findings\n"
            "    - Quantitative measurements\n    - Lab/test results\n    - History and demographics",
            "  q_ = self-reported symptom/history\n  ex_ = observation/exam finding\n"
            "  v_<measurement>_<unit> = quantitative measurement\n  lab_ = test result\n"
            "  hx_ = history\n  demo_ = demographics\n  img_ = imaging (rare)", codebook)),
        ("predicates", _pass_prompt(
            "boolean threshold or computed flag derived from variables",
            "  predicates: boolean conditions computed from variables.\n"
            "    - Age-based cutoffs\n    - Vital sign thresholds\n"
            "    - Danger sign composites\n    - Test result interpretations",
            "  p_<descriptive_flag> (e.g. p_fast_breathing_2_12mo, p_fever_present)", codebook)),
        ("modules", _pass_prompt(
            "clinical decision module or condition topic the manual covers",
            "  modules: distinct clinical decision topics/conditions.\n"
            "    - Each major illness or assessment section = one module",
            "  mod_<topic> (e.g. mod_diarrhoea, mod_pneumonia, mod_fever)", codebook)),
        ("phrase_bank", _pass_prompt(
            "message, instruction, advice, treatment, or referral the CHW communicates or performs",
            "  phrase_bank: things the CHW says, advises, treats, or does.\n"
            "    - Advice, treatment instructions, medication dosing, referral messages, follow-up",
            "  m_ = message\n  adv_ = advice\n  tx_ = treatment instruction\n"
            "  rx_ = medication dosing\n  ref_ = referral\n  fu_ = follow-up", codebook)),
        ("router_integrative", _pass_prompt(
            "routing rule, triage decision, priority ordering, or cross-module interaction",
            "  router: routing/triage decisions, danger sign short-circuits, priority ordering.\n"
            "  integrative: comorbidity rules, cross-module interactions.",
            "  For router items, use type='router'\n  For integrative items, use type='integrative'", codebook)),
    ]


# -----------------------------------------------------------------------
# QC prompt — verifies merged labels from all 6 passes
# -----------------------------------------------------------------------

def build_qc_prompt(codebook):
    return (
        "You are a clinical label quality-control verifier. You examine candidate labels "
        "produced by multiple labeling passes and verify each is grounded in the source text.\n\n"
        "YOUR JOB:\n"
        "1. Verify each label's 'span' appears in the source text (verbatim or near-verbatim). "
        "DROP labels whose span cannot be found — even partial matches are not enough.\n"
        "2. Verify each label's 'id' follows the codebook prefix for its 'type'. Drop wrong-prefixed IDs.\n"
        "3. Drop hallucinations (generic placeholders, concepts not in the text).\n"
        "4. If the same clinical concept appears with two different types (e.g. as both a module and a "
        "phrase_bank entry), KEEP BOTH — this is intentional multi-type labeling, not a duplicate.\n"
        "5. If the same span+type appears twice with different IDs, keep the more descriptive ID and drop the other.\n"
        "6. DO NOT add new labels. Only verify, clean, and deduplicate.\n\n"
        f"CODEBOOK:\n{codebook}\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}'
    )


# -----------------------------------------------------------------------
# Cell C4: Sonnet 6-way + Sonnet QC
# -----------------------------------------------------------------------

async def run_c4(client, codebook, chunk, run_idx):
    passes = build_pass_prompts(codebook)
    user = f"CHUNK TO LABEL:\n{chunk['text']}"

    # 6 parallel labeling passes
    t0 = time.time()
    tasks = [call_anthropic(client, SONNET_MODEL, sys, user) for _, sys in passes]
    pass_results = await asyncio.gather(*tasks)
    t_label = time.time() - t0

    merged_pre_qc = []
    pass_details = []
    for (name, _), res in zip(passes, pass_results):
        merged_pre_qc.extend(res["labels"])
        pass_details.append({"name": name, "label_count": len(res["labels"]),
                             "cost": res["cost"], "elapsed_s": res["elapsed_s"]})

    label_cost = sum(r["cost"] for r in pass_results)

    # QC pass on merged labels
    qc_sys = build_qc_prompt(codebook)
    qc_user = (
        f"SOURCE TEXT:\n{chunk['text']}\n\n"
        f"CANDIDATE LABELS ({len(merged_pre_qc)} from 6 passes):\n"
        f"{json.dumps(merged_pre_qc, indent=1)}\n\n"
        "Verify each against source text. Drop ungrounded labels. Deduplicate same-span+same-type entries."
    )
    qc_res = await call_anthropic(client, SONNET_MODEL, qc_sys, qc_user)
    wall = time.time() - t0

    final = qc_res["labels"] if qc_res["labels"] else merged_pre_qc

    return {
        "cell": "C4", "chunk_id": chunk["chunk_id"], "run": run_idx,
        "labels": final,
        "labels_pre_qc": len(merged_pre_qc),
        "labels_post_qc": len(final),
        "qc_drop_count": len(merged_pre_qc) - len(final),
        "qc_drop_rate": (len(merged_pre_qc) - len(final)) / max(1, len(merged_pre_qc)),
        "passes": pass_details,
        "qc_cost": qc_res["cost"], "qc_elapsed_s": qc_res["elapsed_s"],
        "total_cost": label_cost + qc_res["cost"],
        "total_elapsed_s": wall,
    }


# -----------------------------------------------------------------------
# Scoring
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

def label_set(labels): return {(l.get("id",""), l.get("type","")) for l in labels if l.get("id")}
def codebook_mismatch(labels):
    if not labels: return 0.0
    return sum(1 for l in labels if l.get("type") in PREFIX_OK and not any(l.get("id","").startswith(p) for p in PREFIX_OK[l["type"]])) / len(labels)
def jaccard(a,b):
    u = a|b; return len(a&b)/len(u) if u else 0
def pairwise_jaccard(sets):
    if len(sets)<2: return 1.0
    return sum(jaccard(sets[i],sets[j]) for i in range(len(sets)) for j in range(i+1,len(sets))) / (len(sets)*(len(sets)-1)//2)
def grounded_rate(labels, text):
    if not labels: return 0.0
    t = text.lower()
    return sum(1 for l in labels if (l.get("span","") or "").strip().lower() in t) / len(labels)
def normalize(s): return ' '.join(s.lower().split())
def span_overlap(a,b):
    a_s,b_s = set(normalize(a).split()), set(normalize(b).split())
    return len(a_s&b_s)/min(len(a_s),len(b_s)) if a_s and b_s else 0
def fuzzy_recall(c1_runs, test_runs):
    c1_spans = {normalize(l.get("span","")) for r in c1_runs for l in r if normalize(l.get("span",""))}
    t_spans = {normalize(l.get("span","")) for r in test_runs for l in r if normalize(l.get("span",""))}
    if not c1_spans: return 0,0
    matched = sum(1 for s in c1_spans if max((span_overlap(s,t) for t in t_spans), default=0) > 0.6)
    return matched/len(c1_spans), len(t_spans - c1_spans)


async def main():
    api_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_KEY not set")

    codebook = load_codebook()
    chunks = pick_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # Load baselines
    with open(PRIOR_2WAY) as f: prior2 = json.load(f)
    with open(PRIOR_6WAY) as f: prior6 = json.load(f)

    c1_by = defaultdict(list)
    for r in prior2["results"]:
        if r.get("cell") == "C1" and "labels" in r: c1_by[r["chunk_id"]].append(r)
    c3_by = defaultdict(list)
    for r in prior6["results"]:
        if r.get("cell") == "C3" and "labels" in r: c3_by[r["chunk_id"]].append(r)

    print(f"C1 baseline: {sum(len(v) for v in c1_by.values())} runs")
    print(f"C3 baseline: {sum(len(v) for v in c3_by.values())} runs")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    RUNS = 3
    c4_results = []

    for chunk in chunks:
        print(f"\n=== Chunk {chunk['chunk_id']} ({chunk['est_tokens']} tokens) ===")
        for run_i in range(RUNS):
            print(f"  run {run_i+1}/{RUNS}: ", end="", flush=True)
            try:
                r = await run_c4(client, codebook, chunk, run_i)
                print(f"C4 [{r['labels_pre_qc']} -> {r['labels_post_qc']} labels "
                      f"(QC dropped {r['qc_drop_count']}, {r['qc_drop_rate']*100:.0f}%), "
                      f"${r['total_cost']:.3f}]")
                c4_results.append(r)
            except Exception as e:
                print(f"FAILED: {e}")
                c4_results.append({"cell": "C4", "chunk_id": chunk["chunk_id"], "run": run_i, "error": str(e)})
            with open(OUT / "raw_results_c4.json", "w") as f:
                json.dump({"chunks": chunks, "results": c4_results}, f, indent=2)

    # ------ Score ------
    c4_by = defaultdict(list)
    for r in c4_results:
        if "labels" in r: c4_by[r["chunk_id"]].append(r)

    print("\n\n=== SCORING: C1 vs C3 vs C4 ===\n")
    hdr = (f"{'chunk':6s} | {'C1_lab':7s} {'C3_lab':7s} {'C4_pre':7s} {'C4_post':7s} {'QC_drp':7s} | "
           f"{'C1_jac':7s} {'C3_jac':7s} {'C4_jac':7s} | "
           f"{'C3_fzr':7s} {'C4_fzr':7s} | "
           f"{'C1_mis':7s} {'C3_mis':7s} {'C4_mis':7s} | "
           f"{'C1_gnd':7s} {'C3_gnd':7s} {'C4_gnd':7s} | "
           f"{'C1_$':7s} {'C3_$':7s} {'C4_$':7s}")
    print(hdr)
    print("-" * len(hdr))

    totals = {c: {"labels":0,"cost":0,"jac":[],"mis":[],"gnd":[],"fzr":[]} for c in ["C1","C3","C4"]}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        c1r = c1_by.get(cid,[]); c3r = c3_by.get(cid,[]); c4r = c4_by.get(cid,[])
        if not c1r or not c4r: continue

        def s(runs, txt):
            sets = [label_set(r["labels"]) for r in runs]
            return (sum(len(r["labels"]) for r in runs)/len(runs),
                    pairwise_jaccard(sets),
                    sum(codebook_mismatch(r["labels"]) for r in runs)/len(runs),
                    sum(grounded_rate(r["labels"], txt) for r in runs)/len(runs),
                    sum(r.get("total_cost",0) for r in runs)/len(runs))

        c1s = s(c1r, chunk["text"])
        c3s = s(c3r, chunk["text"]) if c3r else (0,0,0,0,0)
        c4s = s(c4r, chunk["text"])

        c4_pre_avg = sum(r.get("labels_pre_qc",0) for r in c4r)/len(c4r)
        qc_drop_avg = sum(r.get("qc_drop_rate",0) for r in c4r)/len(c4r)

        c3_fzr,_ = fuzzy_recall([r["labels"] for r in c1r],[r["labels"] for r in c3r]) if c3r else (0,0)
        c4_fzr,_ = fuzzy_recall([r["labels"] for r in c1r],[r["labels"] for r in c4r])

        print(f"{cid:6s} | {c1s[0]:7.1f} {c3s[0]:7.1f} {c4_pre_avg:7.1f} {c4s[0]:7.1f} {qc_drop_avg*100:6.1f}% | "
              f"{c1s[1]:7.3f} {c3s[1]:7.3f} {c4s[1]:7.3f} | "
              f"{c3_fzr:7.3f} {c4_fzr:7.3f} | "
              f"{c1s[2]*100:6.1f}% {c3s[2]*100:6.1f}% {c4s[2]*100:6.1f}% | "
              f"{c1s[3]*100:6.1f}% {c3s[3]*100:6.1f}% {c4s[3]*100:6.1f}% | "
              f"${c1s[4]:6.3f} ${c3s[4]:6.3f} ${c4s[4]:6.3f}")

        for cell,st,fzr in [("C1",c1s,1.0),("C3",c3s,c3_fzr),("C4",c4s,c4_fzr)]:
            totals[cell]["labels"]+=st[0]; totals[cell]["cost"]+=st[4]
            totals[cell]["jac"].append(st[1]); totals[cell]["mis"].append(st[2])
            totals[cell]["gnd"].append(st[3]); totals[cell]["fzr"].append(fzr)

    avg = lambda xs: sum(xs)/len(xs) if xs else 0
    print(f"\n{'TOTAL':6s} | {totals['C1']['labels']:7.0f} {totals['C3']['labels']:7.0f} {'':7s} {totals['C4']['labels']:7.0f} {'':7s} | "
          f"{avg(totals['C1']['jac']):7.3f} {avg(totals['C3']['jac']):7.3f} {avg(totals['C4']['jac']):7.3f} | "
          f"{avg(totals['C3']['fzr']):7.3f} {avg(totals['C4']['fzr']):7.3f} | "
          f"{avg(totals['C1']['mis'])*100:6.1f}% {avg(totals['C3']['mis'])*100:6.1f}% {avg(totals['C4']['mis'])*100:6.1f}% | "
          f"{avg(totals['C1']['gnd'])*100:6.1f}% {avg(totals['C3']['gnd'])*100:6.1f}% {avg(totals['C4']['gnd'])*100:6.1f}% | "
          f"${totals['C1']['cost']:6.2f} ${totals['C3']['cost']:6.2f} ${totals['C4']['cost']:6.2f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump({"C1": dict(totals["C1"]), "C3": dict(totals["C3"]), "C4": dict(totals["C4"])}, f, indent=2)
    print(f"\nResults saved to {OUT}/")


if __name__ == "__main__":
    asyncio.run(main())
