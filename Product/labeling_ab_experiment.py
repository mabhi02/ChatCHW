"""Labeling A/B experiment.

Cells:
  C1 = Opus 4.6 monolithic + QC (current production setup)
  C2 = Sonnet 4.6, 2-way decomposition (input pass + output pass), no QC

5 chunks x 3 runs per cell. Measures recall vs C1, novel labels, codebook
compliance, within-cell Jaccard, cost, latency.
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
OUT = REPO / "labeling_ab_results"
OUT.mkdir(exist_ok=True)

# Pricing per million tokens (USD), 2026
PRICING = {
    OPUS_MODEL:   {"in": 5.00, "cache_read": 0.50, "cache_write": 6.25, "out": 25.00},
    SONNET_MODEL: {"in": 3.00, "cache_read": 0.30, "cache_write": 3.75, "out": 15.00},
}


def load_codebook() -> str:
    """Pull NAMING_CODEBOOK string out of backend/system_prompt.py."""
    src = CODEBOOK_FILE.read_text(encoding="utf-8")
    m = re.search(r'NAMING_CODEBOOK\s*=\s*"""(.*?)"""', src, re.DOTALL)
    if not m:
        raise RuntimeError("Could not find NAMING_CODEBOOK in system_prompt.py")
    return m.group(1).strip()


def pick_chunks() -> list[dict]:
    """Build 5 ~2K-token chunks from reconstructed_guide.txt.

    Splits on section markers `[TITLE]`, then greedily packs sections into
    chunks of ~8K chars (~2K tokens). Picks 5 chunks spread across the doc:
    early, mid-early, mid, mid-late, late.
    """
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

    # Pick 5 evenly spaced
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


# -----------------------------------------------------------------------
# Cell C1: Opus monolithic + QC (mirror of backend/gen7/labeler.py)
# -----------------------------------------------------------------------

def build_label_prompt_monolithic(codebook: str) -> str:
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and label EVERY clinical item "
        "with a structured annotation.\n\n"
        "For EVERY clinical item in the text, produce a label with:\n"
        "- span: the exact verbatim text from the chunk\n"
        "- id: a canonical ID using the codebook prefixes below. Descriptive, content-derived, never numbered.\n"
        "- type: one of the 7 artifact types: supply_list, variables, predicates, modules, phrase_bank, router, integrative\n"
        "- subtype: optional finer classification\n"
        "- quote_context: the surrounding sentence for provenance\n\n"
        "ARTIFACT TYPE MAPPING:\n"
        "  supply_list: physical items the CHW must possess (supply_=consumable, equip_=durable equipment)\n"
        "  variables: runtime inputs the CHW collects (q_, ex_, v_, lab_, hx_, demo_, img_)\n"
        "  predicates: boolean thresholds (p_)\n"
        "  modules: clinical decision topics (mod_)\n"
        "  phrase_bank: things the CHW says or does (m_, adv_, tx_, rx_, ref_)\n"
        "  router: routing/triage rows\n"
        "  integrative: comorbidity / cross-module rules\n\n"
        f"NAMING CODEBOOK:\n{codebook}\n\n"
        "CRITICAL RULES:\n"
        "1. Label EVERY clinical item. Missing items is worse than having extras.\n"
        "2. IDs must be descriptive and follow prefix conventions exactly.\n"
        "3. Numeric variables MUST end with unit suffix.\n"
        "4. Use lowercase_with_underscores only.\n"
        "5. Prefer over-labeling to under-labeling.\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}'
    )


def build_qc_prompt(codebook: str) -> str:
    return (
        "You are a clinical label quality-control verifier. You examine candidate labels "
        "and verify each is grounded in source text and codebook-compliant.\n\n"
        "YOUR JOB:\n"
        "1. Verify each label's 'span' appears in source text (verbatim or near-verbatim). "
        "Drop labels whose span cannot be found.\n"
        "2. Verify each label's 'id' follows codebook prefix for its 'type'. Correct or drop wrong-prefixed IDs.\n"
        "3. Drop hallucinations (generic placeholders, concepts not in text).\n"
        "4. DO NOT add new labels. Only verify/clean.\n\n"
        f"CODEBOOK:\n{codebook}\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [...]}'
    )


# -----------------------------------------------------------------------
# Cell C2: Sonnet 2-way decomposition
# Pass A = INPUT side: supply_list, variables, predicates
# Pass B = OUTPUT side: modules, phrase_bank, router, integrative
# -----------------------------------------------------------------------

def build_label_prompt_input_pass(codebook: str) -> str:
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and find EVERY item that is an "
        "INPUT to a clinical decision -- something the CHW collects, measures, has on "
        "hand, or computes from other inputs.\n\n"
        "Label items in these THREE artifact types ONLY:\n"
        "  supply_list: physical items the CHW must possess.\n"
        "    - supply_<name> = consumable that depletes (medication, test kit, ORS packet)\n"
        "    - equip_<name> = durable equipment (thermometer, MUAC strap, timer)\n"
        "  variables: runtime inputs the CHW collects from the patient.\n"
        "    - q_<symptom> = self-reported symptom or history from patient/caregiver\n"
        "    - ex_<finding> = clinician observation or examination finding\n"
        "    - v_<measurement>_<unit> = quantitative measurement (e.g. v_temp_c, v_breaths_per_min)\n"
        "    - lab_<test> = point-of-care test result\n"
        "    - hx_<condition> = baseline history / chronic condition\n"
        "    - demo_<attribute> = demographics (age, sex)\n"
        "    - img_<finding> = imaging finding (rare)\n"
        "  predicates: boolean thresholds computed from variables.\n"
        "    - p_<descriptive_name> (e.g. p_fast_breathing_2_12mo, p_danger_sign_present)\n\n"
        "DO NOT label anything else in this pass. Skip modules, phrase_bank, router, integrative -- those are handled separately.\n\n"
        f"NAMING CODEBOOK (for reference; only the input-side prefixes are relevant here):\n{codebook}\n\n"
        "For EVERY input-side clinical item, produce:\n"
        "- span: the exact verbatim text from the chunk\n"
        "- id: descriptive, content-derived, follows prefix conventions exactly\n"
        "- type: 'supply_list', 'variables', or 'predicates'\n"
        "- subtype: optional (e.g. 'consumable' vs 'equipment' for supply_list)\n"
        "- quote_context: surrounding sentence\n\n"
        "RULES:\n"
        "1. Numeric variables MUST end with unit suffix: _per_min, _c, _mm, _mg, _kg, _days.\n"
        "2. Use lowercase_with_underscores only.\n"
        "3. Prefer over-labeling. Downstream dedup handles duplicates.\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}'
    )


def build_label_prompt_output_pass(codebook: str) -> str:
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and find EVERY item that is an "
        "OUTPUT of a clinical decision -- a module the CHW runs, a message/instruction "
        "the CHW says or does, or a routing/integrative rule.\n\n"
        "Label items in these FOUR artifact types ONLY:\n"
        "  modules: clinical decision topics or conditions.\n"
        "    - mod_<topic> (e.g. mod_pneumonia, mod_diarrhoea, mod_fever)\n"
        "  phrase_bank: things the CHW says, advises, or does.\n"
        "    - m_<message> = generic message/phrase (m_advise_fluids, m_refer_urgently)\n"
        "    - adv_<topic> = advice / counselling text\n"
        "    - tx_<treatment> = treatment instruction\n"
        "    - rx_<drug>_<dose> = medication dosing instruction\n"
        "    - ref_<destination> = referral message\n"
        "  router: routing/triage decisions (danger sign short-circuits, priority ordering rows).\n"
        "  integrative: comorbidity rules, cross-module interactions.\n\n"
        "DO NOT label anything else in this pass. Skip supply_list, variables, predicates -- those are handled separately.\n\n"
        f"NAMING CODEBOOK (for reference; only the output-side prefixes are relevant here):\n{codebook}\n\n"
        "For EVERY output-side clinical item, produce:\n"
        "- span: the exact verbatim text from the chunk\n"
        "- id: descriptive, content-derived, follows prefix conventions exactly\n"
        "- type: 'modules', 'phrase_bank', 'router', or 'integrative'\n"
        "- subtype: optional (e.g. 'treatment' vs 'advice' for phrase_bank)\n"
        "- quote_context: surrounding sentence\n\n"
        "RULES:\n"
        "1. mod_ IDs must describe the clinical topic (mod_diarrhoea_treatment, not mod_5).\n"
        "2. Use lowercase_with_underscores only.\n"
        "3. Prefer over-labeling. Downstream dedup handles duplicates.\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON: {"labels": [{"span": "...", "id": "...", "type": "...", "subtype": "...", "quote_context": "..."}]}'
    )


# -----------------------------------------------------------------------
# JSON parsing (mirror of labeler.py _parse_json_robust)
# -----------------------------------------------------------------------

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
    # truncation repair
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
                if depth == 0:
                    last_complete = i + 1
            elif c == '"':
                i += 1
                while i < len(repaired) and repaired[i] != '"':
                    if repaired[i] == '\\':
                        i += 1
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
        "labels": labels,
        "raw_text": text,
        "elapsed_s": dt,
        "in_tok": in_tok, "out_tok": out_tok,
        "cache_read": cache_read, "cache_write": cache_write,
        "cost": cost_of(model, in_tok, cache_read, cache_write, out_tok),
    }


# -----------------------------------------------------------------------
# Cell runners
# -----------------------------------------------------------------------

async def run_c1_opus_monolithic(client, codebook: str, chunk: dict, run_idx: int) -> dict:
    label_sys = build_label_prompt_monolithic(codebook)
    qc_sys = build_qc_prompt(codebook)
    user_label = f"CHUNK TO LABEL:\n{chunk['text']}"
    label_res = await call_anthropic(client, OPUS_MODEL, label_sys, user_label)
    if not label_res["labels"]:
        return {"cell": "C1", "chunk_id": chunk["chunk_id"], "run": run_idx,
                "labels": [], "calls": [label_res], "total_cost": label_res["cost"],
                "total_elapsed_s": label_res["elapsed_s"]}
    user_qc = (
        f"SOURCE TEXT:\n{chunk['text']}\n\n"
        f"CANDIDATE LABELS ({len(label_res['labels'])}):\n{json.dumps(label_res['labels'], indent=1)}\n\n"
        "Verify, drop hallucinations, canonicalize IDs."
    )
    qc_res = await call_anthropic(client, OPUS_MODEL, qc_sys, user_qc)
    final_labels = qc_res["labels"] if qc_res["labels"] else label_res["labels"]
    return {
        "cell": "C1", "chunk_id": chunk["chunk_id"], "run": run_idx,
        "labels": final_labels,
        "calls": [{"name": "label", **label_res}, {"name": "qc", **qc_res}],
        "label_count_pre_qc": len(label_res["labels"]),
        "label_count_post_qc": len(final_labels),
        "total_cost": label_res["cost"] + qc_res["cost"],
        "total_elapsed_s": label_res["elapsed_s"] + qc_res["elapsed_s"],
    }


async def run_c2_sonnet_2way(client, codebook: str, chunk: dict, run_idx: int) -> dict:
    input_sys = build_label_prompt_input_pass(codebook)
    output_sys = build_label_prompt_output_pass(codebook)
    user = f"CHUNK TO LABEL:\n{chunk['text']}"

    # parallel passes
    t0 = time.time()
    in_res, out_res = await asyncio.gather(
        call_anthropic(client, SONNET_MODEL, input_sys, user),
        call_anthropic(client, SONNET_MODEL, output_sys, user),
    )
    wall = time.time() - t0

    merged = list(in_res["labels"]) + list(out_res["labels"])
    return {
        "cell": "C2", "chunk_id": chunk["chunk_id"], "run": run_idx,
        "labels": merged,
        "calls": [{"name": "input_pass", **in_res}, {"name": "output_pass", **out_res}],
        "label_count_input": len(in_res["labels"]),
        "label_count_output": len(out_res["labels"]),
        "total_cost": in_res["cost"] + out_res["cost"],
        "total_elapsed_s": wall,  # parallel: wall clock, not sum
    }


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------

INPUT_TYPES = {"supply_list", "variables", "predicates"}
OUTPUT_TYPES = {"modules", "phrase_bank", "router", "integrative"}
PREFIX_OK = {
    "supply_list": ("supply_", "equip_"),
    "variables": ("q_", "ex_", "v_", "lab_", "hx_", "demo_", "img_", "sys_", "prev_"),
    "predicates": ("p_",),
    "modules": ("mod_",),
    "phrase_bank": ("m_", "adv_", "tx_", "rx_", "ref_", "fu_", "out_", "proc_"),
    "router": ("router", "rt_"),
    "integrative": ("integrative", "int_"),
}


def label_set(labels: list[dict]) -> set:
    return {(l.get("id", ""), l.get("type", "")) for l in labels if l.get("id")}


def codebook_mismatch_rate(labels: list[dict]) -> float:
    if not labels:
        return 0.0
    bad = 0
    for l in labels:
        t = l.get("type", "")
        i = l.get("id", "")
        ok = PREFIX_OK.get(t, ())
        if ok and not any(i.startswith(p) for p in ok):
            bad += 1
    return bad / len(labels)


def jaccard(a: set, b: set) -> float:
    u = a | b
    if not u: return 0.0
    return len(a & b) / len(u)


def pairwise_jaccard(sets: list[set]) -> float:
    if len(sets) < 2: return 1.0
    pairs = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            pairs.append(jaccard(sets[i], sets[j]))
    return sum(pairs) / len(pairs)


def grounded_rate(labels: list[dict], chunk_text: str) -> float:
    """Fraction of labels whose span appears verbatim or near-verbatim in chunk."""
    if not labels: return 0.0
    text_norm = chunk_text.lower()
    grounded = 0
    for l in labels:
        span = (l.get("span", "") or "").strip().lower()
        if not span: continue
        if span in text_norm:
            grounded += 1
    return grounded / len(labels)


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------

async def main():
    api_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_KEY not set in env")

    codebook = load_codebook()
    chunks = pick_chunks()
    print(f"Loaded {len(chunks)} chunks:")
    for c in chunks:
        print(f"  {c['chunk_id']}: source_idx={c['source_index']:3d}  ~{c['est_tokens']} tokens")
    print(f"Codebook size: {len(codebook):,} chars")
    print()

    client = anthropic.AsyncAnthropic(api_key=api_key)

    RUNS_PER_CELL = 3
    all_results = []

    for chunk in chunks:
        print(f"\n=== Chunk {chunk['chunk_id']} ({chunk['est_tokens']} tokens) ===")
        for run_i in range(RUNS_PER_CELL):
            print(f"  run {run_i+1}/{RUNS_PER_CELL}: ", end="", flush=True)
            try:
                r1 = await run_c1_opus_monolithic(client, codebook, chunk, run_i)
                print(f"C1 [{len(r1['labels'])} labels, ${r1['total_cost']:.3f}] ", end="", flush=True)
                all_results.append(r1)
            except Exception as e:
                print(f"C1 FAILED: {e} ", end="", flush=True)
                all_results.append({"cell": "C1", "chunk_id": chunk["chunk_id"], "run": run_i, "error": str(e)})
            try:
                r2 = await run_c2_sonnet_2way(client, codebook, chunk, run_i)
                print(f"C2 [{len(r2['labels'])} labels, ${r2['total_cost']:.3f}]")
                all_results.append(r2)
            except Exception as e:
                print(f"C2 FAILED: {e}")
                all_results.append({"cell": "C2", "chunk_id": chunk["chunk_id"], "run": run_i, "error": str(e)})

            # Save incrementally
            with open(OUT / "raw_results.json", "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "results": all_results}, f, indent=2)

    # ------ Score ------
    print("\n\n=== SCORING ===\n")
    by_cell_chunk = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if "labels" not in r: continue
        by_cell_chunk[r["cell"]][r["chunk_id"]].append(r)

    summary = {"per_chunk": [], "overall": {}}

    cell_totals = defaultdict(lambda: {"cost": 0.0, "elapsed": 0.0, "labels": 0,
                                         "ground_rates": [], "compl_rates": [], "jaccards": []})

    for chunk in chunks:
        cid = chunk["chunk_id"]
        c1_runs = by_cell_chunk["C1"].get(cid, [])
        c2_runs = by_cell_chunk["C2"].get(cid, [])
        if not c1_runs or not c2_runs:
            continue

        c1_sets = [label_set(r["labels"]) for r in c1_runs]
        c2_sets = [label_set(r["labels"]) for r in c2_runs]

        c1_jaccard = pairwise_jaccard(c1_sets)
        c2_jaccard = pairwise_jaccard(c2_sets)

        # baseline = union of C1 runs (most permissive view of "ground truth")
        baseline = set()
        for s in c1_sets: baseline |= s

        c2_union = set()
        for s in c2_sets: c2_union |= s

        recall_c2_vs_c1 = len(c2_union & baseline) / max(1, len(baseline))
        novel_c2 = len(c2_union - baseline)

        c1_compl = sum(codebook_mismatch_rate(r["labels"]) for r in c1_runs) / len(c1_runs)
        c2_compl = sum(codebook_mismatch_rate(r["labels"]) for r in c2_runs) / len(c2_runs)
        c1_ground = sum(grounded_rate(r["labels"], chunk["text"]) for r in c1_runs) / len(c1_runs)
        c2_ground = sum(grounded_rate(r["labels"], chunk["text"]) for r in c2_runs) / len(c2_runs)

        c1_cost = sum(r["total_cost"] for r in c1_runs) / len(c1_runs)
        c2_cost = sum(r["total_cost"] for r in c2_runs) / len(c2_runs)
        c1_elapsed = sum(r["total_elapsed_s"] for r in c1_runs) / len(c1_runs)
        c2_elapsed = sum(r["total_elapsed_s"] for r in c2_runs) / len(c2_runs)
        c1_labels_avg = sum(len(r["labels"]) for r in c1_runs) / len(c1_runs)
        c2_labels_avg = sum(len(r["labels"]) for r in c2_runs) / len(c2_runs)

        per_chunk = {
            "chunk_id": cid,
            "est_tokens": chunk["est_tokens"],
            "C1_labels_avg": c1_labels_avg, "C2_labels_avg": c2_labels_avg,
            "C1_jaccard": c1_jaccard, "C2_jaccard": c2_jaccard,
            "C1_codebook_mismatch": c1_compl, "C2_codebook_mismatch": c2_compl,
            "C1_grounded_rate": c1_ground, "C2_grounded_rate": c2_ground,
            "C2_recall_vs_C1_union": recall_c2_vs_c1,
            "C2_novel_labels": novel_c2,
            "C1_cost_avg": c1_cost, "C2_cost_avg": c2_cost,
            "C1_elapsed_avg_s": c1_elapsed, "C2_elapsed_avg_s": c2_elapsed,
        }
        summary["per_chunk"].append(per_chunk)

        # accumulate
        cell_totals["C1"]["cost"] += c1_cost; cell_totals["C2"]["cost"] += c2_cost
        cell_totals["C1"]["elapsed"] += c1_elapsed; cell_totals["C2"]["elapsed"] += c2_elapsed
        cell_totals["C1"]["labels"] += c1_labels_avg; cell_totals["C2"]["labels"] += c2_labels_avg
        cell_totals["C1"]["compl_rates"].append(c1_compl); cell_totals["C2"]["compl_rates"].append(c2_compl)
        cell_totals["C1"]["ground_rates"].append(c1_ground); cell_totals["C2"]["ground_rates"].append(c2_ground)
        cell_totals["C1"]["jaccards"].append(c1_jaccard); cell_totals["C2"]["jaccards"].append(c2_jaccard)

    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    summary["overall"] = {
        "C1": {
            "total_cost": cell_totals["C1"]["cost"],
            "total_elapsed_s": cell_totals["C1"]["elapsed"],
            "total_labels": cell_totals["C1"]["labels"],
            "avg_codebook_mismatch": avg(cell_totals["C1"]["compl_rates"]),
            "avg_grounded_rate": avg(cell_totals["C1"]["ground_rates"]),
            "avg_within_jaccard": avg(cell_totals["C1"]["jaccards"]),
        },
        "C2": {
            "total_cost": cell_totals["C2"]["cost"],
            "total_elapsed_s": cell_totals["C2"]["elapsed"],
            "total_labels": cell_totals["C2"]["labels"],
            "avg_codebook_mismatch": avg(cell_totals["C2"]["compl_rates"]),
            "avg_grounded_rate": avg(cell_totals["C2"]["ground_rates"]),
            "avg_within_jaccard": avg(cell_totals["C2"]["jaccards"]),
        },
    }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== PER CHUNK ===")
    print(f"{'chunk':6s} {'C1_lab':7s} {'C2_lab':7s} {'C1_jac':7s} {'C2_jac':7s} {'C2_rec':7s} {'C2_nov':7s} {'C1_$':7s} {'C2_$':7s}")
    for pc in summary["per_chunk"]:
        print(f"{pc['chunk_id']:6s} "
              f"{pc['C1_labels_avg']:7.1f} {pc['C2_labels_avg']:7.1f} "
              f"{pc['C1_jaccard']:7.3f} {pc['C2_jaccard']:7.3f} "
              f"{pc['C2_recall_vs_C1_union']:7.3f} {pc['C2_novel_labels']:7d} "
              f"${pc['C1_cost_avg']:6.3f} ${pc['C2_cost_avg']:6.3f}")

    print("\n=== OVERALL ===")
    for cell, m in summary["overall"].items():
        print(f"{cell}: cost=${m['total_cost']:.2f}  elapsed={m['total_elapsed_s']:.0f}s  "
              f"labels={m['total_labels']:.0f}  jaccard={m['avg_within_jaccard']:.3f}  "
              f"compl_mismatch={m['avg_codebook_mismatch']*100:.1f}%  grounded={m['avg_grounded_rate']*100:.1f}%")

    print(f"\nResults saved to {OUT}/")


if __name__ == "__main__":
    asyncio.run(main())
