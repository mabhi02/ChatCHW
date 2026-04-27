"""Cross-model labeling comparison: Opus 4.6 vs GPT-5.4.

Same monolithic labeling prompt, same chunk, both models.
Shows: label count, codebook compliance, grounding, semantic overlap,
and a detailed diff of what each model finds that the other doesn't.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

OPUS_MODEL = "claude-opus-4-6"
GPT_MODEL = "gpt-5.4"

REPO = Path(__file__).parent
RECONSTRUCTED = REPO / "backend/output/run_d9a39c2d/reconstructed_guide.txt"
CODEBOOK_FILE = REPO / "backend/system_prompt.py"
OUT = REPO / "labeling_cross_model"
OUT.mkdir(exist_ok=True)


def load_codebook():
    src = CODEBOOK_FILE.read_text(encoding="utf-8")
    m = re.search(r'NAMING_CODEBOOK\s*=\s*"""(.*?)"""', src, re.DOTALL)
    return m.group(1).strip()


def pick_chunk():
    text = RECONSTRUCTED.read_text(encoding="utf-8")
    sections = re.split(r'\n(?=\[[^\]]+\])', text)
    target = 8000; chunks = []; cur = ""
    for sec in sections:
        if len(cur) + len(sec) < target: cur += "\n" + sec
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = sec
    if cur.strip(): chunks.append(cur.strip())
    idx = 8
    return {"chunk_index": idx, "text": chunks[idx], "est_tokens": len(chunks[idx]) // 4}


def build_monolithic_prompt(codebook):
    """Same prompt that current production Opus gets."""
    return (
        "You are a clinical document labeler for WHO Community Health Worker manuals.\n"
        "Your job: read a chunk of clinical guide text and label EVERY clinical item "
        "with a structured annotation.\n\n"
        "For EVERY clinical item in the text, produce a label with:\n"
        "- span: the exact verbatim text from the chunk (copy-paste, no paraphrase)\n"
        "- id: a canonical ID using the codebook prefixes below. IDs must be DESCRIPTIVE "
        "and content-derived, never numbered. "
        "GOOD: q_has_cough, p_fast_breathing_2mo, supply_amoxicillin_250mg, equip_thermometer. "
        "BAD: q_1, p_2, supply_3.\n"
        "- type: one of the 7 artifact types: supply_list, variables, predicates, modules, "
        "phrase_bank, router, integrative\n"
        "- subtype: optional finer classification (e.g. 'equipment' vs 'consumable' for supply_list)\n"
        "- quote_context: the surrounding sentence for provenance\n\n"
        "ARTIFACT TYPE MAPPING:\n"
        "  supply_list: physical items the CHW must possess.\n"
        "    - 'consumable' (supply_ prefix): medications, test kits, disposables that deplete\n"
        "    - 'equipment' (equip_ prefix): durable tools reused across patients\n"
        "  variables: runtime inputs the CHW collects during a visit.\n"
        "    - q_ = self-reported symptoms/history from patient\n"
        "    - ex_ = clinician observation/examination finding\n"
        "    - v_ = quantitative measurement requiring equipment\n"
        "    - lab_ = point-of-care test result\n"
        "    - hx_ = baseline history/chronic conditions\n"
        "    - demo_ = demographics\n"
        "  predicates: boolean thresholds computed from variables.\n"
        "    - p_ = threshold flag (e.g. p_fast_breathing when breaths_per_min >= 50)\n"
        "  modules: clinical decision topics/conditions.\n"
        "    - mod_ = decision module (e.g. mod_pneumonia, mod_diarrhoea)\n"
        "  phrase_bank: things the CHW says or advises.\n"
        "    - m_ = message/phrase (e.g. m_advise_fluids, m_refer_urgently)\n"
        "    - adv_ = advice/counseling text\n"
        "    - tx_ = treatment instruction\n"
        "    - rx_ = medication dosing instruction\n"
        "    - ref_ = referral message\n"
        "  router: routing/triage decisions (danger sign short-circuits, priority ordering)\n"
        "  integrative: comorbidity rules, cross-module interactions\n\n"
        "LABEL EVERYTHING you find:\n"
        "- Medications with dosages\n- Equipment and supplies\n"
        "- Questions to ask the patient/caregiver\n- Examination steps\n"
        "- Vital sign thresholds and cutoffs\n- Age-based cutoffs\n"
        "- Danger signs and referral criteria\n- Classification/diagnosis labels\n"
        "- Treatment instructions and dosing\n- Advice and counseling phrases\n"
        "- Follow-up scheduling\n- Referral criteria and destinations\n\n"
        f"NAMING CODEBOOK:\n{codebook}\n\n"
        "CRITICAL RULES:\n"
        "1. Label EVERY clinical item. Missing items is worse than having extras.\n"
        "2. IDs must be descriptive and follow prefix conventions exactly.\n"
        "3. Numeric variables MUST end with unit suffix: _days, _per_min, _c, _mm, _mg, _kg.\n"
        "4. Use lowercase_with_underscores only. No spaces, hyphens, camelCase.\n"
        "5. If a concept does not fit any existing prefix, use the closest match.\n"
        "6. Prefer over-labeling to under-labeling.\n\n"
        'OUTPUT FORMAT: Return ONLY valid JSON (no markdown fences, no explanation). Shape:\n'
        '{"labels": [{"span": "...", "id": "prefix_descriptive_name", "type": "...", '
        '"subtype": "...", "quote_context": "..."}]}'
    )


def parse_json_robust(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    repaired = text.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    try: return json.loads(repaired)
    except json.JSONDecodeError: pass
    return None


PREFIX_OK = {
    "supply_list": ("supply_", "equip_"),
    "variables": ("q_", "ex_", "v_", "lab_", "hx_", "demo_", "img_", "sys_", "prev_"),
    "predicates": ("p_",),
    "modules": ("mod_",),
    "phrase_bank": ("m_", "adv_", "tx_", "rx_", "ref_", "fu_", "out_", "proc_", "dx_"),
    "router": ("router", "rt_", "route_"),
    "integrative": ("integrative", "int_", "comorbid"),
}


def analyze(labels, chunk_text):
    text_lower = chunk_text.lower()
    by_type = defaultdict(int)
    misprefixed = 0
    grounded = 0
    for l in labels:
        by_type[l.get("type", "?")] += 1
        t = l.get("type", "")
        i = l.get("id", "")
        ok = PREFIX_OK.get(t)
        if ok and not any(i.startswith(p) for p in ok):
            misprefixed += 1
        span = (l.get("span", "") or "").strip().lower()
        if span and span in text_lower:
            grounded += 1
    return {
        "total": len(labels),
        "by_type": dict(by_type),
        "misprefixed": misprefixed,
        "mismatch_rate": misprefixed / max(1, len(labels)),
        "grounded": grounded,
        "grounded_rate": grounded / max(1, len(labels)),
    }


def normalize(s): return ' '.join(s.lower().split())


def compute_overlap(a_labels, b_labels):
    """Compute semantic overlap between two label sets."""
    a_spans = {normalize(l.get("span", "")): l for l in a_labels if l.get("span")}
    b_spans = {normalize(l.get("span", "")): l for l in b_labels if l.get("span")}

    a_ids = {(l["id"], l["type"]) for l in a_labels if l.get("id")}
    b_ids = {(l["id"], l["type"]) for l in b_labels if l.get("id")}

    # Exact (id, type) overlap
    exact_overlap = a_ids & b_ids

    # Span-based overlap (exact match)
    exact_span_overlap = set(a_spans.keys()) & set(b_spans.keys())

    # Fuzzy span overlap (>60% word overlap)
    def word_overlap(s1, s2):
        w1, w2 = set(s1.split()), set(s2.split())
        return len(w1 & w2) / min(len(w1), len(w2)) if w1 and w2 else 0

    fuzzy_a_to_b = {}
    for a_span in a_spans:
        best_b = max(b_spans.keys(), key=lambda bs: word_overlap(a_span, bs), default=None)
        if best_b and word_overlap(a_span, best_b) > 0.6:
            fuzzy_a_to_b[a_span] = best_b

    fuzzy_b_to_a = {}
    for b_span in b_spans:
        best_a = max(a_spans.keys(), key=lambda as_: word_overlap(b_span, as_), default=None)
        if best_a and word_overlap(b_span, best_a) > 0.6:
            fuzzy_b_to_a[b_span] = best_a

    # Only in A (Opus)
    a_only_spans = [s for s in a_spans if s not in exact_span_overlap and s not in fuzzy_b_to_a.values()]
    b_only_spans = [s for s in b_spans if s not in exact_span_overlap and s not in fuzzy_a_to_b.values()]

    # Where they agree on span but disagree on type
    type_disagreements = []
    for s in exact_span_overlap:
        a_type = a_spans[s]["type"]
        b_type = b_spans[s]["type"]
        if a_type != b_type:
            type_disagreements.append({
                "span": s[:60],
                "opus_type": a_type, "opus_id": a_spans[s]["id"],
                "gpt_type": b_type, "gpt_id": b_spans[s]["id"],
            })

    # Where they agree on span but disagree on ID naming
    id_disagreements = []
    for s in exact_span_overlap:
        a_id = a_spans[s]["id"]
        b_id = b_spans[s]["id"]
        if a_id != b_id:
            id_disagreements.append({
                "span": s[:60],
                "opus_id": a_id, "gpt_id": b_id,
                "type": a_spans[s]["type"],
            })

    return {
        "a_total": len(a_spans), "b_total": len(b_spans),
        "exact_id_type_overlap": len(exact_overlap),
        "exact_span_overlap": len(exact_span_overlap),
        "fuzzy_a_covered_by_b": len(fuzzy_a_to_b),
        "fuzzy_b_covered_by_a": len(fuzzy_b_to_a),
        "a_only_count": len(a_only_spans),
        "b_only_count": len(b_only_spans),
        "a_only_samples": [(s, a_spans[s]["id"], a_spans[s]["type"]) for s in a_only_spans[:10]],
        "b_only_samples": [(s, b_spans[s]["id"], b_spans[s]["type"]) for s in b_only_spans[:10]],
        "type_disagreements": type_disagreements[:10],
        "id_disagreements": id_disagreements[:15],
    }


async def main():
    anthropic_key = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY_ALT") or os.environ.get("OPENAI_API_KEY")
    if not anthropic_key: raise RuntimeError("ANTHROPIC_KEY not set")
    if not openai_key: raise RuntimeError("OPENAI_API_KEY not set")

    codebook = load_codebook()
    chunk = pick_chunk()
    system_prompt = build_monolithic_prompt(codebook)
    user_msg = f"CHUNK TO LABEL:\n{chunk['text']}"

    print(f"Chunk: index={chunk['chunk_index']}, ~{chunk['est_tokens']} tokens")
    print(f"System prompt: ~{len(system_prompt)//4} tokens")
    print()

    # ---- Call Opus (or reuse saved) ----
    saved = OUT / "comparison.json"
    if saved.exists():
        with open(saved) as f:
            prior = json.load(f)
        if prior.get("opus", {}).get("labels"):
            opus_labels = prior["opus"]["labels"]
            opus_cost = prior["opus"]["cost"]
            opus_time = prior["opus"]["time_s"]
            print(f"  Opus: reusing saved output ({len(opus_labels)} labels, ${opus_cost:.3f})")
        else:
            prior = None

    if not saved.exists() or not prior:
        print("Calling Opus 4.6...", flush=True)
        ant_client = anthropic.AsyncAnthropic(api_key=anthropic_key)
        t0 = time.time()
        opus_resp = await ant_client.messages.create(
            model=OPUS_MODEL, max_tokens=16384, temperature=0.0,
            system=[{"type": "text", "text": system_prompt}],
            messages=[{"role": "user", "content": user_msg}],
        )
        opus_time = time.time() - t0
        opus_text = opus_resp.content[0].text
        opus_parsed = parse_json_robust(opus_text)
        opus_labels = opus_parsed.get("labels", []) if isinstance(opus_parsed, dict) else []
        opus_cost = (opus_resp.usage.input_tokens * 5 + opus_resp.usage.output_tokens * 25) / 1e6
        print(f"  Opus: {len(opus_labels)} labels, {opus_time:.1f}s, ${opus_cost:.3f}")
        print(f"  Tokens: in={opus_resp.usage.input_tokens} out={opus_resp.usage.output_tokens}")

    # ---- Call GPT-5.4 ----
    print("\nCalling GPT-5.4...", flush=True)
    oai_client = openai.AsyncOpenAI(api_key=openai_key)
    t0 = time.time()
    gpt_resp = await oai_client.chat.completions.create(
        model=GPT_MODEL,
        max_completion_tokens=16384,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},  # enforce valid JSON
    )
    gpt_time = time.time() - t0
    gpt_text = gpt_resp.choices[0].message.content
    gpt_parsed = parse_json_robust(gpt_text)
    gpt_labels = gpt_parsed.get("labels", []) if isinstance(gpt_parsed, dict) else []
    gpt_in = gpt_resp.usage.prompt_tokens
    gpt_out = gpt_resp.usage.completion_tokens
    gpt_cost = (gpt_in * 2.50 + gpt_out * 15) / 1e6
    print(f"  GPT-5.4: {len(gpt_labels)} labels, {gpt_time:.1f}s, ${gpt_cost:.3f}")
    print(f"  Tokens: in={gpt_in} out={gpt_out}")

    # ---- Analyze each ----
    print("\n=== PER-MODEL ANALYSIS ===\n")
    opus_stats = analyze(opus_labels, chunk["text"])
    gpt_stats = analyze(gpt_labels, chunk["text"])

    print(f"{'Metric':<25s} {'Opus':>12s} {'GPT-5.4':>12s}")
    print("-" * 50)
    print(f"{'Total labels':<25s} {opus_stats['total']:>12d} {gpt_stats['total']:>12d}")
    print(f"{'Misprefixed':<25s} {opus_stats['misprefixed']:>12d} {gpt_stats['misprefixed']:>12d}")
    print(f"{'Mismatch rate':<25s} {opus_stats['mismatch_rate']*100:>11.1f}% {gpt_stats['mismatch_rate']*100:>11.1f}%")
    print(f"{'Grounded (exact span)':<25s} {opus_stats['grounded']:>12d} {gpt_stats['grounded']:>12d}")
    print(f"{'Grounded rate':<25s} {opus_stats['grounded_rate']*100:>11.1f}% {gpt_stats['grounded_rate']*100:>11.1f}%")
    print(f"{'Cost':<25s} {'$'+f'{opus_cost:.3f}':>12s} {'$'+f'{gpt_cost:.3f}':>12s}")
    print(f"{'Latency':<25s} {f'{opus_time:.1f}s':>12s} {f'{gpt_time:.1f}s':>12s}")

    print(f"\nOpus by type:  {opus_stats['by_type']}")
    print(f"GPT   by type: {gpt_stats['by_type']}")

    # ---- Cross-model overlap ----
    print("\n=== CROSS-MODEL OVERLAP ===\n")
    overlap = compute_overlap(opus_labels, gpt_labels)

    print(f"Opus unique spans:         {overlap['a_total']}")
    print(f"GPT  unique spans:         {overlap['b_total']}")
    print(f"Exact (id,type) overlap:   {overlap['exact_id_type_overlap']}")
    print(f"Exact span overlap:        {overlap['exact_span_overlap']}")
    print(f"Fuzzy Opus covered by GPT: {overlap['fuzzy_a_covered_by_b']}/{overlap['a_total']} "
          f"({overlap['fuzzy_a_covered_by_b']/max(1,overlap['a_total'])*100:.0f}%)")
    print(f"Fuzzy GPT covered by Opus: {overlap['fuzzy_b_covered_by_a']}/{overlap['b_total']} "
          f"({overlap['fuzzy_b_covered_by_a']/max(1,overlap['b_total'])*100:.0f}%)")
    print(f"Opus-only spans:           {overlap['a_only_count']}")
    print(f"GPT-only spans:            {overlap['b_only_count']}")

    if overlap['type_disagreements']:
        print(f"\n--- Type disagreements (same span, different type) [{len(overlap['type_disagreements'])}] ---")
        for d in overlap['type_disagreements']:
            print(f"  \"{d['span'][:55]}\"")
            print(f"    Opus: {d['opus_type']:15s} -> {d['opus_id']}")
            print(f"    GPT:  {d['gpt_type']:15s} -> {d['gpt_id']}")

    if overlap['id_disagreements']:
        print(f"\n--- ID disagreements (same span+type, different ID) [{len(overlap['id_disagreements'])}] ---")
        for d in overlap['id_disagreements'][:10]:
            print(f"  \"{d['span'][:55]}\" ({d['type']})")
            print(f"    Opus: {d['opus_id']}")
            print(f"    GPT:  {d['gpt_id']}")

    if overlap['a_only_samples']:
        print(f"\n--- Opus-only labels (GPT missed) [{overlap['a_only_count']} total, showing 10] ---")
        for span, lid, ltype in overlap['a_only_samples']:
            print(f"  {ltype:15s} {lid:40s} \"{span[:50]}\"")

    if overlap['b_only_samples']:
        print(f"\n--- GPT-only labels (Opus missed) [{overlap['b_only_count']} total, showing 10] ---")
        for span, lid, ltype in overlap['b_only_samples']:
            print(f"  {ltype:15s} {lid:40s} \"{span[:50]}\"")

    # ---- Save everything ----
    results = {
        "chunk": {"index": chunk["chunk_index"], "est_tokens": chunk["est_tokens"]},
        "opus": {"labels": opus_labels, "stats": opus_stats, "cost": opus_cost, "time_s": opus_time},
        "gpt": {"labels": gpt_labels, "stats": gpt_stats, "cost": gpt_cost, "time_s": gpt_time},
        "overlap": {k: v for k, v in overlap.items() if not k.endswith("_samples")},
        "type_disagreements": overlap["type_disagreements"],
        "id_disagreements": overlap["id_disagreements"],
    }
    with open(OUT / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT}/comparison.json")


if __name__ == "__main__":
    asyncio.run(main())
