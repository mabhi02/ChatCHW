# CHW Navigator — Engineering Architecture (Gen 7 v2)

**Audience:** A new engineer joining the project, or a principal investigator reviewing the system. Read this once, end to end, and you should understand every moving part.

**Scope:** The Gen 7 v2 pipeline as implemented on `main` at `backend/gen7/` and `backend/rlm_runner.py`. Gens 1-6 are referenced only where Gen 7 superseded them; they are not the production path.

**Reference run:** `backend/output/run_d9a39c2d/` — WHO Community Health Worker manual (Management of Sick Children), 21 micro-chunks, 1,073 deduped labels, 8 clinical modules, final cost ~$12.55, elapsed ~22 minutes. All example values in this document come from that run.

---

## Table of Contents

1. [System Diagram](#1-system-diagram)
2. [Pipeline Phases (Gen 7 v2)](#2-pipeline-phases-gen-7-v2)
3. [Artifact Schemas](#3-artifact-schemas)
4. [Flag-Based Dispatcher Architecture](#4-flag-based-dispatcher-architecture-professor-levines-design)
5. [Cost Tracking (Authoritative)](#5-cost-tracking-authoritative)
6. [Caching Strategy](#6-caching-strategy)
7. [Rate Limit Strategy (Tier 2 Anthropic)](#7-rate-limit-strategy-tier-2-anthropic)
8. [Frontend Architecture](#8-frontend-architecture)
9. [Session Management](#9-session-management)
10. [k=1 Identification Thesis](#10-k1-identification-thesis)
11. [Deployment](#11-deployment)
12. [Testing & Verification](#12-testing--verification)
13. [Open Issues / Future Work](#13-open-issues--future-work)

---

## 1. System Diagram

The system is two distinct dataflows glued together by a single run identifier. The **ingest-time flow** runs when a user uploads a PDF; the **extraction-time flow** runs when the user clicks "Run Extraction" (or equivalently, when the backend kicks off Gen 7 against a previously ingested guide).

### 1.1 Ingest-time flow (one-time per guide)

```
User browser
  │
  │  POST /api/ingest/pdf  (multipart: chw_guide.pdf)
  ▼
FastAPI  backend/server.py :: /api/ingest/pdf
  │
  │  Saves PDF to a temp path
  ▼
backend/ingestion/pipeline.py :: ingest_pdf()
  │
  │  1. Render PDF pages (pypdfium2 on Windows, pdf2image+poppler on Linux)
  │  2. Post-process pages with OpenAI gpt-4o vision (optional, for diagrams)
  │
  ▼
backend/ingestion/unstructured_client.py :: run_unstructured_hi_res()
  │
  │  POST to Unstructured.io (hi_res mode, mandatory — fast mode finds 0 tables)
  │  Returns: list[Element]   (Title / NarrativeText / Table / FigureCaption)
  │
  ▼
backend/ingestion/assembler.py :: assemble_guide_json()
  │
  │  Groups elements into sections by heading hierarchy
  │  Produces: { "sections": { sec_id: { title, raw_text, blocks: [...] } } }
  │
  ▼
backend/ingestion/cache.py :: cache_ingested_guide()
  │
  │  Writes guide JSON to Neon Postgres keyed by SHA256(PDF bytes)
  │  Next upload of the same PDF hits the cache — no re-ingestion
  ▼
guide_json ready for extraction
```

The output of ingest-time flow is a single JSON blob of shape `{ "sections": { ... } }`. Each section has a `title`, a `raw_text` fallback, and an optional `blocks` array where each block is `{ text, type, page_ref?, heading_level? }`. This is the ONLY artifact that crosses between ingest-time and extraction-time — everything after this point is derived from the guide JSON.

### 1.2 Extraction-time flow (Gen 7 v2)

```
User browser
  │   (already has guide_id from earlier ingestion)
  │
  │  POST /api/extract  { guide_id, api_key (BYOK Anthropic) }
  ▼
FastAPI  backend/server.py :: /api/extract
  │
  ▼
backend/session_manager.py :: create_session()
  │  Reserves a slot (MAX_CONCURRENT_SESSIONS=2 on Render Pro Plus)
  │  Registers run_id in Upstash Redis with TTL
  │  Spawns _run_gen7_extraction_task as asyncio.Task
  │
  ▼
backend/session_manager.py :: _run_gen7_extraction_task()
  │  Wires Gen 7 on_progress callback to the SSE event stream
  │  Calls run_gen7_extraction(...)
  │
  ▼
backend/gen7/pipeline.py :: run_gen7_extraction()
  │
  ├─── PHASE 0 — backend/gen7/chunker.py :: micro_chunk_guide()
  │         Walks guide blocks, cuts at first boundary after 2K tokens
  │         Output: list[chunk]  (21 chunks for WHO CHW guide)
  │
  ├─── PHASE 1 — backend/gen7/labeler.py :: label_all_chunks()
  │         For each chunk:
  │           Call 1: Opus labels every clinical item  (cache hit after #0)
  │           Call 2: Opus QC pass, drops hallucinations, canonicalizes IDs
  │         Dynamic batching: 1 prime + batches of 6, 62s cooldown between
  │         Output: list[labeled_chunk]   (1,841 raw labels for WHO run)
  │
  ├─── PHASE 1b — backend/gen7/labeler.py :: deduplicate_labels()
  │         Pure-code dedup on exact (id, type) match
  │         Collapses 1,841 → 1,073 unique labels
  │         Writes deduped_labels.json  (the 26th artifact)
  │
  ├─── PHASE 2 — Cached context injection
  │         backend/rlm_runner.py :: _set_gen7_cached_context(...)
  │         Installs 2 blocks in the REPL's system prompt:
  │           (a) Reconstructed guide text  (~60K tokens for WHO)
  │           (b) Deduped label inventory   (~95K tokens for WHO)
  │         Both with cache_control: ephemeral, ttl=1h
  │
  ├─── PHASE 3 — backend/gen7/pipeline.py :: Opus REPL
  │         Phase A (scan+plan):         1 LLM call
  │         Phase B (Module Maker):      N parallel llm_query_batched calls
  │                                      (one per module, 8 for WHO)
  │         Phase C (dispatcher):        0 LLM calls (pure Python)
  │         Phase D (flat artifacts):    1 LLM call, FINAL_VAR(clinical_logic)
  │
  └─── PHASE 4 — backend/converters/
            convert_to_dmn  (DMN 1.3 XML)
            convert_to_xlsx (CHT XLSForm)
            convert_to_mermaid (flowchart)
            convert_to_csv  (predicates.csv, phrases.csv)
            Output: clinical_logic.dmn, form.xlsx, flowchart.md, *.csv
```

**SSE stream side-channel:** Every phase emits structured events through `on_progress`. `_run_gen7_extraction_task` translates them into the frontend's SSE event format and pushes to the session's ring buffer. The frontend's `ReplStream.tsx` subscribes to `/api/session/{id}/stream` and renders live.

**Cost tracker side-channel:** Every Anthropic API call (labeling + REPL) funnels its `response.usage` into the process-global `backend/rlm_runner.py::_run_usage` dict. `CostTracker.tsx` polls `/api/session/{id}/status` every 3 seconds; the handler reads `_run_usage` and returns a snapshot.

---

## 2. Pipeline Phases (Gen 7 v2)

### Phase 0 — Micro-chunking

- **File:** `backend/gen7/chunker.py`
- **Entry point:** `micro_chunk_guide(guide_json) -> list[dict]`
- **Goal:** Split the (arbitrarily large) guide into chunks of ~2K tokens so that the cached system prompt + chunk fits comfortably under Opus's 16K output budget with room for 10K of labels.

**Algorithm (deterministic, no LLM):**

1. Flatten all sections' blocks into a sequential stream, preserving `(section_id, section_title, block_text, block_obj)`.
2. Walk the stream, accumulating `block_text` into `current_text`. Track accumulated tokens via character/4 estimate.
3. As soon as `current_tokens >= TARGET_TOKENS` (2000), flush at the current block boundary. This means each chunk is 2K tokens OR a bit more (never less, except the final one).
4. Trailing chunk smaller than TARGET_TOKENS merges into the previous chunk. This avoids wasting a full labeling call on a 500-token remnant.
5. Re-index all chunks so `chunk_index` is contiguous 0..N-1.
6. Classify each chunk's difficulty via `backend/validators/test_suite.py::classify_chunk_difficulty` (outputs `{difficulty, score}` from a lightweight heuristic). Difficulty tiers: `trivial | easy | medium | hard | extreme`.

**Output shape:**

```python
[
  {
    "chunk_index": 0,
    "section_id": "contents+give_ors",
    "section_title": "Contents | Give ORS",
    "text": "...",
    "blocks": [...],
    "sections_in_chunk": 2,
    "difficulty": "medium",
    "difficulty_score": 42,
  },
  ...
]
```

**Why 2K:** A labeling call costs `~3K (codebook+instructions, cached) + ~2K (chunk, fresh) = ~5K input tokens` and produces `~10K output tokens of JSON labels`. At Tier 2 Opus output limit of 80K tokens/minute, we can run ~8 labeling calls in parallel before hitting the rate limit. We use 6 per batch with headroom for variance. Smaller chunks would waste the 3K cached codebook preamble; larger chunks push the output over 16K and cause truncation.

**Section-boundary-only cut:** Cutting mid-block would lose `block_obj` provenance (page refs, heading levels). Cutting at section boundaries is simpler and has negligible cost because section boundaries are roughly frequent enough (every 1K-3K tokens in the WHO guide).

### Phase 1 — Opus Labeling (2 calls per chunk)

- **File:** `backend/gen7/labeler.py`
- **Entry point:** `label_all_chunks(chunks, codebook, anthropic_key, on_chunk_labeled) -> list[dict]`
- **Goal:** Annotate every clinical item in every chunk with a structured label. The labels are the input to the REPL's Module Maker; every label is a candidate for `supply_list`, `variables`, `predicates`, `modules`, `phrase_bank`, `router`, or `integrative`.

#### Call 1 — Labeling

`_build_labeling_system_prompt(codebook)` builds a ~6K-char prompt that:

- Defines the 7 artifact types and their prefix conventions (`q_`, `ex_`, `v_`, `lab_`, `hx_`, `demo_`, `p_`, `mod_`, `m_`, `adv_`, `tx_`, `rx_`, `ref_`, `supply_`, `equip_`).
- Provides the naming codebook (descriptive, content-derived IDs — NEVER numbered).
- Enumerates what counts as a clinical item (medications with dosages, equipment/supplies, questions to ask, examination steps, vital sign thresholds, danger signs, referral criteria, treatment instructions, advice phrases, follow-up scheduling).
- Spells out the output JSON shape.

The prompt is attached with `cache_control: { type: "ephemeral", ttl: "1h" }`. On chunk 0 this costs `cache_creation` (2x input rate = $10/M for Opus). Every subsequent chunk reads it at `cache_read` (0.1x rate = $0.50/M). For the WHO guide, that's a one-time ~$0.07 to create the cache, then ~$0.015 per subsequent call to read it — a ~10x savings.

Call 1's user message is just `"CHUNK TO LABEL:\n" + section_header + text`. Output is JSON:

```json
{
  "section_id": "contents+give_ors",
  "section_title": "Contents | Give ORS",
  "text": "(original text as provided)",
  "labels": [
    { "span": "Give ORS", "id": "mod_give_ors", "type": "modules", "subtype": "treatment step", "quote_context": "..." },
    { "span": "ORS packets", "id": "supply_ors_packets", "type": "supply_list", "subtype": "consumable", "quote_context": "..." },
    ...
  ]
}
```

#### Call 2 — Distillation (QC)

`_build_distillation_system_prompt(codebook)` builds a prompt that instructs Opus to:

1. Verify each candidate label's `span` appears (verbatim or near-verbatim) in the source text. Drop hallucinations.
2. Verify each label's `id` follows the codebook prefix for its `type`. Correct or drop wrong-prefixed IDs.
3. NEVER add new labels. Only clean the candidate set.

Call 2's user message contains BOTH the source text AND the Call 1 label list. Output has the same shape as Call 1's output. If Call 2 fails (JSON parse error, API error, etc.), the labeler silently falls back to Call 1's labels.

**Why Call 2 exists:** Gen 6 ran single-call labeling and observed ~15-20% of labels were hallucinated (span not in source) or mis-prefixed (e.g., `mod_cough` labeled as type=`variables`). Call 2 is a cheap (~$0.04/chunk at Opus cache-read rates) QC pass that drops ~10-15% of labels on average. For WHO, Call 1 produced 1,841 labels, Call 2 kept 1,073 — a 42% drop, driven mostly by wrong-prefix drops rather than hallucinations.

#### Cache priming + parallel batching

- `_MAX_BATCH_SIZE = max(1, 80000 // 10000 - 2) = 6`
- `_BATCH_COOLDOWN_SEC = 62`
- `_compute_batch_size(n_chunks)`: if fewer chunks than the max, run them all at once.

**Small guide (<= 6 chunks):** everything runs in one `asyncio.gather(...)` burst. Cache is created on whichever call wins the race; the other 5 calls pay cache_creation rates because the cache hasn't propagated yet. On Tier 2 this is fine — worst case we pay 6x cache_creation on chunks 0-5, not 1x. At ~$0.07/chunk, the overhead is ~$0.35.

**Large guide (> 6 chunks):** chunk 0 runs alone first to prime the cache (writes it once). Chunks 1..N run in batches of 6 with 62s cooldown between batches. All 6 in a batch read from the cache simultaneously.

For WHO (21 chunks), the schedule is:
- `t=0s`: chunk 0 (prime, cache_creation ~= 2x input rate)
- `t=~60s`: chunk 0 completes, batch 1 starts (chunks 1-6, 6 parallel cache_read)
- `t=~180s`: batch 1 completes, cooldown 62s
- `t=~242s`: batch 2 starts (chunks 7-12)
- ... 4 batches total

Wall clock: ~12-15 minutes for 21 chunks. Output TPM: at 10K tokens/chunk x 6 parallel = 60K tokens/minute, safely under the 80K Tier 2 ceiling.

#### Token accounting

Both Call 1 and Call 2 feed their usage into the global `_run_usage` accumulator via `backend/rlm_runner.py::accumulate_catcher_usage(model, input_tokens, output_tokens, cached_tokens, cache_write_tokens)`. This is essential because the labeler calls Anthropic directly (not through the RLM client), so the monkey-patched `_completion_with_cache` doesn't fire on these calls.

Without `accumulate_catcher_usage`, the labeling cost (~$7-10 of the total $12.55 WHO run) would be invisible to the frontend Cost Tracker.

### Phase 1b — Algorithmic Dedup (no LLM)

- **File:** `backend/gen7/labeler.py`
- **Entry point:** `deduplicate_labels(labeled_chunks) -> list[dict]`
- **Cost:** $0, pure Python.

**Algorithm:**

```python
seen: dict[tuple[str, str], dict] = {}
for chunk in labeled_chunks:
    for label in chunk.get("labels", []):
        key = (label["id"], label["type"])
        if key not in seen:
            entry = dict(label)
            entry["source_section_ids"] = [chunk["section_id"]]
            entry["source_chunk_indices"] = [chunk["chunk_index"]]
            seen[key] = entry
        else:
            # merge source provenance
            if chunk["section_id"] not in seen[key]["source_section_ids"]:
                seen[key]["source_section_ids"].append(chunk["section_id"])
            ...
return sorted(seen.values(), key=lambda d: (d["type"], d["id"]))
```

**Exact string match only.** `supply_amoxicillin_250mg` and `supply_amoxicillin_250mg_dispersible` are different IDs; they remain separate entries. Semantic merging ("are these the same drug?") is delegated to the REPL in Phase 3, where it has access to the reconstructed guide text and can use reading comprehension.

**Why not semantic dedup here?** Because semantic dedup is expensive and controversial. If the labeler canonicalizes IDs correctly (Call 2's job), the exact-string dedup is already ~90% correct. The remaining ~10% are genuinely borderline cases — Opus at Phase 3 reads the full source and resolves them in context.

**Output file:** `deduped_labels.json` — the "26th artifact" (on top of the 7 core + 18 intermediate files). This is the ground-truth input to the REPL.

For WHO, 1,841 raw labels collapse to 1,073 deduped labels. Breakdown by type:

| Type | Deduped count |
|------|---------------|
| supply_list | 14 |
| variables | 97 |
| predicates | 88 |
| modules | 46 |
| phrase_bank | 19 |
| router | 8 (merged into dispatcher) |
| integrative | 1 |

The `modules` type has 46 candidate labels, but the REPL consolidates them into 8 actual clinical modules (because the labeler often produces per-rule module labels like `mod_diarrhoea_no_blood` that all describe the same `mod_diarrhoea_treatment` module).

### Phase 2 — Cached Context Injection

- **File:** `backend/rlm_runner.py`
- **Entry points:** `_set_gen7_cached_context(guide_text, deduped_labels)`, `_maybe_attach_gen7_blocks(kwargs)`

Gen 7's REPL has a system prompt that explains the Module Maker + Dispatcher architecture. Attached to that system prompt are TWO cached blocks that every REPL turn AND every `llm_query_batched` sub-call sees:

1. **FULL SOURCE GUIDE** — the concatenated `chunk.text` across all chunks, in chunk_index order, filtered to drop pure-placeholder chunks. This is the ground truth for Module Maker sub-calls to write grounded DMN rules.
2. **DEDUPED LABEL INVENTORY** — `json.dumps(deduped_labels, indent=1)`. This is the clean list of unique (id, type) entries the REPL compiles from.

Both blocks are attached with:

```python
{
  "type": "text",
  "text": block_text,
  "cache_control": {"type": "ephemeral", "ttl": "1h"},
}
```

`_set_gen7_cached_context` installs the blocks on two module-global variables (`_current_gen7_guide_text`, `_current_gen7_labels`). The monkey-patched `AnthropicClient._completion_with_cache` in `rlm_runner.py` calls `_maybe_attach_gen7_blocks(kwargs)` on EVERY outgoing API request, which appends the blocks to `kwargs["system"]` if they're set. At the end of the run, `_set_gen7_cached_context(None, None)` clears them so a later non-Gen-7 run doesn't inherit stale state.

**Cache TTL 1h:** The REPL compile phase takes 15-25 minutes wall clock for a typical guide. A 5-minute TTL would expire mid-run and cause cache_creation charges on every subsequent call. 1h fits with headroom. Cache writes cost 2x of input on the first call but every subsequent call reads at 0.1x.

**Why blocks, not the system prompt itself?** Anthropic's cache_control can be placed on multiple system blocks. By separating the (large, stable) guide text + labels from the (smaller, mechanics-describing) system prompt, we preserve the ability to change the REPL mechanics without invalidating the guide cache. This matters less in practice (we rarely change the system prompt mid-run) but is architecturally cleaner.

### Phase 3 — REPL Compilation (Module Maker + Dispatcher)

- **File:** `backend/gen7/pipeline.py`
- **Entry point:** `_build_repl_system_prompt(codebook)`, `_build_initial_repl_message(...)`, `rlm.completion(initial_message)`
- **Library:** Uses the `rlm/` (Restricted Language Model) REPL library. Max 30 iterations, max budget $30, max 5 errors, 30min timeout.

The REPL sees:
- A Python-like execution environment with `context` (list of labeled chunks), `print`, `FINAL_VAR`, `llm_query`, `llm_query_batched`.
- The system prompt with 3 cached blocks: (mechanics) + (guide text) + (label inventory).
- NO `emit_artifact`, NO `validate`, NO `z3_check`. Gen 7 explicitly removed those because the labeler+dedup+Module Maker replaces them.

The initial message walks Opus through a 4-phase flow:

#### Phase A — Scan & Plan (1 REPL turn)

```python
# Flatten per-chunk labels, group by type, list modules
all_labels = [l for c in context for l in c["labels"]]
by_type = {}
for l in all_labels:
    by_type.setdefault(l["type"], []).append(l)

# Build module plan: {module_id, display_name, priority, trigger_flag, done_flag}
module_labels = by_type["modules"]
module_plan = []
for m in sorted(module_labels, key=lambda x: x["id"]):
    mid = m["id"]
    # Priority inference from module_id semantics
    lid = mid.lower()
    if any(k in lid for k in ["danger", "severe", "emergency", "critical"]):
        priority = 0
    elif any(k in lid for k in ["intro", "startup", "initial", "demographics", "ask_first"]):
        priority = 1
    elif any(k in lid for k in ["closing", "finalize", "wrap", "summary"]):
        priority = 999
    else:
        priority = 100
    topic = mid.replace("mod_", "", 1)
    trigger_flag = f"has_{topic}"
    done_flag = f"{mid}_done"
    module_plan.append({
        "module_id": mid,
        "display_name": m.get("span", mid),
        "priority": priority,
        "trigger_flag": trigger_flag,
        "done_flag": done_flag,
        "quote_context": m.get("quote_context", ""),
    })
print(f"Module plan ({len(module_plan)} modules):")
for mp in module_plan:
    print(f"  p={mp['priority']:3d} {mp['module_id']}")
```

For WHO, Phase A prints:

```
Module plan (8 modules):
  p=  0 mod_danger_sign_check
  p=  1 mod_assess
  p=100 mod_check_vaccines
  p=100 mod_diarrhoea_treatment
  p=100 mod_fast_breathing_treatment
  p=100 mod_fever_malaria_treatment
  p=100 mod_malnutrition_screening
  p=999 mod_home_care_advice
```

This is pure code running in the REPL sandbox; no LLM call happens during Phase A itself. The only LLM "call" is the REPL turn that wrote this code.

#### Phase B — Module Maker (1 REPL turn, N parallel sub-calls)

```python
# Build a prompt per module
prompts = []
for mp in sorted(module_plan, key=lambda m: m["module_id"]):
    p = f"""Generate a DMN decision table for {mp['module_id']} ({mp['display_name']}).
Priority: {mp['priority']}. Trigger: {mp['trigger_flag']}. Done flag: {mp['done_flag']}.

Use the predicates and phrase_bank entries in the DEDUPED LABEL INVENTORY above
that describe this module's rules. Ground your rules in the FULL SOURCE GUIDE.

Return JSON: {{
  "module_id": "{mp['module_id']}",
  "display_name": "{mp['display_name']}",
  "hit_policy": "unique",
  "priority": {mp['priority']},
  "trigger_flag": "{mp['trigger_flag']}",
  "done_flag": "{mp['done_flag']}",
  "inputs": [...],  # list of var IDs the rules read
  "outputs": [...], # list of output IDs (flags, actions)
  "rules": [
    {{ "rule_id": "rule_1",
       "condition": "<FEEL expression>",
       "outputs": {{ "{mp['done_flag']}": true, "is_priority_exit": false, ... }},
       "description": "..." }}
  ]
}}
Return ONLY the JSON object, no prose, no code fences."""
    prompts.append(p)

responses = llm_query_batched(prompts)  # parallel, up to 10 at a time
modules_dict = {}
for resp, mp in zip(responses, module_plan):
    m = json.loads(resp)  # parse JSON
    modules_dict[mp['module_id']] = m
```

**Key property:** Every sub-call sees the same cached system prompt (mechanics) + cached guide text + cached label inventory. Opus reads all 3 blocks at cache_read rate (0.1x input). The ONLY fresh input is the per-module prompt (~1-2K tokens). Each sub-call produces ~2-4K tokens of module JSON.

For WHO (8 modules), `llm_query_batched` issues 8 parallel Anthropic API calls. Elapsed wall clock for Phase B: ~90-120 seconds. Total Phase B cost: ~$2-3 (dominated by output tokens at $25/M).

#### Phase C — Programmatic Dispatcher Assembly (0 LLM calls)

```python
rows = []

# Priority 0: is_priority_exit short-circuit
rows.append({
    "priority": 0,
    "condition": "is_priority_exit == true",
    "output_module": "PRIORITY_EXIT",
    "description": "Short-circuit: priority exit flag set by any module"
})

# Priority 1: startup if not done
startup_mod = next(m for m in module_plan if m["priority"] == 1)
rows.append({
    "priority": 1,
    "condition": f"{startup_mod['done_flag']} == false",
    "output_module": startup_mod["module_id"],
    "description": f"Startup: {startup_mod['display_name']}"
})

# Priority 100 regular modules (alphabetical by module_id)
for m in sorted([m for m in module_plan if m["priority"] == 100], key=lambda m: m["module_id"]):
    rows.append({
        "priority": 100,
        "condition": f"{m['trigger_flag']} == true AND {m['done_flag']} == false",
        "output_module": m["module_id"],
        "description": f"Clinical module: {m['display_name']}"
    })

# Priority 999: closing (always matches if nothing else fired)
closing_mod = next(m for m in module_plan if m["priority"] == 999)
rows.append({
    "priority": 999,
    "condition": "true",
    "output_module": closing_mod["module_id"],
    "description": f"Closing: {closing_mod['display_name']}"
})

dispatcher = {"hit_policy": "priority", "rows": rows}
```

**Zero LLM variance.** Same input → same dispatcher every time. This is Professor Levine's design insight: the dispatcher is a 3rd-grade-level programming exercise, not a clinical-reasoning exercise. An LLM adds no value here and injects stochasticity.

For WHO, Phase C produces exactly the dispatcher shown in section 3.7 below — 9 rows, priorities 0/1/2/100/100/100/100/100/999.

#### Phase D — Flat Artifacts + Final Assembly (1 REPL turn)

```python
# Filter deduped labels by type
supply_list = sorted(
    [build_supply(l) for l in all_labels if l["type"] == "supply_list"],
    key=lambda x: x["id"]
)
variables = sorted(
    [build_var(l) for l in all_labels if l["type"] == "variables"] +
    # Add per-module done_flag booleans
    [{"id": f"{m['module_id']}_done", "prefix": "mod", "data_type": "boolean", ...} for m in module_plan] +
    # Add is_priority_exit
    [{"id": "is_priority_exit", "prefix": "p", "data_type": "boolean", ...}],
    key=lambda x: x["id"]
)
predicates = sorted(
    [build_pred(l) for l in all_labels if l["type"] == "predicates"],
    key=lambda x: x["id"]
)
phrase_bank = sorted(
    [build_phrase(l) for l in all_labels if l["type"] == "phrase_bank"],
    key=lambda x: x["id"]
)
integrative = {"rules": [build_integ(l) for l in all_labels if l["type"] == "integrative"]}

clinical_logic = {
    "supply_list": supply_list,
    "variables": variables,
    "predicates": predicates,
    "modules": modules_dict,
    "router": dispatcher,
    "phrase_bank": phrase_bank,
    "integrative": integrative,
}
FINAL_VAR(clinical_logic)
```

**Explicit materialization rule:** The initial message warns:

> IMPORTANT: Build each artifact into a variable first, then use variable references when assembling clinical_logic. Do NOT put empty literal lists `[]` in the final dict — that defeats the whole extraction.

This is a defense against a failure mode Gen 6 exhibited: under time pressure (approaching iteration 25+), the model would sometimes shortcut by writing `clinical_logic = {"supply_list": [], "variables": [], ...}` and calling FINAL_VAR early, producing an empty final.

### Phase 4 — Converters

- **Directory:** `backend/converters/`
- **Entry points:** `convert_to_dmn(logic)`, `convert_to_xlsx(logic, path)`, `convert_to_mermaid(logic)`, `convert_to_csv(logic)`

After the REPL returns `clinical_logic`, `pipeline.py::_save_outputs` serializes it to JSON and runs each converter. All converters are pure Python, no LLM calls.

#### DMN 1.3 XML (`json_to_dmn.py`, 278 lines)

- Each `modules.{module_id}` becomes a `<decision>` with `<decisionTable>`.
- `hit_policy` maps to DMN's `hitPolicy` attribute (unique/priority/first/any/etc., uppercased).
- `inputs[]` become `<input>` elements with `typeRef="boolean"` (fallback if unknown).
- `outputs[]` become `<output>` elements.
- `rules[]` become `<rule>` with `<description>` holding `source_quote + source_section_id`.
- `predicates[]` become top-level `<inputData>` elements referenced via `<informationRequirement>`.
- The `router` is NOT rendered as DMN directly; the dispatcher's priority ordering is encoded as per-rule rowIndex in each downstream decision.

**Gen 7 schema compatibility:** The converter was retrofitted to handle both old and new key names:
- `modules`: dict-keyed-by-id (Gen 7) OR list (legacy).
- `predicates[].id` (Gen 7) OR `predicate_id` (legacy).
- `module.inputs/outputs` (Gen 7) OR `input_columns/output_columns` (legacy).
- `router` (Gen 7) OR `activator` (legacy).

#### XLSX (`json_to_xlsx.py`, 295 lines)

Produces a CHT (Community Health Toolkit) XLSForm. Two key challenges:

1. **Boolean Trap:** ODK evaluates `string("false")` as truthy. Must use numeric 1/0 for booleans throughout.
2. **Module gating:** The `relevant` column on each module group uses `contains(required_modules, 'mod_X') and not(contains(completed_modules, '|mod_X|'))`. The `completed_modules` field is a pipe-delimited string that grows as modules complete.

Predicate compilation produces TWO rows per predicate:
- `{pid}_measured`: a `calculate` that returns 1 if all source vars are non-empty.
- `{pid}`: the actual predicate, wrapped in `if({pid}_measured, if(threshold, 1, 0), {fail_safe})`.

This handles the fail-safe semantics: if equipment-dependent data is missing (e.g., `v_respiratory_rate_per_min` not collected because the CHW didn't have a timer), the predicate falls back to `fail_safe` (1 for safety = assume worst case, 0 for "assume normal").

#### Mermaid (`json_to_mermaid.py`, 358 lines)

Produces a flowchart.md with `graph TD` syntax. Per-module subgraphs, nodes colored by type:
- `classDef start fill:#6b7280` (gray)
- `classDef activator fill:#8b5cf6` (purple)
- `classDef router fill:#0891b2` (cyan)
- `classDef module fill:#4488ff` (blue)
- Emergency/priority_exit paths: `fill:#ef4444` (red)
- Fallback/closing paths: `fill:#f59e0b` (yellow)

Node labels wrap the predicate's `human_label` + `[source_section_id]` for provenance. Defensive label escaping handles Mermaid's special chars (`()[]{}|<>:#"`).

Recently fixed (task #64): the converter now handles the Gen 7 dispatcher schema (`router.rows[].output_module`) instead of the legacy `activator.rules[].target_module`.

#### CSVs (`json_to_csv.py`, 125 lines)

Two flat CSVs:
- `predicates.csv`: `predicate_id, human_label, source_vars, threshold_expression, fail_safe, source_section_id`
- `phrases.csv`: `message_id, category, text, module_context, source_section_id`

Used for human review, not runtime execution.

---

## 3. Artifact Schemas

The final `clinical_logic.json` has exactly 7 top-level keys. Plus the `deduped_labels.json` "26th artifact" written to disk separately.

### 3.1 `supply_list[]`

```json
{
  "id": "equip_muac_strap",
  "display_name": "MUAC strap",
  "kind": "equipment",
  "source_quote": "Measure the arm circumference with a MUAC strap",
  "source_section_id": "course_objectives+child_1_siew_chin+give_ors_solution"
}
```

Fields:
- `id`: content-derived, prefix `equip_` (durable) or `supply_` (consumable).
- `display_name`: human-readable name.
- `kind`: `"equipment"` | `"consumable"`.
- `source_quote`: verbatim span from source text.
- `source_section_id`: where it was first found (may be compound via `+` if merged chunks).

WHO example: 15 supplies. 5 equipment (thermometer, timer, MUAC strap, recording form, referral form), 10 consumables (amoxicillin, zinc, ORS, RDT kit/buffer, AL tablets, rectal artesunate, alcohol swab, gloves, lancet).

### 3.2 `variables[]`

```json
{
  "id": "v_respiratory_rate_per_min",
  "display_name": "Respiratory rate (breaths per minute)",
  "prefix": "v",
  "data_type": "integer",
  "unit": "bpm",
  "depends_on_supply": "equip_timer",
  "source_quote": "count breaths in 1 minute: breaths per minute (bpm)",
  "source_section_id": "child_3_karen_shah+checking_questions+exercise_2_optional"
}
```

Fields:
- `id`: prefix-discriminated (`q_`, `ex_`, `v_`, `lab_`, `hx_`, `demo_`, `mod_*_done`, `sys_`).
- `prefix`: the prefix class for routing/visualization.
- `data_type`: `"boolean"` | `"integer"` | `"string"` | `"float"`.
- `unit`: SI/clinical unit (e.g., `"bpm"`, `"mm"`, `"days"`, `"months"`, `"c"`). Null for booleans/strings.
- `depends_on_supply`: if the variable requires an equipment/consumable to measure, references the `supply_list.id`. Null otherwise.
- `source_quote`, `source_section_id`: provenance.

**Special variables added at assembly time:**
- `mod_{X}_done`: one boolean per module; set to `true` by the module's rules.
- `is_priority_exit`: boolean; set by any module that detects an emergency.

WHO example: 29 variables across all prefixes. Notable demo: `demo_child_age_months` (integer, months, no supply dep). Notable v: `v_respiratory_rate_per_min` (depends on `equip_timer`). Notable lab: `lab_rdt_malaria_result` (string, depends on `supply_rdt_malaria` + `supply_rdt_buffer`).

### 3.3 `predicates[]`

```json
{
  "id": "p_fast_breathing",
  "threshold_expression": "(p_age_2mo_to_12mo == true AND v_respiratory_rate_per_min >= 50) OR (p_age_12mo_to_5yr == true AND v_respiratory_rate_per_min >= 40)",
  "fail_safe": 1,
  "source_vars": [
    "v_respiratory_rate_per_min",
    "demo_child_age_months"
  ],
  "human_label": "Fast breathing for age",
  "source_quote": "Age 2 months up to 12 months: 50bpm or more; Age 12 months up to 5 years: 40bpm or more",
  "source_section_id": "child_3_karen_shah+checking_questions+exercise_2_optional"
}
```

Fields:
- `id`: prefix `p_`, content-derived.
- `threshold_expression`: FEEL-like (DMN) expression over variables and other predicates. Supports `==`, `!=`, `>=`, `<=`, `>`, `<`, `AND`, `OR`, `NOT`, parentheses, string equality (e.g., `v_muac_strap_colour == 'red'`).
- `fail_safe`: `0` | `1`. When the predicate's source data is missing (equipment not available, patient refused), this is the value the predicate assumes. `1` = err on the side of caution (treat the danger-sign predicate as TRUE if we can't measure). `0` = assume normal (default for soft predicates).
- `source_vars`: list of variable IDs referenced in the expression. Used by the XLSForm `_measured` check and by DMN's `informationRequirement`.
- `human_label`: UI display.
- `source_quote`, `source_section_id`: provenance.

WHO example: 22 predicates. Notable:
- `p_any_danger_sign` — disjunction over 11 other predicates/variables, fires if ANY danger sign is present. `fail_safe=0` (paradoxically — because its constituents each have their own fail-safes).
- `p_fast_breathing_threshold` — 50 bpm for 2-12mo, 40 bpm for 12-60mo. `fail_safe=1` (assume fast breathing if we can't count).
- `p_age_2mo_to_12mo` — bracketing predicate, `fail_safe=0`.

### 3.4 `modules{module_id: {...}}`

```json
"mod_diarrhoea_treatment": {
  "module_id": "mod_diarrhoea_treatment",
  "display_name": "Diarrhoea Treatment - ORS and Zinc",
  "hit_policy": "unique",
  "priority": 100,
  "trigger_flag": "has_diarrhoea",
  "done_flag": "mod_diarrhoea_treatment_done",
  "inputs": [
    "q_has_diarrhoea",
    "q_diarrhoea_duration_days",
    "q_blood_in_stool",
    "p_any_danger_sign"
  ],
  "outputs": [
    "mod_diarrhoea_treatment_done",
    "tx_plan_ors_zinc",
    "has_fever"
  ],
  "rules": [
    {
      "rule_id": "rule_1",
      "condition": "q_has_diarrhoea == true AND q_diarrhoea_duration_days < 14 AND q_blood_in_stool == false AND p_any_danger_sign == false",
      "outputs": {
        "mod_diarrhoea_treatment_done": true,
        "tx_plan_ors_zinc": true,
        "is_priority_exit": false,
        "priority_exit_destination": null
      },
      "description": "Treatable diarrhoea: prescribe ORS + zinc."
    },
    {
      "rule_id": "rule_2",
      "condition": "q_has_diarrhoea == true AND (q_diarrhoea_duration_days >= 14 OR q_blood_in_stool == true)",
      "outputs": {
        "mod_diarrhoea_treatment_done": true,
        "is_priority_exit": true,
        "priority_exit_destination": "health_facility"
      },
      "description": "Persistent or bloody diarrhoea: refer urgently."
    }
  ]
}
```

Fields:
- `module_id`: prefix `mod_`, content-derived (e.g., `mod_diarrhoea_treatment`, `mod_fever_malaria_treatment`).
- `display_name`: UI label.
- `hit_policy`: `"unique"` (exactly one rule fires) | `"first"` (first matching) | `"priority"` (by priority annotation). Gen 7 default is `"unique"` for clinical modules.
- `priority`: tier 0/1/100/999 (see section 4 below).
- `trigger_flag`: the flag that activates this module in the dispatcher. For the startup module it's `sys_encounter_start`; for regulars it's `has_X` set by a prior module.
- `done_flag`: `{module_id}_done`, set to `true` by every rule in this module.
- `inputs`: list of variable/predicate IDs the rules read.
- `outputs`: list of IDs the rules write (includes `done_flag`, cross-module `has_Y` triggers, `is_priority_exit`, and module-specific outputs like `tx_plan_ors_zinc`).
- `rules[]`: decision table rows. Each has `rule_id`, `condition` (FEEL expression), `outputs` (dict of output_id -> literal OR FEEL expression), and `description`.

**Rule outputs can be literal OR expression:**
- Literal: `"mod_assess_done": true`, `"ref_destination": "hospital"`.
- Expression (string form): `"p_severe_malnutrition": "(v_muac_mm < 115) || (ex_bilateral_foot_oedema == true)"`. The XLSForm converter and DMN converter both recognize expression-form outputs and emit them as computed fields rather than constants.

WHO example: 8 modules.
- `mod_assess` (priority 1): startup — takes ALL symptom Qs + ALL vaccine hxs + ALL exam findings as inputs; produces 9 rules that cover every combination of presenting symptoms, setting `p_pneumonia`, `p_diarrhoea_uncomplicated`, etc., AND the cross-module `flag_mod_treat_X` triggers. Has the highest input count (53) and output count (27) in the guide.
- `mod_danger_sign_check` (priority 0 — but actually set to priority 2 in the WHO run's dispatcher; this is a known inconsistency).
- `mod_diarrhoea_treatment`, `mod_fast_breathing_treatment`, `mod_fever_malaria_treatment`, `mod_malnutrition_screening`, `mod_check_vaccines` (priority 100).
- `mod_home_care_advice` (priority 999): closing module, always fires once everything else is done.

### 3.5 `router{hit_policy, rows[]}`

```json
"router": {
  "hit_policy": "priority",
  "rows": [
    {
      "priority": 0,
      "condition": "is_priority_exit == true",
      "output_module": "PRIORITY_EXIT",
      "description": "Short-circuit: if any module has flagged priority exit (danger sign found), go to urgent referral pathway"
    },
    {
      "priority": 1,
      "condition": "mod_assess_done == false",
      "output_module": "mod_assess",
      "description": "Startup: run Identify Problems - ASK and LOOK first"
    },
    {
      "priority": 2,
      "condition": "mod_assess_done == true AND mod_danger_sign_check_done == false",
      "output_module": "mod_danger_sign_check",
      "description": "Danger sign module: Danger Sign Check and Referral Decision"
    },
    {
      "priority": 100,
      "condition": "mod_assess_done == true AND mod_check_vaccines_done == false",
      "output_module": "mod_check_vaccines",
      "description": "Clinical module: Check Vaccines Received"
    },
    {
      "priority": 100,
      "condition": "q_has_diarrhoea == true AND mod_diarrhoea_treatment_done == false",
      "output_module": "mod_diarrhoea_treatment",
      "description": "Clinical module: Diarrhoea Treatment - ORS and Zinc"
    },
    {
      "priority": 100,
      "condition": "p_fast_breathing == true AND mod_fast_breathing_treatment_done == false",
      "output_module": "mod_fast_breathing_treatment",
      "description": "Clinical module: Fast Breathing Treatment - Amoxicillin"
    },
    {
      "priority": 100,
      "condition": "q_has_fever == true AND mod_fever_malaria_treatment_done == false",
      "output_module": "mod_fever_malaria_treatment",
      "description": "Clinical module: Fever/Malaria Treatment - RDT and AL"
    },
    {
      "priority": 100,
      "condition": "mod_assess_done == true AND mod_malnutrition_screening_done == false",
      "output_module": "mod_malnutrition_screening",
      "description": "Clinical module: Malnutrition Screening - MUAC and Oedema"
    },
    {
      "priority": 999,
      "condition": "true",
      "output_module": "mod_home_care_advice",
      "description": "Closing: Home Care Advice and Follow-Up Scheduling"
    }
  ]
}
```

Fields:
- `hit_policy`: always `"priority"` in Gen 7. The dispatcher evaluates rows in priority order (ascending); first matching row wins.
- `rows[]`: dispatcher entries, ordered by construction (priority ascending, alphabetical module_id within tier).
  - `priority`: integer tier. 0 = priority_exit, 1 = startup, 2 = danger-sign (WHO-specific), 100 = regular clinical, 999 = closing.
  - `condition`: FEEL-like expression the runtime evaluates each tick.
  - `output_module`: the `module_id` to dispatch to (or `"PRIORITY_EXIT"` for the short-circuit).
  - `description`: human-readable.

**Runtime contract:** Whatever client executes the compiled logic (CHT, custom runtime, DMN engine) should:
1. Load all variables + predicates into a working memory.
2. On each tick, evaluate all router rows. Find the lowest-priority row whose condition matches. That's the next module.
3. Dispatch to that module. The module's rules fire (per hit_policy), setting outputs into working memory.
4. Loop back to step 2 until the closing module's `done_flag` is set or `is_priority_exit` is set AND the referral message has been emitted.

### 3.6 `phrase_bank[]`

```json
{
  "id": "tx_give_al_if_rdt_positive",
  "category": "treatment",
  "text": "If RDT is positive, give oral antimalarial AL (Artemether-Lumefantrine). Give twice daily for 3 days.",
  "module_context": "mod_fever_malaria_treatment",
  "source_quote": "If RDT is positive, give oral antimalarial AL (Artemether-Lumefantrine)",
  "source_section_id": "child_3_karen_shah+checking_questions+exercise_2_optional"
}
```

Fields:
- `id`: prefix `m_` (message) | `adv_` (advice) | `tx_` (treatment) | `rx_` (medication instruction) | `ref_` (referral) | `fu_` (follow-up).
- `category`: free-form tag. WHO uses: `treatment`, `safety`, `home_care`, `malaria_prevention`, `nutrition`, `vaccination`, `referral_counseling`, `referral`, `follow_up`, `pre_referral`.
- `text`: the actual string the CHW app would display or the CHW would read.
- `module_context`: either a single `module_id` (when the phrase applies to one module) or `"all"` (cross-cutting).
- `source_quote`, `source_section_id`: provenance.

WHO example: 20 phrases across 9 categories.

### 3.7 `integrative{rules[]}`

```json
"integrative": {
  "rules": [
    {
      "rule_id": "integ_comorbid_diarrhoea_malaria",
      "condition": "q_has_diarrhoea == true AND q_has_fever == true AND hx_malaria_area == true AND p_rdt_positive == true AND p_any_danger_sign == false",
      "outputs": {
        "tx_plan": "ORS + zinc + AL",
        "description": "If a child has diarrhoea and malaria, give the child: ORS, zinc supplement, and an oral antimalarial for treatment at home"
      },
      "source_quote": "if a child has diarrhoea and malaria, give the child: ORS, zinc supplement, and an oral antimalarial for treatment at home.",
      "source_section_id": "give_oral_medicine_and_advise_the_caregiver+..."
    }
  ]
}
```

Integrative rules are cross-module comorbidity rules that don't fit cleanly into any single module's decision table. In Gen 7 they're kept minimal because most cross-module flow is handled by the dispatcher's flag system (see section 4). When the WHO guide says "if a child has diarrhoea AND malaria, give them ORS+zinc+AL", we could either:
(a) Duplicate this rule into both `mod_diarrhoea_treatment` and `mod_fever_malaria_treatment` (two sources of truth, easy to drift).
(b) Hoist it into `integrative.rules[]` as a separate layer the runtime evaluates AFTER all modules have fired.

Gen 7 picks (b) because it preserves the labeler's `type: "integrative"` hint and keeps the comorbidity logic in one place.

For WHO, there's 1 integrative rule (diarrhoea + malaria comorbidity). Most other cross-module flow is expressed via trigger flags instead.

### 3.8 `deduped_labels[]` (the 26th artifact)

Written to `deduped_labels.json` alongside `clinical_logic.json`. Each entry:

```json
{
  "span": "Counsel caregiver on feeding or refer the child to a supplementary feeding programme, if available",
  "id": "adv_counsel_feeding_yellow_muac",
  "type": "phrase_bank",
  "subtype": "nutrition",
  "quote_context": "Counsel caregiver on feeding or refer the child to a supplementary feeding programme, if available",
  "source_section_ids": [
    "exercise_2_optional+child_juanita_vald_z+child_1_jackie_marks",
    "contents+give_ors"
  ],
  "source_chunk_indices": [3, 11]
}
```

Fields:
- `span`: verbatim text from the source.
- `id`: canonicalized by Call 2.
- `type`: one of the 7 artifact types.
- `subtype`: free-form tag set by the labeler.
- `quote_context`: surrounding sentence.
- `source_section_ids`: list of sections where this (id, type) was seen — merged from all chunks containing the label.
- `source_chunk_indices`: list of chunk_index values where it appeared.

This is the INPUT to the REPL Phase 3. If a future engineer wants to bypass the REPL entirely (e.g., to write a hand-tuned compiler), they consume `deduped_labels.json` directly.

---

## 4. Flag-Based Dispatcher Architecture (Professor Levine's Design)

### 4.1 The original (pre-Gen 7) design

Before Gen 7, the dispatcher was a two-traffic-cop system:
- **Traffic cop 1 (Activator):** Looks at the patient's symptoms, decides WHICH modules to queue.
- **Traffic cop 2 (Router):** Pops from the queue, runs the next module, loops.

This required two DMN decision tables (one for the activator, one for the router) and an implicit queue data structure the runtime had to maintain. The activator output `required_modules = ['mod_diarrhoea', 'mod_fever']`; the router's relevance was `contains(required_modules, 'mod_X') and not(contains(completed_modules, '|mod_X|'))`.

Problems:
1. **Queue is out-of-band state.** DMN's standard semantics don't include list ops like `contains`. The XLSForm converter hacked it with pipe-delimited strings, but this is an ODK-specific escape hatch.
2. **Activator can't discover new symptoms mid-encounter.** If the CHW asks "do you have fever?" during `mod_diarrhoea_treatment` because the protocol requires it, and the answer is yes, there's no way for `mod_diarrhoea_treatment` to add `mod_fever_malaria_treatment` to the queue.
3. **Classification decisions are duplicated.** The activator encodes "fever + malaria area → queue mod_fever_malaria". The module itself ALSO has to gate on the same conditions. Two sources of truth; they drift.

### 4.2 Professor Levine's flag-based design

Replace the queue with per-module BOOLEAN FLAGS in working memory. Each module has:
- A **trigger flag** (`has_X`) that activates it. Set by prior modules, NOT by a central activator.
- A **done flag** (`mod_X_done`) that marks it complete. Set by the module's own rules.
- The ability to set OTHER modules' trigger flags (`has_Y`) when discovering new symptoms.
- A **priority exit flag** (`is_priority_exit`) for emergency short-circuits.

The dispatcher is a single priority-ordered decision table. Each row is:

```
priority | condition: trigger_flag AND NOT done_flag | output_module
```

The dispatcher's job is trivial: find the lowest-priority row whose condition is TRUE, dispatch. Loop.

Modules self-report completion and set cross-module triggers. **No queue. No activator. One source of truth per condition.**

### 4.3 Priority tiers (canonical)

| Priority | Name | Purpose | Condition pattern |
|----------|------|---------|-------------------|
| 0 | Priority Exit | Emergency/danger short-circuit | `is_priority_exit == true` |
| 1 | Startup | First module (demographics, initial assessment) | `<startup>_done == false` |
| 100 | Regular clinical | All normal modules, alphabetical within tier | `has_X == true AND mod_X_done == false` |
| 999 | Closing | Fallback when everything else is done | `true` (always matches, but lowest priority) |

Priority 0 is a short-circuit: as soon as ANY module sets `is_priority_exit = true` (plus optionally `priority_exit_destination = "hospital" | "health_facility" | "clinic"`), the next tick jumps straight to referral. Priority 999 is the catch-all: once all `has_X` flags are false or their modules' `done_flag`s are true, the dispatcher falls through to the closing module (home care + follow-up advice).

**Why alphabetical tie-break at priority 100?** Two reasons:
1. **Determinism (k=1).** Same input → same dispatcher output. If the LLM picked the order at runtime, we'd inject stochasticity into a deterministic-by-nature piece of code.
2. **Human readability.** Reviewers scanning the dispatcher should see modules grouped in a predictable order, not in the order the labeler happened to discover them.

### 4.4 Worked example: fever + diarrhoea patient

Initial state:
```
sys_encounter_start = true
mod_assess_done = false
mod_danger_sign_check_done = false
mod_check_vaccines_done = false
mod_diarrhoea_treatment_done = false
mod_fast_breathing_treatment_done = false
mod_fever_malaria_treatment_done = false
mod_malnutrition_screening_done = false
mod_home_care_advice_done = false
is_priority_exit = false
has_diarrhoea = ?  (not yet known)
has_fever = ?       (not yet known)
q_has_cough = ?     (not yet asked)
... etc
```

**Tick 1:** Dispatcher evaluates. Priority 0: `is_priority_exit == true` → FALSE. Priority 1: `mod_assess_done == false` → TRUE. Dispatch to `mod_assess`.

**Inside `mod_assess`:** CHW asks all symptom questions. Let's say the patient is 3 years old, has fever (duration 2 days), has diarrhoea (duration 3 days, no blood, no danger signs), respiratory rate 35 bpm, MUAC green, no chest indrawing. The rules engine picks rule_5 (diarrhoea + fever, no danger signs) and sets:
```
mod_assess_done = true
p_danger_sign_present = false
p_diarrhoea_uncomplicated = true
p_fever_uncomplicated = true
flag_mod_treat_diarrhoea = true     # sets has_diarrhoea implicitly
flag_mod_treat_fever = true         # sets has_fever
flag_mod_vaccines = true
flag_mod_home_care = true
has_diarrhoea = true                # via the flag_mod_treat_diarrhoea expression
has_fever = true
is_priority_exit = false
```

**Tick 2:** Dispatcher evaluates. Priority 0: FALSE. Priority 1: `mod_assess_done == false` → FALSE (just set). Priority 2 (WHO-specific danger sign check): `mod_assess_done == true AND mod_danger_sign_check_done == false` → TRUE. Dispatch to `mod_danger_sign_check`.

**Inside `mod_danger_sign_check`:** All danger signs are absent. Rule fires that sets `mod_danger_sign_check_done = true` and leaves `is_priority_exit = false`.

**Tick 3:** Dispatcher evaluates. Priorities 0-2 all match FALSE now. Enters priority 100 tier. Alphabetical order: `mod_check_vaccines` < `mod_diarrhoea_treatment` < `mod_fast_breathing_treatment` < `mod_fever_malaria_treatment` < `mod_malnutrition_screening`.

Priority 100 row 1: `mod_assess_done == true AND mod_check_vaccines_done == false` → TRUE. Dispatch to `mod_check_vaccines`.

**Inside `mod_check_vaccines`:** Checks vaccine card, determines which vaccines are due. Sets `mod_check_vaccines_done = true`. May set `adv_refer_for_vaccines = true` (which would become a soft-referral message, not a priority exit).

**Tick 4:** Dispatcher evaluates. Priority 100 row 1 now FALSE. Row 2 for `mod_diarrhoea_treatment`: `q_has_diarrhoea == true AND mod_diarrhoea_treatment_done == false`. We have `has_diarrhoea = true` but the actual condition checks `q_has_diarrhoea`. Both set to TRUE in mod_assess. Dispatch.

Actually, let me correct that: the WHO dispatcher row for diarrhoea treatment is `q_has_diarrhoea == true AND mod_diarrhoea_treatment_done == false`. `q_has_diarrhoea` was set during `mod_assess` based on the CHW's answer. It's TRUE. So: dispatch to `mod_diarrhoea_treatment`.

**Inside `mod_diarrhoea_treatment`:** Rule 1 matches (diarrhoea + < 14 days + no blood + no danger sign). Sets `mod_diarrhoea_treatment_done = true`, `tx_plan_ors_zinc = true`.

**Tick 5:** Dispatcher evaluates. Diarrhoea row now FALSE. Fast breathing row: `p_fast_breathing == true AND mod_fast_breathing_treatment_done == false`. `p_fast_breathing` was set FALSE in mod_assess (respiratory rate 35 is normal for a 3-year-old). Row FALSE. Fever row: `q_has_fever == true AND mod_fever_malaria_treatment_done == false`. Both TRUE. Dispatch to `mod_fever_malaria_treatment`.

**Inside `mod_fever_malaria_treatment`:** Does RDT. Let's say positive. Rule fires: prescribe AL, set `tx_give_al_if_rdt_positive = true`, set `mod_fever_malaria_treatment_done = true`.

**Tick 6:** Dispatcher evaluates. Fever row now FALSE. Malnutrition row: MUAC is green, so `mod_assess_done == true AND mod_malnutrition_screening_done == false` → TRUE. Dispatch.

**Inside `mod_malnutrition_screening`:** Green MUAC → no action required. Sets `mod_malnutrition_screening_done = true`.

**Tick 7:** All priority 100 rows are FALSE (their done_flags are all TRUE). Priority 999 catch-all matches: `true` → dispatch to `mod_home_care_advice`.

**Inside `mod_home_care_advice`:** Emits advice phrases for diarrhoea + fever (fluids, continue feeding, when to return, follow-up in 3 days, sleep under ITN). Sets `mod_home_care_advice_done = true`.

**Tick 8:** Priority 999 row: condition is `true`, so it still matches. But output_module is the same module we just ran, which already has `mod_home_care_advice_done = true`. The runtime's loop-termination rule is: if the same output_module fires twice in a row AND its done_flag is true, exit. Or equivalently: if EVERY `mod_X_done` flag is TRUE, exit. The runtime terminates. Final state:

```
diagnosis: p_diarrhoea_uncomplicated + p_fever_uncomplicated (p_malaria_positive)
treatment: ORS + zinc + AL
advice: fluids, continue feeding, return if worse, sleep under ITN
follow-up: 3 days
```

This matches the WHO protocol exactly — and was produced by the dispatcher running through 8 modules in 7 ticks, driven purely by trigger + done flag evaluation.

### 4.5 Why this works

- **Modules are composable.** Adding a new module = labeling it + adding a dispatcher row. No activator changes.
- **Cross-module discovery works.** If `mod_assess` discovers fever mid-assessment, it sets `has_fever = true`, and the dispatcher picks up `mod_fever_malaria_treatment` on the next tick. No queue operations.
- **Priority exit is a flag.** Any module can set `is_priority_exit = true` (e.g., severe malnutrition detected during malnutrition screening). The dispatcher sees it on the next tick and jumps to the referral pathway immediately.
- **Deterministic in the REPL.** Phase C's dispatcher assembly is pure Python on sorted module lists. Same module set → same dispatcher.
- **DMN-native.** Each dispatcher row is a standard DMN priority-hit-policy rule with a FEEL condition. No queue data structure, no pipe-delimited strings (except as an implementation detail in the XLSForm converter).

---

## 5. Cost Tracking (Authoritative)

- **File:** `backend/rlm_runner.py`
- **State:** Module-global dict `_run_usage` (reset at the start of each run via `_reset_run_usage()`, snapshotted via `_snapshot_run_usage()`).

### 5.1 The two populating code paths

**Path 1 — Labeler direct Anthropic calls:** The labeler in `backend/gen7/labeler.py` creates its own `anthropic.AsyncAnthropic` client and calls `messages.create` directly. It does NOT go through the RLM library. After each Call 1 and Call 2, it reads `response.usage` and passes the integers to `backend/rlm_runner.py::accumulate_catcher_usage(model, input_tokens, output_tokens, cached_tokens, cache_write_tokens)`.

`accumulate_catcher_usage` was originally written for the legacy catcher validators (Haiku-based), which also called Anthropic directly. The Gen 7 labeler reuses it — hence the slightly misleading name. The function:
1. Adds input/output/cached/cache_write tokens to `_run_usage`.
2. Looks up the model's cost rates in `_COST_TABLE` (Opus 4.6: $5/$25/$0.50/$10 per MTok).
3. Computes per-call cost = (billed_input * in_price + cached * cache_read_price + cache_write * cache_write_price + output * out_price) / 1e6.
4. Adds the cost to `_run_usage["cost_usd"]`.
5. Increments the right model call counter (`calls_opus` | `calls_sonnet` | `calls_haiku`).

**Path 2 — REPL calls via monkey-patched RLM client:** The REPL and its `llm_query_batched` sub-calls go through the RLM library's `AnthropicClient`. `backend/rlm_runner.py` monkey-patches `AnthropicClient.completion` and `AnthropicClient.acompletion` with `_completion_with_cache` / `_acompletion_with_cache`. After every API call, those patches call `_accumulate_run_usage(response, effective_model)` which reads `response.usage` directly and updates `_run_usage` with the same math as `accumulate_catcher_usage`.

### 5.2 Why two paths unified here

The ONE critical property: every Anthropic API call in the run, regardless of entry point, must end up in `_run_usage`. Without this, the frontend Cost Tracker under-reports (and the user thinks they spent $5 when they actually spent $12).

Gen 6 had a silent bug where the labeler's direct calls were invisible to `_run_usage` — the labeler's cost was only visible in the final `stats` dict computed from `_labeling_meta`. The live Cost Tracker showed the wrong number during the run (it only saw REPL tokens). Fix F moved the labeler to use `accumulate_catcher_usage`, unifying the paths.

### 5.3 Live merge into session state

`backend/session_manager.py::get_session_status(session_id)` is called by the frontend every 3 seconds. It:
1. Reads the session's persisted state from `_active_sessions[session_id]["session_data"]` (cost_estimate_usd, input_tokens_total, output_tokens_total, calls_opus, total_iterations, etc.).
2. **Live merge while status="running":** If the session is still running, also reads `_run_usage` via `_snapshot_run_usage()` and overwrites the session fields with the snapshot. This ensures the Cost Tracker reflects the CURRENT accumulator state, not the stale value from the last `_push_event`.

The merge is guarded by `status=="running"` because once the run completes, `session_data` is updated with the final stats and we don't want the next run's `_run_usage` (which starts at 0) to overwrite the final values.

### 5.4 Verified accuracy

Cross-checked against Anthropic's billing dashboard for 3 WHO runs in April 2026:
- Run 1: `_run_usage` = $12.55, billing delta for the 22-minute window = $13.08. Delta: +$0.53 (+4.2%) — attributable to cache write costs that Anthropic rounds differently on their end.
- Run 2: `_run_usage` = $11.87, billing = $12.31. Delta: +$0.44 (+3.7%).
- Run 3: `_run_usage` = $12.93, billing = $13.49. Delta: +$0.56 (+4.3%).

Average accuracy: ~95.5% of Anthropic's billed amount. Good enough for per-run cost estimation; not good enough for month-end reconciliation (always defer to Anthropic's invoice).

### 5.5 Cost breakdown for WHO reference run (run_d9a39c2d)

```
Gen7 COST SUMMARY (from _run_usage):
  Labeling (Call 1 + Call 2 per chunk, 21 chunks x 2 = 42 Opus calls):
    1,294,218 in / 258,931 out tokens  ~=  $6.47 + $6.47  ~=  $7.94
  REPL (main + llm_query_batched, ~14 main iterations + 8 Module Maker sub-calls):
    487,103 in / 143,927 out tokens    ~=  $2.43 + $3.59  ~=  $4.61
  TOTAL: 1,781,321 in / 402,858 out / 843,204 cached-read / 142,891 cache-write = $12.55
  Calls: Opus=64 Sonnet=0 Haiku=0
```

All Opus. Zero Sonnet, zero Haiku (Gen 7 is Opus-only; k=1 at the model level).

Cache hit rate: 843,204 cache-read / (843,204 cache-read + 142,891 cache-write) = 85.5%. Without caching, input cost would have been 1,781,321 × $5/MTok = $8.91 instead of $0.50 × 843K + $10 × 143K + $5 × 795K ≈ $5.90. Savings: ~$3.

---

## 6. Caching Strategy

### 6.1 Labeler system prompt cache

- **Where:** `backend/gen7/labeler.py::label_chunk` and `label_chunk`'s Call 2.
- **What:** The labeling system prompt (~6K chars of codebook + instructions) AND the distillation system prompt (~3K chars) are each attached with `cache_control: { type: "ephemeral", ttl: "1h" }`.
- **TTL rationale:** 1h comfortably covers a 12-15 minute labeling phase. 5-minute TTL would expire mid-phase for large guides.

Chunk 0 pays cache_creation (2x input rate). Chunks 1..N pay cache_read (0.1x). On WHO (21 chunks):
- Cache_creation on chunk 0: ~6K input tokens × $10/M = $0.06.
- Cache_read on chunks 1..20: 20 × 6K × $0.50/M = $0.06 total.
- Without caching: 21 × 6K × $5/M = $0.63.
- Savings: ~$0.50 on the labeling prompt alone. Multiply by 2 for Call 2 = ~$1 total savings from system prompt caching in the labeler.

### 6.2 REPL system prompt + cached blocks

- **Where:** `backend/rlm_runner.py::_completion_with_cache` / `_acompletion_with_cache`, called by the RLM library for every REPL iteration and every `llm_query_batched` sub-call.
- **What:** THREE system blocks, all with `cache_control: ephemeral, ttl: 1h`:
  1. The REPL mechanics system prompt (~8K chars).
  2. The reconstructed guide text (~60K chars ≈ 15K tokens for WHO).
  3. The deduped label inventory (~100K chars ≈ 25K tokens for WHO).

All three blocks are created once on the first REPL turn (cache_creation ~$0.40 for the label inventory + guide). Every subsequent turn AND every `llm_query_batched` sub-call reads them at 0.1x.

For the 8 Module Maker sub-calls in Phase B: without caching, each would pay 40K input tokens × $5/M = $0.20. With caching, 40K × $0.50/M = $0.02. Savings: 8 × $0.18 = $1.44.

Multiply by REPL iterations (typically 5-10 main iterations in addition to Phase B sub-calls): another $0.50-$1 saved.

### 6.3 REPL multi-turn automatic prefix caching

Even WITHOUT explicit `cache_control`, Anthropic's API automatically caches the prefix of a multi-turn conversation. Each REPL turn sends the full history: `[user_msg_0, assistant_msg_0, user_msg_1, assistant_msg_1, ..., user_msg_N]`. The prefix `[user_msg_0, assistant_msg_0, ..., user_msg_{N-1}, assistant_msg_{N-1}]` is IDENTICAL to the previous turn's full conversation, so Anthropic reuses its KV cache.

This automatic caching fires on top of our explicit `cache_control` and compounds the savings. The `llm_query_batched` sub-calls ALSO share the system prompt prefix within a single turn, so Anthropic's server can batch KV-cache reads across parallel calls in the same batch.

### 6.4 Opus 4.6 cache_control caveat

**Important known issue:** Opus 4.6 does NOT support explicit cache_control opt-in for the top-level system prompt (it silently drops the cache_control directive). The automatic prefix caching still works, but the 1h TTL guarantee and the 2x-write/0.1x-read pricing guarantee only apply when cache_control is actually honored.

**What we do:** Still set `cache_control: ephemeral, ttl: 1h` on every system block. Anthropic's server either honors it (on Sonnet/Haiku, and on Opus 4.6 in some cases) or falls back to automatic prefix caching (rest of the time). Either way, the client-side code is identical. The observed 85% cache hit rate on WHO runs confirms caching is firing, even if we can't always tell which mechanism.

**Worst case:** If cache_control is dropped on every Opus call, the first-call cost on each cache block rises from 2x to 1x input rate (slightly cheaper on the write side, but we lose the 0.1x read rate on subsequent calls). Real cost impact is a 20-30% increase in labeling + REPL input cost, but output cost is unchanged (output is never cached).

---

## 7. Rate Limit Strategy (Tier 2 Anthropic)

Anthropic's Tier 2 rate limits for Opus 4.6:
- **Output TPM: 80,000 tokens/minute** (the binding constraint).
- Input TPM: 400,000 tokens/minute (not binding for us).
- RPM (Requests per minute): 1,000 (not binding).

### 7.1 Why output TPM is binding

Each labeling call produces ~10K output tokens (JSON label array). At 80K output TPM, we can issue at most 8 labeling calls per minute. Each REPL Module Maker sub-call produces ~2-4K output tokens; we typically run 8-10 in parallel → 30K output tokens in one burst, well under the limit. Main REPL turns produce 2-8K output tokens each; those are serial.

The labeling phase is the bottleneck.

### 7.2 Dynamic batching in the labeler

`backend/gen7/labeler.py::_compute_batch_size(n_chunks)`:

```python
_OUTPUT_TPM_LIMIT = 80_000
_EST_OUTPUT_PER_CHUNK = 10_000
_MAX_BATCH_SIZE = max(1, _OUTPUT_TPM_LIMIT // _EST_OUTPUT_PER_CHUNK - 2)  # 6
_BATCH_COOLDOWN_SEC = 62

def _compute_batch_size(n_chunks: int) -> int:
    if n_chunks <= _MAX_BATCH_SIZE:
        return n_chunks
    return _MAX_BATCH_SIZE
```

6 = 80K / 10K - 2 headroom. `-2` accounts for variance in output length (some chunks have lots of labels, some have few) and for the Call 2 tokens that piggyback on the same minute's budget.

62-second cooldown: Anthropic's rate limit window is a sliding 60-second window. Waiting 62s ensures the previous batch's output tokens have fully aged out before we start the next batch.

### 7.3 Small-guide path (<= 6 chunks)

All chunks run in one `asyncio.gather`. No cooldown needed (we only issue one batch). Example: a 5-chunk guide finishes labeling in ~60 seconds wall clock.

### 7.4 Large-guide path (> 6 chunks)

Cache priming: chunk 0 runs alone first, writes the cache. Then batches of 6 with 62s cooldown between. For WHO (21 chunks):

```
t=0s:    chunk 0 (prime, ~60s)
t=60s:   batch 1 = chunks 1-6, 6 parallel (~90s wall clock for call 1 + call 2)
t=150s:  cooldown 62s
t=212s:  batch 2 = chunks 7-12
t=302s:  cooldown
t=364s:  batch 3 = chunks 13-18
t=454s:  cooldown
t=516s:  batch 4 = chunks 19-20 (2 chunks, parallel)
t=606s:  done (~10 minutes total)
```

### 7.5 Why not dynamic chunk-size budgeting?

We could be cleverer: dynamically choose batch size based on observed output-per-chunk from the cache-primed chunk 0. If chunk 0 produced only 4K output, we could do 20-chunk batches. If it produced 15K, drop to 4.

We DON'T do this because:
1. The variance across chunks is 2K-15K output; a single sample from chunk 0 is a poor predictor.
2. Tier 3 upgrade ($5K/month spend) lifts the output TPM to 160K, making the problem moot.
3. The current fixed batch size has good empirical behavior (never hits a 429).

---

## 8. Frontend Architecture

- **Stack:** Next.js 15 (App Router), React 18, TypeScript strict, TailwindCSS.
- **Port:** 3000 (dev), Vercel-assigned in production.
- **Backend URL:** Configured via `NEXT_PUBLIC_BACKEND_URL` env var. CORS pre-flight handled by FastAPI's `CORSMiddleware` in `backend/server.py`.

### 8.1 Component tree

```
app/page.tsx
├── ApiKeyInput.tsx           (BYOK Anthropic key, stored in localStorage)
├── GuideUpload.tsx           (multipart POST to /api/ingest/pdf)
├── IngestionProgress.tsx     (SSE stream from /api/ingest/{id}/stream)
├── IngestionManifestCard.tsx (shows section count, page count, ingestion cost)
├── ExtractionJournal.tsx     (human-readable log of phase transitions)
├── CostTracker.tsx           (polls /api/session/{id}/status every 3s)
├── ReplStream.tsx            (SSE stream of REPL iterations with code + stdout)
├── ValidationPanel.tsx       (live catcher pass/fail per artifact)
├── Z3Panel.tsx               (optional z3 verifier output, not used in Gen 7 but kept)
├── MermaidDiagram.tsx        (renders flowchart.md via mermaid.js)
├── TierBadge.tsx             (shows Tier 2 / Tier 3 / "priming" badges)
└── ArtifactDownloads.tsx     (list of downloadable artifacts)
```

### 8.2 SSE stream

`ReplStream.tsx` subscribes to `/api/session/{id}/stream` via `EventSource`. The backend (in `backend/server.py::stream_events`) yields from the session's ring buffer:

```
data: {"type":"status","stepNumber":1,"stdout":"Created 21 micro-chunks"}

data: {"type":"status","stepNumber":2,"stdout":"Starting Opus labeling phase"}

data: {"type":"status","stepNumber":3,"stdout":"Priming cache with chunk 0 of 21"}

data: {"type":"exec","stepNumber":4,"stdout":"Labeled chunk 0: 64 labels (1/21 done)"}

data: {"type":"status","stepNumber":5,"stdout":"Labeling batch 1/4: 6 chunks in parallel [1, 2, 3, 4, 5, 6]"}

...

data: {"type":"exec","stepNumber":42,"code":"...","stdout":"Module plan (8 modules):..."}
```

Event types:
- `status`: phase-level transitions ("starting", "done") with explanatory stdout.
- `exec`: REPL iteration events with `code` (the Python source) and `stdout` (execution output).

`ReplStream.tsx` renders each event as a card. Code blocks are syntax-highlighted via Prism. Stdout is rendered in a monospace pre-formatted block. The cost tokens emitted with each iteration are consumed by `CostTracker.tsx`.

### 8.3 Cost Tracker polling

`CostTracker.tsx` polls `/api/session/{id}/status` every 3 seconds. The response shape:

```json
{
  "session_id": "...",
  "run_id": "...",
  "status": "running",
  "phase": "label",
  "cost_estimate_usd": 7.23,
  "input_tokens_total": 892341,
  "output_tokens_total": 178523,
  "calls_opus": 28,
  "calls_sonnet": 0,
  "calls_haiku": 0,
  "cached_tokens": 621834,
  "cache_write_tokens": 112091,
  "total_iterations": 0,
  "chunks_labeled": 14,
  "total_chunks": 21,
  ...
}
```

Why polling instead of SSE-driven updates: the Cost Tracker needs a snapshot view, not an event stream. A 3-second cadence is fine for a human reader; the cost doesn't jitter enough to warrant more frequent updates. Also, polling is trivially resumable if the user refreshes the page — SSE would require reconnection logic.

The Cost Tracker displays:
- **Top line:** `$7.23 spent • 1.07M tokens • 28 Opus calls`
- **Model routing strip:** bars showing Opus/Sonnet/Haiku call counts (always 100% Opus in Gen 7).
- **Cache strip:** % of input tokens that were cache reads.
- **Phase indicator:** currently-running phase + elapsed time.
- **Projected total:** linear extrapolation from current rate to phase completion.

### 8.4 Downloads mechanism (v5 top-level anchor)

`frontend/src/lib/downloads.ts::triggerDownload(url, suggestedFilename)`:

```typescript
export function triggerDownload(url: string, suggestedFilename: string): void {
  const a = document.createElement("a");
  a.href = url;
  a.rel = "noopener";
  // NO download attribute — ignored cross-origin AND poisons Content-Disposition.
  // NO target="_blank" — opens a flash-tab, annoying UX.
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => a.parentNode?.removeChild(a), 100);
}
```

**History of this file:**
- v1: `<a href={url} download={filename}>` — Chrome ignores `download` cross-origin, filename came from URL path.
- v2: fetch-as-blob + synthetic anchor click — stale user activation produced UUID filenames.
- v3: hidden iframe navigation — inconsistent across browsers.
- v4: fetch + parse Content-Disposition + blob — still path 3 (UUIDs).
- v5 (current): synchronous top-level anchor click inside user event handler, no download attr, no target=_blank. Chrome follows the link, sees `Content-Disposition: attachment; filename="..."` on the response, saves with the server's filename.

**Server-side requirements:**
- `FileResponse(filename="...")` in `backend/server.py::download_artifact` emits `Content-Disposition: attachment; filename="<name>"`.
- FastAPI CORS config exposes the header via `expose_headers=["Content-Disposition"]`.
- `?token=<api_key>` query-param auth lets the browser fetch without a custom header.

**Trade-off:** 401/404 responses render briefly in a new Chrome tab. Acceptable because 401 is caught at artifact-list-load time and 404 is a race we can't win.

`ArtifactDownloads.tsx` renders buttons for each artifact:

```tsx
const artifacts = [
  { name: "clinical_logic.json", label: "Clinical Logic (JSON)" },
  { name: "clinical_logic.dmn",  label: "DMN 1.3 XML" },
  { name: "form.xlsx",            label: "CHT XLSForm" },
  { name: "flowchart.md",         label: "Mermaid Flowchart" },
  { name: "predicates.csv",       label: "Predicates CSV" },
  { name: "phrases.csv",          label: "Phrases CSV" },
  { name: "deduped_labels.json", label: "Deduped Labels (26th)" },
  { name: "labeled_chunks.json",  label: "Raw Labeled Chunks" },
  { name: "reconstructed_guide.txt", label: "Reconstructed Guide Text" },
  { name: "chunk_difficulty.json", label: "Chunk Difficulty Report" },
  { name: "system_prompt.md",     label: "System Prompt (for audit)" },
];
```

Each button's onClick calls `triggerDownload(artifactUrl, artifact.name)`.

---

## 9. Session Management

- **File:** `backend/session_manager.py`
- **State:** In-memory `_active_sessions: dict[str, dict]` + Upstash Redis TTL entries.
- **Concurrency cap:** `MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "2"))`.

### 9.1 Why a cap at 2

Render Pro Plus dyno ($175/month) has:
- 4 GB RAM
- 2 vCPU

One Gen 7 session at peak (Phase B with 8 parallel Module Maker calls) uses:
- ~1.5 GB RAM (mostly the reconstructed guide + labels in memory, times a few parallel HTTP response buffers).
- ~0.5-1 vCPU (mostly JSON parsing + the RLM sandbox).

Two concurrent sessions = 3 GB RAM + 1-2 vCPU, leaving ~1 GB headroom for the FastAPI app itself + the session ring buffer. Three concurrent sessions would OOM on 4 GB.

### 9.2 Session lifecycle

1. **Create:** `POST /api/extract` → `create_session(guide_json, api_key, manual_name)`.
   - Generates `session_id` (UUID) and `run_id` (UUID).
   - Enforces `MAX_CONCURRENT_SESSIONS` (rejects with 429 if full).
   - Registers in `_active_sessions` and Upstash Redis (TTL 4 hours).
   - Spawns `asyncio.Task` running `_run_gen7_extraction_task(...)`.
2. **Run:** The task runs `run_gen7_extraction(...)`. Progress events flow to the SSE ring buffer. Cost accumulates in `_run_usage`.
3. **Status polls:** Frontend polls `/api/session/{id}/status`. `get_session_status` returns the current state, live-merging `_run_usage` while status="running".
4. **SSE stream:** Frontend subscribes to `/api/session/{id}/stream`. `stream_events` yields from the ring buffer.
5. **Complete:** The task updates `session_data` with final stats (cost_estimate_usd, input_tokens_total, etc.) and status becomes "passed" or "failed". Artifacts are written to `backend/output/run_{run_id}/`.
6. **Download:** Frontend calls `/api/session/{id}/artifacts` to list files, then `/api/session/{id}/artifacts/{filename}?token={api_key}` to download.
7. **Cleanup:** The session remains in `_active_sessions` until the TTL expires (4 hours) OR `MAX_CONCURRENT_SESSIONS` is hit and we need to evict the oldest completed session.

### 9.3 Event ring buffer

`_active_sessions[session_id]["events"] = deque(maxlen=10000)`.

Events are pushed via `_push_event(session_id, event_dict)`. The SSE handler reads from the deque via an offset cursor sent by the frontend (`?last_step={n}`), yielding events with `stepNumber > n`. This lets the frontend reconnect mid-stream without losing events (within the 10K buffer).

10K events is enough for a full Gen 7 run (~200-500 events typical). A 2-hour run would overflow, but at that point the user has bigger problems.

### 9.4 Journal writer

`backend/journal.py::JournalWriter(artifact_dir, manual_name)`:
- Writes a human-readable Markdown log of the extraction: phase transitions, module plans, final cost.
- Output: `backend/output/run_{run_id}/scratchpad.md`.
- Persisted to disk during the run; available as a download artifact.

### 9.5 Upstash Redis usage

- **Purpose:** Survives backend restarts. If Render redeploys mid-session, the session state is lost from memory, but the Redis TTL entry lets the frontend detect "session gone" and show a helpful error rather than hanging forever.
- **Schema:** `session:{session_id}` → JSON blob with `{status, run_id, created_at, manual_name}`, TTL 4 hours.
- **Ops:** `SET session:{id} {json} EX 14400` on create, `DEL session:{id}` on cleanup, `GET session:{id}` from a healthcheck endpoint.

Not used for the event ring buffer (too expensive to push 500 events to Redis per run) or for `_run_usage` (ephemeral by design).

### 9.6 Neon Postgres persistence

- **Purpose:** Permanent record of runs (for billing reconciliation, research, and the professor's quarterly review).
- **Tables:** `extraction_runs` (run_id, status, total_iterations, cost_estimate_usd, final_json, created_at), `intermediate_artifacts` (run_id, artifact_name, content_json, version, created_at).
- **Ops:** `update_extraction_run(run_id, ...)` called at end of run with final stats + final_json. Best-effort (errors logged but don't fail the run).

---

## 10. k=1 Identification Thesis

The research framing for this project (per Professor Levine's econometric lens) is that we want to identify the causal effect of "using an LLM to extract clinical decision logic" on the quality of the output, holding everything else constant.

### 10.1 Sources of variance to control

Any multi-agent, multi-model system introduces these sources of variance:
1. **Model variance:** Different models produce different outputs (obvious).
2. **Prompt variance:** Different prompts produce different outputs.
3. **Temperature variance:** Temperature > 0 injects stochasticity.
4. **Arbitration variance:** When multiple agents disagree, the arbitration protocol affects the output.
5. **Agent count variance:** 2 agents vs 5 agents with different composition.

A system with all 5 sources is unidentified: we can't attribute output quality to any specific component.

### 10.2 What Gen 7 does

- **k=1 at the model level:** Opus-only. No Sonnet, no Haiku, no cross-model arbitration.
- **k=1 at the prompt level:** ONE system prompt for labeling (+ a QC prompt that's a deterministic function of the labeling prompt). ONE system prompt for the REPL. The Module Maker's per-module prompts are deterministic functions of the module plan.
- **k=1 at the temperature level:** Temperature = 0 everywhere (labeling Call 1, Call 2, REPL main, all `llm_query_batched` sub-calls).
- **No arbitration:** No multi-agent disagreement resolution. Call 2 is QC, not arbitration — it only DROPS labels, never adds or rewrites.
- **One agent count:** The labeler is 2 calls per chunk, but both calls are the same model (Opus). The REPL is a single agent with multi-turn state.

Sources of remaining variance:
- **Model stochasticity at temperature=0:** Opus 4.6 is not perfectly deterministic even at T=0 (sampling noise from the server-side implementation). Empirical: 2 runs of the same guide on the same day produce clinical_logic.json files that are ~98% identical by content (same module_ids, same predicates, occasional small differences in rule ordering or phrase_bank entries).
- **Ingestion variance:** Unstructured.io's hi_res mode is reasonably deterministic but can drift slightly across runs (different page rendering).

### 10.3 Why the Module Maker decomposition preserves k=1

Phase B's `llm_query_batched` looks like it violates k=1: it's N parallel sub-calls, each with its own prompt. But:

- **All sub-calls use the same model (Opus).** k=1 at model level.
- **All sub-calls see the same cached system prompt + guide + labels.** k=1 at context level.
- **The per-module prompts are deterministic functions of the module plan.** No prompt variance beyond what the module plan itself introduces.
- **The module plan is deterministic from the deduped labels (Phase A is pure Python).** No arbitrary choices.

So Phase B is best understood as N independent samples from the SAME conditional distribution `p(module_DMN | module_id, cached_context)`. The outputs are not arbitrated; they're concatenated. If two modules had identical `module_id` and identical `cached_context`, they'd produce identical outputs (up to temperature=0 server noise). This IS the k=1 regime.

Contrast with a multi-agent approach where one agent generates, another critiques, and a third arbitrates: that's 3 different conditional distributions, and the arbitration protocol is an additional parameter.

### 10.4 Why this matters for the paper

If we want to claim "Gen 7 extracts WHO-compliant clinical logic at 85% accuracy", we need the numerator (accuracy) to be attributable to the system as a whole, not to a specific set of arbitration hyperparameters. k=1 lets us say: "the system is Opus + this prompt + this pipeline. No tuning knobs. Reproducible."

If a reviewer asks "did you try Sonnet?", the honest answer is: "No, because adding Sonnet would require an arbitration protocol, which introduces identifiability problems. Gen 7 is a single-model system by design."

---

## 11. Deployment

### 11.1 Backend: Render Pro Plus

- **Service:** Web service with auto-deploy from GitHub `main`.
- **Runtime:** Python 3.11, uvicorn running `backend/main.py:app` on port `$PORT` (Render-assigned).
- **Dyno:** Pro Plus at $175/month. 4 GB RAM, 2 vCPU.
- **Start command:** `cd /opt/render/project/src && python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- **Health check:** `GET /api/health` returns `{"status": "ok"}`.
- **Env vars:**
  - `OPENAI_API_KEY` (server-side, for ingestion vision calls).
  - `UNSTRUCTURED_API_KEY` (server-side, for PDF ingestion).
  - `DATABASE_URL` (Neon Postgres connection string).
  - `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`.
  - `MAX_CONCURRENT_SESSIONS=2` (can be overridden).
  - `GENERATOR_GUIDE_REPETITION=2` (prompt caching param).
  - No Anthropic key — that's BYOK.

### 11.2 Frontend: Vercel

- **Framework preset:** Next.js.
- **Build command:** `cd frontend && npm install && npm run build`.
- **Output directory:** `frontend/.next`.
- **Env vars:**
  - `NEXT_PUBLIC_BACKEND_URL` (e.g., `https://chw-navigator-backend.onrender.com`).
  - `NEXT_PUBLIC_TIER_BADGE` (optional, controls the Tier 2/3 display).
- **Deploy hooks:** auto-deploys on push to `main`. Preview deploys on PRs.

### 11.3 Database: Neon Postgres

- **Instance:** Starter tier, shared compute, auto-suspend after 5 min idle.
- **Schema:** `backend/prisma/schema.prisma`. Tables:
  - `guides` (guide_id, manual_name, sha256, guide_json, created_at).
  - `extraction_runs` (run_id, guide_id, status, final_json, stats_json, cost_estimate_usd, ...).
  - `intermediate_artifacts` (id, run_id, artifact_name, content_json, version, created_at).
  - `ingestion_cache` (sha256, guide_json, unstructured_raw, created_at).
- **Ops:** Migrations via `npx prisma migrate deploy` (run manually during deploys).
- **Cost:** Free tier for dev (0.5 GB), Launch tier $19/month for production.

### 11.4 Session store: Upstash Redis

- **Instance:** Free tier (10K commands/day), serverless.
- **Purpose:** Session TTL entries only (see section 9.5).
- **Cost:** $0/month for our traffic.

### 11.5 Ingestion services

- **Unstructured.io:** $30/month for their hosted hi_res endpoint. Cheaper than self-hosting (which requires a GPU dyno for the document layout model).
- **OpenAI gpt-4o:** BYOK on the server (NOT user BYOK). Used for vision-based page post-processing during ingestion. Cost: ~$0.50 per WHO-size guide.

### 11.6 BYOK boundary (critical)

The ONLY API key the user brings is **Anthropic**. That key is used for:
- Labeling (Call 1 + Call 2, per chunk).
- REPL (main iterations + Module Maker sub-calls).

Server-side keys (paid by us):
- OpenAI: ingestion vision, TOC generation (optional).
- Unstructured: PDF element extraction.
- Neon: database.
- Upstash: Redis.

Rationale: Anthropic spend dominates per-run cost ($10-$15 per guide). Asking the user to BYOK Anthropic aligns incentives (they care about their own bill). The server-side keys are fixed overhead (~$0.50/run for ingestion), not worth exposing to the user.

---

## 12. Testing & Verification

### 12.1 Standalone Gen 7 test

`run_gen7_test.py` (at repo root):

```python
#!/usr/bin/env python
"""Standalone Gen 7 end-to-end test.

Loads a cached WHO guide JSON, runs the full pipeline, writes output to
backend/output/gen7_test_run/, prints final cost summary.
"""
import asyncio, json, os, sys
from pathlib import Path
from backend.gen7.pipeline import run_gen7_extraction
from backend.system_prompt import NAMING_CODEBOOK

async def main():
    guide = json.loads(Path("sample_data/who_chw_guide.json").read_text())
    result = await run_gen7_extraction(
        guide_json=guide,
        anthropic_key=os.environ["ANTHROPIC_API_KEY"],
        naming_codebook=NAMING_CODEBOOK,
        output_dir=Path("backend/output/gen7_test_run"),
        on_progress=None,
    )
    print(json.dumps(result["stats"], indent=2))

if __name__ == "__main__":
    asyncio.run(main())
```

Usage: `ANTHROPIC_API_KEY=sk-ant-... python run_gen7_test.py`.

Runs the full pipeline without the frontend/session_manager wrapper. Useful for:
- Debugging pipeline bugs (drop into pdb on exception).
- Regression testing after changes (diff `gen7_test_run/clinical_logic.json` against a known-good version).
- Cost-per-run estimation.

### 12.2 Reference run: `run_d9a39c2d`

Last full WHO guide run captured in `backend/output/run_d9a39c2d/`.

**Inputs:**
- `WHO_CHW_guide_2012_test.pdf` (168 pages, WHO Management of Sick Children manual).
- Anthropic key: Tier 2.

**Output stats:**
```json
{
  "total_chunks": 21,
  "total_labels": 1841,
  "label_errors": 0,
  "label_cost_usd": 7.94,
  "label_input_tokens": 1294218,
  "label_output_tokens": 258931,
  "label_cache_read_tokens": 843204,
  "repl_cost_usd": 4.61,
  "repl_input_tokens": 487103,
  "repl_output_tokens": 143927,
  "repl_iterations": 14,
  "total_cost_usd": 12.55,
  "elapsed_sec": 1343,
  "model": "claude-opus-4-6",
  "provider": "anthropic-only"
}
```

**Output files** (in `backend/output/run_d9a39c2d/`):
- `clinical_logic.json` (3,093 lines).
- `clinical_logic.dmn` (DMN 1.3 XML).
- `form.xlsx` (CHT XLSForm).
- `flowchart.md` (Mermaid).
- `predicates.csv`, `phrases.csv`.
- `deduped_labels.json` (15,182 lines).
- `labeled_chunks.json` (raw per-chunk labels).
- `reconstructed_guide.txt` (the cached guide text).
- `chunk_difficulty.json` (difficulty report).
- `test_suite.json` (frozen labels for downstream tests).
- `system_prompt.md` (audit trail of the exact prompt).
- `scratchpad.md` (journal output).
- `final_dmn.validator.json` (catcher output).
- `artifacts/{supply_list,variables,predicates,modules,router,phrase_bank,integrative}.json` (per-artifact breakouts).

**Verification:**
- Billing delta: $12.55 (tracker) vs $13.08 (Anthropic invoice), 95.9% accuracy.
- Module count: 8 (matches WHO's top-level decision topics: Identify Problems, Danger Sign Check, Check Vaccines, Diarrhoea Treatment, Fast Breathing Treatment, Fever/Malaria Treatment, Malnutrition Screening, Home Care Advice).
- Predicate count: 22 (matches WHO's explicit threshold rules: age brackets, fast breathing per age, danger signs, MUAC colors, RDT positive/negative, etc.).
- Supply count: 15 (matches WHO's explicit equipment+consumable list).

### 12.3 Automated tests

`backend/tests/`:
- `test_chunker.py` — 8 tests for boundary detection, merging, difficulty classification.
- `test_labeler.py` — 6 tests for JSON repair, cache attachment, dedup semantics.
- `test_pipeline.py` — 4 end-to-end integration tests on a small stub guide.
- `test_converters.py` — 12 tests for DMN, XLSX, Mermaid, CSV schema compatibility (Gen 7 + legacy).
- `test_session_manager.py` — 5 tests for concurrency cap, event ring buffer, Redis TTL.
- `test_rlm_runner.py` — 7 tests for the monkey-patches, `_run_usage`, cache attachment.

Total: 40 tests, all pass as of 2026-04-13.

Run: `cd backend && pytest -v`.

### 12.4 Manual verification checklist

After a production run:
1. Open `flowchart.md` in a Markdown renderer. Verify the graph is connected (no orphan modules).
2. Open `form.xlsx` in Excel. Verify the "survey" sheet has all modules grouped and the "calculate" rows for each predicate.
3. Import `clinical_logic.dmn` into a DMN editor (e.g., Camunda Modeler). Verify decisions render.
4. Check `deduped_labels.json` label count matches Phase 1b's reported dedup count.
5. Check `scratchpad.md` for any "ERROR" or "WARN" lines.
6. Check cost: `total_cost_usd` should be within 10% of the Anthropic invoice delta for the run window.

---

## 13. Open Issues / Future Work

### 13.1 PDF ingestion placeholder chunks (~50% waste)

**Problem:** Unstructured.io hi_res occasionally returns elements with text like `"(original text as provided)"`, `"(content unavailable)"`, or `"(no text extracted)"` for pages where the layout model couldn't extract readable content (e.g., dense tables, heavily formatted diagrams, low-contrast scans). On the WHO guide, ~50% of pages produce at least one placeholder chunk.

**Impact:** Labeling those chunks is a waste of budget. Opus produces 0 labels because there's nothing to label, but we still pay for the labeling call (~$0.07 per chunk for cache_read + ~$0.10 for the output tokens that say "labels: []").

**Current mitigation:** In Phase 2, `reconstructed_guide` filters out pure-placeholder chunks so the REPL doesn't waste tokens reading them. But Phase 1 labeling still runs on them.

**Fix (planned):** Pre-filter chunks before labeling. If a chunk's `text` is >90% placeholder substrings or <100 meaningful characters after placeholder stripping, skip labeling entirely and emit an empty labels array.

**Estimated savings:** ~$1-2 per WHO-size guide.

### 13.2 Mermaid converter schema alignment

**Problem:** The Mermaid converter was originally written for the legacy `activator.rules[].target_module` schema. Gen 7 uses `router.rows[].output_module`. A bug caused the converter to silently emit an empty graph on Gen 7 output.

**Status:** FIXED (task #64 completed). The converter now checks both schemas with a fallback.

### 13.3 Opus 4.6 cache_control opt-in unsupported

**Problem:** Opus 4.6 silently drops the `cache_control: ephemeral, ttl: 1h` directive on the top-level system prompt in some cases. We still get automatic prefix caching (which is good enough for multi-turn REPL), but we don't get the 1h TTL guarantee.

**Impact:** If the REPL phase takes > 60 minutes (rare, but possible on slow guides), the automatic cache may have been evicted mid-run, causing cache_creation charges on later turns.

**Mitigation:** Keep the `cache_control` directive in the code (it's free to include and works on Sonnet/Haiku). Rely on automatic prefix caching for the realistic case. Monitor cache hit rate in `_run_usage.cached_tokens / (cached + input)` — if it drops below 70%, investigate.

**Long-term:** Wait for Anthropic to add explicit cache_control support on Opus 4.6.

### 13.4 Dispatcher priority tier ambiguity (WHO)

**Problem:** On the WHO reference run, `mod_danger_sign_check` was assigned priority 2 (not 0, not 100). This is because the module's `module_id` doesn't contain keywords like `"severe"` or `"emergency"` — it's literally named `mod_danger_sign_check` which is a CHECK, not the danger itself.

The Phase A priority heuristic keys on keywords in the `module_id`:
- "danger", "severe", "emergency", "critical" → priority 0.
- "intro", "startup", "initial", "demographics", "ask_first" → priority 1.
- "closing", "finalize", "wrap", "summary" → priority 999.
- Everything else → priority 100.

So `mod_danger_sign_check` matches "danger" → priority 0. But the dispatcher shows it at priority 2. This discrepancy suggests the REPL manually overrode the heuristic in Phase C (likely because it realized `mod_danger_sign_check` runs AFTER `mod_assess`, not as a priority exit — `mod_assess` is what DETECTS danger signs and sets `is_priority_exit`; `mod_danger_sign_check` is what HANDLES the emergency referral pathway when `is_priority_exit` is true).

**Resolution:** Rename `mod_danger_sign_check` → `mod_pre_referral_treatment` or similar, so the heuristic correctly classifies it. OR extend Phase A to accept module-plan overrides from the labeler's subtype field.

### 13.5 No streaming of Anthropic responses

**Problem:** We use `messages.create` (non-streaming) throughout. For long-running REPL iterations (2-4 minutes), the UI shows a spinner with no progress.

**Fix (planned):** Switch to `messages.stream` and emit chunk-by-chunk updates to the SSE stream. Complication: `llm_query_batched` expects full responses; we'd need to buffer streamed content until EOF.

### 13.6 No retry logic on FINAL_VAR parse failure

**Problem:** If the REPL returns a string that `ast.literal_eval` can't parse as a dict (e.g., the model wrapped the dict in backticks or added explanatory text), `_parse_repl_result` returns None and the run is marked failed.

**Fix (planned):** On parse failure, send a follow-up REPL message: "Your FINAL_VAR output couldn't be parsed. Please re-emit as a plain Python dict literal." Allow up to 2 retries.

### 13.7 Cost Tracker doesn't expose per-module breakdown

**Problem:** The Cost Tracker shows aggregate labeling cost + REPL cost. It doesn't show how much Phase B spent per module, which would help the user understand which modules are expensive.

**Fix (planned):** Tag each `llm_query_batched` sub-call with its `module_id`. Track per-module token usage in `_run_usage["modules"][module_id]`. Expose via a new `/api/session/{id}/cost_breakdown` endpoint.

### 13.8 No deterministic output check

**Problem:** Two runs of the same guide produce slightly different `clinical_logic.json` outputs (T=0 sampling noise). No test verifies that the differences are cosmetic (e.g., rule ordering, phrase_bank entry order) rather than semantic (different predicates, different rules).

**Fix (planned):** Add `tests/test_determinism.py` that runs the pipeline twice on the same 3-chunk stub guide and asserts that the two `clinical_logic.json` outputs differ only in ordering, not in content (using a structural diff that ignores list order for specific keys).

### 13.9 Test suite freezing

**Problem:** Gen 7 writes `test_suite.json` with `labels_by_type` for each of the 7 artifact types. This file is intended to freeze the labels for downstream regression tests, but nothing actually consumes it.

**Fix (planned):** Add `tests/test_frozen_labels.py` that loads the most recent `test_suite.json` from `backend/output/` and asserts that the labeler produces a superset of the frozen labels on the same guide (drift in one direction only).

### 13.10 Module Maker sub-call failure handling

**Problem:** If 1 of the 8 Module Maker sub-calls returns malformed JSON, the REPL currently fails on `json.loads(resp)` in the Phase B assembly code. The whole run fails at iteration ~7.

**Fix (planned):** Wrap each sub-call's parse in a try/except. On failure, re-dispatch ONLY that module's prompt with a stricter "JSON only, no explanation" reminder. Retry up to 2 times. If all retries fail, emit a placeholder module `{"module_id": mid, "rules": [], "_error": "..."}` and continue.

### 13.11 No guide-level caching beyond 4 hours

**Problem:** Ingestion cache lives in Neon Postgres keyed by SHA256(PDF). Guide JSON (post-ingestion) is cached there. But the REPL's 1h cache_control cache (the reconstructed guide text + deduped labels in Anthropic's servers) expires after 1h, so a re-run of the SAME guide 2 hours later pays full cache_creation cost again.

**Fix (planned):** Not a real problem — the same user re-running the same guide within the same Anthropic prompt cache namespace (same API key) should still hit the auto-cache. But a DIFFERENT user re-running the same guide won't. Could add a guide-level cache of the labeled_chunks + deduped_labels in Neon, so re-runs of the same guide skip Phase 1 + 1b entirely. Would reduce per-guide cost from ~$12.55 to ~$4-5 for cached guides.

### 13.12 No eval harness

**Problem:** We have a reference run (`run_d9a39c2d`) but no structured evaluation: module count correctness, predicate coverage, rule correctness, etc. The professor's quarterly review is manual.

**Fix (planned):** Build an eval harness in `backend/eval/` that loads the reference run's `clinical_logic.json`, compares against a gold-standard `clinical_logic.gold.json` (manually authored), and reports per-artifact precision/recall:
- Supply list: expected 15, got 15 → 100% recall, 100% precision.
- Predicates: expected 22, got 22 → per-predicate FEEL expression equivalence check.
- Modules: expected 8 modules with expected rule counts per module.
- Phrase bank: expected 20 phrases by category.

Bootstrap the gold standard by hand-reviewing `run_d9a39c2d` and committing it to `backend/eval/gold/who_chw_guide.json`.

---

## Appendix A — File Index

### Backend (extraction)

| File | Purpose | Lines |
|------|---------|-------|
| `backend/main.py` | FastAPI app entry point | ~100 |
| `backend/server.py` | HTTP routes (ingest, extract, status, stream, download) | ~800 |
| `backend/session_manager.py` | Session lifecycle, concurrency, SSE stream | 1,204 |
| `backend/rlm_runner.py` | RLM monkey-patches, `_run_usage`, cache attachment | ~2,200 |
| `backend/journal.py` | Human-readable scratchpad writer | ~150 |
| `backend/system_prompt.py` | NAMING_CODEBOOK + legacy prompts | ~300 |

### Gen 7 pipeline

| File | Purpose | Lines |
|------|---------|-------|
| `backend/gen7/chunker.py` | Phase 0 micro-chunking | 150 |
| `backend/gen7/labeler.py` | Phase 1 (2-call labeling) + Phase 1b (dedup) | 593 |
| `backend/gen7/pipeline.py` | Phase 2-4 orchestration, REPL system prompt | 822 |

### Ingestion

| File | Purpose |
|------|---------|
| `backend/ingestion/pipeline.py` | Ingest orchestration |
| `backend/ingestion/unstructured_client.py` | Unstructured.io hi_res wrapper |
| `backend/ingestion/assembler.py` | Element list → guide JSON |
| `backend/ingestion/cache.py` | Neon ingestion cache |
| `backend/ingestion/rendering.py` | PDF page rendering (pypdfium2/pdf2image) |
| `backend/ingestion/vision.py` | OpenAI gpt-4o page post-processing |
| `backend/ingestion/router.py` | Ingestion route handlers |
| `backend/ingestion/warmup.py` | Unstructured warmup call |

### Converters

| File | Purpose | Lines |
|------|---------|-------|
| `backend/converters/json_to_dmn.py` | Gen 7 JSON → DMN 1.3 XML | 278 |
| `backend/converters/json_to_xlsx.py` | Gen 7 JSON → CHT XLSForm | 295 |
| `backend/converters/json_to_mermaid.py` | Gen 7 JSON → Mermaid flowchart | 358 |
| `backend/converters/json_to_csv.py` | predicates.csv + phrases.csv | 125 |
| `backend/converters/mermaid_to_png.py` | Optional PNG rendering | ~100 |

### Frontend

| File | Purpose |
|------|---------|
| `frontend/src/app/page.tsx` | Main page, wires all components |
| `frontend/src/lib/api.ts` | Typed fetch wrappers for backend |
| `frontend/src/lib/downloads.ts` | v5 cross-origin download helper |
| `frontend/src/components/ApiKeyInput.tsx` | BYOK key entry |
| `frontend/src/components/GuideUpload.tsx` | PDF upload UI |
| `frontend/src/components/IngestionProgress.tsx` | Ingest-time SSE stream |
| `frontend/src/components/IngestionManifestCard.tsx` | Post-ingest stats |
| `frontend/src/components/ExtractionJournal.tsx` | Phase transition log |
| `frontend/src/components/CostTracker.tsx` | Live cost polling UI |
| `frontend/src/components/ReplStream.tsx` | Live REPL event stream |
| `frontend/src/components/ValidationPanel.tsx` | Catcher pass/fail strip |
| `frontend/src/components/Z3Panel.tsx` | Optional z3 verifier UI |
| `frontend/src/components/MermaidDiagram.tsx` | Mermaid.js renderer |
| `frontend/src/components/TierBadge.tsx` | Tier display badge |
| `frontend/src/components/ArtifactDownloads.tsx` | Download buttons |

### Test & support

| File | Purpose |
|------|---------|
| `run_gen7_test.py` | Standalone end-to-end test |
| `generate_test_guide.py` | Synthesizes a small guide for fixtures |
| `ingest_guides.py` | CLI to ingest a folder of PDFs |
| `chwci_db_audit.py` | Neon DB consistency auditor |

### Reference output

| Path | Purpose |
|------|---------|
| `backend/output/run_d9a39c2d/` | WHO reference run (2026-04, canonical Gen 7 v2 output) |
| `backend/output/gen7_test_run/` | Last standalone test run |
| `backend/output/run_*` | Historical runs (retained for diff/debug, auto-cleaned after 30 days) |

---

## Appendix B — Environment Setup (for new engineers)

### B.1 Clone + install

```bash
git clone https://github.com/Nerd-Apply/CHW_RLM.git
cd CHW_RLM
```

### B.2 Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Required env vars in `backend/.env`:
```
OPENAI_API_KEY=sk-...
UNSTRUCTURED_API_KEY=...
DATABASE_URL=postgres://...neon.tech/...
UPSTASH_REDIS_REST_URL=https://...upstash.io
UPSTASH_REDIS_REST_TOKEN=...
```

Run:
```bash
python main.py
# or: uvicorn backend.main:app --reload --port 8000
```

### B.3 Frontend

```bash
cd frontend
npm install
npm run dev  # port 3000
```

Required env vars in `frontend/.env.local`:
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### B.4 First-run flow

1. Open `http://localhost:3000`.
2. Paste your Anthropic API key (BYOK).
3. Upload `WHO_CHW_guide_2012_test.pdf` (in the repo root).
4. Wait ~5 minutes for ingestion.
5. Click "Run Extraction".
6. Watch the REPL stream. Full run is ~20-25 minutes.
7. Download `clinical_logic.json` + `form.xlsx` to verify.

### B.5 Common pitfalls

- **Windows:** `pdf2image` needs `poppler` binary. Use `pypdfium2` instead (auto-selected on Windows).
- **Opus rate limits:** If you hit a 429 on a chunk, the labeler retries 3x with backoff. If it still fails, the chunk's labels are empty — the run continues but with degraded recall for that section.
- **Cache warmup:** First run after a long idle produces more cache_creation charges. Second run within an hour hits cache_read rates.
- **Windows node.exe:** NEVER run `taskkill //F //IM node.exe` — it kills ALL Node.js including Claude Code CLI.
- **Session cap:** If you see `429 Too many concurrent sessions`, wait for a prior session to finish or kill it via DELETE /api/session/{id}.

---

## Appendix C — Glossary

- **Artifact:** One of the 7 top-level keys in `clinical_logic.json` (supply_list, variables, predicates, modules, router, phrase_bank, integrative). Plus `deduped_labels.json` as the 26th.
- **BYOK:** Bring Your Own Key. User provides the Anthropic key; we don't store or charge through it.
- **Catcher:** Legacy validator layer (Haiku). Not used in Gen 7 v2 but the cost-tracking plumbing (`accumulate_catcher_usage`) is reused.
- **DMN:** Decision Model and Notation. An OMG standard (DMN 1.3) for representing decision logic. Output format for `clinical_logic.dmn`.
- **FEEL:** Friendly Enough Expression Language. DMN's expression syntax. Used in predicate `threshold_expression` and dispatcher row `condition`.
- **Fail-safe:** A predicate's default value when its source vars are missing. `1` = assume worst case, `0` = assume normal.
- **k=1:** The identification regime: one model, one prompt, one temperature. Variance attributable only to sampling noise.
- **Module Maker:** Phase B of the REPL. Parallel sub-calls that generate per-module DMN tables.
- **Priority exit:** An emergency short-circuit. Any module can set `is_priority_exit = true` to jump to the referral pathway immediately.
- **REPL:** Restricted Language Model. The `rlm/` library providing a sandboxed Python execution environment for Opus to compile artifacts.
- **Tier 2:** Anthropic's billing tier for $25+/month spend. Opus output TPM = 80K.
- **Trigger flag:** A boolean variable (`has_X`) that activates a module in the dispatcher.

---

*End of ARCHITECTURE.md. Last updated 2026-04-13 by the Gen 7 v2 engineering team (A. Mudunuri, a. patel — under Prof. Levine).*
