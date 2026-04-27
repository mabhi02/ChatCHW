# Sequential Pipeline Upgrade Plan

Bringing the sequential pipeline to REPL gen 2.3 accuracy parity, plus the sequential-only optimizations REPL structurally cannot use.

**Status**: planning doc. Nothing in this file is shipped yet.
**Last updated**: 2026-04-11
**Related**: `INFRA.md` section 9 (REPL catcher gen log), section 15 (Pareto analysis), `arena_style.md` (the A/B test this upgrade feeds).

---

## Contents

1. [Current state](#1-current-state)
2. [Upgrade targets](#2-upgrade-targets)
3. [Phase S1: catcher gen 2.1 port (chunking + 5-voter majority + artifact cache)](#3-phase-s1-catcher-gen-21-port)
4. [Phase S2: catcher gen 2.2 port (Citations, contextual headers, two-stage enumerate-then-diff, dict criticals, programmatic quote verification)](#4-phase-s2-catcher-gen-22-port)
5. [Phase S3: generator gen 2.3 port (2× guide repetition, Opus TOC preamble, explicit enumeration)](#5-phase-s3-generator-gen-23-port)
6. [Phase S4: sequential-only wins (the Pareto improvements REPL cannot use)](#6-phase-s4-sequential-only-wins)
7. [Phase S5: cross-family maker/red team pairing (optional)](#7-phase-s5-cross-family-optional)
8. [Cost and wall-clock estimate](#8-cost-and-wall-clock-estimate)
9. [Where sequential and REPL remain structurally different after parity](#9-where-the-two-architectures-remain-different)
10. [Implementation plan (ordered execution list)](#10-implementation-plan-ordered)
11. [Open questions](#11-open-questions)

---

## 1. Current state

The sequential pipeline is the Pipeline A half of the arena A/B test (Professor Levine's proposal). It runs extraction as a chain of stateless stages, each with its own maker → red team → repair loop.

### Files

| File | Responsibility |
|---|---|
| `backend/sequential/runner.py` | Top-level orchestrator, iterates `PARALLEL_GROUPS`, emits artifacts via SSE |
| `backend/sequential/executor.py` | Single-stage executor, runs maker → red team → repair loop, builds the cached guide context |
| `backend/sequential/prompts.py` | Stage definitions (B1, B2, B3, B4, B5b, B6) + shared `REDTEAM_PROMPT` and `REPAIR_PROMPT` |
| `backend/sequential/llm_client.py` | Raw Anthropic + OpenAI API wrapper with token accounting and majority-vote helpers |

### Stages and DAG

From `backend/sequential/prompts.py` lines 579-598:

```
B1_make                         (facts + workflow)
   ├→ B2_make   ┐
   │            │  parallel within group
   └→ B3_make   ┘
              ↓
           B4_make                (enrich factsheet + supply inventory)
              ↓
           B5b_make                (boolean literals)
              ↓
           B6_make                 (rule criteria + modules + router + integrative + phrase bank)
```

Six stages total. B2 and B3 run in parallel, everything else is serial.

### Stage loop (maker → red team → repair)

From `backend/sequential/executor.py`:

1. **Maker** (`call_anthropic` at line 140): Sonnet 4.6, full guide cached in system prompt, one call, temperature 0, max_tokens 16384.
2. **Red team** (`_run_redteam` at line 289): Sonnet 4.6, single call (no majority vote), the guide appended after the redteam template in the system prompt, parses a JSON `{passed, critical_issues, ...}` object.
3. **Repair** (`_run_repair` at line 372): Sonnet 4.6, system is maker system + repair template, user message is the current artifact plus the red team report. Loops up to `MAX_REPAIR_RETRIES = 2` times. On retry, the prior red team report is injected into the next red team call to prevent the "moving goalpost" problem (line 305).

### Guide handling

`_build_guide_context` (`executor.py` lines 25-84):
- Emits metadata header, then all `pages[*].raw_text`, then all `sections[*].content[*].text`
- Caps the result at 700K chars with a warning (line 76)
- Same context string used for maker/repair and red team, but wired into different system prompts (different prefixes, so Anthropic prompt caching does not cross between stage types)

`_build_system_prompt` (lines 87-102):
- Fixed preamble text + the guide context appended
- Used for maker and repair

Red team builds its own system block (`executor.py` lines 343-346):
```python
full_system = (
    f"{redteam_system}\n\nSOURCE MANUAL CONTEXT (for cross-reference):\n"
    f"{self._guide_context[:500_000]}"
)
```

Note the second truncation: red team only sees the first 500K of a 700K-capped context. That is a recall hole all by itself.

### Caching state today

Three distinct prompt cache lines across a sequential run:

| Cache line | Prefix | Shared across | Writes per run |
|---|---|---|---|
| Maker cache | `_system_prompt` (preamble + guide) | all 6 maker calls | 1 |
| Red team cache | `redteam_system + guide` (different prefix because redteam template is first) | all red team calls (6-18 depending on retries) | 1 |
| Repair cache | `_system_prompt + "\n\n" + repair_system` (shares maker prefix up to the concat point) | all repair calls | shared with maker |

So the guide is effectively paid for **twice** in cache writes: once for the maker/repair prefix, once for the red team prefix.

### Cost and recall (estimated, pre-upgrade)

| Axis | Current value |
|---|---|
| $/run on full WHO | ~$3-8 (mostly Sonnet maker/red team/repair calls, with 1h cache keeping subsequent stage reads cheap) |
| Wall clock | ~5-12 min |
| Multi-needle artifact recall | ~70-80% estimated (no voting, no chunking, no quote verification, single-shot red team) |

### Known recall holes

1. **Red team is a single Sonnet call against the full guide** (`executor.py:348-355`). No chunking, no voting, no Citations, no programmatic quote verification. This is exactly the failure mode that drove REPL gen 2.0 → 2.1 → 2.2.
2. **Red team sees only the first 500K chars of the 700K-capped guide context** (`executor.py:345`). On any guide larger than ~130K tokens raw, the red team is blind to the tail of the guide.
3. **Prior-artifact truncation at 200K chars** (`executor.py:276-282`). B6's inputs include B4's factsheet, B5b's boolean_literals, B3's consolidation_rules. On full WHO these routinely exceed 200K combined. The current implementation silently truncates each, which means downstream stages build on a half-visible prior.
4. **No two-stage enumeration** in the maker prompts. Each stage's maker prompt is a single-pass extraction instruction, which underperforms the enumerate-then-structure pattern on dense lists (see Google 2512.14982 and MNIAH-R 2504.04150).
5. **No prompt repetition**. The guide is in the cached system block once. Bidirectional-attention pressure is weaker than REPL's gen 2.3 2× repetition on the same content.
6. **No TOC preamble**. The maker has no structural focus prior before reading the guide. Each stage re-discovers the guide's section structure on its own.
7. **Deterministic gates are not split out**. The xlsx has 160 quality standards; many are expressible as Python predicates (MECE, else-row presence, source-quote presence, no-raw-numerics). Today these ride inside the red team's natural-language critic, which wastes LLM budget on checks that a parser would catch in microseconds.

Items 1-3 are the catcher-gen-2.x-shaped holes. Items 4-6 are the generator-gen-2.3-shaped holes. Item 7 is sequential-specific: the bounded per-stage output schema makes deterministic gates much easier to express in sequential than in REPL.

---

## 2. Upgrade targets

At the end of this upgrade the sequential pipeline should:

1. Hit **~94-98% artifact-level multi-needle recall**, matching REPL gen 2.3.
2. Cost **~$25-35 per full WHO run** (up from ~$3-8, offset by fewer re-extractions).
3. Land in **~25-35 min wall clock at Tier 2** concurrency, or **~15-20 min at Tier 3**.
4. Add **four sequential-only optimizations** that REPL structurally cannot use: content-hash stage memoization, per-stage model routing without context penalty, cross-family maker/red team pairing per stage, and deterministic pre-red-team gates.
5. Maintain the architectural property that makes sequential useful in the first place: **each stage is a stateless LLM call with a bounded schema**. No REPL, no continuous session, no shared Python state across stages.

After this upgrade, the sequential and REPL pipelines should be accuracy-equivalent on first-pass runs and the A/B test in `arena_style.md` actually has a fair comparison to measure.

---

## 3. Phase S1: catcher gen 2.1 port

Replaces the single-shot Sonnet red team with the REPL-catcher pattern: chunked guide, 5-voter Haiku majority per chunk, per-chunk artifact cache.

### What changes

The red team stage in `executor.py:_run_redteam` (lines 289-370) becomes a wrapper around the existing `run_chunk_voters` / `validate_artifact` helpers in `backend/validators/phases.py`. The Sonnet call is removed.

### Why it ports cleanly

The sequential red team is structurally a multi-needle recall task: "given this guide and this artifact, what guide items are missing from the artifact?" That is the exact task REPL's catchers solve. Sequential does not lose anything by using the same solution.

Three REPL gen 2.1 interventions apply verbatim:

1. **50K-char section-boundary chunking.** Use `chunk_guide_for_catcher()` from `phases.py:987`. Full WHO splits into ~23 chunks, each sitting in Haiku 4.5's near-perfect attention plateau.
2. **5 voters with temperature diversity** (0.0, 0.1, 0.2, 0.3, 0.4). Use `run_chunk_voters` or call `_call_catcher_chunked_majority` at `phases.py:1157`. Fail-closed aggregation: any critical from any chunk fails the stage.
3. **Per-chunk artifact cache with 5m TTL.** Already implemented in `phases.py:_call_catcher` — the chunk lives in a cached system block, the artifact lives in the user message, so vote 2..5 read the chunk cache for free.

The sequential retry loop actually **amplifies** the artifact-cache win relative to REPL: each retry changes the artifact but not the chunks, so the chunk cache amortizes across maker → red team → repair → red team → repair (up to 3 red team rounds per stage). REPL's catcher only runs once per emit.

### Artifact-name mapping

Sequential stage outputs need to map onto catcher validator names. `phases.py:valid_artifact_names()` exposes the valid set. Current REPL catchers exist for:

- `supply_list`
- `variables`
- `predicates`
- `modules`
- `router`
- `integrative`
- `phrase_bank`

Sequential stage → REPL catcher mapping:

| Sequential stage | Sequential output | REPL catcher | Mapping strategy |
|---|---|---|---|
| B1_make | `raw_facts`, `workflow_pattern` | (none direct) | Add new catcher prompt `completeness_raw_facts.txt` + `completeness_workflow_pattern.txt`. Follow the recall-only dict-critical template from gen 2.1. |
| B2_make | `context_mappings` | (none direct) | Add `completeness_context_mappings.txt`. Small artifact, 1-2 chunks usually. |
| B3_make | `consolidation_rules` | (none direct) | Add `completeness_consolidation_rules.txt`. |
| B4_make | `supply_inventory`, `factsheet` | `supply_list`, `variables` | Reuse REPL's existing `completeness_supply_list.txt` for `supply_inventory`. Reuse `completeness_variables.txt` for the `factsheet`'s observation fields. |
| B5b_make | `boolean_literals` | `predicates` | Reuse REPL's `completeness_predicates.txt`. |
| B6_make | `rule_criteria`, `modules`, `router`, `integrative`, `phrase_bank` | `modules`, `router`, `integrative`, `phrase_bank` | Reuse REPL's catchers directly. Add `completeness_rule_criteria.txt` for the atomized rules. |

Sequential gets five catchers from REPL for free (`supply_list`, `variables`, `predicates`, `modules`, `router`, `integrative`, `phrase_bank`) and needs ~4 new catcher prompts for the stages where sequential's output shape differs from REPL's emit checkpoints.

### File-level changes

**`backend/sequential/executor.py`**:

- Delete `_run_redteam` lines 289-370.
- Replace with a thin wrapper that calls `phases.validate_artifact(artifact_name, artifact_json, guide_json, api_key)` and adapts the `CatcherResult` to the shape the stage loop expects (`{content, passed, input_tokens, output_tokens, cached_tokens, duration_ms, cost_usd}`).
- Remove the `self._guide_context[:500_000]` truncation on line 345 — chunking replaces it.
- Keep `_build_guide_context` as-is for the maker side (Phase S3 will modify it for 2× repetition).

**`backend/sequential/prompts.py`**:

- Delete `REDTEAM_PROMPT` (lines 34-102) — sequential no longer uses a qualitative red team. Or keep it as a fallback for stages without a dedicated catcher.
- Keep `REPAIR_PROMPT` (lines 109-138) — repair is still an LLM call that consumes the catcher's critical_issues output.

**`backend/prompts/validators/`**:

- Add `completeness_raw_facts.txt`
- Add `completeness_workflow_pattern.txt`
- Add `completeness_context_mappings.txt`
- Add `completeness_consolidation_rules.txt`
- Add `completeness_rule_criteria.txt`

Each follows the gen 2.2 template: two-stage enumerate-then-diff, dict critical_issues with isolated `guide_quote`, guide-agnostic language (no CHW/pediatric-specific vocabulary).

### Cost delta

- Replaces 1 Sonnet red team call (~$0.30 per call × 6-18 calls per run ≈ $2-5/run) with 5 Haiku voters × ~23 chunks × 6 stages ≈ 690 Haiku calls per run.
- Haiku pricing with per-chunk cache: ~$15-20/run for the full catcher round.
- Net delta: **+$10-15/run**.

### Wall clock delta

- At `CATCHER_CONCURRENCY_CAP=3`: 690 calls / 3 concurrent ≈ 230 sequential batches. At ~5s per batch ≈ 19 min catcher wall clock alone.
- Counterbalanced by: per-stage maker/repair work still takes ~10-15 min total.
- Net sequential wall clock: **~25-35 min** (up from ~5-12 min, but now includes rigorous recall auditing).
- At Tier 3 with `CATCHER_CONCURRENCY_CAP=15`, catcher wall drops to ~4 min, total ~15-20 min.

### Recall delta

**+8-15 pp** on artifact-level multi-needle recall, bringing sequential from ~70-80% to ~85-92% on first-pass.

---

## 4. Phase S2: catcher gen 2.2 port

All five gen 2.2 interventions port directly to the sequential red team once Phase S1 has replaced it with the chunked-voter pattern. No restructuring needed.

### 4.1 Anthropic Citations API on red team calls

Already wired inside `phases.py:_call_catcher` when `CATCHER_USE_CITATIONS=true`. The chunk is passed as a document block with citations enabled. Sequential inherits this for free once Phase S1 is done. No sequential-specific code change.

Caveat: Citations go dormant when `cache_control` is also set on the document block. The load-bearing grounding check is item 4.2, not Citations.

### 4.2 Programmatic quote verification

Every critical_issue dict emitted by a catcher has a `guide_quote` field. Before accepting the critical, substring-match the `guide_quote` against the chunk it came from. If the substring is not found verbatim, drop the critical as a hallucination.

Already implemented in `phases.py` for REPL. Sequential inherits it via the `validate_artifact` entry point.

This is the **load-bearing grounding mechanism** that lets us trust the sequential red team without human review. Without it, a voter can invent a plausible-sounding critical and block a correct artifact. With it, every critical survives only if a byte-for-byte quote actually appears in the source.

### 4.3 Per-chunk contextual headers

Each chunk gets a 2-3 sentence structural context header generated by Haiku before being passed into the voter prompts. Headers are cached by chunk content hash in `phases.py:_chunk_header_cache`. First sequential run on a guide pays ~$0.12 per unique chunk (~$2.80 for 23 chunks); subsequent runs on the same guide read from the cache for free.

Already wired in `phases.py` when `CATCHER_CONTEXTUAL_HEADERS=true`. Sequential inherits this for free.

**Bonus**: the header cache is module-level, shared across any pipeline that imports `phases.py`. Running sequential after a REPL run on the same guide means sequential starts with headers already populated. Zero-cost for the second pipeline.

### 4.4 Two-stage enumerate-then-diff catcher prompts

The new catcher prompts added in Phase S1 (`completeness_raw_facts.txt`, `completeness_consolidation_rules.txt`, etc.) must follow the gen 2.2 two-stage structure from the MNIAH-R pattern:

```
STAGE 1 (enumerate): For every {supply | variable | predicate | rule | ...}
described anywhere in the guide chunk, emit a dict entry:
  {verbatim_quote, section_id, kind_hint, ...}
Do not filter, do not deduplicate. Enumerate everything.

STAGE 2 (diff): For each enumerated item, check whether the submitted
artifact contains a corresponding entry. If not, emit a critical_issue
with the verbatim guide_quote.
```

This is stricter than the REPL catchers were in gen 2.1 and matches gen 2.2's rewrites. Sequential benefits more than REPL because the red team runs up to 3 times per stage in the repair loop — each two-stage enumeration pass sharpens the critical set.

### 4.5 Dict-typed critical_issues with isolated guide_quote

Every catcher emits criticals in this shape (not plain strings):

```json
{
  "description": "Missing danger sign: severe chest indrawing",
  "guide_quote": "If the child has severe chest indrawing, refer URGENTLY",
  "page_ref": 27,
  "section_id": "sec_danger_signs"
}
```

The isolated `guide_quote` field is what enables both Citations and programmatic verification. Without it, the critical's quote is embedded in natural-language prose and neither grounding mechanism fires.

Already standard in REPL gen 2.2. New sequential catcher prompts follow the same format.

### Cost delta for Phase S2

- Citations API: $0
- Programmatic quote verification: <10ms per critical, $0
- Contextual headers: +$2.80/run first-time, $0 on retests (shared cache with REPL)
- Two-stage enumerate: +$0.30/run (more output tokens on the enumerate stage)
- Dict criticals: $0

Net: **+$3/run first-time, $0 on retests**.

### Recall delta

**+3-8 pp** per-chunk. Largest gains on the dense list stages (B1 raw_facts, B6 modules). Combined with S1, sequential first-pass recall should be around **~90-96%**.

---

## 5. Phase S3: generator gen 2.3 port

The three gen 2.3 interventions that landed on the REPL's main extraction layer. These are maker-side wins and apply to every stage's maker call.

### 5.1 2× guide repetition in the cached system block

**Where**: `backend/sequential/executor.py:_build_system_prompt` (lines 87-102).

**What**: Currently the guide is embedded once in the cached system text. Replace with the repetition pattern from `rlm_runner.py:_maybe_attach_guide_block` (line 601): emit the guide N times with a separator, where N defaults to 2 via `GENERATOR_GUIDE_REPETITION`.

Size-aware fallback (same rule as REPL):
- If `2 * len(guide_text) + preamble` exceeds `_GUIDE_CACHE_MAX_CHARS = 3_500_000`, fall back to 1× (gen 2.2 behavior).
- If 1× exceeds the cap, fall back to splitting the extraction into per-section passes (not implemented yet, tracking as an open question).

**Important**: sequential has **three distinct system-prompt prefixes** (maker, red team, repair), each with its own cache line. If you apply 2× repetition to all three, you pay the cache write 3×. To avoid this, apply 2× repetition **only to the maker system block**. Red team uses chunked guide (one chunk per call, no repetition needed). Repair reuses the maker system, so it inherits the repetition.

Cost of 2× on maker only: **+$10-15/run** (cache write ~doubled on the maker prefix).

### 5.2 Opus-generated guide TOC preamble

**Where**: `backend/sequential/runner.py:run_sequential_pipeline` (called at session start, before any stage fires).

**What**: Reuse `rlm_runner.py:_generate_guide_toc` directly. This is an Opus 4.6 call that reads the guide JSON and emits a 500-800 word dense factual preamble describing:
- What the guide is (age range, clinical domain, organizing principle)
- How it is structured (major sections, table of contents, cross-reference conventions)
- The specific terminology the guide uses (drug names, threshold formats, severity labels)

The preamble is cached by SHA256 content hash in `_toc_cache` (a module-level dict). First run on a new guide pays ~$2.45 and adds 60-90s at session start. Retests on the same guide read from cache for free.

**Shared cache with REPL**: `_toc_cache` is a module-level dict in `rlm_runner.py`. If sequential imports from there, a REPL run on guide X populates the TOC that a subsequent sequential run on guide X reads for free. One-time Opus cost per unique guide, across both pipelines.

**Where to prepend**: the TOC goes at the **top** of the maker's cached system block, before the 2× guide repetition. Structure:

```
[Role preamble, stage instructions, etc.]

=== GUIDE TABLE OF CONTENTS (structural focus prior) ===
{toc_text}

=== GUIDE CONTENT (repetition 1 of 2) ===
{guide_text}

=== GUIDE CONTENT (repetition 2 of 2) ===
{guide_text}
```

Sonnet reads the TOC before diving into the guide body, which gives it a structural prior that reduces "I have no idea where I am in this document" drift. Matches the REPL behavior exactly.

### 5.3 Explicit two-stage enumeration in maker prompts

**Where**: `backend/sequential/prompts.py` — every stage's `maker_prompt` string.

**What**: The current prompts are single-pass: "read the guide, emit the artifact." Rewrite them to two-stage:

```
STAGE 1 (enumerate candidates):
Before building the structured output, emit a flat list of every {fact | supply | predicate | rule | ...} mentioned in the guide. Include:
  - verbatim_quote: the exact text from the guide
  - section_id: where it appears
  - kind_hint: what kind of item this is
Do not filter. Do not deduplicate. Enumerate everything, even borderline items.

STAGE 2 (structure):
For each enumerated candidate, build the structured output entry. Preserve provenance back to the enumerated entry.

OUTPUT FORMAT:
{
  "_enumerated_candidates": [...],
  "{stage_output_key}": [...]
}
```

This is the JSON-output analog of the REPL's `supply_candidates.append(...)` Python scratchpad pattern. Same cognitive discipline: enumerate exhaustively, then structure. The `_enumerated_candidates` field is useful as a debugging/replay surface — you can look at any stage's run and see exactly what the maker "noticed" before committing to a structure.

Apply to:
- B1_MAKER: enumerate `raw_fact_candidates` before `raw_facts`
- B4_MAKER: enumerate `supply_candidates` before `supply_inventory`, `observation_candidates` before `factsheet`
- B5b_MAKER: enumerate `literal_candidates` before `boolean_literals`
- B6_MAKER: enumerate `module_candidates` and `rule_candidates` before `modules` and `rule_criteria`

Stages B2 and B3 are small enough that two-stage enumeration is over-engineering. Skip them.

### Cost delta for Phase S3

- 2× repetition on maker prefix: +$10-15/run
- Opus TOC: +$2.45/run first-time per unique guide, $0 thereafter
- Explicit enumeration scratchpad: +$0.50/run (more output tokens)

Net: **+$13-18/run first-time, +$10-15/run on retests**.

### Recall delta

**+10-20 pp** first-pass recall on the list-building stages (B1, B4, B5b, B6). Combined with S1 + S2, sequential first-pass recall should be around **~92-96%**, approaching REPL gen 2.3 parity.

---

## 6. Phase S4: sequential-only wins

These are the Pareto improvements that are **only available to sequential** because they exploit the stateless-per-stage architecture. They are not ports from REPL gen 2.x — REPL structurally cannot use them.

### 6.1 Content-hash stage memoization

**The biggest practical win for test-retest reliability studies.**

**Mechanism**: Hash `(guide_content_hash, stage_id, maker_prompt_hash, sorted(input_artifact_hashes), model_id, temperature)`. On every `run_stage` call, check a Neon table for a cached output keyed by this hash. If present, skip the stage and load the artifact directly.

**New table**:

```sql
CREATE TABLE sequential_stage_cache (
    cache_key       TEXT PRIMARY KEY,      -- SHA256 of the hash tuple above
    run_id          UUID NOT NULL,         -- the run that originally produced it
    stage_id        TEXT NOT NULL,         -- B1_make, B2_make, etc.
    guide_hash      TEXT NOT NULL,         -- for cleanup / invalidation by guide
    artifact_json   JSONB NOT NULL,
    validation_json JSONB,                 -- the catcher result when it passed
    cost_usd        NUMERIC,               -- cost of the original production
    duration_ms     INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_stage_cache_guide ON sequential_stage_cache(guide_hash);
CREATE INDEX idx_stage_cache_stage ON sequential_stage_cache(stage_id);
```

**Gate behind** `ENABLE_STAGE_MEMOIZATION=true` env var so the default behavior is unchanged — only test-retest runs that opt in get memoization.

**What this unlocks**:

- Run 1 of a guide: full cost (~$25-35 with S1-S3 ported).
- Run 2 of the same guide with identical prompts: **near-zero cost**. Every stage hits the cache. Total run: ~$0-2 for the bookkeeping + any stages where temperature was deliberately varied.
- Run 3: same as run 2.

A 3-run test-retest costs **~$30 total**, not ~$90. This is the single largest cost win in the upgrade, and it is available only to sequential.

**Variance isolation**: to measure B6 variance specifically, memoize B1-B5b and re-run B6 with a fresh call. You get a controlled variance measurement: everything except B6 is deterministic, so any differences are attributable to B6. REPL cannot do this — its variance is stochastic across the whole session.

**REPL cannot use this** because the REPL's Python variable state is continuously coupled across phases. You cannot resume from phase 4 with a cached phase 1 output because phase 4's reasoning depends on reading variables the model itself populated earlier.

### 6.2 Per-stage model routing without context penalty

REPL uses phase-aware Opus↔Sonnet routing but pays a small context-switch cost: the model has to re-parse its own prior variables. Sequential has no such cost because each stage is a fresh call reading from disk.

**Mechanism**: add a `model_override` field to the `StagePrompt` dataclass. Route stages to their optimal model:

| Stage | Recommended model | Reasoning |
|---|---|---|
| B1_make | `claude-opus-4-6` | Raw facts drive every downstream stage. A missed fact cascades. Opus has the best first-pass enumeration. |
| B2_make | `claude-sonnet-4-6` | Mechanical mapping from symptoms to modules. Bounded output, low ambiguity. |
| B3_make | `claude-opus-4-6` | Cross-section consolidation synthesis. A missed consolidation rule causes a safety bug (e.g., missing "no oral meds if unconscious"). |
| B4_make | `claude-sonnet-4-6` | Mostly referential enrichment of B1's output. |
| B5b_make | `claude-sonnet-4-6` | Deterministic threshold parsing. Sonnet is already near-perfect here. |
| B6_make | `claude-opus-4-6` | The most important single artifact. Modules + router + integrative + phrase bank. Highest stakes, most benefit from Opus-grade synthesis. |

**Cost delta**: +$5-8/run vs all-Sonnet.
**Recall delta**: +5-10 pp on the Opus-routed stages.
**Net**: favorable. The Opus-heavy routing pays for itself via fewer repair retries on B1/B3/B6.

### 6.3 Cross-family maker/red team pairing per stage

Because each stage is isolated, you can use **different model families** for maker vs red team at each stage. This is the cross-family optimization from Pareto config H, but shipped inside the sequential pipeline without running a parallel extraction.

**Mechanism**: add a `redteam_model_family` field to `StagePrompt`. Route red teams to decorrelating families on the stages that most need blind-spot coverage:

| Stage | Maker | Red team | Why cross-family here |
|---|---|---|---|
| B1 | Sonnet | Haiku × 5 (same family) | Cheap enumeration check, Haiku is fine |
| B3 | Opus | **GPT-5-mini × 5** | Consolidation rules are where training-data blind spots bite hardest — Claude and GPT have different priors on "what counts as a consolidation rule" |
| B6 | Opus | **Gemini 2.5 Flash** with a Gemini 2.5 Pro 1M-context window single-call audit | Gemini's 1M-token context means it can audit the full guide in one call — no chunking. Good second opinion on the final assembly. |

**Prerequisites**:
- `BYOK_GOOGLE_API_KEY` env var (optional; fall back to Haiku if unset)
- `BYOK_OPENAI_API_KEY` env var (already present for ingestion — reuse for catcher calls, but scope it as catcher-only so it doesn't touch extraction)
- Fallback path: if a family's key is missing, degrade gracefully to Haiku × 5 for that stage

**Cost delta**: +$3-5/run.
**Recall delta**: +3-5 pp on the cross-family stages (B3, B6).

**REPL cannot use this** because the REPL session locks to one model family for the duration. You cannot mid-session swap the underlying model without losing context coherence.

### 6.4 Deterministic gates before the LLM red team fires

The xlsx has 160 quality standards. Many are expressible as Python predicates that run in microseconds and never need an LLM.

**New file**: `backend/sequential/deterministic_gates.py`.

**Structure**:

```python
from typing import Callable, Any

# Per-stage dispatch: artifact name -> list of predicate checks
DETERMINISTIC_GATES: dict[str, list[Callable[[Any], list[str]]]] = {
    "modules": [
        check_every_table_has_else_row,
        check_no_raw_numerics_in_antecedents,
        check_all_predicates_defined,
        check_router_row_0_is_danger_sign,
        check_hit_policies_valid,
    ],
    "boolean_literals": [
        check_every_literal_has_negation,
        check_threshold_expressions_parse,
        check_source_vars_referenced_exist,
        check_missing_policy_set,
    ],
    "supply_inventory": [
        check_every_supply_has_stockout_fallback,
        check_every_supply_has_source_quote,
        check_prefixes_valid,  # equip_ or supply_
    ],
    "factsheet": [
        check_variable_codes_have_valid_prefix,
        check_every_fact_has_source_quote,
        check_bidirectional_supply_links,
    ],
    "consolidation_rules": [
        check_policy_type_in_valid_set,
        check_every_rule_has_source_quote,
    ],
    "raw_facts": [
        check_every_fact_has_required_fields,
        check_category_in_13_valid_categories,
        check_data_type_valid,
    ],
}
```

**Execution order per stage**: maker produces artifact → deterministic gates run (fast) → if any gate fails, emit a targeted repair prompt with the specific failures → repair → deterministic gates run again → only if deterministic gates pass, fire the LLM red team.

**What this saves**: ~30-40% of LLM red team invocations on first-pass runs. The red team is the biggest wall-clock item in S1-S3; cutting 30% of its calls drops wall clock by ~6-10 min and cost by ~$5-8.

**What this catches that the LLM red team would miss**: structural invariants like "every table has an else row" are hard for an LLM to enumerate exhaustively but trivial for a Python parser. You get a stricter structural check at near-zero cost.

**REPL partially has this via** `backend/validators/deterministic.py` but the REPL variables live in an in-memory REPL where schema is less bounded. Sequential's per-stage output has a known schema and deterministic gates fit naturally.

### 6.5 Targeted backtracking without full stage restart

When a downstream stage's red team finds an upstream gap (e.g., "B6 references predicate X but it is not in boolean_literals"), the current behavior is to fail B6. With targeted backtracking, you emit a **patch call** to the upstream stage with only the missing item as input.

**Mechanism**: new method `executor.py:patch_stage_output(stage_id, patch_instructions)`. Signature:

```python
async def patch_stage_output(
    self,
    stage_id: str,
    patch_instructions: str,  # "Add predicate X with threshold Y"
    max_patch_retries: int = 1,
) -> bool:
    """Emit a targeted repair for an already-completed stage.

    Returns True if the patch succeeded and the stage output was updated.
    Capped at 1 retry to avoid infinite propagation loops.
    """
```

**Flow**:

1. B6 red team finds a missing item attributed to B5b.
2. Instead of failing B6, call `patch_stage_output("B5b_make", "Add predicate p_fast_breathing_unknown_age with ...")`.
3. B5b re-runs with a tiny prompt: "Here is your current boolean_literals output. Add the following predicate: [description]. Return the complete updated artifact."
4. B5b's deterministic gates + LLM red team re-validate the patched output.
5. B6 red team re-runs against the patched B5b.

**Guardrails**:
- Cap at 1 patch per downstream stage to prevent loops.
- If the patch itself fails the upstream stage's red team, fall back to failing the downstream stage (current behavior).
- Log every patch as a `_patch_log` entry in the stage output so runs are auditable.

**This gives sequential REPL-level self-correction** without REPL's continuous session. The difference: REPL edits a Python variable in place with full context; sequential patches a JSON artifact with a targeted prompt. Same observable behavior on the downstream stages, different failure mode analysis.

### 6.6 Decomposed per-stage catchers

REPL catchers are per-artifact (one catcher per emit checkpoint). Sequential can decompose further because each stage's red team only looks at that stage's output surface.

**Example: decompose B6's single red team into 5 parallel specialized catchers**:

| Catcher | Focus | Model | Chunks |
|---|---|---|---|
| `b6_redteam_coverage` | Does every fact from B1 flow into at least one module rule? | Haiku × 5 voters | Yes, 50K chunks |
| `b6_redteam_safety` | Danger-sign monotonicity, row-0 short-circuit, no-oral-if-unconscious | Haiku × 5 voters | Yes |
| `b6_redteam_reachability` | Every module endpoint reachable from at least one input | Z3 (no LLM) | No |
| `b6_redteam_traceability` | Every rule has a source_quote that substring-matches the guide | Deterministic Python | No |
| `b6_redteam_mece` | Z3 unsat check on table inputs (no two rules can fire together) | Z3 (no LLM) | No |

Five parallel specialized red teams per stage. Each has a tiny mental model, tighter attention, and three of the five are deterministic (Z3 or Python) with zero LLM cost.

**Cost**: minor — the LLM catchers (coverage, safety) cost about the same as the single combined red team they replace, and the deterministic ones are free.

**Recall delta**: +2-5 pp on first-pass B6 correctness.

**REPL analog**: REPL would need to thread 5 separate emit checkpoints through one continuous session, which is architecturally disruptive. Sequential gets this essentially for free because each stage's red team is already a separate call.

### 6.7 Stage journaling for resume, replay, and diff

Write every stage's full state to a JSONL log per run. One line per stage, containing:

```json
{
  "stage_id": "B4_make",
  "run_id": "seq-abc123",
  "inputs": {...},
  "output": {...},
  "redteam_report": {...},
  "deterministic_gate_results": {...},
  "cost_usd": 3.42,
  "duration_ms": 58000,
  "model": "claude-sonnet-4-6",
  "cache_hit": false,
  "timestamp": "2026-04-11T15:32:18Z"
}
```

**New file**: `output/{run_id}/journal.jsonl`.

**Unlocks**:

1. **Resume crashed runs** from the last successful stage. REPL cannot resume mid-session. Sequential runner gets a `--resume-from <run_id>` CLI flag that reads the journal, finds the last successful stage, and re-runs from there.
2. **Replay a run with one stage's prompt swapped** — debug a specific stage without re-running the whole pipeline. Useful for A/B testing prompt changes.
3. **Diff two runs at stage granularity** — instead of diffing two final DMN outputs, diff each stage's output to isolate where variance entered.
4. **Audit trail for the research group** — every run has a complete, append-only record of what the pipeline saw and what it produced.

**REPL does not have this** because REPL state is in Python variables inside a continuous session. The only way to snapshot REPL state is to serialize the full variable namespace at each emit, which is what REPL's `emit_artifact` already does — but the REPL snapshots are smaller than sequential's journals because REPL does not have per-stage red team reports and deterministic gate outputs to include.

---

## 7. Phase S5: cross-family (optional)

Only do this if the research group actually wants the "decorrelate training-data blind spots" property. Otherwise, skip.

### What

Add GPT-5-mini and Gemini 2.5 Flash as optional red team backends for specific stages (configured via `StagePrompt.redteam_model_family`). Add a Gemini 2.5 Pro single-call full-guide audit as an optional post-B6 gate.

### How

- Add `backend/sequential/openai_catcher.py` — wraps GPT-5-mini with an equivalent chunked-voter pattern, parameterized to match `phases.py:_call_catcher`'s interface.
- Add `backend/sequential/gemini_catcher.py` — wraps Gemini 2.5 Flash similarly. Gemini's 1M context also enables an unchunked full-guide audit path.
- Add BYOK routing in `runner.py`: if the caller provides `BYOK_GOOGLE_API_KEY` and `BYOK_OPENAI_API_KEY`, use the cross-family paths for stages that opted in. Otherwise fall back to Haiku.

### Cost

+$3-5/run when enabled. Can be reduced by only running the Gemini full-guide audit once on B6 (the highest-stakes stage) instead of on every stage.

### Recall

+1-3 pp. The cross-family win is smaller per stage than the gen 2.1-2.3 wins, but it catches categorical blind spots the single-family pipeline is structurally unable to see.

---

## 8. Cost and wall-clock estimate

All numbers for a full WHO run (140 pages, 1.94M chars pretty, 485K tokens).

| Config | $/run | Wall clock (Tier 2) | Wall clock (Tier 3) | Artifact recall | Notes |
|---|---|---|---|---|---|
| **Current sequential (pre-upgrade)** | $3-8 | ~5-12 min | ~5-12 min | ~70-80% | single Sonnet red team, no chunking, no voting, no quote verification |
| **+ S1 (chunked 5-voter catchers)** | $13-22 | ~20-28 min | ~12-15 min | ~85-92% | largest single accuracy jump, catcher overhaul |
| **+ S2 (Citations, headers, two-stage, dict criticals, verification)** | $16-25 | ~22-30 min | ~13-17 min | ~90-96% | grounding guarantees, tighter red team |
| **+ S3 (2× guide repetition, Opus TOC, explicit enumeration)** | $25-35 | ~28-38 min | ~17-22 min | ~92-96% | maker-side prompt-repetition wins; REPL-equivalent first-pass recall |
| **+ S4 (stage memoization, per-stage routing, cross-family, deterministic gates, decomposed catchers, journaling)** | $28-40 | ~30-40 min | ~18-22 min | **~94-98%** | sequential-only wins; REPL parity achieved |
| **+ S5 (cross-family)** | $32-45 | ~30-40 min | ~18-22 min | ~95-98% | optional |
| **+ Stage memoization on retest (runs 2 and 3)** | **$1-3** | **~3-6 min** | **~2-4 min** | same as run 1 | the biggest practical win for test-retest reliability |

### Test-retest reliability study comparison

A 3-run test-retest on full WHO at each config:

| Config | 3-run total cost | Notes |
|---|---|---|
| Current sequential | ~$15-25 | low accuracy baseline |
| Sequential gen 2.3 equivalent (S1-S3) | ~$75-105 | REPL-equivalent accuracy, but expensive per retest |
| Sequential gen 2.3 + stage memoization (S1-S4) | **~$30-45** | retests hit cache, dominant cost is run 1 |
| REPL gen 2.3 | ~$120-135 | equivalent accuracy, cannot memoize stages across retests |

**Sequential with stage memoization is ~3× cheaper than REPL for the test-retest reliability studies** the arena is structurally designed to run. This is the headline result of the upgrade.

---

## 9. Where the two architectures remain different

Even after full parity, sequential and REPL remain structurally different on two axes. The A/B test exists to measure which set of trade-offs matters more in practice.

### 9.1 Backward self-correction vs forward DAG re-entry

- **REPL**: when phase 4 discovers a phase 2 gap, the model edits a Python variable in place. Single continuous context, single session. Self-correction is free — the model has full memory of why phase 2 produced what it did.
- **Sequential (even with targeted backtracking)**: when B6 discovers a B5b gap, the orchestrator fires a patch call to B5b. Fresh context, the patch prompt is the only input. The new B5b invocation has no memory of the original B5b reasoning.

Both produce equivalent **outputs** after gen 2.3 (both hit ~94-98% recall), but different **failure modes**:

- REPL's failure mode: "phase 4 cannot fix what phase 2 missed because the model forgot the phase-2 context the fix depends on."
- Sequential's failure mode: "the targeted B5b patch loses nuance because the patch prompt does not know what B6 actually needed."

The arena A/B test measures how often each failure mode fires on real guides.

### 9.2 Variance under repetition

- **REPL**: variance is dominated by single-session randomness. Even at temperature 0, you get small differences across runs because of floating-point non-determinism in the backend, cache hit/miss timing, and llm_query_batched race ordering.
- **Sequential (with stage memoization enabled)**: variance is **structurally controlled**. Memoize all stages except the one under test, and retest runs are deterministic except for the deliberately-varied stage.

This is why Levine's test-retest framing favors sequential structurally: sequential gives a controlled variance mechanism (memoize all but one stage), REPL gives stochastic variance (everything varies together). Both can be measured, but sequential gives finer-grained diagnostics.

### 9.3 What sequential cannot do even after upgrade

- **Self-repair within a single stage** across multiple artifact types. REPL can emit supply_list, then during variables emit realize supply_list was wrong, and patch it in-place. Sequential has to flag the B1 mistake, run targeted backtracking to B1, and re-validate. Targeted backtracking is one retry deep, so a cascading error that touches B1 → B3 → B4 → B6 would still fail in sequential while REPL could handle it inside one session.
- **Continuous cache TTL refresh** across phases. REPL's 1h TTL survives the whole session because reads refresh the TTL. Sequential's cache lines are separate per stage type, so if a stage takes longer than the TTL the cache expires. At 1h TTL and ~30-40 min total wall clock this is not a problem today, but it constrains how long sequential can run before it starts paying cache-miss costs.

---

## 10. Implementation plan (ordered)

Execute in this order. Each phase is independently shippable and testable.

### Phase 0: preconditions

- [ ] Confirm the REPL gen 2.3 baseline is stable (Run #9 or later passes cleanly on WHO sample and full WHO)
- [ ] Confirm `CATCHER_CONCURRENCY_CAP`, `CATCHER_CHUNK_MAX_CHARS`, `CATCHER_USE_CITATIONS`, `CATCHER_CONTEXTUAL_HEADERS`, `GENERATOR_GUIDE_REPETITION` are all set to gen 2.3 defaults in `.env`
- [ ] Read `backend/validators/phases.py` end-to-end to understand the catcher surface sequential will call into
- [ ] Read `backend/rlm_runner.py:_generate_guide_toc` and `_maybe_attach_guide_block` to understand the TOC + repetition helpers

### Phase S1: catcher gen 2.1 port (red team chunking + 5-voter majority)

1. [ ] Add new catcher prompts in `backend/prompts/validators/`:
   - `completeness_raw_facts.txt`
   - `completeness_workflow_pattern.txt`
   - `completeness_context_mappings.txt`
   - `completeness_consolidation_rules.txt`
   - `completeness_rule_criteria.txt`
2. [ ] Extend `phases.py:valid_artifact_names()` to include the five new catcher targets
3. [ ] Extend `phases.py:validate_artifact()` dispatch to route the new artifact names to their catchers
4. [ ] In `backend/sequential/executor.py`:
   - Delete the Sonnet red team path in `_run_redteam` (lines 289-370)
   - Replace with a wrapper that calls `phases.validate_artifact(artifact_name, artifact_json, guide_json, api_key)` and adapts the `CatcherResult` to the sequential stage-loop shape
   - Remove `self._guide_context[:500_000]` truncation (line 345)
5. [ ] Run a smoke test on WHO sample: `python -m backend.sequential.runner --guide who_sample.json`
6. [ ] Verify the sequential red team now uses chunked voters (check logs for `chunk_voters` calls)

### Phase S2: catcher gen 2.2 port

1. [ ] Rewrite the five new catcher prompts (and the reused REPL catchers if sequential uses them verbatim) to follow the gen 2.2 two-stage enumerate-then-diff structure with dict `critical_issues` and isolated `guide_quote`
2. [ ] Verify `CATCHER_USE_CITATIONS=true` and `CATCHER_CONTEXTUAL_HEADERS=true` are set — no code changes needed, sequential inherits from `phases.py`
3. [ ] Run sequential smoke test and verify that hallucinated criticals are being caught by programmatic quote verification (check the `verified` field on critical_issues in the output)

### Phase S3: generator gen 2.3 port

1. [ ] In `backend/sequential/executor.py:_build_system_prompt`:
   - Import `_GUIDE_REPETITION_N` and `_GUIDE_CACHE_MAX_CHARS` from `rlm_runner.py` (or lift them into a shared module)
   - Implement 2× repetition with size-aware fallback (same rule as `_maybe_attach_guide_block`)
   - Apply only to the maker system block (not the red team's, since red team no longer builds one after S1)
2. [ ] In `backend/sequential/runner.py:run_sequential_pipeline`:
   - Call `_generate_guide_toc(guide_json, anthropic_key)` at session start (before the first stage)
   - Pass the TOC string into `StageExecutor` so `_build_system_prompt` can prepend it
3. [ ] In `backend/sequential/prompts.py`:
   - Rewrite B1_MAKER, B4_MAKER, B5B_MAKER, B6_MAKER to use two-stage enumeration with the `_enumerated_candidates` scratchpad field
   - Leave B2_MAKER and B3_MAKER as-is (small outputs, over-engineering)
4. [ ] Smoke test on WHO sample + full WHO
5. [ ] Compare output recall to REPL gen 2.3 on the same guide (use the existing REPL harness output as ground truth for what a "good" extraction looks like)

### Phase S4: sequential-only wins

1. [ ] **Stage memoization** (the biggest single win):
   - Create `sequential_stage_cache` table in Neon (add migration)
   - Add memoization check at the top of `executor.py:run_stage`
   - Add `ENABLE_STAGE_MEMOIZATION=true` env var gating
   - Add unit test: run the same guide twice, verify run 2 hits the cache for every stage
2. [ ] **Per-stage model routing**:
   - Add `model_override` field to `StagePrompt` dataclass in `prompts.py`
   - Route B1, B3, B6 to `claude-opus-4-6`
   - Plumb `model_override` through `executor.run_stage` into `call_anthropic`
3. [ ] **Deterministic gates**:
   - Create `backend/sequential/deterministic_gates.py`
   - Implement the per-artifact check functions from section 6.4
   - Wire into `run_stage` to fire before the LLM red team
   - Add targeted repair path: deterministic failures generate a specific repair prompt, not a full red team round
4. [ ] **Stage journaling**:
   - Add `output/{run_id}/journal.jsonl` writer to `run_stage`
   - Include full inputs, output, red team result, deterministic gate result, cost, duration, model, cache_hit flag
5. [ ] **Resume from journal**:
   - Add `--resume-from <run_id>` CLI flag to `runner.py`
   - On resume, read the journal, find the last successful stage, skip to the next stage
6. [ ] **Decomposed B6 catchers**:
   - Split `completeness_modules` catcher into coverage + safety + reachability + traceability + mece
   - Wire Z3 for reachability and MECE (reuse existing `backend/z3_verifier/`)
7. [ ] **Targeted backtracking**:
   - Add `patch_stage_output` method to `StageExecutor`
   - Cap at 1 patch per downstream stage
   - Emit `_patch_log` on patched stages

### Phase S5: cross-family (optional, gated on BYOK keys)

1. [ ] Add `backend/sequential/openai_catcher.py`
2. [ ] Add `backend/sequential/gemini_catcher.py`
3. [ ] Add `redteam_model_family` field to `StagePrompt`
4. [ ] Route B3 red team to GPT-5-mini and B6 post-validation to Gemini 2.5 Pro when keys are present
5. [ ] Fall back to Haiku if keys are missing

### Verification

After each phase, run the sequential pipeline on:

1. WHO sample (120K chars, 6 pages) — fast smoke test, ~3-5 min per run
2. Full WHO (1.94M chars, 140 pages) — full cost test, ~25-40 min per run
3. A second guide (e.g., the RLMS PDF in the repo root) — generalization test

Compare output artifacts to REPL gen 2.3 outputs on the same guides. The target is **same items captured, same quote verification, same recall numbers**, with the sequential pipeline achieving this via a different architectural path.

---

## 11. Open questions

These are design decisions not yet locked. Revisit before starting Phase S1.

1. **Split extraction for guides larger than 3.5M chars** (where 2× repetition fails even the 1× fallback). The REPL fallback is to skip the guide block entirely, relying on the system prompt's other context. Sequential does not have that fallback because every stage needs the guide in its cached system block. Options:
   - Split the guide by top-level section, run each stage N times (one per section), merge the outputs. Breaks the "each stage is one call" property.
   - Pre-compress the guide via a Haiku summarization pass that keeps every dosage, threshold, and danger sign verbatim. Adds a cost layer and a correctness risk.
   - Only support guides up to ~1.1M chars raw (2× repetition fits comfortably). This is the current REPL constraint and most clinical guides fit. Document the limitation and move on.

2. **What to do when stage memoization serves a stale artifact.** If the guide is updated (a new version is uploaded), the `guide_content_hash` changes and the memoization cache misses correctly. But what if the catcher prompts are updated? The `maker_prompt_hash` should change, but what about the catcher prompt hash? Should we include it in the cache key? Leaning toward yes — invalidate on any prompt change, even if only the catcher changed, because a newer catcher might have caught a hallucination the older catcher missed.

3. **Should the deterministic gates block or warn?** REPL's architecture checks are warn-only (non-blocking). Sequential could go stricter and block on hard failures (else-row missing, raw numerics present). Block-on-hard-failures matches sequential's "fail the stage, run repair" pattern more naturally. Recommend blocking on hard failures and warning on soft ones.

4. **Cross-family red team on which stages specifically.** Section 6.3 proposes B3 and B6. Is that the right split? The argument for B3: consolidation rules are where Claude's training data is thinnest (few examples of "cross-module clinical consolidation" in pretraining). The argument for B6: it is the most important artifact and benefits most from a second opinion. Both are plausible. First pass: enable on B3 and B6 together, measure, cut one if the second is not pulling weight.

5. **How to surface the journal to the research group.** A JSONL file is fine for automated tools but not for humans. Do we need a journal viewer in the frontend that shows each stage's input/output/redteam-report/cost? Probably yes for the arena, but can be deferred until after Phase S4 ships.

6. **Stage memoization security**. If the cache key hashes are content-addressed but the guide contains PHI, the cache table leaks content identity across runs. For a research tool this is fine, for a production deployment it would need per-tenant partitioning. Document as a known limitation and move on.

---

## Appendix A: file-level change summary

| File | Phase | What happens |
|---|---|---|
| `backend/sequential/executor.py` | S1, S3 | Replace `_run_redteam` with catcher wrapper; add 2× repetition + TOC to `_build_system_prompt`; add memoization check, deterministic gates, journaling, targeted backtracking to `run_stage` |
| `backend/sequential/prompts.py` | S3, S4 | Rewrite maker prompts with two-stage enumeration; add `model_override` and `redteam_model_family` fields to `StagePrompt` |
| `backend/sequential/runner.py` | S3, S4 | Call `_generate_guide_toc` at session start; add `--resume-from` CLI flag; wire journal writer |
| `backend/sequential/llm_client.py` | S4 | Add model parameter plumbing for per-stage routing; no structural change |
| `backend/sequential/deterministic_gates.py` | S4 | **New file.** Per-artifact Python predicate checks |
| `backend/sequential/openai_catcher.py` | S5 | **New file** (optional). GPT-5-mini catcher backend |
| `backend/sequential/gemini_catcher.py` | S5 | **New file** (optional). Gemini catcher backend |
| `backend/validators/phases.py` | S1 | Extend `valid_artifact_names()` and `validate_artifact()` dispatch with the new sequential catcher targets |
| `backend/prompts/validators/completeness_raw_facts.txt` | S1 | **New file.** Two-stage enumerate-then-diff catcher for B1 raw facts |
| `backend/prompts/validators/completeness_workflow_pattern.txt` | S1 | **New file.** Catcher for B1 workflow_pattern |
| `backend/prompts/validators/completeness_context_mappings.txt` | S1 | **New file.** Catcher for B2 context_mappings |
| `backend/prompts/validators/completeness_consolidation_rules.txt` | S1 | **New file.** Catcher for B3 consolidation_rules |
| `backend/prompts/validators/completeness_rule_criteria.txt` | S1 | **New file.** Catcher for B6 rule_criteria |
| `backend/prisma/schema.prisma` | S4 | Add `sequential_stage_cache` table |
| `.env` | S4 | Add `ENABLE_STAGE_MEMOIZATION`, optionally `BYOK_OPENAI_API_KEY` / `BYOK_GOOGLE_API_KEY` |

## Appendix B: environment variables added by this upgrade

```bash
# Sequential-specific knobs (optional, defaults are safe)
ENABLE_STAGE_MEMOIZATION=false      # Opt-in for test-retest runs
SEQUENTIAL_DETERMINISTIC_GATES=true # Pre-red-team Python gates on by default
SEQUENTIAL_DECOMPOSED_B6=true       # Split B6 red team into 5 specialized catchers

# Cross-family catchers (optional, Phase S5)
BYOK_OPENAI_CATCHER_KEY=             # If present, B3 red team uses GPT-5-mini
BYOK_GOOGLE_CATCHER_KEY=             # If present, B6 post-validation uses Gemini 2.5

# Shared with REPL (already set in .env today)
GENERATOR_GUIDE_REPETITION=2         # 2x guide repetition in cached system block
CATCHER_USE_CITATIONS=true
CATCHER_CONTEXTUAL_HEADERS=true
CATCHER_PROMPT_REPETITION=3
CATCHER_CONCURRENCY_CAP=3
CATCHER_CHUNK_MAX_CHARS=50000
```

## Appendix C: relationship to arena_style.md

This upgrade makes the arena A/B test (`arena_style.md`) actually meaningful. Before this upgrade, the sequential pipeline is running ~70-80% recall while REPL is running ~94-98%. Any A/B test on the current sequential would show REPL dominating for reasons that have nothing to do with architecture choice, everything to do with sequential not having had the catcher gen 2.x treatment.

After this upgrade, sequential and REPL both hit ~94-98% first-pass recall on the same guides. The A/B test then measures the actual architectural differences (section 9 above): backward self-correction vs forward DAG re-entry, and variance under repetition. That is the comparison Professor Levine asked for, and it is the comparison the arena was designed to measure.

The sequential-only wins (Phase S4) give sequential a structural advantage on the specific question Levine framed as "test-retest reliability" — stage memoization makes 3-run retests ~3× cheaper than the REPL equivalent, and variance isolation lets the research group answer "which stage is responsible for run-to-run differences" in a way REPL structurally cannot answer.
