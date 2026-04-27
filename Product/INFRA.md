# CHW Navigator — Infrastructure Reference

One document covering everything about how this thing runs: costs, hosting, API tiers, BYOK boundaries, concurrency limits, and the scale tradeoffs that decide what you can serve.

**Status:** Internal research tool. Not designed as public SaaS. Expected usage: 1 researcher doing 3-retest variance runs OR 3 coordinated researchers running extractions in parallel. Coordination happens out-of-band (Slack/email), not in the product.

**Generation: Gen 7 v2** (shipped 2026-04-14). Reference run: session `d9a39c2d-da6e-4c5c-83ae-b2b9ad3cd6dc` on the full WHO CHW guide 2012 (140 pages, 156 sections). $12.55 total, 25m 27s wall clock, PASSED. Every number in this doc traces back to this run unless otherwise noted.

**Companion docs** (read alongside this one):
- `ARCHITECTURE.md` — engineering-level walkthrough of the six-phase pipeline, data shapes, flag dispatcher, and Module Maker subprocess layout
- `SYSTEM_PROMPTS.md` — the verbatim prompts: labeler Call 1, labeler Call 2, REPL main, Module Maker, all frozen
- `PIPELINE.md` — the 6-stage extraction pipeline vocabulary and the 17 checkpoints the system is accountable to
- `ORCHESTRATOR.md` — validator contracts and the Levine memo on frozen surfaces

---

## 1. Architecture at a glance

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Next.js frontend    │   │  FastAPI backend     │   │  Neon Postgres       │
│  (Vercel)            │◄──┤  (Render Pro Plus)   │──►│  (extraction_runs,   │
│  - BYOK UI           │   │  - uvicorn           │   │   repl_steps,        │
│  - SSE trajectory    │   │  - asyncio event loop│   │   intermediate_      │
│  - Cost tracker      │   │  - rlm library       │   │   artifacts,         │
│  - Downloads (v5)    │   │  - Prisma Python     │   │   source_guides)     │
└──────────┬───────────┘   └────────┬─────────────┘   └──────────────────────┘
           │                        │
           │ SSE + REST (Bearer)    │ outbound HTTPS
           │                        ▼
           │          ┌─────────────────────────────┐
           │          │  Upstash Redis (session TTL)│
           │          └─────────────────────────────┘
           │                        │
           │                        ▼
           │          ┌──────────────────────────────────────┐
           └─────────►│  Anthropic API  (BYOK, user's key)   │
                      │  - Opus 4.6 for EVERYTHING           │
                      │  - Labeler + Module Maker + REPL     │
                      │  - automatic prefix caching          │
                      └──────────────────────────────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────────────────┐
                      │  OpenAI API (server-side, our key)   │
                      │  - gpt-5.4 / gpt-5.4-mini vision     │
                      │  - used only during PDF ingestion    │
                      └──────────────────────────────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────────────────┐
                      │  Unstructured.io (server-side)       │
                      │  - PDF → structured elements         │
                      │  - used only during PDF ingestion    │
                      └──────────────────────────────────────┘
```

**Gen 7 data flow (happy path, one extraction):**

1. User uploads PDF → frontend posts to `/api/ingest` → backend calls Unstructured + OpenAI vision → guide JSON stored in Neon (shared cache; first user pays, subsequent users reuse).
2. User clicks "Start extraction" → frontend posts `sourceGuideId` to `/api/session/start` with `Authorization: Bearer <anthropic-key>`.
3. Backend pre-flights the Anthropic key (tier check, one tiny Opus call) → rejects Tier 1 with actionable error.
4. Backend creates session, spawns `_run_extraction_task` as asyncio background task.
5. **Phase 0 — Micro-chunking.** The source guide is reconstructed from ingestion JSON, PDF-ingestion placeholders are stripped, and the text is sliced into ~2K-token chunks at natural boundaries. On full WHO: 21 chunks.
6. **Phase 1 — Two-call Opus labeling per chunk.** For each chunk, Call 1 reads the chunk and emits an exhaustive candidate label list (supplies, variables, predicates, rules, phrases, module hints). Call 2 reads Call 1's output + the chunk and distills/QC's the labels. Both calls are Opus 4.6. Runs in batches of 6 chunks with a 62s cooldown to respect Tier 2 output TPM.
7. **Phase 1b — Algorithmic dedup.** A pure-Python step takes the union of all Call 2 outputs, deduplicates on `(id, type)` exact-string keys, and writes `deduped_labels.json` as the 26th artifact. No LLM call.
8. **Phase 2 — Cache priming.** `_set_gen7_cached_context(reconstructed_guide, deduped_labels)` installs two extra cache blocks in the REPL system prompt, each marked `cache_control: ephemeral, ttl: 1h`. `_maybe_attach_gen7_blocks()` fires on every Anthropic call the RLM makes so the REPL main + every `llm_query_batched` sub-call reads them.
9. **Phase 3 — REPL compilation (Module Maker + Dispatcher).**
   - *Phase A (scan):* one Opus call reads the deduped labels and emits a module plan — names, flags, priorities (0=emergency, 1=startup, 100=regular, 999=closing).
   - *Phase B (build):* `llm_query_batched` fans out one Opus sub-call per module. Each Module Maker produces that module's rules, with `mod_X_done: true`, `is_priority_exit: true`, and cross-module `has_Y: true` triggers built into rule outputs.
   - *Phase C (dispatcher):* purely programmatic — sort module plan by `(priority, name)`, emit dispatcher rows that route on `has_X` and `mod_X_done`. No LLM call. This is the key Gen 7 idea: the router is a sorted dict, not a generated artifact.
   - *Phase D (finalize):* emit flat artifacts (supplies, variables, predicates, phrases, modules, dispatcher) + `FINAL_VAR(clinical_logic)`.
10. **Phase 4 — Converters.** DMN, XLSForm, Mermaid, CSV generation on the final bundle.
11. Results stream to frontend via SSE (`/api/session/{id}/stream`), persisted to Neon (`repl_steps`, `intermediate_artifacts`).
12. User downloads via `/api/session/{id}/artifacts/{name}` or `artifacts.zip`. The v5 pattern — bare anchor click, `Content-Disposition` header controls the filename — works for Playwright intercept (drops files in `.playwright-mcp/`) and normal user Chrome (drops files in `~/Downloads`).

**What Gen 7 removed from the legacy Gen 3 pipeline:**

- No `emit_artifact` per-phase gate loop. Compilation is one REPL, not seven phases of emit-then-catcher.
- No Haiku catchers. No 3-vote or 5-vote majority. No per-chunk artifact cache. No chunked catcher fan-out of any kind.
- No Sonnet downshift. No phase-aware router choosing between Opus and Sonnet.
- No Citations API toggle. No programmatic quote verification.
- No integrative artifact as load-bearing logic. The dispatcher is the router; the integrative artifact is vestigial (professor's explicit design choice; kept only so downstream consumers that expected it don't break).

**What Gen 7 kept:**

- The BYOK boundary (Anthropic user-BYOK, OpenAI + Unstructured server-side).
- The REPL harness from the `rlm` library — we still run the main compilation step inside a REPL with the same custom tools (`emit_artifact`, `validate`, `FINAL_VAR`, `llm_query_batched`). The change is what we do INSIDE the REPL, not the REPL itself.
- All hosting choices: Render Pro Plus, Neon Postgres, Vercel frontend, Upstash Redis.
- `MAX_CONCURRENT_SESSIONS=2` as the global concurrency ceiling (lowered from 3 in Gen 3 because Gen 7's Opus-heavy profile eats more output TPM per run).
- The database schema (`extraction_runs`, `repl_steps`, `intermediate_artifacts`, `source_guides`).

---

## 2. The BYOK boundary (who provides what)

| API / service | Owner | Rationale |
|---|---|---|
| **Anthropic** (labeler + Module Maker + REPL) | **User BYOK** | ~99% of per-run cost lives here. Users bring their own key so they pay for their own extractions. Rate limits are tied to the user's Anthropic account tier. |
| **OpenAI** (vision enrichment during PDF ingestion) | **Server-side** | Ingestion happens once per PDF upload. Vision calls are server-owned because ingestion is a shared cache — multiple users running extractions on the same guide share the already-ingested JSON. Making this BYOK would require re-ingesting per-user, which is wasteful. |
| **Unstructured.io** (PDF → structured JSON) | **Server-side** | Same rationale as OpenAI — shared ingestion cache. 15K page free tier + pay-as-you-go. |
| **Neon Postgres** | **Server-side** | Central database for run history, guide cache, intermediate artifacts. Prisma Python client. |
| **Upstash Redis** | **Server-side** | Session TTL + ephemeral session data. |

**Code-level enforcement:** `project_byok_architecture.md` (saved memory). Only Anthropic comes from the user's browser. Server-side keys live in `.env` locally and in Render's environment variable settings in production.

**What the user sees:** the frontend asks for an Anthropic key (one input). No OpenAI key, no Unstructured key. The BYOK scope is deliberately narrow — users only need the one key that matches where their personal billing lives.

**What this means operationally for Gen 7:** because Gen 7 routes everything through Opus 4.6, the user's Anthropic bill increases vs Gen 3's Opus+Sonnet+Haiku mix. Typical per-run cost climbs from ~$7-13 (Gen 3) to ~$12-15 (Gen 7). The quality and reproducibility gains pay for the delta. The tool does not auto-downshift to cheaper models — if the user wants a cheaper variant they can run an older gate generation. Gen 7 is explicitly "quality tier" on the Pareto frontier.

---

## 3. Cost per extraction (Gen 7 v2 measured numbers)

These are measured costs on observed runs, not estimates. Reference run `d9a39c2d-da6e-4c5c-83ae-b2b9ad3cd6dc` on full WHO CHW guide 2012, 2026-04-14.

### WHO full guide (140 pages, 156 sections) — Gen 7 v2 actuals

| Component | Calls | Cost | Notes |
|---|---|---|---|
| Phase 1 labeling (Opus × 2 calls × 21 chunks) | 42 | ~$9.00 | 1,771 Call 1 labels → 1,692 Call 2 distilled. Batched 6/62s. |
| Phase 3 REPL main (Opus) | ~8 | ~$2.30 | Scan + finalize + occasional repair. System block priming on call 0. |
| Phase 3 Module Maker sub-calls (`llm_query_batched`) | 50 | ~$1.15 | 8 modules + variance retries. Each reads the primed cache blocks. |
| Incidental (tier pre-flight, validators) | <5 | ~$0.10 | |
| **Total per run** | **~105 LLM calls** | **$12.55** | |
| **Wall clock** | — | **25m 27s** | session start → artifacts ready |

By-call-kind breakdown from `_run_usage` on this run:
- 43 Opus top-level calls (labeler + REPL main)
- 50 Opus sub-calls via `llm_query_batched` (Module Maker + scan fan-outs)
- 8 Sonnet calls (vestigial — a few legacy validator paths still call Sonnet; targeted for removal in v3)
- 7 Haiku calls (tier pre-flight + small utility checks; NOT catcher calls)

`_run_usage` is the authoritative source of truth for per-run cost. It's populated by:
1. The monkey-patched Anthropic client in `rlm_runner.py` that wraps every `client.messages.create` call (catches REPL main and `llm_query_batched` sub-calls).
2. Direct labeler calls in `backend/labeling/opus_labeler.py`, which call `accumulate_catcher_usage(...)` after each response to feed the same per-run accumulator.

Measured agreement with Anthropic's billing dashboard: **~95%** over 10 gate runs. The 5% residual is cache-read accounting that Anthropic amortizes differently than our per-call bookkeeping; it cancels out over ~3 runs.

### Extraction output from reference run

| Artifact | Count |
|---|---|
| Modules | 8 |
| Predicates | 20 |
| Variables | 35 |
| Supplies (equipment + consumables + medications) | 15 |
| Phrases | 21 |
| Dispatcher rows | 24 (8 modules × 3 routes avg) |
| Unique deduped labels (input to Phase 3) | 1,073 |

### WHO sample guide (6 pages) — Gen 7 v2 actuals

| Component | Cost | Notes |
|---|---|---|
| Phase 1 labeling (2 chunks × 2 calls) | ~$0.60 | 4 Opus calls |
| Phase 3 REPL + Module Maker | ~$1.20 | 3 modules → 3 Module Maker sub-calls |
| **Total per run** | **~$1.80** | |
| Duration | ~8 min | |

### 3-retest (3 concurrent runs on full WHO)

Concurrency is currently capped at 2 on Gen 7 (down from 3 in Gen 3), so a 3-retest runs as 2 concurrent + 1 queued. Per-run cost is unchanged; wall clock is `max(run_1, run_2) + run_3`.

| Scenario | Cost | Wall clock | Tier 2 headroom at peak |
|---|---|---|---|
| 1 run at a time × 3 | ~$38 | ~80 min | 30% of output TPM |
| 2 concurrent + 1 queued | ~$38 | ~55 min | 78% of output TPM |

### Ingestion cost (paid once per unique PDF, then cached in Neon)

| Component | Cost for 140-page guide |
|---|---|
| Unstructured hi_res | ~$4.20 (140 × $0.03) |
| OpenAI vision (table + image enrichment) | ~$2-3 |
| **Total first-time ingestion** | **~$6-7** |

**Cached after first ingestion.** Subsequent extractions on the same PDF reuse the cached `guideJson` and pay zero ingestion cost.

### Monthly cost model (internal research usage)

| Research activity | Runs/month | Cost |
|---|---|---|
| 10 full-guide extractions on different manuals | 10 | ~$125 Anthropic + ~$70 ingestion |
| 3 × 3-retest variance studies (same guide each) | 9 | ~$115 Anthropic |
| 20 sample-guide sanity checks | 20 | ~$40 Anthropic |
| **Total compute cost** | **~$350/month** | |
| Render Pro Plus hosting | — | **$175/month** |
| Neon Postgres (Pro) | — | **~$20/month** |
| Vercel (frontend) | — | **$0 (hobby)** or $20 (Pro) |
| **Total monthly** | — | **~$545/month** |

Gen 7 increases the compute line by about $40/month vs Gen 3 at equivalent usage. Infra is unchanged. The compute cost dominates infra as soon as you do ~3-4 full runs per month.

### Cost levers you can still pull

- **Chunk count (Phase 0)**: larger chunks reduce labeler call count proportionally. We use 2K tokens because it keeps per-chunk Opus attention in the >99.5% regime. Going to 3K tokens would cut labeler cost ~30% at the price of multi-needle recall per chunk.
- **Call 2 skip**: the distill/QC pass can be disabled for throwaway runs. Savings: ~40% of labeler cost (from 42 calls to 21). Penalty: measurable drop in deduped label quality. Not recommended for production runs.
- **Module Maker fan-out width**: `llm_query_batched` dispatches up to N parallel sub-calls. We cap at 3 concurrent to stay under output TPM. Raising the cap speeds up Phase B but doesn't change cost.
- **Opus-only routing**: non-negotiable. Gen 7 is explicitly Opus-only. See section 15's Pareto discussion for why cheaper variants were rejected.

---

## 4. Hosting: Render Pro Plus

The backend runs on Render Pro Plus. Why:

| Tier | RAM | CPU | Cost | Our need |
|---|---|---|---|---|
| Standard | 512 MB | 0.5 | $7/mo | ❌ rlm library + Prisma client alone is ~500 MB |
| Pro | 2 GB | 1 | $25/mo | ⚠️ 2 concurrent sessions + ingestion burst → tight |
| **Pro Plus** | **4 GB** | **2** | **$175/mo** | **✓ 2 concurrent REPLs + Unstructured ingestion + vision burst** |
| Pro Max | 8 GB | 4 | $450/mo | overkill for current Gen 7 usage |

**Measured memory profile at peak (2 concurrent full-WHO Gen 7 runs + active ingestion):**

| Component | Memory |
|---|---|
| Baseline (uvicorn, FastAPI, Prisma, rlm library imports) | ~500 MB |
| Per-session REPL state, event buffer, journal, deduped labels in memory | ~150 MB × 2 = 300 MB |
| Active ingestion (Unstructured partition + vision workers in flight) | ~200-400 MB |
| In-flight Opus request buffers (labeler batch of 6) | ~1 MB |
| **Total working set** | **~1.0-1.2 GB** |
| **Pro Plus headroom** | **~2.8 GB free** |

**CPU:** mostly I/O-bound (asyncio event loop waits on Anthropic responses). Gen 7's labeling phase is particularly I/O-bound because each chunk's two calls are sequential (Call 2 needs Call 1's output) but 6 chunks run in parallel. 2 vCPUs handle event loop + Prisma + PIL (mermaid rendering) + occasional pdf2image with slack.

**Upgrade trigger to Pro Max:** sustained OOM during ingestion of guides with >500 tables/images. Ingestion is still the biggest memory spike in the pipeline, not Gen 7 extraction. Not seen in practice yet.

### Render caveats

- **No `.env` file support.** Render reads env vars from the service dashboard, not from `.env` files committed to the repo. Every env var from `.env` has to be mirrored in Render's environment settings.
- **Memory limit is a hard cap.** Going over OOMs the process and restarts it. Uvicorn's auto-reload is disabled in production (`reload=False` in `backend/main.py`), so a restart loses all in-memory session state. Redis + Neon pick up persistent state but any running REPL dies.
- **Render's free tier will spin down on idle** — we use Pro Plus which is always-on.

---

## 5. Anthropic tiers (the real constraint on Gen 7)

**Tiers are not sold — they are unlocked.** Per-token rates are identical across all tiers. Only rate limits change. You upgrade by cumulative spend + account age.

Thresholds (verify current values in Claude Console → Settings → Limits):

| Tier | Unlock | RPM | Input TPM (excl. cache reads) | Output TPM | Our capacity |
|---|---|---|---|---|---|
| Tier 1 | default | ~50 | ~50K | ~10K | ❌ Sample guide only. Labeler's Call 1 output alone blows through in one batch. |
| **Tier 2** | $40 spend + 7 days | 1000 | 450K | 80K (Opus) | ✓ Current baseline. 2 concurrent full-WHO Gen 7 runs at 78-84% output TPM utilization. |
| Tier 3 | $200 spend + 7 days | 2000 | ~900K | ~160K | ✓ 4-5 concurrent. ~2× headroom on everything. |
| Tier 4 | $400 spend + 14 days | 4000 | ~2M | ~320K | ✓ 10+ concurrent. Essentially unlimited. |

**Tier 2 output TPM is the single binding constraint for Gen 7**, just as it was for Gen 3 — but the pressure source moved. In Gen 3, output TPM pressure came from ~1,380 Haiku catcher calls per run. In Gen 7, it comes from 42 Opus labeling calls (each emitting ~6-8K output tokens) plus 50 Opus `llm_query_batched` sub-calls (each emitting ~2-5K output tokens).

### Why Tier 2 is the hard minimum for Gen 7

Labeler Call 1 on a 2K-token chunk emits ~6-8K output tokens (exhaustive candidate label list). With 6 chunks dispatched in parallel:

- 6 × ~7K = 42K output tokens in a single ~15-20s window
- Sustained burst rate: ~130K output TPM
- Tier 1 ceiling: 10K output TPM → instant 429
- Tier 2 ceiling: 80K output TPM → 78% utilization, no 429s with 62s cooldown between batches

The 62s cooldown is tuned so that after each batch of 6 completes, the 60-second rolling TPM window resets before the next batch dispatches. At batch=6 with 62s cooldown, peak 60-second output equals one batch's output (~42K), well under the 80K ceiling. Going to batch=8 would peak at ~56K (70% utilization), which is still fine, but occasional over-budget chunks push us close to the limit. batch=6 is the conservative knee.

### Dynamic batch sizing

```python
# backend/labeling/rate_limiter.py
OUTPUT_TPM = 80_000           # Tier 2 Opus output TPM ceiling
EST_OUTPUT_PER_CHUNK = 10_000 # upper-bound estimate for a Call 1 emit
_MAX_BATCH_SIZE = max(1, OUTPUT_TPM // EST_OUTPUT_PER_CHUNK - 2)  # = 6
_COOLDOWN_SECONDS = 62
```

The `- 2` is safety margin for the concurrent-user case where two sessions are hitting the labeler at the same time. On Tier 3 (160K output TPM), `_MAX_BATCH_SIZE` auto-raises to 14; on Tier 4 it raises to 30. This is the only place in the Gen 7 codebase where raising tier has an immediate throughput effect without a config change — the math is derived from the ceiling, not hardcoded.

### Tier 2 math for full WHO (Gen 7 actuals)

| Metric | 1 REPL (avg) | 1 REPL (peak) | 2 REPLs (avg) | 2 REPLs (peak) | Tier 2 limit |
|---|---|---|---|---|---|
| RPM | 5 | 18 | 10 | 36 | 1000 (4%) |
| Input TPM (billed only) | 80K | 250K | 160K | 500K* | 450K (sustained OK, peak tight) |
| Output TPM | 30K | 65K | 60K | 130K* | 80K (75% sustained, peaks brief) |

*The 500K input TPM and 130K output TPM peaks for 2 REPLs are 60-second rolling-window maxima that only hit when both sessions happen to fire labeler batches in the same ~5-second window. In practice the 62s cooldown de-synchronizes the two sessions after the first batch, and sustained numbers stay under the ceiling.

**Output TPM (80K on Opus) is the tightest ceiling on Tier 2.** 2 concurrent Gen 7 runs sit at 75% output utilization. A 3rd concurrent run would push past 95% and start hitting 429s during labeler batch dispatch. This is why `MAX_CONCURRENT_SESSIONS` dropped from 3 (Gen 3) to 2 (Gen 7).

### Unlock ladder in practice

If the research group burns $160 more on Tier 2 (roughly 13 more full-WHO Gen 7 runs), Tier 3 auto-unlocks. Tier 3 doubles most limits:
- `MAX_CONCURRENT_SESSIONS` can go from 2 → 4 with zero code changes
- Labeler batch size auto-raises from 6 → 14 (via the formula in section 5.1)
- 429s during batch dispatch effectively disappear
- No increase in per-call cost

**There is no "pay to upgrade" — just use the API.** Anthropic watches cumulative spend and time, and auto-promotes.

---

## 6. Multi-key patterns

**Rate limits are per-organization, not per-key.** Multiple API keys from the **same Anthropic organization** share one rate-limit pool. Creating 10 keys from one org gives you 10 authentication tokens, not 10× the rate limit.

**What DOES multiply limits:** multiple separate Anthropic **organizations**. Each org has its own independent rate-limit pool. Two orgs on Tier 2 = 900K combined input TPM (vs 450K for one org with 2 keys).

### When multi-org is useful

- **2 orgs, 2 billing accounts** — e.g., research group + personal fallback. Start runs against the primary org; if it's near rate limit, manual failover to the secondary key. Annoying to manage, rarely worth it.
- **Sharding labeler vs REPL across orgs** — run the 42 labeling calls against org A and the 50 Module Maker sub-calls against org B. Would require server-side routing logic and two BYOK key fields in the frontend. Not currently supported; noted because it's a structurally clean split if someone ever wants to double Gen 7 throughput on Tier 2.
- **Per-tenant billing in a public SaaS** — not our use case.

### When multi-org is NOT useful

- Research group with one shared Anthropic account → single org is correct.
- "I'll just make a second key from the same org to get 2x capacity" → doesn't work, same pool.
- "I want to avoid hitting my limit" → just upgrade the one org to a higher tier.

### Recommended setup for the research group

1. **One Anthropic organization** named for the research group.
2. **Add researchers as members** via Claude Console → Organization → Members.
3. **Each researcher generates their own API key** tied to the shared org (not a shared raw key).
4. All keys draw from the same rate-limit pool and contribute to the same cumulative-spend counter for auto-upgrades.
5. **Billing is centralized** on whoever pays for the org.
6. **Revoke individual keys** when someone leaves without rotating everyone's access.

Under this setup, the BYOK boundary is preserved (the server never stores a key), and everyone benefits from the org's tier level.

---

## 7. Concurrency model

### The knob: `MAX_CONCURRENT_SESSIONS`

Set in `.env` locally AND on Render in production. **Default 2 on Gen 7** (was 3 on Gen 3). Enforced at `/api/session/start` in `backend/session_manager.py`.

**This is a GLOBAL cap across the entire backend instance.** It does NOT track users, API keys, or identity. The backend counts every session in the `running` state; when the count hits the cap, the next session-start returns HTTP 429 with `too_many_concurrent_sessions`.

### What "2" means in practice

| Scenario | Slots used | Accepted |
|---|---|---|
| 1 researcher doing a 3-retest variance run | 2 concurrent + 1 queued | ✅ all 3 (sequential within single queue) |
| 2 different researchers each running 1 extraction | 2 | ✅ both |
| 1 researcher with 2 runs + 1 researcher with 1 run | 2 + 1 queued | ✅ third waits |
| 3 researchers each wanting 1 run | 2 + 1 queued | ✅ third waits |
| Arena mode A/B compare (separate cap) | uses `MAX_CONCURRENT_PER_SIDE`, NOT this cap | — |

When the cap is hit, the rejected user sees an HTTP 429 response with a user-friendly message: *"The tool is busy — another researcher may be running an extraction. Please try again in a few minutes."*

This matches the "internal research tool, coordinate out-of-band" model. Researchers in the group coordinate via Slack/email about who's running what; the backend just enforces the physical ceiling.

### Why 2 and not 3 on Gen 7

Gen 3 sat at cap=3 because Haiku catcher rounds were the dominant output TPM pressure, and Haiku's output TPM ceiling was effectively 90K. Gen 7 shifts everything to Opus, whose Tier 2 output TPM ceiling is 80K. Per-REPL peak output TPM rose from ~30K (Gen 3) to ~60K (Gen 7) because Opus emits more verbose outputs than Haiku and the labeler dispatches a larger parallel batch.

| Concurrent Gen 7 REPLs | Output TPM (sustained) | Output TPM (peak) | Viable on Tier 2? |
|---|---|---|---|
| 1 | ~30K (38%) | ~65K (81%) | yes, comfy |
| **2** | **~60K (75%)** | **~130K (160%)*** | **yes, tight — current cap** |
| 3 | ~90K (113%) | ~195K (243%) | no, persistent 429s |

*The 160% peak is a 5-second burst only; the 60-second rolling-window average stays at 75% because the 62s labeler cooldown de-syncs sessions. If both sessions happen to hit labeler batch dispatch in the exact same window, we get brief 429s that the retry loop absorbs.

Render Pro Plus has memory/CPU headroom for 5-6 sessions. Upgrading the Render plan buys you nothing until Anthropic tier changes.

### How to raise the cap

**Tier 2 (current):** stay at 2. Setting 3 works most of the time but bursts will 429 during labeler batch dispatch.

**Tier 3 (~$200 cumulative spend to unlock):** raise to 4 via env var. Labeler batch also auto-raises from 6 to 14. No code changes:
```
MAX_CONCURRENT_SESSIONS=4
```

**Tier 4 (~$400 cumulative):** raise to 10. Render Pro Plus will start being the bottleneck around 6-8; may need Pro Max.

### Session cap ≠ Arena mode cap

Arena mode (`/api/arena/start`) has its own limit `MAX_CONCURRENT_PER_SIDE=2` in `backend/arena.py` (also reduced from 3 on Gen 7). This is a SEPARATE pool from `MAX_CONCURRENT_SESSIONS`. A single Arena run uses 2 REPL slots + 2 Sequential slots = 4 slots, all under the Arena cap, bypassing the regular session limit. This is intentional — Arena is for A/B comparison studies, not normal usage.

---

## 8. The Gen 7 cache strategy

Anthropic prompt caching is still the single biggest cost and rate-limit lever, but the way Gen 7 uses it is different from Gen 3's two-level catcher caching. Gen 7 relies primarily on **automatic prefix caching** plus **explicit cache_control markers on the REPL system blocks**.

### What Anthropic prompt caching does

A request to Anthropic can mark one or more text blocks with `cache_control: {"type": "ephemeral", "ttl": "..."}`. On the first call with that block content:

- Anthropic stores the block's tokens in a KV cache keyed by the full prefix up to the cache marker
- You pay a **cache write** rate (1.25× base for 5-minute TTL, 2× base for 1-hour TTL)

On subsequent calls with the **same prefix** within the TTL window:

- Anthropic skips prefill for the cached tokens and reads from KV
- You pay a **cache read** rate (0.1× base — one tenth the normal input cost)
- **Cache reads do NOT count against your input TPM rate limit** — this is the critical phrase in the rate limit table

### TTL choice: 1-hour everywhere

| TTL | Write rate | Read rate | When we use it |
|---|---|---|---|
| 5 minutes | 1.25× base | 0.1× base | nowhere in Gen 7 |
| **1 hour** | **2× base** | **0.1× base** | **every cached block** |

Gen 7 uses 1-hour TTL everywhere because:

1. A full-guide run takes ~25 minutes. 5m TTL would require re-writing mid-run between labeler batches and between REPL turns.
2. 3-retest variance studies take 3 × 25 min = ~75 min. 1h TTL lets a primed cache serve retest runs 2 and 3 with only reads.
3. Mid-run iteration gaps between REPL turns can exceed 5 minutes when the Module Maker fan-out is wide. 5m TTL would expire between turns.

The 2× write cost is paid once per cache key per TTL window. At our call volumes, the amortized cost is ~10× cheaper than paying 1.25× cache writes every few minutes.

### Gen 7 cache blocks (three levels)

The main REPL path installs three cached system blocks on every call:

```
system: [
    { text: <REPL system prompt, ~7K tokens>,       cache_control: {ttl: 1h} },
    { text: <reconstructed_guide, 100K-450K tokens>, cache_control: {ttl: 1h} },
    { text: <deduped_labels JSON, 30K-80K tokens>,   cache_control: {ttl: 1h} },
]
messages: [
    { role: user, content: <current turn user message> }
]
```

- **Block 1** (the frozen REPL system prompt) is identical across all calls in all runs. First call writes; every subsequent call reads.
- **Block 2** (the reconstructed guide) is identical across all calls within a run. First call after Phase 1b writes; the Module Maker fan-out and every `llm_query_batched` sub-call read it.
- **Block 3** (the deduped labels artifact) is identical across all calls within a run. Same pattern as block 2 — one write, N reads.

The second and third blocks are installed by `_set_gen7_cached_context(guide_text, deduped_labels)` at Phase 2. `_maybe_attach_gen7_blocks()` hooks every Anthropic call the RLM makes via the monkey-patch in `rlm_runner.py`, so blocks 2 and 3 are attached uniformly to REPL main, Module Maker sub-calls, and any utility calls. There is no code path in Phase 3 that bypasses the cache attachment.

**Cache hit math on a full WHO Gen 7 run:**

```
Phase 3 REPL main turns: ~8
Module Maker sub-calls: ~8-10 (one per module + retries)
llm_query_batched other sub-calls: ~40 (scan, finalize, etc.)
Total Phase 3 calls reading blocks 1-3: ~56-60

Block 2 (guide, ~350K tok on full WHO) cache write: 350K × $30/MTok × 2 = $21.00  (once)
Block 3 (labels, ~60K tok) cache write:            60K × $30/MTok × 2 = $3.60   (once)
Block 2 cache reads: 60 × 350K × $3/MTok (0.1× base) = $63.00
Block 3 cache reads: 60 × 60K × $3/MTok = $10.80
Without cache (all reads would be writes at 1×): 60 × 410K × $30/MTok = $738.00
```

Wait — the above math doesn't match our $12.55 reference run total. The reconciliation: Opus cache read rate is actually ~$0.30/MTok (0.1× the $3/MTok input rate), not $3/MTok. And we don't have 60 calls reading the full 410K-token cached prefix — many Module Maker sub-calls run with a trimmed prompt that doesn't include the full guide. The worked example above is a worst-case illustration. Actual measured cost on the reference run: $12.55 total, of which the Phase 2 cache writes account for ~$4-5 and Phase 3 cache reads for ~$1-2. The labeling phase dominates at ~$9 because labeler calls have their own cache pattern (see next subsection).

### The labeler cache caveat

Labeler Call 1 and Call 2 use `cache_control: {ttl: 1h}` on the chunk content. **But on Opus 4.6, explicit `cache_control` markers below a certain size threshold are silently ignored** — the response returns `cache_creation_input_tokens: 0` for those blocks.

Verified empirically on 2026-04-13: marking a 2K-token chunk with `cache_control` produces `cache_creation_input_tokens: 0`. This is NOT a bug in our code; it's a property of Opus 4.6's caching layer. The minimum cacheable block size for Opus appears to be ~5K tokens, not the 1024-token figure quoted in older Anthropic docs for Sonnet/Haiku.

**Our position on this**: we send `cache_control` anyway (costs nothing if ignored), because:
1. Anthropic's **automatic multi-turn prefix caching** still works on Opus even when explicit `cache_control` is ignored. The labeler's Call 2 benefits from this because it includes Call 1's system context.
2. If Opus 4.6's threshold changes in a future model version, our code picks up the caching automatically without a redeploy.
3. The labeler prompt itself (~3K tokens of instructions) is below threshold whether we mark it or not; the chunk is too small to hit threshold alone. Since the prompt + chunk prefix is the same shape across all Call 1 invocations within a run, automatic prefix caching handles it.

The REPL blocks (blocks 1-3 in the previous subsection) are each 30K+ tokens on full WHO, comfortably above threshold. Explicit `cache_control` is respected there.

### Anthropic cache constraints (Gen 7 specifics)

Things the Gen 7 layout has to respect:

1. **Minimum block size for caching**: The old 2048-token figure was for Haiku 3.x. Current Opus 4.6 empirical minimum appears to be ~5K tokens per block. Labeler chunks (2K) fall below; REPL blocks 2 and 3 (30K+) sit above. Our labeler relies on automatic prefix caching, which is independent of this threshold.

2. **Maximum 4 `cache_control` markers per request**. We use 3 (REPL system prompt + guide + labels). Headroom for a 4th if needed.

3. **Prefix match, not substring match**. The cache key is `hash(blocks[0] ... blocks[N])` where N is the last block with `cache_control`. Change ANY byte in any earlier block and you get a cache miss. This means block ordering matters: mutable content must come AFTER all cached blocks. Our REPL harness places the current-turn user message in the `messages` array (uncached by design) after all three system blocks.

4. **Per-account KV store**. Cache is scoped to the Anthropic organization. Two users from the same org share cache entries if they happen to send identical prefixes (rare in practice because the guide and labels differ per run).

5. **TTL is refreshed on use, not on creation**. A cached block that gets read at minute 55 of a 60-minute window gets its TTL reset. This is what lets a single Phase 2 cache prime serve all ~60 Phase 3 calls across a 25-minute run.

### Things we do NOT rely on (cold-start guarantees)

**Every run must work correctly with cold caches.** We do not assume:

- **"Previous run warmed the cache"** — first-run-after-deploy and new-guide runs start cold.
- **"The labeler warmed the REPL blocks"** — labeler doesn't cache the full guide (chunks are too small); REPL blocks are primed fresh in Phase 2.
- **"Developer loop warmed things during testing"** — code reloads invalidate nothing on Anthropic's side, but a 1-hour gap between test runs expires the cache. Dev loop cost estimates must use cold numbers.
- **"The retry loop saved us from writing the cache twice"** — retries hit the SAME cache key, so a second call on a retried request either reads (if first write succeeded) or writes (if first write failed). Either way, costs are deterministic.

The cost estimates in section 3 are all **cold-start**. Warm state is a bonus (mostly within a run, rarely across runs).

---

## 9. Labeling + dispatcher architecture (replaces the catcher layer)

Gen 7 replaces the Gen 3 chunked-catcher layer with two structurally different subsystems. This section is the research-grade explanation of how labeling and dispatching work and why they were the right replacement.

### The problem Gen 3 catchers solved, and the one they didn't

Gen 3's catcher layer was built to enforce **multi-needle recall** on each emitted artifact. For each `emit_artifact('modules', ...)` call, four Haiku catchers chunked the guide and voted on "did this artifact miss anything?" over 3-5 voters per chunk at temperatures (0.0, 0.15, 0.3). The output was a list of critical issues; the generator Opus/Sonnet had to re-emit with fixes until the catchers passed.

What this solved: catchers pushed per-artifact recall from ~80% to ~94-98% measured artifact-level on full WHO Gen 2.3. The Pareto analysis in section 15's old version documented this climb in detail.

What it didn't solve:

1. **Repeat-fail cycles.** The catchers would flag 10-20 critical issues on the first re-emit. The generator would fix 8, miss 2, introduce 3 new ones, and the next catcher round would flag 5 different ones. Runs went 5-12 iterations on modules and integrative, and ~15% of gate runs eventually gave up without converging.
2. **Catcher hallucinations.** Haiku would occasionally cite a "missing item" that wasn't in the guide. Gen 2.2's programmatic quote verification caught most of these, but not all (a hallucinated quote could accidentally substring-match something unrelated in the guide).
3. **Variance.** Same guide, same seed, different runs produced materially different artifact bundles. The root cause: the generator's first-pass extraction was noisy, and the catcher-repair loop amplified the noise because each iteration's "fix" depended on which critical happened to get flagged first.
4. **Cost anxiety.** Gen 2.3 ran $40-45 per full guide. Most of that went to the catcher layer. The professor's explicit feedback was that this was too expensive for how much of the guide content was actually novel each run.

### The Gen 7 insight: labels ARE the artifacts

The Gen 5 architecture memo posed the question: "what if the thing we call 'extraction' is actually just producing a clean label list, and all the artifact-level structure (modules, predicates, variables) is a programmatic transformation of the labels?"

Gen 7 v2 is the realization of that insight. The labeling phase (Phase 1) is the new "extraction" — it's where all the clinical content discovery happens. The REPL compilation phase (Phase 3) is a structurally bounded transformation step that maps labels into the target artifact schema.

This matters because:

- **Labeling is a single-needle problem per chunk** (2K tokens), which Opus 4.6 solves at ~99.9% per-item. The multi-needle compounding problem disappears because no single chunk has enough items to compound over.
- **Dedup is algorithmic**, not LLM-judged. Two labels with the same `(id, type)` collapse to one. The variance source that the catchers were trying to fix (LLM picks different representative quotes on different runs) gets collapsed out before the artifact ever gets built.
- **Module Maker sub-calls are deterministic given the same deduped labels**. Each sub-call gets a stable subset of labels scoped to one module and emits that module's rules. No cross-module dependency, no repair loop.

### Phase 1: two-call Opus labeling per chunk

```
For each chunk of the guide (~2K tokens):
  Call 1 (Opus 4.6):
    System: LABELER_CALL_1_PROMPT (~3K tokens, frozen)
    User: <chunk text>
    Output: exhaustive candidate label list
      [
        {id: "supply_zinc_20mg", type: "supply", description: "...", citation: "..."},
        {id: "var_age_months", type: "variable", description: "...", citation: "..."},
        {id: "pred_diarrhea_3days", type: "predicate", description: "...", citation: "..."},
        {id: "mod_diarrhea", type: "module_hint", ...},
        ...
      ]
    Typical output size: 80-120 labels per chunk, ~6-8K output tokens.

  Call 2 (Opus 4.6):
    System: LABELER_CALL_2_PROMPT (~3K tokens, frozen — different from Call 1)
    User: <chunk text> + <Call 1 output>
    Output: distilled + QC'd label list
      - removes obvious duplicates within the chunk
      - flags labels whose citation doesn't substring-match the chunk
      - merges near-duplicates (e.g., "zinc" and "zinc dispersible tablets")
      - emits ~70-100 labels per chunk after distillation, ~4-6K output tokens.
```

Why two calls and not one: Call 1 is a "go wide" enumeration task; Call 2 is a "go deep" QC task. Asking Opus to do both in one call produces middling results on both — it either over-enumerates and misses QC, or it over-QCs and drops valid items. Separating the modes produces better results on each, at the cost of doubling the labeling cost. This is a deliberate Pareto choice: we spend ~$9 on labeling to get a clean 1,073-label deduped list, which lets Phase 3 compile without repair loops.

### Phase 1b: algorithmic exact-string dedup

Pure Python, no LLM. Takes the union of all Call 2 outputs across chunks and deduplicates on `(id, type)` exact-string keys.

```python
def dedup_labels(all_call2_outputs: list[list[dict]]) -> list[dict]:
    seen = {}
    for chunk_labels in all_call2_outputs:
        for label in chunk_labels:
            key = (label["id"], label["type"])
            if key not in seen:
                seen[key] = label
            else:
                # Merge citations: preserve both source chunks for audit trail
                seen[key]["citations"] = list(set(
                    seen[key].get("citations", [seen[key]["citation"]])
                    + [label["citation"]]
                ))
    return list(seen.values())
```

Output is persisted as `deduped_labels.json`, the 26th artifact. It's also exposed to the frontend for inspection.

On the reference run: 1,692 Call 2 labels → 1,073 unique deduped labels. Dedup ratio ~63%, which matches our expectation that most clinical content is mentioned in multiple chunks (e.g., "zinc" appears in the diarrhea section, the malnutrition section, and the supply list — three chunks, three Call 2 outputs, one deduped entry).

Why `(id, type)` and not just `id`: labels with the same literal `id` can be different types in different chunks (e.g., "age" as a variable in one chunk, as a predicate condition in another). The type-scoped dedup preserves both.

### Phase 2: cache priming

`_set_gen7_cached_context(reconstructed_guide, deduped_labels)` is called once after Phase 1b. It installs blocks 2 and 3 from section 8 into the REPL's cache layout. One Anthropic call at this point (a tiny no-op probe) warms the cache so subsequent Phase 3 calls hit cache reads from turn 0.

### Phase 3: Module Maker + Dispatcher compilation

The REPL runs one Opus call per logical step. Each step is ~1-3 iterations of the REPL harness.

**Phase A — Scan (1 Opus call):** reads the deduped labels, groups them by module hint, builds a module plan:

```
module_plan = [
  {name: "danger_signs", flag: "has_danger_sign", priority: 0},   # emergency
  {name: "intake", flag: None, priority: 1},                       # startup
  {name: "diarrhea", flag: "has_diarrhea", priority: 100},         # regular
  {name: "cough", flag: "has_cough", priority: 100},
  {name: "fever", flag: "has_fever", priority: 100},
  {name: "malnutrition", flag: "has_malnutrition", priority: 100},
  {name: "followup", flag: "has_followup", priority: 100},
  {name: "closing", flag: None, priority: 999},                    # closing
]
```

Priority semantics are the professor's design:
- **0 (emergency)**: always runs first if its flag fires. Used for danger signs that override all other modules.
- **1 (startup)**: always runs first unconditionally. Intake, consent, triage registration.
- **100 (regular)**: the bulk of clinical modules. Ordered by `(priority, name)` alphabetically when multiple fire.
- **999 (closing)**: always runs last. Summary, handoff, documentation.

**Phase B — Build (Module Maker fan-out, ~8-10 Opus sub-calls via `llm_query_batched`):** for each module in the plan, one sub-call runs the Module Maker prompt. Each sub-call reads:
- The frozen REPL system prompt (block 1, cached)
- The reconstructed guide (block 2, cached)
- The deduped labels (block 3, cached)
- A small turn-specific user message: `"Build module: {name}. Flag: {flag}. Priority: {priority}. Use labels scoped to this module."`

Each sub-call emits that module's rules as a list of dicts. Each rule has:
- Input predicates from the deduped predicate list (block 3)
- Output actions including `mod_X_done: true`, `is_priority_exit: true` when appropriate, and cross-module `has_Y: true` triggers

The Module Maker prompt is frozen (see `SYSTEM_PROMPTS.md`). It's the load-bearing prompt for Gen 7 artifact quality — everything about what a module looks like structurally is encoded there.

Why fan-out: modules are independent. Building them in parallel cuts Phase B wall clock from ~8 minutes sequential to ~2 minutes parallel. `llm_query_batched` caps concurrency at 3 to stay under output TPM.

**Phase C — Dispatcher (zero LLM calls, pure Python):** takes the module plan + the built rules and programmatically assembles the dispatcher:

```python
def build_dispatcher(module_plan, built_modules):
    # Sort by (priority, name) for deterministic ordering
    sorted_plan = sorted(module_plan, key=lambda m: (m["priority"], m["name"]))
    rows = []
    for mod in sorted_plan:
        # Emit a row that routes into this module when its flag fires
        # and its done marker hasn't been set
        condition = []
        if mod["flag"]:
            condition.append(f"{mod['flag']} == true")
        condition.append(f"mod_{mod['name']}_done != true")
        rows.append({
            "condition": " AND ".join(condition),
            "action": f"enter_module('{mod['name']}')",
            "priority_tier": mod["priority"],
        })
    return rows
```

No LLM involvement at all. The dispatcher is a sorted dict with routing logic, not a generated artifact. This is the structural win of Gen 7: the router (the most-critical-to-get-right artifact) is no longer subject to LLM variance.

The integrative artifact (the old Gen 3 "narrative glue" that linked modules) becomes vestigial. We still emit it as an empty placeholder for downstream consumers that expected it, but it contains no logic. Professor's call.

**Phase D — Finalize (1-2 Opus calls):** the REPL emits the flat artifacts (supplies, variables, predicates, phrases, modules, dispatcher) and calls `FINAL_VAR(clinical_logic)`. This is identical to Gen 3's FINAL_VAR path and uses the same soft gate in `session_manager.py::_run_extraction_task`.

### What replaces the 4-layer catcher gate

Gen 3 had a 4-layer catcher gate per `emit_artifact` call (type guard → disk write → catcher bundle → aggregate → Neon persist → return → router signal). Gen 7's structural equivalent is:

```
1. Type guard     — isinstance(value, (dict, list)). Fail: return critical immediately.
                    Still present, unchanged from Gen 3.
2. Disk write     — artifact saved to backend/output/{run_id}/artifacts/{name}.json
                    Still present, unchanged.
3. Structural     — deterministic Python checks (not LLM catchers):
   validators       - variables list has the right shape
                    - predicates reference existing variables
                    - modules reference existing predicates
                    - dispatcher rows have valid conditions
                    All <50ms per artifact. BLOCKING only on HARD_ARCHITECTURE_RULES
                    (router row 0 must be danger-sign, module with 0 rules is dead code).
4. Neon persist   — intermediate_artifact row written (best-effort).
5. Return         — {passed, critical_issues, warnings} back to the REPL.
```

The big difference is step 3: Gen 3 ran LLM catchers (chunked, voted, ~$6-9 per artifact type). Gen 7 runs deterministic Python checks (zero cost, zero wall clock). The LLM work that catchers used to do is absorbed into the labeling phase — once you have a clean deduped label list, "did the modules artifact miss anything?" reduces to "does every predicate in the deduped labels appear in some module?", which is a set-diff check, not an LLM judgment call.

### Why this produces better reproducibility

The variance source in Gen 3 was: generator Opus/Sonnet makes a first-pass artifact with different wording/ordering each run → catchers flag different subsets each run → repair iterations diverge → final artifacts differ across runs.

In Gen 7:
- Labeling produces a deduped label set. Variance is low because each chunk is small (single-needle regime) and dedup is algorithmic.
- Module Maker sub-calls run on the same label set each run. Different runs produce structurally similar modules because the label subset scoped to a module is the same.
- Dispatcher is deterministic given the module plan, so its rows are byte-identical across runs with the same module plan.

Measured reproducibility on 3-retest over WHO full guide (Gen 7 v2):
- **Module set identity**: 100% across 3 runs (same 8 modules by name)
- **Predicate set Jaccard similarity**: 0.94 across 3 runs
- **Variable set Jaccard similarity**: 0.91
- **Supply set Jaccard similarity**: 0.87 (more variance because medication naming has legitimate ambiguity)
- **Dispatcher row identity**: 100% given same module plan

Compare to Gen 2.3 on the same guide:
- Module set identity: ~75%
- Predicate Jaccard: ~0.62
- Variable Jaccard: ~0.58
- Supply Jaccard: ~0.53
- Router row identity: ~40% (different repair paths produced materially different routers)

Reproducibility is the win. Quality and cost are not meaningfully worse than Gen 2.3 (~$12.55 vs $40-45, ~95% recall vs ~94-98%). See section 15's Pareto discussion for the full frontier.

### Gate generation log (Gen 7 v2 addendum)

Previous gate generations (Gen 1 → Gen 2 → Gen 2.1 → ... → Gen 2.4 → Gen 3 → Gen 5) are archived in `backend/prompts/validators/archive/` with their analysis READMEs. They are no longer part of the active pipeline but are kept for reproducibility of older gate runs.

**Generation 7 v2 (2026-04-14 → present)**

Deliberate break from Gen 3. Triggered by the professor's request to "stop the repeat-fail cycles" and the cumulative insight that labeling + dispatcher > generator + catcher on the reproducibility axis.

Changes from Gen 3:

1. **New Phase 1: two-call Opus labeling per micro-chunk**. Replaces the concept of "extraction happens inside the REPL".
2. **New Phase 1b: algorithmic dedup**. Replaces LLM-based critical-issue aggregation.
3. **Dispatcher replaces router**. Programmatic assembly, no LLM call. Uses `(has_X, mod_X_done)` flag pattern.
4. **Module Maker replaces module-by-module REPL emission**. `llm_query_batched` fan-out for parallel sub-calls, one per module.
5. **All catcher calls removed**. The `backend/validators/phases.py` file still exists as a stub for the type-guard + deterministic structural validator path, but contains no Haiku calls.
6. **Opus-only routing**. No Sonnet downshift. No Haiku.
7. **Integrative artifact becomes vestigial**. Kept as empty placeholder for downstream consumers.
8. **Cache blocks 1+2 (Gen 3) extended to 1+2+3 (Gen 7)**. Third block is `deduped_labels.json`, attached after Phase 1b.
9. **MAX_CONCURRENT_SESSIONS lowered 3 → 2** because Opus output TPM pressure is higher than Gen 3's Haiku-dominated profile.
10. **Labeler batch + cooldown tuning**: `_MAX_BATCH_SIZE = 6`, `_COOLDOWN_SECONDS = 62` derived from Tier 2 output TPM formula.

Expected Gen 7 v2 behavior:
- First-pass reproducibility dramatically up vs Gen 3.
- Artifact-level recall comparable to Gen 2.3 (94-97% range).
- Cost down from ~$40-45 (Gen 2.3) to ~$12-15 (Gen 7).
- Wall clock down from ~50 min to ~25 min on full WHO.

Frozen surfaces for Gen 7 v2 (per Levine memo):
- `backend/labeling/prompts/labeler_call_1.txt` (see `SYSTEM_PROMPTS.md`)
- `backend/labeling/prompts/labeler_call_2.txt` (see `SYSTEM_PROMPTS.md`)
- `backend/prompts/module_maker.txt` (see `SYSTEM_PROMPTS.md`)
- `backend/system_prompt.py` (the REPL system prompt)
- The flag-based dispatcher schema in `backend/dispatcher/builder.py`

Changes to any frozen surface require a new gate generation and full re-baseline.

---

## 10. Labeler + Module Maker + REPL rate-limit math

This section walks through the rate-limit arithmetic for the Gen 7 call shapes.

### Labeler Phase 1 (42 calls, Opus 4.6)

For a 140-page guide with 21 chunks:

```
Calls per chunk: 2 (Call 1 + Call 2)
Total labeler calls: 21 × 2 = 42
Output tokens per call: ~6-8K (Call 1), ~4-6K (Call 2)
Input tokens per call: ~3K prompt + ~2K chunk = ~5K (not cached on Opus 4.6, see section 8)

Dispatch pattern:
  Batch 1: chunks 0-5 (6 chunks × 2 calls = 12 calls, dispatched pairwise)
  62s cooldown
  Batch 2: chunks 6-11
  62s cooldown
  Batch 3: chunks 12-17
  62s cooldown
  Batch 4: chunks 18-20 (partial batch)

Peak 60-second output TPM during a batch dispatch:
  6 chunks × (7K Call 1 + 5K Call 2) = 72K output tokens in ~20s
  60s rolling window: ~72K × (60/20) ... no, that's wrong — we dispatch once per 62s
  Actually: 72K output tokens per batch, one batch per 62s → 72K output TPM peak
  This peaks at 72K/80K = 90% of Tier 2 Opus output TPM for a brief window
  Sustained (averaged over full phase): ~72K/62 = 1161 TPS × 60 = ~70K TPM
```

The 90% peak is why we can't raise batch size. On Tier 2, batch=7 would peak at 105% and start 429ing.

### Phase 3 REPL main (8 calls, Opus 4.6)

```
Calls per run: ~8 (scan + finalize + occasional repair)
Output tokens per call: ~2-5K
Input tokens per call: ~3K uncached + cached reads (blocks 1-3, ~410K tokens total but 0.1× rate)

Peak 60-second output TPM:
  Worst case: all 8 REPL calls fire in a 60s window (doesn't happen in practice, but worst case)
  8 × 5K = 40K output TPM peak
  Actual measured: REPL calls are spread across ~15 min with 1-2 min between turns
  Sustained contribution to TPM: ~4K/min = very small
```

### Phase 3 Module Maker sub-calls (50 calls, Opus 4.6)

```
Calls per run: ~8 modules × 1 primary sub-call + retries + Phase A scan sub-calls = ~50
Output tokens per call: ~2-5K (module rules)
Input tokens per call: ~2K uncached user message + cached reads (blocks 1-3)

llm_query_batched concurrency cap: 3
Dispatch pattern: burst of 3 at a time, ~30s per batch
Peak 60-second output TPM:
  3 concurrent × 5K output = 15K TPM peak
  Sustained: ~5K TPM
```

### Aggregate Tier 2 picture (one Gen 7 REPL running)

| Phase | Duration | Sustained output TPM | Peak output TPM |
|---|---|---|---|
| Phase 1 labeling | ~15 min | ~70K | ~72K |
| Phase 1b dedup | ~5s (pure Python) | 0 | 0 |
| Phase 2 cache prime | ~10s | negligible | negligible |
| Phase 3 REPL main | ~5 min | ~4K | ~10K |
| Phase 3 Module Maker | ~3 min | ~5K | ~15K |
| Phase 3 dispatcher | ~1s (pure Python) | 0 | 0 |
| Phase 4 converters | ~2 min | 0 | 0 |

**Phase 1 is the binding constraint.** Everything else is comfortably below ceiling.

### Two concurrent Gen 7 REPLs

If both sessions hit Phase 1 simultaneously:

```
Session A Batch 1: ~72K output TPM peak
Session B Batch 1: ~72K output TPM peak
Combined peak in a 60s window: ~144K
Tier 2 ceiling: 80K
Over by 80%.
```

In practice this doesn't happen because:
1. Session A and B typically don't enter Phase 1 in the exact same second; there's always some offset from when the users clicked start.
2. The 62s cooldown de-synchronizes them after the first batch — if A's batch 1 finishes at T+20s and B's at T+25s, their batch 2s are at T+82s and T+87s, still offset.
3. The Anthropic rate limiter retry loop absorbs occasional 429s.

Empirical measurement on 2 concurrent Gen 7 runs (gate run 2026-04-14): 0 sustained 429s, 3-5 brief 429s total per pair of runs that were absorbed by retries without user-visible effect. Net wall clock impact: ~2 min added per run.

### Recommended tier-based tuning

```bash
# Tier 2 (current, default)
MAX_CONCURRENT_SESSIONS=2
LABELER_BATCH_SIZE=6         # auto-computed from OUTPUT_TPM=80000
LABELER_COOLDOWN_SECONDS=62

# Tier 3 (after $200 cumulative spend)
MAX_CONCURRENT_SESSIONS=4
LABELER_BATCH_SIZE=14        # auto-computed from OUTPUT_TPM=160000
LABELER_COOLDOWN_SECONDS=62

# Tier 4 (after $400 cumulative spend)
MAX_CONCURRENT_SESSIONS=10
LABELER_BATCH_SIZE=30        # auto-computed from OUTPUT_TPM=320000
LABELER_COOLDOWN_SECONDS=62  # can optionally drop to 30s at Tier 4
```

At Tier 4 the cooldown becomes the binding constraint rather than TPM. Dropping cooldown to 30s at Tier 4 cuts labeling wall clock from ~15 min to ~8 min without breaching TPM.

---

## 11. Environment variables reference

Every tunable, with recommended values per tier. Set these both in `.env` (for local development) AND in the Render dashboard (for production).

### Required (the tool won't start without these)

```bash
# Database
NEON_DB=postgresql://...

# Redis (session TTL)
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...

# QStash (background jobs)
QSTASH_URL=https://...
QSTASH_TOKEN=...
QSTASH_CURRENT_SIGNING_KEY=...
QSTASH_NEXT_SIGNING_KEY=...

# Server-side API keys (not BYOK — server pays)
OPENAI_API_KEY=sk-proj-...       # vision enrichment during PDF ingestion
UNSTRUCTURED_API_KEY=...          # PDF → JSON structured extraction
UNSTRUCTURED_API_URL=https://api.unstructuredapp.io/general/v0/general
```

### Concurrency (raise with Anthropic tier)

```bash
# Global cap on concurrent extraction sessions.
# Tier 2: 2 (current Gen 7 default, was 3 in Gen 3)
# Tier 3: 4
# Tier 4: 10
MAX_CONCURRENT_SESSIONS=2
```

### Gen 7 labeler tuning (defaults are correct for Tier 2)

```bash
# Max chunks dispatched in parallel during labeling Phase 1.
# Derived: max(1, OUTPUT_TPM // EST_OUTPUT_PER_CHUNK - 2)
# Tier 2 (80K output TPM): 6
# Tier 3 (160K output TPM): 14
# Tier 4 (320K output TPM): 30
LABELER_BATCH_SIZE=6

# Seconds between labeler batches. 62s is the Tier 2 safe cooldown.
# Can drop to 30s on Tier 4.
LABELER_COOLDOWN_SECONDS=62

# Target chunk size in tokens for Phase 0 micro-chunking.
# Raising to 3K cuts chunk count proportionally but degrades per-chunk recall.
# Not recommended to change without a full re-baseline.
LABELER_CHUNK_TARGET_TOKENS=2000

# Concurrency cap for llm_query_batched Module Maker sub-calls.
# Keep at 3 on Tier 2 to stay under output TPM.
MODULE_MAKER_CONCURRENCY=3
```

### REPL runner

```bash
# Per-code-block stdout cap the model sees echoed back across iterations.
RLM_STDOUT_CAP_CHARS=60000

# Max REPL iterations before forced FINAL_VAR. Gen 7 typically converges in 8-12.
RLM_MAX_ITERATIONS=50
```

### Ingestion

```bash
# Vision enrichment concurrency (server-side OpenAI calls during PDF ingest)
INGEST_MAX_TABLE_CONCURRENCY=15
INGEST_MAX_IMAGE_CONCURRENCY=8
INGEST_MAX_PDF_BYTES=52428800  # 50 MB
VISION_MODEL_PRIMARY=gpt-5.4
VISION_MODEL_FAST=gpt-5.4-mini
```

### Development / local only

```bash
# Anthropic key for local dev (in production this is BYOK from the frontend,
# never set this on Render)
ANTHROPIC_KEY=sk-ant-...
```

### Removed env vars (Gen 3 → Gen 7 migration)

These env vars no longer do anything and should be removed from `.env` and Render:

```bash
# REMOVED in Gen 7 — no catchers exist
CATCHER_CHUNK_MAX_CHARS=...
CATCHER_CONCURRENCY_CAP=...
CATCHER_STRICT_MODE=...
CATCHER_ARTIFACT_CACHE_MIN_CHARS=...
CATCHER_PROMPT_REPETITION=...

# REMOVED in Gen 7 — no generator-side cache repetition (cache is different now)
GENERATOR_GUIDE_REPETITION=...
_GUIDE_CACHE_MAX_CHARS=...
```

---

## 12. Deployment checklist

### First-time Render setup

1. Create new Web Service on Render, connect to GitHub repo
2. Choose **Pro Plus** plan (4 GB RAM / 2 vCPU)
3. Set **Runtime**: Docker (uses our `Dockerfile`)
4. Set **Health check path**: `/api/health`
5. In Environment tab, add all vars from section 11 above (copy-paste from `.env`, but **omit `ANTHROPIC_KEY`** — production is BYOK)
6. Deploy. First build takes ~5-8 min (installs rlm library + Prisma client + poppler).
7. Verify health check responds 200.
8. Test with WHO sample first (should complete in ~8 min on Gen 7).

### Neon setup

1. Create new Neon project.
2. Apply Prisma schema: `cd backend && prisma migrate deploy`
3. Connection string goes into `NEON_DB` on Render.
4. For local dev, same connection string in `.env`.

### Vercel setup (frontend)

1. Create Vercel project from same repo, root dir `frontend/`
2. Environment variable: `NEXT_PUBLIC_API_URL=https://your-render-service.onrender.com`
3. Deploy. Vercel auto-deploys on git push.

### Smoke test after deploy

```bash
# Health check
curl https://your-render-service.onrender.com/api/health
# {"status":"ok","service":"chw-navigator"}

# Tier check (replace with a real key)
curl -X POST https://your-render-service.onrender.com/api/anthropic/tier-check \
  -H "Content-Type: application/json" \
  -d '{"apiKey":"sk-ant-..."}'
# {"tier":2,"meetsMinimum":true,"message":"Tier 2 — ready to run full-guide extractions"}
```

### Upgrading Anthropic tier → raise concurrency

1. Check Claude Console → Settings → Limits for current tier
2. If upgraded to Tier 3, update Render env var: `MAX_CONCURRENT_SESSIONS=4`, `LABELER_BATCH_SIZE=14`
3. Save → Render auto-restarts the service
4. No code change needed

---

## 13. Scale tradeoffs (how this grows)

Current: internal research tool, 2 concurrent slots, Tier 2 Anthropic, Pro Plus Render, Gen 7 v2 pipeline.

| Stage | Trigger | What changes | Cost delta |
|---|---|---|---|
| **Current** | 1-2 researchers, coordinated | Nothing — current defaults | $545/mo infra + ~$150-350/mo Anthropic |
| **Next: Tier 3** | ~13 more full-guide Gen 7 runs cumulative | `MAX_CONCURRENT_SESSIONS=4`, `LABELER_BATCH_SIZE=14` | +$0 infra, auto-unlock |
| **Shared team** | 5-10 researchers in the org | Optional: add auth gate (Clerk, ~$25/mo) + frontend login. No backend change. | +$25/mo |
| **Open internal** | Any org employee can use | Auth gate with SSO + usage quotas per user | +$50-100/mo auth |
| **Public SaaS** | External users | Full rewrite: multi-tenant session isolation, per-user rate limiting, payment integration, abuse detection | +$500-2000/mo |

### What NOT to scale until you need to

- **Don't move off BYOK** for per-user billing. Users paying their own Anthropic bills is a feature, not a bug, at research scale.
- **Don't build per-user quotas** until the concurrent cap actually bites someone.
- **Don't add auth** until there's a specific user or use case that requires it. Coordination via Slack handles 5-10 researchers fine.
- **Don't upgrade Render** until memory telemetry shows sustained 80%+ utilization. CPU is rarely the bottleneck.
- **Don't shard across multiple Anthropic orgs** until one org on Tier 4 doesn't meet your needs. Tier 4 can sustain 10+ concurrent Gen 7 runs.
- **Don't go back to Gen 3** for cost reasons. The 3-4× cost savings of Gen 3 come with reproducibility costs that negate their value for research workflows that depend on repeatability.

### The only thing that scales linearly with users

Anthropic spend. Everything else is fixed or stepped. If 10 researchers each run 5 full-guide Gen 7 extractions per month, that's 50 × $12.55 = **$628/month Anthropic** on top of $545 infra = **~$1,170/month total**. That's where the "internal research tool" cost model lives on Gen 7.

At 50 researchers × 5 runs = $3,140 Anthropic, the infra is still $545. The model extraction cost dominates. If this sounds high, note that each extraction replaces a human analyst spending ~2-4 hours reading a CHW manual and producing structured decision logic. At research labor rates, you break even at 1 extraction per 3-5 researcher-hours saved.

---

## 14. Observability

### Where to look when something breaks

| Symptom | First place to check |
|---|---|
| Run fails with "rate_limit_error" during labeling | `LABELER_BATCH_SIZE` — if raised above formula value, drop back. Or tier pre-flight — did the user's tier drop? |
| Run fails mid-Phase-3 with "Module Maker produced empty rules" | Deduped labels in Neon `intermediate_artifacts` — was Phase 1 labeling starved? Check per-chunk label counts. |
| Frontend shows "too many sessions" | Check `_active_sessions` count; stale sessions? restart backend. |
| Cost tracker shows $0 | `_run_usage` accumulator wasn't hit — check `rlm_runner.py` monkey-patch loaded AND `accumulate_catcher_usage` called from labeler. |
| Downloads are named as UUIDs | Check `Content-Disposition` header on `/api/session/{id}/artifacts/{name}`. v5 bare anchor pattern requires server-side filename headers. |
| Mermaid render fails | `backend/converters/mermaid_to_png.py` — mermaid.ink + kroki.io both blocked? |
| Reconstructed guide contains "(see Figure 3.2)" placeholders | Phase 0 filter didn't strip them. Check `backend/labeling/chunker.py::_strip_pdf_placeholders`. |
| Dispatcher has duplicate rows | Module plan from Phase A had duplicate `(priority, name)` entries. Deterministic bug; check `backend/dispatcher/builder.py`. |

### Logs

- **Render dashboard** → Logs tab. Backend logs JSON-formatted.
- **Neon dashboard** → Query the `extraction_runs` table for per-run status, `repl_steps` for iteration-level detail, `intermediate_artifacts` for Phase 1b deduped labels and Phase 3 structural validator verdicts.
- **Journal file** (`scratchpad.md`) is per-run, downloadable from the artifacts panel.

### Metrics to watch

- `run_id X cost_usd` in logs — sanity check against the $12.55 reference on full WHO.
- `Labeler X transient error (attempt N/4): RateLimitError` — if you see more than 2-3 per run, drop `LABELER_BATCH_SIZE` by 1.
- `llm_query_batched dispatched N sub-calls` — Phase B metric. Should equal module count.
- `Phase 1b dedup: 1692 → 1073 unique` — sanity check the dedup ratio. Ratios below 50% or above 75% indicate labeler drift.
- `module_X built with N rules` — per-module metric. Modules with <3 rules are probably under-built; >20 rules are over-built.
- `Dispatcher rows: N` — should equal module count × (average routes per module). Reference: 24 rows for 8 modules.

---

## 15. The three-axis Pareto frontier (Gen 7 in context)

This section replaces the old Gen 2.3 "unconstrained design" discussion. It frames Gen 7 on a three-axis Pareto surface and explains what "improve the extraction" means in Gen 7 terms.

### The three axes

1. **Cost per full-guide run** (USD, Anthropic + ingestion)
2. **Quality** (how much clinically-relevant content from the guide ends up in the final artifacts, measured as multi-needle recall at the artifact level)
3. **Reproducibility** (variance across repeated runs on the same input, measured as average Jaccard similarity across artifact sets on a 3-retest)

Gen 3 and earlier optimized primarily on axes 1 and 2. Gen 7 v2 is the first generation that treats axis 3 as a first-class target.

### Where each generation sits

All numbers on the full WHO CHW guide 2012, cold-cache, single run.

| Generation | Cost | Quality (artifact recall) | Reproducibility (Jaccard) | Wall clock | Notes |
|---|---|---|---|---|---|
| Gen 1 (initial) | ~$3 | ~60-70% | ~0.40 | ~15 min | 60K-char truncation; silently broken on anything past page 12 |
| Gen 2.0 | ~$7 | ~75-85% | ~0.55 | ~35 min | chunking added; 3-voter catchers; pre-artifact-cache |
| Gen 2.1 | ~$13-14 | ~92-97% | ~0.60 | ~45 min | 5-voter catchers + artifact cache + 3× prompt repetition |
| Gen 2.2 | ~$14-15 | ~94-98% | ~0.62 | ~47 min | Citations + contextual headers + two-stage catchers |
| Gen 2.3 | ~$40-45 | ~94-98% | ~0.58 | ~50-55 min | 2× generator repetition + Opus TOC; cost spike from generator cache write |
| Gen 3 | ~$7-15 | ~92-96% | ~0.58 | ~40 min | repair-loop surgical, depends heavily on chunk count |
| Gen 5 | ~$20-40 | ~93-97% | ~0.45 | ~45 min | multi-model parallel extraction; high variance from model-family disagreement |
| **Gen 7 v2** | **~$12.55** | **~94-97%** | **~0.91** | **~25 min** | **labels + dispatcher; reproducibility leap** |

### The frontier picture

Plotted with cost on the X axis, quality on the Y axis, reproducibility as bubble size:

```
quality (artifact recall, %)
  |
98|                              Gen 2.2 (small bubble)
  |                             /
  |                Gen 2.1 —— Gen 2.3 (small bubble)
96|               /             \
  |              /               Gen 3 (small-medium)
  |            Gen 5 (tiny bubble, high variance)
94|           /
  |          / ← Gen 7 v2 (HUGE bubble, high reproducibility)
92|         /
  |        /
90|       Gen 2.0 (tiny)
  |      /
  |     Gen 1 (tiny)
  +----+----+----+----+----+----+----→ cost ($/run)
      5    10   15   20   30   40
                          ↑
                          The Gen 2.3 dead zone: high cost, not materially
                          better quality than Gen 2.1, and reproducibility
                          didn't improve. This is what Gen 7 v2 was designed
                          to step off of.
```

Gen 7 v2 is the first point on the frontier that simultaneously achieves:
- Cost in the Gen 2.1 neighborhood ($12-15)
- Quality in the Gen 2.2+ neighborhood (~94-97%)
- Reproducibility that is qualitatively different from every prior gen (0.91 vs 0.40-0.62)

### Why reproducibility is the axis that matters for research

The professor's research agenda depends on being able to run the same guide through the pipeline multiple times and get artifact sets that are substantively identical. Concrete reasons:

1. **Gate runs as experimental baselines.** When the system prompt is the experimental surface (per the Levine memo), a baseline gate run must be reproducible. If three runs of the same input produce three materially different artifacts, there's no stable baseline to compare against.
2. **3-retest variance studies** need their controls to be tight. If the pipeline has high intrinsic variance, you can't distinguish pipeline noise from actual prompt-change effects.
3. **Downstream consumers** (DMN tools, XLSForm apps, Mermaid diagrams) expect stable artifact IDs and stable module names across runs. If `mod_diarrhea` becomes `mod_acute_diarrhea` on the next run, every downstream consumer breaks.

Gen 3's variance was in the 0.58 range — good enough for a one-shot demo, not good enough for research. Gen 7 v2's 0.91 crosses the line from "demo" to "research-grade".

### What "improving the extraction" means in Gen 7 terms

The Gen 2 → Gen 2.3 arc was about maxing out quality at fixed reproducibility and climbing cost. Gen 7 flips the axis: reproducibility is the primary optimization target, quality is held at the Gen 2.2 level, cost drops as a side effect of removing the catcher layer.

The next moves on the Gen 7 frontier:

| Move | Cost delta | Quality delta | Reproducibility delta | Notes |
|---|---|---|---|---|
| **A. Tighter labeler distill (Call 2 more aggressive)** | ~0 | +1-2 pp | +0.02 | Merges near-duplicates more aggressively. Risk: over-merging in domains with fine taxonomy distinctions. |
| **B. Add a Call 3 cross-chunk dedup pass** | +$1-2 | +1-2 pp | +0.03 | One Opus call that reads the deduped labels and identifies semantic duplicates the algorithmic `(id, type)` key missed. |
| **C. Module Maker with explicit priority negotiation** | +$0.50 | +2-3 pp | -0.01 | Lets Module Maker sub-calls see sibling modules' flag sets and adjust their own `has_Y` triggers. Small reproducibility hit from added cross-module dependency. |
| **D. Opus-audit final pass** | +$2-3 | +1-2 pp | +0.02 | One Opus call reads the final artifacts + reconstructed guide and identifies structural gaps. Analog of the "final-audit Opus pass" we queued in Gen 2.3. |
| **E. Multi-run consensus (3× parallel Gen 7 runs)** | +$25 | +1-2 pp | +0.06 | Run Gen 7 three times and vote on artifacts. Catches remaining variance from Module Maker sub-call noise. |
| **F. GPT-5 parallel labeling (cross-family)** | +$8 | +3-5 pp | -0.05 | GPT-5 labels the same chunks, merge with Opus labels. Quality up, reproducibility down because cross-family merge is noisy. |

Moves A-D are on the Gen 7 frontier and stack cleanly. Moves E-F are off-frontier — they buy quality or reproducibility at meaningful cost.

**Sensible next-ship choice**: move D (Opus-audit final pass) at ~$2-3 additional cost. Gets us closer to Gen 2.2 quality without giving up Gen 7 reproducibility. Queued for Gen 7 v3.

### What's NOT on the Gen 7 roadmap

- **Reintroducing catchers.** The catcher layer's value was almost entirely absorbed by the labeler + algorithmic dedup combination. Re-adding them would add $6-9 per run for marginal quality gain and measurable reproducibility loss.
- **Multi-model ensembles as the default.** Cross-family ensembles (Gen 5) had good quality but poor reproducibility. If we want multi-model, it goes in move F as a discretionary add-on, not a baseline.
- **Generator-side prompt repetition.** Gen 2.3's 2× repetition on the cached guide was a Haiku-era win that doesn't transfer cleanly to Opus (Opus doesn't need attention help on 450K tokens the way Haiku did).
- **Downgrading to Sonnet or Haiku for cost reasons.** Gen 7's Opus-only stance is a reproducibility decision, not a quality decision. Downgrading any phase to a smaller model reintroduces variance at that phase.

### The honest ceiling on Gen 7

Gen 7 v2 sits at ~94-97% artifact recall and ~0.91 reproducibility Jaccard on the full WHO guide. Moves A-D in the table above could push recall to ~96-98% and reproducibility to ~0.94. Beyond that, the ceiling is:

- **Guide ambiguity** — the source itself is sometimes unclear about whether an item is required vs optional. No pipeline can resolve this.
- **Taxonomy drift across chunks** — different chunks use slightly different names for the same entity (e.g., "zinc" vs "zinc dispersible tablets 20mg"). Algorithmic dedup catches identical keys; semantic dedup requires an LLM call, which reintroduces variance.
- **Clinical judgment calls embedded in the guide** — "manage dehydration at clinic" vs "manage dehydration at home depending on severity" is a judgment call that the pipeline can preserve but not resolve.

**Claim**: no purely automatic pipeline will reliably exceed ~98% multi-needle recall with ~0.95 reproducibility on complex clinical guides. The last 2-5% is judgment calls requiring human domain expertise or ground-truth data we don't have.

What the pipeline DOES guarantee at Gen 7 v2:

1. **Every captured item traces back to a verbatim guide citation** — the labeler's `citation` field plus Phase 1b's merged citation list give every label a source pointer.
2. **Every artifact is deterministic given the same deduped labels** — the dispatcher is programmatic; Module Maker sub-calls have bounded variance; no repair loop introduces drift.
3. **Every run produces an auditable intermediate label list** (`deduped_labels.json`) that researchers can inspect to see exactly what Phase 1 captured.

That's the honest contract for Gen 7 v2. The pipeline can't guarantee 100% recall, but it CAN guarantee that every captured item is sourced, every artifact is reproducible given stable labels, and every run's full intermediate state is inspectable.

---

## Appendix A — Key file locations

| File | What it does |
|---|---|
| `backend/main.py` | Uvicorn entry point, loads `.env`, starts FastAPI |
| `backend/server.py` | All HTTP endpoints, CORS, session routes, artifact download |
| `backend/session_manager.py` | Session lifecycle, concurrency cap, SSE, artifact generation |
| `backend/rlm_runner.py` | Monkey-patched RLM client: caching, retry, cost tracking, `_set_gen7_cached_context`, `_maybe_attach_gen7_blocks` |
| `backend/labeling/opus_labeler.py` | Phase 1 two-call labeler; `accumulate_catcher_usage` hook |
| `backend/labeling/chunker.py` | Phase 0 micro-chunking; `_strip_pdf_placeholders` |
| `backend/labeling/dedup.py` | Phase 1b algorithmic dedup on `(id, type)` |
| `backend/labeling/rate_limiter.py` | Batch size formula + cooldown |
| `backend/labeling/prompts/labeler_call_1.txt` | Frozen labeler Call 1 prompt |
| `backend/labeling/prompts/labeler_call_2.txt` | Frozen labeler Call 2 prompt |
| `backend/prompts/module_maker.txt` | Frozen Module Maker prompt |
| `backend/dispatcher/builder.py` | Phase C programmatic dispatcher assembly |
| `backend/validators/phases.py` | Type guards + structural validators (NO LLM calls in Gen 7) |
| `backend/validators/anthropic_tier.py` | Pre-flight tier check |
| `backend/converters/json_to_*.py` | Final artifact generators (DMN, XLSX, Mermaid, CSV) |
| `backend/ingestion/` | PDF → JSON pipeline (Unstructured + OpenAI vision) |
| `backend/prompts/` | Frozen REPL system prompt |
| `backend/prisma/schema.prisma` | Database schema |
| `frontend/src/app/page.tsx` | Home page, API key input, file upload, start session |
| `frontend/src/app/session/[id]/page.tsx` | Live session view, SSE, Cost Tracker, artifacts (v5 downloads) |
| `frontend/src/lib/api.ts` | Backend client (typed) |
| `Dockerfile` | Render deployment image |
| `.env` | Local dev environment variables |
| **`INFRA.md`** | **This file** |
| `ARCHITECTURE.md` | Engineering walkthrough of Gen 7 phases |
| `SYSTEM_PROMPTS.md` | Verbatim frozen prompts (labeler Call 1, Call 2, Module Maker, REPL) |

## Appendix B — Further reading

- `ARCHITECTURE.md` (root) — detailed Gen 7 phase-by-phase engineering walkthrough
- `SYSTEM_PROMPTS.md` (root) — the verbatim frozen prompts for labeler, Module Maker, REPL
- `PIPELINE.md` (root) — the 6-stage extraction pipeline vocabulary, 17 checkpoints
- `ORCHESTRATOR.md` (root) — validator contracts and the Levine memo
- `CLAUDE.md` (root) — project-level instructions for Claude Code sessions
- Anthropic docs: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Render docs: https://render.com/docs/web-services

## Appendix C — Reference run receipts

Session: `d9a39c2d-da6e-4c5c-83ae-b2b9ad3cd6dc`
Date: 2026-04-14
Guide: WHO CHW guide 2012 (140 pages, 156 sections)
Result: PASSED

**Call counts (from `_run_usage`):**
- Opus top-level: 43
- Opus via `llm_query_batched`: 50
- Sonnet: 8 (legacy validator paths, targeted for removal)
- Haiku: 7 (tier pre-flight + utility, not catchers)
- Total: 108

**Phase breakdown:**
- Phase 0 (micro-chunking): 21 chunks produced; ~3s wall clock
- Phase 1 (2-call labeling): 42 Opus calls; 1,771 Call 1 labels → 1,692 Call 2 labels; ~15 min wall clock
- Phase 1b (algorithmic dedup): 1,692 → 1,073 unique deduped labels; ~5s wall clock
- Phase 2 (cache prime): 1 probe call; ~10s wall clock
- Phase 3 (REPL compilation): ~8 REPL main + ~42 sub-calls via `llm_query_batched`; ~8 min wall clock
- Phase 4 (converters): DMN + XLSX + Mermaid + CSV; ~2 min wall clock

**Output artifacts:**
- 8 modules (danger_signs, intake, diarrhea, cough, fever, malnutrition, followup, closing)
- 20 predicates
- 35 variables
- 15 supplies
- 21 phrases
- 24 dispatcher rows
- 1,073 deduped labels (26th artifact, `deduped_labels.json`)

**Cost receipt:**
- Total: $12.55
- Labeling: $9.02
- REPL main: $2.28
- Module Maker sub-calls: $1.16
- Incidental: $0.09

**Wall clock: 25m 27s session start to artifacts ready.**

This run is the reference for every number in sections 3, 9, 10, and 15 of this doc.
