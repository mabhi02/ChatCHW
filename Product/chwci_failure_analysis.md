# CHW-Chat-Interface — Failure Analysis and Postmortem

> **Author:** Atharva Patel
> **Date:** April 8, 2026
> **Audience:** Prof. David Levine (Haas); Emett Mendel; CHW Navigator team
> **Purpose:** Document the three identification failures observed running CHW-Chat-Interface in March 2026, with quantitative evidence from the production Neon database, the contemporaneous March 18 audit report, and a clear account of the architectural decisions that led there. This is the postmortem that should have accompanied the March 31 pivot proposal; it did not, and that process gap is the subject of a separate apology in the April 8 memo.

---

## 0. Project definition and the identification constraint

The system we are building takes a Community Health Worker clinical manual (PDF) as input and produces a validated DMN decision model plus a CHT-deployable XLSForm (XLSX) as output, with **full auditability of every intermediate artifact**: supply list, variable definitions, predicates, decision tables, phrase bank. Every artifact must be traceable to a page and quote in the source manual. The system must be **reproducible: same manual, same output, regardless of who runs it.**

The core engineering challenge is that there is a stochastic function (an LLM) in the middle of this pipeline. **Perfect determinism is not achievable.** What is achievable is an architecture where any remaining variance is attributable *solely* to model stochasticity, not to confounders in the orchestration, the retrieval, the prompting strategy, or the human operator. **That is the identification problem this memo is about.**

This constraint is load-bearing for everything that follows. It is not "the system should be reliable" as a soft goal. It is "the system must be architected so that variance has only one possible source." Any architecture that introduces additional variance sources (multiple prompts, human-in-the-loop decisions, environment drift) makes the research program — measuring how prompt edits affect DMN quality — impossible, because we cannot tell what produced what variance. The rest of this document evaluates CHW-Chat-Interface against this constraint and explains why we concluded on March 18-30 that the architecture was structurally incapable of satisfying it.

---

## 1. Executive summary

Between January and March 2026 the team built CHW-Chat-Interface: a Claude-plus-Gemini sequential pipeline that converts clinical Community Health Worker (CHW) manuals into Decision Model and Notation (DMN) artifacts. The system went live on March 12 and was paused on March 18 after a 15-agent audit loop surfaced ten BLOCKING findings (eight fixed, two deferred) and several additional medium-priority deferred items. Between March 19 and March 30 I reviewed the audit outputs and concluded that three of those deferred items were not individual bugs but symptoms of three distinct *identification* failures that could not be repaired in place, given the project's identification constraint (Section 0). On March 31 I sent the ORCHESTRATOR.md spec proposing the REPL pivot. This document is the evidence chain behind that conclusion.

The three failures, each framed in your own reliability/validity vocabulary from your March 3 email:

1. **High-dimensional treatment vector** — 10 independently tunable prompts create a treatment space we cannot sample at the rate needed to identify the marginal effect of any single prompt edit.
2. **Unobservable mediator** — the user's [Auto-fix All] / [Fix with Instructions] / [Ignore] decisions on catcher reports re-enter the pipeline but are not recorded in the schema. In the entire production history, zero out of 129 messages captured a mediator decision.
3. **Non-stationary baseline** — the cache is keyed by chat session rather than by PDF content hash, and in practice was never populated at all: zero of 45 LLM calls had a cache hit, and all 21 cache state rows have NULL observability fields. The same PDF uploaded into 55 different chats produced 21 distinct DMN outputs with 21 distinct content hashes.

Each failure corresponds to a named econometric identification problem. Together they explain why I paused the system: the architecture could not, in principle, produce the reliability measurements our research program requires.

---

## 2. How CHW-Chat-Interface worked

CHW-Chat-Interface was designed as a chat-authoring environment for DMN generation. A medical reviewer uploaded a CHW clinical manual (usually a PDF), started a chat session, and the system produced a DMN in iterative turns: the generator proposed rules, a panel of validator agents (catchers) flagged issues, and the reviewer accepted or ignored each flag until the DMN was finalized. Three deterministic transpilers then converted the validated DMN to Mermaid flowcharts, XLSForm files for the CHT Community Health Toolkit, and CSV exports for downstream tooling.

The user flow, end to end:

1. **Upload.** A medical reviewer uploads a PDF manual through a React frontend authenticated with Clerk. The PDF is parsed into structured JSON by a dedicated GPT-4o extraction service and stored in Vercel Blob.
2. **Generate.** The reviewer opens a chat against the parsed guide and asks the system to produce a DMN. A generator prompt running on Anthropic Claude (routed through a provider abstraction that also supports Google Gemini) reads the guide from a prompt-cached context block and produces an initial DMN artifact.
3. **Validate.** Seven catcher agents run in parallel against the generated DMN, each checking a different concern: `provenance` (do rules trace to manual quotes), `clinical_review` (are the rules medically sound), `consistency` (do rules contradict each other), `completeness` (are there gaps), `dmn_audit` (is the XML syntactically valid), `boundary_condition` (are edge cases covered), and `module_architecture` (does the Activator/Router/Integrative two-table pattern hold). The catchers run through QStash as a fan-out queue to parallelize across Anthropic.
4. **Resolve.** Each catcher report appears in the chat UI with three buttons: [Auto-fix All], [Fix with Instructions], and [Ignore]. The reviewer picks one per report. A fix_generator prompt then produces a revised DMN based on the accepted fixes.
5. **Version.** Every generation and every fix produces a new row in `dmn_versions` in Neon Postgres with the full DMN XML, a Mermaid rendering, and an XLSForm payload. The relationship to the prior version is tracked in a `fix_of_version_id` field.
6. **Export.** At any point the reviewer can download the current DMN in any of three formats via deterministic TypeScript transpilers that live in a separate `converters/` package.

The supporting infrastructure: Clerk for auth, Vercel Blob for guide PDFs, Upstash Redis for cache state tracking, QStash for catcher fan-out, and Neon Postgres for persistent state across 26 Prisma models including a full audit trail (`dmn_versions`, `catcher_runs`, `cache_state`, `llm_call_log`, `prompt_log`, `edit_log`, `replay_tokens`, `api_request_log`, and more).

From a reliability-testing standpoint, the design was ambitious. The replay_tokens table, the llm_call_log with token-level cost tracking, the edit_log, and the dmn_versions audit trail were all intended to make the system empirically measurable: "run the same PDF twice, diff the outputs, compute test-retest reliability." That was the whole point.

The problem the rest of this document documents is that while the *schema* was designed to support this measurement, the *system* never produced data we could measure against. The measurement infrastructure was architecturally correct and operationally hollow.

---

## 3. Architectural decisions and their rationale

Each decision was made for good reasons. None was obviously wrong when made. The failure modes in Section 4 are emergent consequences of decisions that, individually, were defensible.

### 3.1 Use an LLM to produce DMN at all

**Rationale.** DMN generation from a free-text clinical manual is not a lookup; it requires interpreting the manual's clinical intent, choosing how to decompose conditions into boolean predicates, and matching the facts in the manual to a formal decision-table shape. No existing NLP pipeline produces this output shape reliably. LLMs are the only technology that can.

**Consequence downstream.** Once the system relies on an LLM, every knob becomes a prompt, and every prompt is an independent variable we have to control.

### 3.2 Break the generator into a generator plus multiple catcher validators

**Rationale.** The generator alone produced DMN that was clinically plausible but often incomplete or internally inconsistent (missing fail-safe predicates, contradictory rules, unprovable boundary conditions). Rather than fattening the generator prompt with every possible check, we separated concerns: the generator produces a candidate, and specialized catcher prompts validate specific dimensions. Each catcher is short (5 to 26 lines of prompt text) and focused. This separation of concerns is a standard software engineering practice and it worked, in the sense that the catchers reliably surface real issues.

**Consequence downstream.** The number of prompts grew. By March 18 we had seven catchers plus the generator, plus a fix_generator that consolidates catcher reports back into a revised DMN, plus a PDF extractor prompt for the upstream parsing stage. Ten prompts in total. Each is an independent degree of freedom.

### 3.3 The auto-accept versus manual-change decision (the critical tension)

**Rationale.** This is the architectural decision that most needs unpacking, because it is both *necessary* for clinical safety and *structurally unsustainable* as a reliability apparatus.

The catchers are needed to flag clinically important issues. The generator does not always catch, for example, a fail-safe that defaults to the wrong direction for a vital sign measurement. A provenance catcher can verify that every dosing threshold traces to a quote in the manual. These checks are essential. A system that just generates DMN without checking would ship wrong clinical rules into production, which is unacceptable.

But once a catcher flags an issue, you have three choices for how to handle it, and all three are legitimate in different contexts:

- **Auto-fix all.** Let the model apply the catcher's suggested fix automatically. This is fast but dangerous because the fix might be wrong. For trivial issues (formatting, naming, missing provenance tags) it is fine. For clinical issues (dosing thresholds, severity classifications) it is not.
- **Fix with instructions.** The reviewer writes a natural-language instruction telling the fix_generator *how* to fix the issue. This is the safe middle ground: the reviewer's clinical judgment guides the fix, but the LLM does the mechanical work of updating the DMN.
- **Ignore.** The reviewer determines that the catcher is wrong or that the flagged issue is acceptable. Catchers produce false positives (especially the boundary_condition and completeness catchers), so Ignore is a real and frequent choice.

The clinical logic here is correct: **you cannot auto-accept every fix, because some fixes would introduce clinical errors the catcher correctly flagged but fixed wrong.** Human review is not optional at this stage. It is the load-bearing safety guarantee.

But the *operational consequence* is:

- Every catcher report becomes a blocking decision point. A DMN with twenty catcher reports is twenty decision points before the reviewer can move on.
- Different reviewers will make different choices at each decision point. Same input, different outputs, depending on who is clicking.
- The reviewer's decisions become part of the system's state but are not automatically captured in a structured form. The schema records only *whether* a fix was applied (`fix_applied` boolean), not *which of the three buttons the reviewer clicked* nor *what instructions they wrote* for the Fix-with-Instructions flow.
- Over a 100-page manual the reviewer is making hundreds of such decisions, and the cumulative time cost makes the system impractical for real CHW manuals.

This is the central tension: **the catchers are clinically necessary and the human review of their output is unavoidable, but the resulting workflow is too slow for real use AND it reintroduces the very prompt-path divergence we built the system to eliminate, just at a different level of the stack.** Last semester the divergence was Zach and you clicking differently in the Claude.ai chat interface (10 queries versus 5). With CHW-Chat-Interface the divergence would be Zach and you clicking [Auto-fix] versus [Ignore] on different catcher reports. The system moved the variance one layer down; it did not shrink it.

There is no way to resolve this tension inside the sequential-with-catchers architecture. You cannot auto-accept for clinical safety reasons, you cannot skip the catchers without losing the safety guarantee they provide, and you cannot make the human review step deterministic without eliminating the reviewer entirely.

### 3.4 Cross-model validation (Claude generator, Gemini red team)

**Rationale.** To defend against sycophancy (a model agreeing with its own earlier wrong output), we built a provider abstraction that would let the generator run on Claude and a subset of catchers run on Gemini. This mirrors classical instrumental-variable thinking in econometrics: if you want to check whether an estimator is biased, check it with a different instrument.

**Consequence downstream.** The abstraction worked but was never used in practice. The system ran with `DEV_MODE=true` throughout, which forces all calls to Claude Haiku regardless of configured provider. The cross-model validation was never exercised in the production data (the llm_call_log shows all 45 calls hit `claude-haiku-4-5-20251001`). The architectural investment was real; the operational benefit was zero.

### 3.5 Prompt caching

**Rationale.** A 100-page CHW manual is 20 thousand to 50 thousand tokens. Sending it on every LLM call would multiply costs by the number of turns. Anthropic's `cache_control: {type: "ephemeral", ttl: "1h"}` breakpoint was supposed to cache the guide JSON for the duration of a chat session and give us a 90% discount on input tokens.

**Consequence downstream.** It never worked. The March 18 audit flagged that the `ttl` field in our cache_control header may not be a valid Anthropic API format; we never verified which. The production data confirms it: zero of 45 LLM calls had a cache hit. All 127,217 input tokens were fresh. The caching investment was real; the realized discount was zero. And guides larger than the cache would be impossible turn into a DMN.

### 3.6 Persistent DMN versioning

**Rationale.** Every DMN generation and every fix creates a new row in `dmn_versions` with the full content, linked to the prior version via `fix_of_version_id`. This is standard event-sourcing. It gives us a full history of every output and lets us diff across time, reviewers, and chat sessions.

**Consequence downstream.** The infrastructure worked as designed but was barely exercised. The production data shows 24 DMN versions total across the entire operating history, and zero of them have `fix_of_version_id` populated. The corrected analysis in Section 4.3 attributes this to either non-use of the [Auto-fix] / [Fix with Instructions] flows in production, or to a runtime path I did not trace end-to-end — the static code path is functional but I have not verified the runtime end-to-end. Either way, no successor lineage is recorded in the audit trail across the entire 21-day operating window.

### 3.7 Deterministic IR converters

**Rationale.** DMN XML, Mermaid, and XLSForm are three different output formats. Rather than prompting an LLM for each, we wrote pure TypeScript transpilers that take an intermediate representation (IR) and emit any of the three formats deterministically. No AI in the conversion layer. This is the one decision that held up entirely — the converters work, they are tested (54 passing tests on 8 test files as of March 18), and they are the only part of CHW-Chat-Interface that ports directly into the REPL hybrid unchanged.

---

## 4. The three identification failures, with numbers

All numbers in this section are pulled directly from the Neon Postgres database on April 8, 2026. The system's total operating window was March 12 (first chat) to April 1 (last chat), with the bulk of activity on March 12 (31 chats) and March 18 (23 chats, mostly from the automated overnight audit loop).

### 4.1 Row counts across the audit trail

| Table | Rows |
|---|---|
| guides | 3 |
| chats | 61 |
| messages | 129 |
| dmn_versions | 24 |
| catcher_runs | 42 |
| cache_state | 21 |
| llm_call_log | 45 |
| prompt_log | 0 |
| edit_log | 0 |
| replay_tokens | 0 |

The three observability tables at the bottom (`prompt_log`, `edit_log`, `replay_tokens`) are entirely empty. These were designed to hold the reproducibility-testing data that the REPL hybrid will need to replicate in a cleaner form. CHW-Chat-Interface never populated them.

The `guides` table has three rows: one named "Test Guide" (really just `sample-data/guide_full.json`), one named "iccm_guide.json", and one named "iCCM Pediatric Guidelines v2024". **Not one of them is a real production CHW manual.** The system was tested exclusively against sample data.

### 4.2 Failure 1 — High-dimensional treatment vector

**Plain English.** Ten distinct prompts control the system's behavior. Editing any single prompt in isolation requires holding the other nine constant and running multiple trials per edit. The system cannot be sampled densely enough to identify the marginal effect of a single-prompt edit.

**Econometric equivalent.** High-dimensional treatment with insufficient observations, leading to severe under-identification. Formally: if we treat each prompt as an independent treatment variable *T_i* and the DMN output as the outcome *Y*, the design matrix for estimating **β = (β_1, ..., β_10)** has far fewer rows (runs) than columns (prompts). The marginal effect of any *β_i* is not recoverable without drastically more observations than we collected.

**Numbers.**
- **10 prompts total.** Seven catcher prompts (`boundary_condition`, `clinical_review`, `completeness`, `consistency`, `dmn_audit`, `module_architecture`, `provenance`), one generator prompt, one fix_generator prompt, and one PDF extractor prompt. The `module_architecture` catcher was added on March 18 as part of the convergence cycle, raising the count from nine to ten.
- **Actually exercised in production: only 3 catchers** (`consistency`, `dmn_audit`, `provenance`). The other four catchers have zero rows in `catcher_runs`. Of the seven catchers we designed, four never ran against real data.
- **Only 3 `generate` calls total** in the `llm_call_log`. The generator prompt ran only three times in the entire 21-day operating window. Yet 24 DMN versions exist in the database — meaning 21 generations happened outside the logged code path. This is the March 18 audit's deferred item #6 manifesting in the data: *"Generator streaming calls never logged — Need to capture usage from Anthropic stream events."* The most important prompt in the system was also the least observed.
- **All 42 catcher runs happened on a single day: March 12, 2026.** After March 12, the system ran 31 more chats across March 14, 18, 19, 20, and April 1 without a single catcher firing. The mediator half of the architecture effectively stopped working on day one, and we did not notice because the catcher_runs count was not on any dashboard.
- **Average 3 catchers per DMN version.** Of 24 DMN versions, 14 have catcher runs attached. On those 14, an average of 3.0 catchers ran per version. The intended panel of 6-7 catchers per DMN never ran in production.
- **All catchers ran on `claude-haiku-4-5-20251001`.** The cross-model architecture (Section 3.4) was never exercised because `DEV_MODE=true` in the environment forced all calls through the cheap Claude Haiku endpoint. We have zero empirical data on whether the Gemini red-team pass would have caught different issues.

**What this implies.** Even if we wanted to identify the effect of editing, say, the `provenance.txt` prompt on downstream DMN quality, we do not have enough observations (14 catcher-equipped DMN versions, 3 generate calls, 1 day of catcher activity) to run that experiment. The treatment vector is 10-dimensional; the observation vector is effectively 1-dimensional. Under-identified by a factor of ten.

### 4.3 Failure 2 — Unobservable mediator

**Plain English.** The reviewer's decisions on catcher reports ([Auto-fix All], [Fix with Instructions], [Ignore]) change the final DMN but are not recorded in the schema. We cannot separate variance due to the model from variance due to the reviewer's choices.

**Econometric equivalent.** An unobservable mediator between treatment and outcome. In mediation analysis, identifying the direct effect of the treatment (prompt) on the outcome (DMN) requires observing the mediator (reviewer choice). If the mediator is unobserved, the model-versus-user decomposition is unidentified, and every reliability estimate conflates prompt noise with reviewer noise.

**Numbers and structural status.** The mediator failure has two distinct sub-problems that the data alone does not separate. To distinguish them I traced the frontend-to-backend code path for each of the three reviewer actions:

| Reviewer button | Frontend handler | Backend reach | Structural status | What the 0 rows in the audit trail tell us |
|---|---|---|---|---|
| [Auto-fix All] | `handleAutoFix` -> `handleFix(reports)` -> POST `/api/chat/fix` with `dmn_version_id` of the parent version | `chat.py` writes a new `dmn_versions` row with `fixOfVersionId` populated | **Functional end-to-end.** The code path is intact. | Nobody clicked the button in production, OR clicked it before a DMN was loaded and was silently rejected by the `latestVersionIdRef.current` guard. The data is consistent with non-use, not with a broken pipe. |
| [Fix with Instructions] | `handleFixWithInstructions` -> `handleFix(reports, instructions)` -> same POST | Same as above plus a `user_instructions` field passed into the fix prompt | **Functional end-to-end.** The code path is intact. | Same as above. |
| [Ignore] | `handleIgnore` is `useCallback(() => {}, [])` — an empty function that does nothing beyond collapsing the report card visually | No backend call. No log. No state change. Nothing. | **Structurally invisible.** | We cannot tell whether the button was clicked at all. A reviewer clicking [Ignore] one hundred times leaves exactly the same trace in the audit trail as a reviewer who never opened the system. |

The other audit-trail numbers, separately:

- **0 of 129 messages have `catcher_results` populated.** The `Message.catcher_results` field is a JSONB column designed to store the full catcher report that prompted a reviewer decision. The frontend's fix flow does not appear to write to this column at all — it sends the catcher reports as part of the fix POST body, but the backend does not snapshot them onto the message row. So even if a fix was applied, the catcher report that prompted it would not appear here. This is a pure observability gap, not a mediator gap.
- **0 of 24 `dmn_versions` have `fix_of_version_id` set.** Combined with the structural status above, this means we have zero recorded instances of any reviewer completing the [Auto-fix] or [Fix with Instructions] flow against any DMN, on any guide, in the entire 21-day operating window of the system.
- **`Message.fix_applied` is a boolean**, capturing at most "a fix was applied (true)" or "no fix was applied (false)." The reviewer's action space is ternary ([Auto-fix All] / [Fix with Instructions] / [Ignore]) and the instruction text for the second option is unbounded. Even *if* the column were populated, it could not distinguish a fix prompted by Auto-fix from one prompted by Fix-with-Instructions, nor capture the instruction text for the second case.
- **Two browser tabs on the same chat can race.** The March 18 audit's Causal Chain item B2 identified that two tabs on the same chat can corrupt each other's mediator state because there is no server-side lock and no cross-tab BroadcastChannel. The mediator is not just unobservable; it is multi-valued when two browser sessions collide.

**What this implies.** The mediator problem is more nuanced than "the schema is broken." The structural reality is:

- The [Auto-fix] and [Fix with Instructions] code paths are intact and would produce auditable evidence if used. They were not used. Reliability data exists in principle but was never collected.
- The [Ignore] code path is a no-op. Even if reviewers had used it, the audit trail would show nothing. This part of the mediator is genuinely unobservable by construction.
- The catcher_results column on the message row is unwritten regardless of which button was clicked, so we cannot reconstruct from messages alone what catcher reports prompted what decisions.
- The `Message.fix_applied` boolean is a binary projection of a ternary action space, but in practice was never populated, so the binary-vs-ternary point is moot — the field is empty.

The overall identification problem stands: we cannot decompose the variance in the final DMN into "model error" versus "user choice error" because we have no recorded user choices. But the *reason* we have no recorded user choices is partly structural (the [Ignore] no-op, the unwritten catcher_results column) and partly behavioral (nobody used the [Auto-fix] / [Fix with Instructions] flows in production despite the code paths being intact). I should not have collapsed both into "the schema was broken" in an earlier draft of this document. The honest version is that the structural and behavioral failure modes coexist, and the data alone cannot fully separate them without re-running the system end-to-end on a fresh fix-flow attempt to verify the static code path actually produces audit rows at runtime — which we did not do for this postmortem.

### 4.4 Failure 3 — Non-stationary baseline

**Plain English.** The cache state is scoped to a chat session, not to the content of the PDF. Two chats on the same PDF have independent cache state. The test-retest reliability measurement we designed (same input, different runs, diff the outputs) is violated because the environment is not stationary across runs.

**Econometric equivalent.** Non-stationary baseline violates the stationarity assumption required for test-retest reliability estimation. When the environment drifts between observations, the variance in the outcome across observations cannot be attributed to the thing you are studying (the prompt); it could also be attributed to the drifting environment. The estimator is biased toward the noise floor of the environment.

**Numbers.**
- **Test Guide was uploaded into 55 distinct chat sessions.** The same PDF (a single file, one content hash) was opened in 55 separate sessions across the system's operating window. This is the test-retest reliability scenario written in the data: same input, 55 runs, should produce substantively identical DMN.
- **It produced 21 DMN versions with 21 distinct md5 content hashes.** Of the 21 DMN versions generated from the Test Guide across 55 chats, every single one is a distinct artifact. **Test-retest reliability measured at the DMN level is zero percent.** Not one pair of runs produced the same output.
- **21 cache_state rows exist — all 21 have NULL `last_turn_at` and NULL `expires_at`.** The observability fields the schema was designed to populate were never written to. This is the March 18 audit's deferred item #5 (*"CacheState lastTurnAt/expiresAt never written — Schema exists but never populated"*) manifesting exactly as predicted.
- **0 of 21 cache_state rows are currently cached.** The `is_cached` flag is false for every row. The cache was never in the "cached" state for any session. 
- **0 of 21 cache_state rows have a `gemini_cache_name` or `claude_cache_hash`.** The provider-side cache references were never written either. The cache was, operationally, a null object.
- **0 of 45 LLM calls had a cache hit.** The `cached_tokens` column in `llm_call_log` is 0 for every single row. **127,217 input tokens** were sent to the LLM providers as fresh context. The intended 90 percent cache discount was realized at 0 percent.
- **The `cache_control: {type: "ephemeral", ttl: "1h"}` header we sent to Claude may have been malformed.** The March 18 audit's Gotcha #4 flagged that the `ttl` field is not a documented Anthropic API key. We never verified whether it was ignored silently or whether the cache would have worked with a different field name. The answer may be "the cache would have worked if we had used the correct header." We will never know from the production data because the data is empty.

**What this implies.** The single most important reliability number we wanted to measure — test-retest reliability of DMN generation on the same PDF — is 0% in the actual data. Every run produced a different DMN. Whether this is because the prompt is brittle, the model is stochastic, the cache was broken, or all three, we cannot tell from the data alone. The non-stationarity of the baseline (no two cache states alike, no reuse across sessions) makes attribution impossible.

---

### 4.5 Why observability cannot fix the dimensionality problem

A natural reaction to the empty audit-trail findings in Section 4.3 and 4.4 is "the team should have wired up the observability infrastructure they had already designed." That reaction is correct as a critique of the engineering execution, but wrong as a critique of the architecture. Wiring up the observability would have produced visibility into what happened, but it would not have produced *tractability* — the ability to fix what happened.

Imagine the counterfactual. Suppose every March 18 deferred observability item was repaired: `cache_state.last_turn_at` and `expires_at` are populated on every turn, generator streaming calls are logged into `llm_call_log`, the [Ignore] button writes a row into a new `mediator_decisions` table, two-tab races are eliminated by a server-side lock. We now have full, unbiased records of every variable in the system.

A bad DMN comes out of a run. Where did the variance come from?

- Was the bad output caused by `generator.txt`?
- Or by `provenance.txt` flagging the wrong things?
- Or by `consistency.txt` failing to flag a real conflict?
- Or by the reviewer's [Auto-fix] choice on catcher report #4?
- Or by the reviewer's [Ignore] choice on catcher report #6?
- Or by `fix_generator.txt` synthesizing a wrong fix from the accepted catcher feedback?
- Or by a transient model failure on one of the dozen LLM calls in the run?
- Or by a cache hit returning stale state?
- Or by an interaction between several of the above?

Even with complete logs, we have nine-plus candidate causes per failure and no way to isolate which one mattered. We cannot edit one prompt and observe the marginal effect on the output, because edits to one prompt cascade through the others — `generator.txt` produces a candidate that all the catchers then validate, so an edit to `generator.txt` changes what `provenance.txt` flags, which changes what `fix_generator.txt` is asked to fix, which changes the final DMN. The system is not a set of independent variables we can tune one at a time; it is a network of coupled variables where every edit propagates.

Just because we can see the environment variables does not mean they stop existing. The team could spend months tuning prompts and never converge, because each tuning pass would change downstream behavior in ways no amount of logging could let us attribute. **This is not a problem of insufficient observability. It is a problem of degrees of freedom relative to observations.** The system has more knobs than we can afford test runs, and the knobs interact, so even perfect logs cannot tell us which knob to turn.

### 4.6 Dimensionality scales nonlinearly with prompt count

The cost of going from one prompt to many is not linear; it is closer to combinatorial. Edits to any prompt cascade through the prompts downstream of it, so the effective degrees of freedom we have to control grow much faster than the prompt count itself.

| Prompt count | Approximate effective degrees of freedom | Identification tractability |
|---|---|---|
| **1 prompt** | 1 (the prompt itself) | Identification by design. Edit the prompt, hold everything else constant by construction, observe the output, attribute the change. This is the architectural ideal. |
| **2 prompts** | ~4 (each prompt independently, plus the two interaction directions: A's effect through B, B's effect through A) | Acceptable. Identification cost is real but manageable. Within-subject experiments are still feasible: hold one prompt constant while editing the other and run a few trials. The marginal cost of each tuning iteration is bounded. |
| **9 prompts** | Combinatorial (each prompt independently, all pairwise interactions, all higher-order interaction terms) | Effectively un-tunable. The search space grows exponentially in the number of prompts. Two engineers tuning the same system from the same starting state will end up at different points because the local gradient varies with where in the space they start. |

Going from 1 to 2 is acceptable. Going from 2 to 9 is a phase change. The reason is not the absolute number of prompts; it is the ratio of degrees of freedom to test runs we can afford. Once you have more knobs than observations, identification is gone, and adding more knobs (catchers, validators, fix prompts, extractor prompts) makes it strictly worse.

A single-prompt extraction architecture is the only architecture that puts this ratio safely above one for the budget our research program can afford. The reason single-prompt extraction has not historically been the default approach for long-document tasks is that there was no mechanism for a model to operate over a 100-page document inside a single prompt: the document exceeded the effective context window, so the engineer had to chunk and pipeline, which forced multi-prompt orchestration. **The technical mechanism that makes single-prompt extraction feasible at our document scale is Recursive Language Models (Zhang, Kraska, Khattab, MIT CSAIL, January 2026).** Before that paper, multi-prompt was the only feasible approach for documents longer than the context window. After that paper, single-prompt became feasible — and once feasible, it became the only architecture that satisfies the project's identification constraint.

### 4.7 Why the kill decision was correct at the time it was made

The standard objection to the March 30 kill decision is: "CHW-Chat-Interface was not completed; you stopped before you had the full reliability data; how can you justify killing it on incomplete evidence?" This objection has the question backwards. The right question is not "did you collect enough data" but **"was the architecture capable of collecting the right kind of data, given the project's identification constraint?"**

By March 18 we had four pieces of evidence sufficient to answer that question with no:

1. **Architectural dimensionality.** The system contained 9-10 prompts (Section 4.2) — generator, fix_generator, seven catchers (the seventh, `module_architecture`, was added on March 18 as part of the convergence cycle, *raising* the count from nine to ten), and a PDF extractor. By the dimensionality scaling argument in Section 4.6, this puts the system well past the phase change beyond which identification is intractable. This is a structural fact independent of any production data; we did not need to wait for more reliability runs to know that the architecture had too many degrees of freedom.

2. **Empirical zero reliability where we did have data.** The Test Guide had been uploaded into 55 chats and produced 21 distinct DMN outputs (Section 4.4). Test-retest reliability measured at exactly 0%: no two runs produced the same artifact. This was already in the database before the March 18 audit ran.

3. **Observability infrastructure was operationally empty.** The fields the team designed for measuring reliability — `cache_state.last_turn_at`, `Message.catcher_results`, generator streaming logs, the entire `prompt_log` and `edit_log` and `replay_tokens` tables — were not populated, by some combination of design oversight and behavioral disuse. The deferred items in the March 18 CONVERGENCE-REPORT documented this contemporaneously. By the argument in Section 4.5, even if we had repaired all of the observability infrastructure, doing so would have given us visibility but not tractability — we still would have had nine-to-ten knobs and no way to identify which one to turn.

4. **A single-prompt mechanism became available.** Recursive Language Models was published in January 2026. This was the first published mechanism for single-prompt extraction on documents longer than the model's effective context window. Before this paper, single-prompt was technically infeasible at our document scale; you had to chunk, which forced multi-prompt orchestration, which made the identification problem unavoidable. After this paper, single-prompt became feasible and the alternative architecture was finally on the table.

The first three pieces of evidence say *the current architecture cannot be repaired in place to satisfy the identification constraint*. The fourth piece of evidence says *a different architecture that can satisfy the constraint has just become technically feasible*. Together they justify killing the multi-prompt approach on the evidence available on March 30, without waiting for more data from a system that was structurally incapable of producing the right kind of data.

The kill decision is not "we tried CHW-Chat-Interface and it failed." It is "we built CHW-Chat-Interface as the best multi-prompt approach we could design, learned from running it that the multi-prompt class is structurally wrong-shape for our identification constraint, and pivoted to a single-prompt architecture as soon as a published mechanism made it technically feasible at our document scale." The first half of that sentence is the work CHW-Chat-Interface did. The second half is the conclusion the work supports.

This is also why I want to avoid the framing in your April 7 note that the pivot was "throwing out something that kind of worked by hand." CHW-Chat-Interface did work in the sense that it produced DMN output; it did not work in the sense the project requires, which is producing DMN output that is reproducible across runs and attributable to its causes. The latter is the only definition of "work" that satisfies the identification constraint. Any architecture that fails the identification constraint fails the project regardless of how much output it produces.

---

## 5. The March 18 audit and what it documented

The March 18 overnight audit loop produced `overnight_loop_2026-03-18/CONVERGENCE-REPORT.md`, a 120-line document that summarizes a 15-agent sequential audit of the CHW-Chat-Interface codebase. The headline numbers from that report:

- **15 sequential audit agents run** (one per concern: backend trace, frontend flow, cross-service contracts, state lifecycle, causal chain, timing/race, edge/boundary, observability, idempotency, product/UX, render efficiency, fetch timing, query efficiency, streaming pipeline, perceived speed).
- **10 unique BLOCKING findings** after deduplication across agents.
- **8 BLOCKING fixed in the same cycle** (B2 through B10 plus MermaidViewer M2).
- **2 BLOCKING deferred** (B1: QStash signature verification; B8: Anthropic stream context manager).
- **Test coverage:** 54 passing tests across 8 test files, up from 10 passing tests across 3 files before the cycle.

Three of the deferred issues in the CONVERGENCE-REPORT map directly onto the three failure modes in Section 4:

- **Causal Chain B2 — "Two-tab same-chat corruption":** *"Needs either server-side full-request lock or cross-tab BroadcastChannel guard."* This is the unobservable-mediator problem made worse by a race condition. If two browser tabs on the same chat make different [Auto-fix]/[Ignore] choices at the same time, the resulting mediator state is not just missing from the audit trail, it is structurally multi-valued.
- **State Lifecycle B3 — "CacheState lastTurnAt/expiresAt never written":** *"Schema exists but never populated. Needs migration to add default values or code to write them."* This is exactly the non-stationarity evidence that the production data confirms. The team knew on March 18 that the cache observability fields were dead. The production data shows all 21 rows have these fields NULL, confirming the finding.
- **Observability B1 — "Generator streaming calls never logged":** *"Need to capture usage from Anthropic stream events."* This is the treatment-vector observability problem. The generator is the most important prompt in the system (it produces the candidate DMN that all catchers validate against), and the team documented on March 18 that we could not even audit which generator prompts fired on which turn.

Two additional gotchas from the same report:

- **Gotcha #4 — the `ttl` field may not be valid Anthropic API format.** The cache header we were passing to Claude might have been silently ignored. The production data (0 cache hits) is consistent with this hypothesis.
- **Gotcha #5 — "Rate limit `check_rate_limit()` was defined but never wired to any endpoint."** A pattern of "designed but not wired" that applies throughout the system.

The pattern across all three deferred items and both gotchas is the same: the team repeatedly designed observability and control infrastructure, did not wire it up, and did not notice until the March 18 audit because the infrastructure was never exercised by real usage.

**Crucially, the March 18 audit was not a "we are pausing because this is broken" document.** It was a disciplined engineering convergence report that fixed 8 of 10 BLOCKING issues and recommended a Cycle 2 for the remaining MEDIUMs. The decision to pause and pivot was made *after* the audit, between March 19 and March 30, based on a review of the deferred items plus a realization that the three that remained were not individual bugs but symptoms of the three identification failures in Section 4.

---

## 6. Implications for the REPL pivot

The REPL/RLM hybrid I proposed on March 31 (and that you approved on April 2) addresses each of the three failure modes by construction:

**Failure 1 (treatment vector) is addressed by collapsing the prompt surface to one.** The REPL approach gives the model a single frozen system prompt and a Python REPL with the manual loaded as a variable. All intermediate artifacts (supply list, variables, predicates, decision tables) are produced in one continuous session with the same prompt. To identify the effect of a prompt edit on output quality, we need 2 runs, not 18. The catcher prompts survive as *frozen utility validators* that the REPL calls between phases — they are not part of the experimental control surface, in the same sense that Python's standard library is not part of the variable space when you are testing your own code.

**Failure 2 (unobservable mediator) is addressed by removing the user from the inner loop entirely.** The REPL runs autonomously: generate candidate, validate, fix, re-validate, until stable. There is no [Auto-fix]/[Ignore] decision point because the model itself makes the fix decisions based on the validator output. The reviewer comes in at the end of the run, not in the middle, and reviews the final artifact plus the trace. The mediator problem disappears because the mediator disappears.

**Failure 3 (non-stationary baseline) is addressed by using content-hashed PDF as the session key.** The REPL session is keyed by `SHA-256(pdf_bytes)`. Same hash, same starting REPL state, same initial context. Test-retest reliability becomes well-defined: fix the system prompt, fix the PDF, run twice, diff the outputs. Any difference is model stochasticity (which we can bound by running with a fixed seed) or a bug. No environment drift.

The trade-off is that the REPL approach gives up the per-decision clinical review step. The clinician no longer sees every catcher report and chooses how to handle each. This is the right trade-off for a research-grade reliability tool, because the reliability bar is the primary metric and clinical review can happen at the artifact level (review the final DMN, not every intermediate step). It is probably *not* the right trade-off for a production clinical authoring tool, which would need to reintroduce a reviewer-in-the-loop step once the reliability question is settled.

What carries forward from CHW-Chat-Interface to the REPL hybrid, unchanged:

- **The seven catcher prompts.** They stay as frozen utility validators called between REPL phases.
- **The 317-line prompt corpus.** The generator prompt, the fix prompt, and the catcher prompts are the starting point for the REPL's single system prompt and its validator suite. The variable naming convention (`q_`, `ex_`, `v_`, `lab_`, `p_`, `dx_`, `tx_`, `ref_`, etc.) ports directly.
- **The TypeScript IR converters.** DMN → IR, IR → DMN, IR → Mermaid, IR → XLSForm are deterministic and tested. They are the strongest part of CHW-Chat-Interface and they are what we show CHT as the deliverable end of the pipeline.
- **The Prisma schema design.** The `dmn_versions`, `catcher_runs`, and `cache_state` tables (and the observability tables below them) are the right shape for capturing reliability data. The REPL hybrid will populate them for real.

What does not carry forward:

- **The 9-prompt pipeline architecture.** Replaced by one REPL session with one system prompt.
- **The [Auto-fix]/[Ignore]/[Fix with Instructions] UI.** Removed entirely. The REPL has no per-decision human checkpoint.
- **The cross-model (Claude + Gemini) abstraction.** Kept as infrastructure but not part of the critical path; the REPL runs on one model at a time.
- **The chat-based session model.** Replaced by content-hash-keyed REPL sessions with a full execution trace.

---

## 7. Notes on the process gap

Your April 7 email raised a process concern: I pivoted without writing down the evidence. That concern is valid. I made two process mistakes between March 18 and April 7:

1. **I did not send the CONVERGENCE-REPORT review.** Between March 19 and March 30 I read the March 18 audit outputs and formed the conclusion that the three deferred items were symptoms of deeper identification failures. I did not write this reasoning down in a form I could send to you. On March 31 I sent the ORCHESTRATOR.md spec proposing the pivot, but it led with the REPL's architectural properties, not with the CHW-Chat-Interface failure analysis that motivated the pivot. From your perspective the pivot looked sudden because the document chain does not contain the review step.

2. **I did not quantify the failures when I observed them.** The numbers in Section 4 of this document were generated on April 8, three weeks after the pivot decision. They confirm everything I concluded qualitatively in late March, but I should have pulled them on March 30 and included them in the ORCHESTRATOR spec. The delay meant the pivot proposal was evidence-free from your standpoint even though the evidence existed in the database the whole time.

Both mistakes trace to the same root cause: I prioritized moving forward on the REPL build over closing the loop on the postmortem. The REPL had complex infrastructure ahead and a four-week semester clock, and I judged that writing up the failure analysis would slow the build more than it would clarify the decision. That judgment was wrong on the clarification axis. This document is the correction.

A third, smaller methodological note for honesty's sake: an earlier draft of Section 4.3 of this document stated that the fix-flow audit fields were "designed but never wired up," conflating the structural and behavioral failure modes. I traced the actual code path while preparing this version and found that the [Auto-fix All] and [Fix with Instructions] flows are functional end-to-end at the static-code-path level — they POST to `/api/chat/fix`, the backend writes a new `dmn_versions` row with `fixOfVersionId` set, and the SSE stream returns the new version. The 0 rows in `fix_of_version_id` therefore mean either "nobody clicked the button" or "there is a runtime bug in a path I did not trace." I did not run the system end-to-end to disambiguate. The genuine structural mediator failure is narrower than the original draft claimed: it is the [Ignore] button (a literal no-op `useCallback(() => {}, [])` in `ChatPanel.tsx:500`) and the unwritten `Message.catcher_results` JSONB column. The Section 4.3 table in this document shows the corrected per-button structural status. I am noting the correction here because making absence-of-data inferences without distinguishing structural from behavioral causes is exactly the methodological error econometric training is meant to prevent, and I should not have done it in a postmortem written for an econometrically-trained reader.

---

## 8. Appendix: full database query results

All queries run against the Neon Postgres production database on April 8, 2026, via the `CHW-Chat-Interface/.env::NEON_DB_URL` connection string. The query script used is `chwci_db_audit.py` in the CHW_RLM repo root and can be re-run at any time.

### 8.1 Table row counts (full audit trail snapshot)

| Table | Rows | Notes |
|---|---|---|
| guides | 3 | "Test Guide", "iccm_guide.json", "iCCM Pediatric Guidelines v2024" — all sample data, no real CHW manuals |
| chats | 61 | across March 12 to April 1 |
| messages | 129 | 0 with catcher_results populated |
| dmn_versions | 24 | 24 distinct md5 hashes, 0 with fix_of_version_id set |
| catcher_runs | 42 | all on 2026-03-12, only 3 catchers fired |
| cache_state | 21 | all with NULL last_turn_at, NULL expires_at, 0 currently cached |
| llm_call_log | 45 | 0 cache hits, all on claude-haiku |
| prompt_log | 0 | designed but never populated |
| edit_log | 0 | designed but never populated |
| replay_tokens | 0 | designed but never populated |

### 8.2 Test Guide cross-chat fan-out (Failure 3 primary evidence)

| Metric | Value |
|---|---|
| Distinct chats using Test Guide | 55 |
| Messages across those chats | 103 |
| DMN versions generated | 21 |
| Distinct md5(dmn_xml) outputs | 21 |
| **Test-retest reliability** | **0%** (no two runs produced the same DMN) |
| cache_state rows for those chats | 16 |
| catcher_runs for those chats | 42 |

### 8.3 Catcher activity breakdown

| Catcher | Model | Runs | Avg ms | Severity distribution |
|---|---|---|---|---|
| consistency | claude-haiku-4-5-20251001 | 14 | 4736 | 3 error, 6 pass, 5 warning |
| dmn_audit | claude-haiku-4-5-20251001 | 14 | 4287 | 8 error, 6 warning |
| provenance | claude-haiku-4-5-20251001 | 14 | 3845 | 14 error |

Four of the seven designed catchers (`boundary_condition`, `clinical_review`, `completeness`, `module_architecture`) have zero rows in `catcher_runs`. They were defined in the prompts directory but never fired against real data.

### 8.4 LLM call log — operation breakdown

| Operation | Provider | Model | Calls | Input tokens | Output tokens | Cached tokens | Cost USD |
|---|---|---|---|---|---|---|---|
| catch | anthropic | claude-haiku-4-5-20251001 | 42 | 124,744 | 21,663 | 0 | $0.1864 |
| generate | anthropic | claude-haiku-4-5-20251001 | 3 | 2,473 | 8,820 | 0 | $0.0373 |

Total: 45 calls, 127,217 input tokens, 30,483 output tokens, **0 cache hits (0.0%)**, $0.2237 cumulative cost. The entire production LLM bill over 21 days was 22 cents.

### 8.5 CacheState observability fields

| Field | Rows | Null / default |
|---|---|---|
| total rows | 21 | — |
| last_turn_at NULL | 21 | all |
| expires_at NULL | 21 | all |
| cached_at NULL | 0 | none |
| is_cached = true | 0 | none |
| gemini_cache_name not null | 0 | none |
| claude_cache_hash not null | 0 | none |

The `cached_at` field is populated (all 21 rows have a timestamp for when the cache row was created) but the downstream observability fields (`last_turn_at`, `expires_at`, `is_cached`, and the provider-side cache references) are all uniformly empty. The schema exists, the creation code ran, and the population code never did. This is exactly the CONVERGENCE-REPORT's deferred item #5, observable in the data.

### 8.6 Activity timeline

| Date | Chats created | Catcher runs |
|---|---|---|
| 2026-03-12 | 31 | 42 |
| 2026-03-14 | 1 | 0 |
| 2026-03-18 | 23 | 0 |
| 2026-03-19 | 1 | 0 |
| 2026-03-20 | 1 | 0 |
| 2026-04-01 | 4 | 0 |

The large spike on March 18 corresponds to the 15-agent overnight audit loop running automated tests against the system. None of those tests exercised the catcher pipeline, which is why catcher runs are zero on that day. The small April 1 cluster happened after the REPL pivot proposal was sent.

### 8.7 DMN version re-run distribution

| Chat ID (abbreviated) | Version count | Time span |
|---|---|---|
| 98125cac | 2 | 74 seconds |
| b74089bc | 2 | 35 seconds |
| (all other chats) | 1 each | n/a |

Only two chat sessions in the entire history produced more than one DMN version within the same chat. In both cases the second version appeared within 75 seconds of the first. Neither second version has `fix_of_version_id` set. Two interpretations are consistent with this evidence:

1. **Re-generation, not fix.** The reviewer triggered a fresh generation rather than the fix flow, so the lineage column legitimately stays NULL.
2. **Fix flow ran but the lineage column was not written.** The static code path I traced (Section 4.3) populates `fixOfVersionId` from the request body; if either the frontend stopped sending the field or the backend dropped it on a code path I did not trace, the lineage would be missing even after a successful fix.

I have not run the system end-to-end to disambiguate these two interpretations on the historical data. What the database does support unambiguously is the weaker claim: across 61 chat sessions in 21 days, 59 sessions produced exactly one DMN version each. The iterative workflow (generate, catch, fix, re-validate) reached the "fix" stage at most twice in the entire production history of CHW-Chat-Interface, and we cannot tell from the audit trail alone whether those two cases were fixes or re-generations.

---

*End of document. Total runtime of CHW-Chat-Interface as an operating system: approximately 21 days. Total production LLM cost: $0.22. Total observed test-retest reliability on the single repeatedly-used test guide: 0%. Total reviewer decisions captured in the audit trail: 0 — across an unknown number that may have been made. The system was architecturally elaborate and operationally hollow, and the three failure modes in Section 4 are the structural reasons it was operationally hollow in exactly the ways the schema was designed to prevent. The kill decision (Section 4.7) was justified by the project's identification constraint (Section 0), not by the engineering failures that confirmed it.*
