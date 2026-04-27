# CHW-Chat-Interface Failure Demonstration — Video Plan

> **Target repo:** `CHW-Chat-Interface` (drop this file at the repo root before recording)
> **Audience:** Prof. David Levine, Haas School of Business
> **Length:** 7-9 minutes, plus 1 minute intro, 1 minute conclusion
> **Goal:** Empirical demonstration of the three identification failures the team observed in March 2026 that motivated the pivot to RLM. This video is the postmortem artifact that should have accompanied the March 31 ORCHESTRATOR.md pivot proposal.

---

## 0. The single sentence to anchor the video

> *"Reliability is necessary for validity, but not sufficient. CHW-Chat-Interface had three structural reasons it could not achieve reliability. Here is each one, measured."*

(Quote from David's own March 3 email, which the video should cite verbatim at the 10-second mark.)

---

## 1. What the video needs to show

Three failure modes, each demonstrated with code and data from the repo, each tied to a named econometric problem. Each has contemporaneous evidence from the March 18 `CONVERGENCE-REPORT.md` audit that validates it.

| # | Failure mode | Code evidence (file:line) | Contemporaneous evidence | Econometric name |
|---|--------------|---------------------------|--------------------------|------------------|
| 1 | 10-prompt treatment vector | `backend/src/prompts/*.txt` (9 files, ~180 lines) + `backend/pdf_extractor/prompt.py` (134 lines). Note: `module_architecture.txt` was ADDED on March 18, making it the 7th catcher. | `CONVERGENCE-REPORT.md` deferred item #6: *"Generator streaming calls never logged — Need to capture usage from Anthropic stream events."* Translation: we could not even audit which prompt fired on a given run. | High-dimensional treatment vector with insufficient observations |
| 2 | User-in-the-loop mediator | `frontend/src/components/CatcherReportCard.tsx:189-211` (three button handlers: `onAutoFix`, `onFixWithInstructions`, `onIgnore`). `backend/prisma/schema.prisma:63-77` (Message model — note `fixApplied` is a **Boolean** at line 72, but the user action is **ternary**). | `CONVERGENCE-REPORT.md` Causal Chain B2: *"Two-tab same-chat corruption — Needs either server-side full-request lock or cross-tab BroadcastChannel guard."* The mediator isn't just unobservable — it's racy. | Unobservable mediator |
| 3 | Cache state non-stationarity | `backend/prisma/schema.prisma:49-61` — `CacheState` has `chatId` as the **primary key**, not a content hash. `backend/src/routes/cache.py:43-49` auto-uncaches per-chat, not per-content. `backend/src/llm/claude_client.py:87-92` uses `cache_control: {"type": "ephemeral", "ttl": "1h"}`. | `CONVERGENCE-REPORT.md` State Lifecycle B3: *"CacheState lastTurnAt/expiresAt never written — Schema exists but never populated."* AND Gotcha #4: *"the `ttl` field may not be valid Anthropic API format."* Translation: the cache was not only non-stationary, it was partially broken and the team knew it. | Non-stationary baseline |

Plus a closing walkthrough of `CONVERGENCE-REPORT.md` itself (54 tests passing, 15 audit agents, 10 BLOCKING identified, 8 fixed, 2 deferred) that demonstrates the team had disciplined engineering processes and the pause decision was post-audit, not ad-hoc.

---

## 2. Prerequisites checklist

Before you start recording, verify each item. **Do NOT skip this step** — a failed recording mid-session wastes 30 minutes of work each attempt.

### Environment
- [ ] `CHW-Chat-Interface` cloned and on `main` branch
- [ ] `backend/.venv` exists with dependencies installed (`pip install -r backend/requirements.txt`)
- [ ] `frontend/node_modules` exists (`cd frontend && npm install`)
- [ ] `.env` populated with: `DATABASE_URL` (Neon or local Postgres), `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `CLERK_SECRET_KEY` (dev), `VERCEL_BLOB_TOKEN`, `UPSTASH_REDIS_*`, `QSTASH_*`
- [ ] `npx prisma generate` run without error
- [ ] `npx prisma db push` to sync schema to the dev DB
- [ ] Dev DB reachable: `npx prisma studio` opens without connection errors

### Data
- [ ] `CHW Navigator v1.pdf` in repo root (verified: present, 1.5 MB)
- [ ] At least one historical test run in `DmnVersion` table (query: `SELECT COUNT(*) FROM "DmnVersion"`). If zero: see Section 7 for fresh-data generation.

### Recording tools
- [ ] Windows: **Xbox Game Bar** (Win+G) — built into Windows 11, captures the focused window plus mic audio. **Recommended.**
- [ ] Alternate: **Loom** browser extension — free tier allows up to 5 min per clip on the free plan (may need to split into 2 clips).
- [ ] Alternate: **OBS Studio** — full control, higher setup cost.
- [ ] Microphone tested: record a 10-second test clip, play it back. Is audio audible? Is the environment quiet?

### Browser and editors
- [ ] VS Code open on `CHW-Chat-Interface` repo
- [ ] Chrome/Edge browser with a fresh profile (no extensions bloating the toolbar)
- [ ] Terminal open at `CHW-Chat-Interface/backend` (for running `psql` or `prisma studio`)
- [ ] Zoom level in VS Code set so line numbers are readable at recording resolution (Ctrl+= two or three times)

### Optional: Playwright for deterministic browser actions
If you want the browser navigation to be scripted rather than manually clicked:
- [ ] Playwright installed: `cd frontend && npx playwright install`
- [ ] Create `demo/scene_3_catcher_flow.spec.ts` (script provided in Section 6)
- [ ] Run with `--headed` so it's visible on screen while Game Bar records

---

## 3. Recording setup

### Recommended: Xbox Game Bar (Windows 11 native)

1. Press **Win+G** to open Game Bar
2. Click the microphone icon — verify mic is ON and not muted
3. Click the camera-with-dot icon (recording) — confirm it captures the **focused window only**
4. Click the gear icon → **Capturing** → set quality to **High** (1080p), frame rate to 30fps, audio bitrate to 128 Kbps
5. Click the recording icon to start, **Win+Alt+R** to stop
6. Recordings save to `C:\Users\athar\Videos\Captures\` as MP4

### Recording hygiene

- **Do the intro twice.** The first take is always stiff. Throw it out.
- **Leave a 3-second pause** between scenes. Post-production cuts are easier at silent gaps.
- **Do not narrate while you type commands.** Type, pause, then narrate. Game Bar audio + keyboard clacks is hard to listen to.
- **Keep the narration above the code**, not below it. You are talking to David, not reading to him.

---

## 4. Test data

### Primary test PDF
`CHW Navigator v1.pdf` — 1.5 MB, ~100 pages, already in repo root. This is the same manual you used during development, so any historical runs in the DB will be against this input.

### Ideal: at least 3 DmnVersion rows from this PDF
Query to verify:
```sql
SELECT id, "chatId", "createdAt", jsonb_array_length(dmn_json->'decisions') AS decision_count
FROM "DmnVersion"
WHERE "chatId" IN (
  SELECT id FROM "Chat" WHERE "guideId" IN (
    SELECT id FROM "Guide" WHERE filename LIKE '%CHW Navigator%'
  )
)
ORDER BY "createdAt" ASC;
```

If this returns ≥3 rows across ≥2 chats, you have empirical reliability data. If not, see Section 7 (fresh data generation).

---

## 5. Scene-by-scene script

**Total runtime target:** 8 minutes. Keep each scene tight. Pause between scenes — do not rush.

### Scene 1 — Intro (30 seconds)

**What to show on screen:** A single text slide or VS Code with this memo open to the top.

**Narration (verbatim):**
> "Hi David. This is the postmortem on CHW-Chat-Interface that I should have sent you on March 18 when we paused the system. I'm recording it now because you asked for empirical evidence before Friday's meeting, and showing you is faster than writing it. This video is 8 minutes. I'm going to walk through the three identification failures we observed running the system, grounded in the code and the Postgres audit trail you can query yourself. You said on March 3 that reliability is necessary for validity but not sufficient. CHW-Chat-Interface had three structural reasons it could not achieve reliability. Here's each one."

---

### Scene 2 — Nine-prompt treatment vector (90 seconds)

**What to show on screen:**
1. VS Code with `backend/src/prompts/` folder expanded in the explorer
2. Open `generator.txt` (77 lines) and scroll the full file
3. Quickly open each catcher: `boundary_condition.txt` (25), `module_architecture.txt` (10), `completeness.txt` (9), `clinical_review.txt` (8), `dmn_audit.txt` (8), `consistency.txt` (7), `provenance.txt` (5)
4. Open `fix_generator.txt` (34 lines)
5. Open `backend/pdf_extractor/prompt.py` (134 lines — note this is a Python file with a prompt string inside)
6. Terminal: `wc -l backend/src/prompts/*.txt backend/pdf_extractor/prompt.py` — show the total: **317 lines across 10 files**

**Narration (verbatim):**
> "The first failure is the treatment vector. CHW-Chat-Interface uses ten distinct prompts. One generator, one fix prompt, seven catcher validators, and one PDF extractor. Three hundred and seventeen lines of prompt text total. Each of these files is an independent degree of freedom. If I edit the generator prompt to improve one clinical rule, I have no way to hold the other nine prompts fixed while I measure the effect. Two engineers tuning this system will diverge across all ten prompts simultaneously, which is exactly what we saw happen in Fall 2025 when Zach and you extracted the same manual through the chat interface and took different numbers of queries. The system moved the variance one layer of abstraction down from 'human prompting style' to 'engineer prompt-tuning style,' but the variance did not shrink. In econometric terms, we have a ten-dimensional treatment vector with roughly five runs per iteration. Ten dimensions, five observations per trial — the system is massively under-identified. We cannot attribute any reliability problem to a specific prompt. That's failure mode one."

---

### Scene 3 — Unobservable mediator (2 minutes)

**What to show on screen (option A — live system):**
1. Frontend running at localhost:3000 (or wherever)
2. Upload `CHW Navigator v1.pdf`, click "Extract DMN"
3. Wait for catcher reports to populate (7 of them in the post-March-18 build, including `module_architecture`)
4. Hover over one report — point at the **THREE** buttons: `[Auto-fix All]`, `[Fix with Instructions]`, `[Ignore]`
5. Click `[Ignore]` on one report, `[Auto-fix All]` on another
6. Switch to VS Code, open `frontend/src/components/CatcherReportCard.tsx`, navigate to **lines 189-211**. Show the three onClick handlers: `onAutoFix(results)`, `setShowInstructions(true)` (which opens the Fix with Instructions flow), and `onIgnore()`.
7. Switch to VS Code, open `backend/prisma/schema.prisma`, navigate to **lines 63-77** (`model Message`). Point at **line 72**: `fixApplied Boolean @default(false)`.
8. **The kill shot:** Say out loud: *"The button is ternary. The audit field is binary. We can tell you a fix happened. We cannot tell you which of the three buttons the user clicked to cause it."*

**What to show on screen (option B — static, if live won't run):**
1. VS Code: open `frontend/src/components/CatcherReportCard.tsx`, jump to lines 189-211. Read the three onClick handlers out loud.
2. VS Code: open `backend/prisma/schema.prisma`, navigate to `model Message` at **line 63**. Read the fields aloud, including `fixApplied Boolean` at **line 72**.
3. State explicitly on camera: "There is no `userAction`, `userChoice`, or `fixVariant` field anywhere in this schema. The user's decision is not in the audit trail."

**Narration (verbatim):**
> "The second failure is what econometricians call an unobservable mediator. When a catcher flags a problem, the user has three choices: Auto-fix All, Fix with Instructions, or Ignore. That decision changes the DMN downstream. Here's where the decision lives in the frontend — `CatcherReportCard.tsx` lines 189 through 211. Three distinct button handlers. [Show the three buttons.] Now here's the database table where we record what happened — `Message` in `schema.prisma` line 63 through 77. Look at line 72: `fixApplied` is a **Boolean**. The user's action space is ternary. The audit field is binary. We can tell you *whether* a fix was applied. We cannot tell you *which of the three buttons the user clicked* to cause it, and we cannot distinguish "user ignored the report" from "user clicked Auto-fix but the model rejected the fix." The mediator isn't just hidden — it's structurally un-recordable in the current schema. In mediation analysis, an unobservable mediator is fundamentally unidentified. We cannot decompose variance into 'model error' versus 'user choice error,' which means we cannot attribute reliability problems to the system itself. The `CONVERGENCE-REPORT.md` from March 18 also flags a related issue — two browser tabs on the same chat can corrupt each other's mediator state. The team documented this as a deferred BLOCKING item that would need either a server-side lock or a cross-tab BroadcastChannel guard. That's failure mode two."

---

### Scene 4 — Non-stationary cache (2 minutes)

**What to show on screen:**
1. VS Code: open `backend/prisma/schema.prisma`, navigate to **lines 49-61** (`model CacheState`). Point specifically at line 49 where `chatId String @id` makes the chat ID the **primary key** — there is no content hash anywhere in this table.
2. VS Code: open `backend/src/routes/cache.py`, navigate to **lines 43-49**. Show the auto-uncache logic: when a new chat claims the cache, the old chat's cache is invalidated. This is per-chat, not per-content.
3. VS Code: open `backend/src/llm/claude_client.py`, navigate to **lines 87-92**. Point at the `cache_control: {"type": "ephemeral", "ttl": "1h"}` block. Note on camera: the team flagged in `CONVERGENCE-REPORT.md` Gotcha #4 that the `ttl` field may not even be a valid Anthropic API format — meaning the cache header could have been silently ignored in production.
4. VS Code: open `CONVERGENCE-REPORT.md` at the "Deferred Issues → Requires Schema Changes (PROTECTED)" section. Read item #5 aloud: *"CacheState lastTurnAt/expiresAt never written — Schema exists but never populated."*
5. If you have live data: Prisma Studio, query `SELECT DISTINCT "chatId", "cachedAt", "lastTurnAt" FROM "CacheState" ORDER BY "cachedAt"`. Show that `lastTurnAt` is NULL for every row. The tracking fields exist but were never wired up.

**Narration (verbatim):**
> "The third failure is non-stationary baseline. We designed prompt caching to stabilize the environment across runs. Here's the cache state schema — `CacheState` model in `schema.prisma` lines 49 through 61. Look at line 49: `chatId String @id`. The chat ID is the **primary key**. There is no content hash anywhere in this table. That means if you upload the same PDF into two different chats, the system creates two completely separate cache rows. Here's the confirming evidence — `cache.py` lines 43 through 49 — when a new chat claims the cache, the old chat's cache is auto-invalidated. Per-chat, not per-content. And here's the ephemeral cache header we were passing to Claude — `claude_client.py` lines 87 through 92, `type: ephemeral, ttl: 1h`. The team flagged this on March 18 as potentially malformed — the `ttl` field may not be a valid Anthropic API key, meaning the cache header might have been silently ignored in production. We never verified which. Now the kill shot: open `CONVERGENCE-REPORT.md` deferred item #5 — `CacheState lastTurnAt and expiresAt never written. Schema exists but never populated.` We designed the tracking fields, we never wired them up. The cache was not just non-stationary across sessions; it was non-observable within a session. Any reliability estimate we computed across runs was biased toward the noise floor of the environment differences, because we couldn't tell whether the cache was working, partially working, or silently broken. That's failure mode three."

---

### Scene 5 — Postgres audit trail walkthrough (2 minutes)

**What to show on screen:**
1. Terminal or Prisma Studio open to the CHW-Chat-Interface database
2. Run three queries in sequence (prepared in advance, pasted from the clipboard):

```sql
-- Q1: How many DMN versions do we have for the same PDF?
SELECT g.filename, COUNT(DISTINCT dv.id) AS version_count, COUNT(DISTINCT c.id) AS chat_count
FROM "Guide" g
JOIN "Chat" c ON c."guideId" = g.id
JOIN "DmnVersion" dv ON dv."chatId" = c.id
WHERE g.filename LIKE '%CHW Navigator%'
GROUP BY g.filename;

-- Q2: For one specific guide, how many rules did each run produce?
SELECT dv.id, dv."chatId", dv."createdAt",
       jsonb_array_length(dv.dmn_json->'decisions') AS decision_count
FROM "DmnVersion" dv
JOIN "Chat" c ON c.id = dv."chatId"
JOIN "Guide" g ON g.id = c."guideId"
WHERE g.filename LIKE '%CHW Navigator%'
ORDER BY dv."createdAt" ASC;

-- Q3: For the most recent run of each chat, what catchers fired and at what severity?
SELECT cr."catcherName", cr."severityLevel", COUNT(*) AS n
FROM "CatcherRun" cr
JOIN "DmnVersion" dv ON dv.id = cr."dmnVersionId"
JOIN "Chat" c ON c.id = dv."chatId"
JOIN "Guide" g ON g.id = c."guideId"
WHERE g.filename LIKE '%CHW Navigator%'
GROUP BY cr."catcherName", cr."severityLevel"
ORDER BY cr."catcherName", cr."severityLevel";
```

**Narration (verbatim):**
> "Here's the evidence you can query yourself. [Run Q1.] We have N runs of the CHW Navigator manual across M chats. [Run Q2.] Here's the rule count per run. You can see the variation across runs of the same PDF. This is the test-retest reliability test you asked about on March 3, and the variance here is the answer. [Run Q3.] And here's what the catchers flagged. Notice that the same catchers fire different severities on different runs of the same PDF. If the system were reliable, these counts would be identical across runs. They aren't. The audit trail is queryable and reproducible. I can give you read-only access to this database if you want to run queries yourself."

---

### Scene 5b — The March 18 audit trail (90 seconds) ⭐ STRONGEST EVIDENCE

**What to show on screen:**
1. VS Code: open `overnight_loop_2026-03-18/CONVERGENCE-REPORT.md`
2. Scroll through the summary: *"15 audit agents run sequentially, 10 unique BLOCKING identified, 8 fixed, 2 deferred"*
3. Scroll to the "Deferred Issues" section. Read aloud:
   - **Item #5** (State Lifecycle B3): *"CacheState lastTurnAt/expiresAt never written — Schema exists but never populated."*
   - **Item #6** (Observability B1): *"Generator streaming calls never logged — Need to capture usage from Anthropic stream events."*
   - **Item #2** (Causal Chain B2): *"Two-tab same-chat corruption — Needs either server-side full-request lock or cross-tab BroadcastChannel guard."*
4. Scroll to "Gotchas Discovered" section. Read aloud item #4: *"`cache_control: {'type': 'ephemeral', 'ttl': '1h'}` — the `ttl` field may not be valid Anthropic API format."*
5. Scroll to "Test Results": *"Before: 10/10 (3 test files). After: 54/54 (8 test files)."*

**Narration (verbatim):**
> "Before I close, I want to show you the contemporaneous evidence that these failures aren't retrospective rationalizations. On March 18, the team ran a 15-agent sequential audit cycle against CHW-Chat-Interface. The report is in `overnight_loop_2026-03-18/CONVERGENCE-REPORT.md`. Fifteen audit agents found ten unique BLOCKING issues. We fixed eight. Two were deferred. Several of the deferred items map directly to the three failure modes I just walked through. Item five — `CacheState lastTurnAt and expiresAt never written, schema exists but never populated` — that's the non-stationary baseline evidence. Item six — `Generator streaming calls never logged` — that means for the main generator, the most important prompt in the system, we couldn't even audit which prompt fired on which turn, which is the treatment-vector observability problem. Causal Chain item B2 — `Two-tab same-chat corruption` — that's the mediator problem made worse by a race condition. And Gotcha number four — `the ttl field may not be valid Anthropic API format` — the cache header we designed might have been silently ignored in production. The test count went from ten to fifty-four. This is a disciplined engineering process, not a cowboy pivot. When I reviewed this report between March 18 and March 31 and concluded the architecture had structural measurement problems we couldn't fix in place, that's when I started the ORCHESTRATOR.md pivot proposal. The timeline is: audit cycle March 18, review period March 19 through 30, pivot proposal March 31, your approval April 2, your reversal April 7. What's new here — and what I should have sent you on March 19 — is the audit artifact itself."

---

### Scene 6 — Conclusion (30 seconds)

**What to show on screen:** Split screen or VS Code with the memo draft visible.

**Narration (verbatim):**
> "To be clear about what this video is and isn't. It is not me saying CHW-Chat-Interface was wrong to build — it was the right next step after the Fall chat experiments, and the team shipped real infrastructure and real audit discipline. It IS me showing you three structural reasons it could not reach the reliability bar we need for CHT, each grounded in specific code and validated by our own March 18 audit. Each of these three failures maps to something the REPL substrate addresses by design: one prompt instead of ten, no user in the loop during extraction, and content-hashed PDF as a stationary starting state. The memo I sent today walks through how the REPL hybrid keeps the intermediate artifacts visible while fixing the three identification failures. I owe you this walkthrough three weeks earlier than I'm giving it to you. I'm happy to walk through any of this in person on Friday. Thank you."

---

## 6. Optional: Playwright script for Scene 3

If you want Scene 3 (catcher flow) to be scripted rather than manually clicked, create this file in `CHW-Chat-Interface/demo/scene_3_catcher_flow.spec.ts`:

```typescript
import { test, expect } from '@playwright/test';

test('scene 3 — catcher flow with mediator highlight', async ({ page }) => {
  // Assumes dev server at localhost:3000, Clerk dev session already authed
  await page.goto('http://localhost:3000');

  // Upload the test PDF
  await page.getByRole('button', { name: /upload|new chat/i }).click();
  const fileChooserPromise = page.waitForEvent('filechooser');
  await page.getByText(/attach.*pdf/i).click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles('../CHW Navigator v1.pdf');

  // Wait for generation to complete and catchers to fire
  await page.waitForSelector('[data-testid="catcher-report-card"]', { timeout: 120_000 });

  // Demonstrate the mediator: hover each report, do nothing, just let David see the buttons
  const reports = page.locator('[data-testid="catcher-report-card"]');
  const count = await reports.count();
  for (let i = 0; i < count; i++) {
    await reports.nth(i).hover();
    await page.waitForTimeout(800); // slow pan for narration
  }

  // Click [Ignore] on the first, [Auto-fix] on the second — DO show the choice being made
  await reports.nth(0).getByRole('button', { name: /ignore/i }).click();
  await page.waitForTimeout(500);
  await reports.nth(1).getByRole('button', { name: /auto.?fix/i }).click();

  // Pause long enough for narration to say "and this choice is not captured in the audit trail"
  await page.waitForTimeout(4000);
});
```

Run with: `npx playwright test demo/scene_3_catcher_flow.spec.ts --headed --project=chromium`

While it runs, Xbox Game Bar (or your recorder of choice) is capturing the screen.

---

## 7. Fresh data generation (if historical runs are missing)

If Section 4's query returns zero DmnVersions for `CHW Navigator v1.pdf`, you need to generate fresh runs before recording. Plan:

1. **Run 1:** New chat, upload `CHW Navigator v1.pdf`, run extraction to completion. Do NOT touch catcher reports — let them fire, record the output as-is.
2. **Run 2:** New chat (fresh session, different cache state), same PDF. Same extraction. Do not touch reports.
3. **Run 3:** New chat, same PDF, but this time click `[Auto-fix All]` on every catcher report. Record the result.

Runs 1 and 2 give you **test-retest reliability evidence** (same input, no intervention). Run 3 vs Runs 1/2 gives you **mediator evidence** (same input, different user choices).

Time budget: ~15 minutes per run = 45 minutes of setup before recording.

This is also the fallback plan if the dev env spins up but historical data is missing.

---

## 8. Fallback: No dev env (pure static walkthrough)

If CHW-Chat-Interface won't run locally in under an hour, switch to a **code-only walkthrough**. You lose the Postgres queries and the live UI but keep the three failure modes. Total runtime: ~6 minutes.

Scenes adapted:
- **Scene 2 (prompts):** unchanged — just VS Code.
- **Scene 3 (mediator):** skip the live UI, go directly to `CatcherFeedback.tsx` onClick handlers + `CatcherRun` schema. Narrate the absence of the `userAction` field.
- **Scene 4 (cache):** unchanged — schema walkthrough.
- **Scene 5 (audit trail):** replace the Postgres queries with a read of `backend/prisma/schema.prisma` showing the three tables exist and could be queried. State that you can run the queries live on Friday.

The static version is weaker evidence but still much stronger than the memo alone. Any evidence beats no evidence.

---

## 9. Post-production

- [ ] Open the MP4 in any free video editor (DaVinci Resolve Free, Shotcut, or even Windows Video Editor)
- [ ] Trim the first 3 seconds and last 3 seconds (recorder noise)
- [ ] Cut any long dead air between scenes (keep pauses to ≤1 second)
- [ ] Export as 1080p MP4, target ≤50 MB
- [ ] Upload to: Google Drive (Berkeley account), get a "Anyone with the link can view" URL
- [ ] Paste the URL into the email as a new section: "**8. Recorded walkthrough**: I've recorded an 8-minute video walking through each of the three failure modes with code and database evidence. It's at [URL]. I'd rather you watch this than read more text on the failure modes."

---

## 10. Checklist: ready to record?

Run through this list one more time right before hitting record. If any item is a No, stop and fix it first.

**Files to have open in VS Code tabs, in this order (left to right):**
- [ ] `backend/src/prompts/generator.txt` — for Scene 2
- [ ] `backend/src/prompts/fix_generator.txt` — for Scene 2
- [ ] `backend/src/prompts/boundary_condition.txt` — for Scene 2
- [ ] `backend/src/prompts/module_architecture.txt` — for Scene 2 (new as of March 18)
- [ ] `backend/src/prompts/completeness.txt` — for Scene 2
- [ ] `backend/src/prompts/clinical_review.txt` — for Scene 2
- [ ] `backend/src/prompts/dmn_audit.txt` — for Scene 2
- [ ] `backend/src/prompts/consistency.txt` — for Scene 2
- [ ] `backend/src/prompts/provenance.txt` — for Scene 2
- [ ] `backend/pdf_extractor/prompt.py` — for Scene 2
- [ ] `frontend/src/components/CatcherReportCard.tsx` scrolled to line 189 — for Scene 3
- [ ] `backend/prisma/schema.prisma` scrolled to line 63 (Message model) — for Scene 3
- [ ] `backend/prisma/schema.prisma` second window scrolled to line 49 (CacheState model) — for Scene 4
- [ ] `backend/src/routes/cache.py` scrolled to line 43 — for Scene 4
- [ ] `backend/src/llm/claude_client.py` scrolled to line 87 — for Scene 4
- [ ] `overnight_loop_2026-03-18/CONVERGENCE-REPORT.md` — for Scene 5b

**Running systems:**
- [ ] Dev env: backend running, frontend running, DB reachable
- [ ] Prisma Studio OR psql open to the correct DB
- [ ] Queries in Section 5 Q1/Q2/Q3 are prepared in a text file ready to copy-paste
- [ ] Chrome on a clean profile with localhost:3000 already loaded
- [ ] `CHW Navigator v1.pdf` ready to drag into the uploader

**Recording setup:**
- [ ] VS Code zoom level readable (Ctrl+= 2-3 times)
- [ ] Terminal visible and font-zoomed
- [ ] Mic tested, environment quiet
- [ ] Game Bar or recorder configured, recording to `C:\Users\athar\Videos\Captures\`

**Rehearsal:**
- [ ] You have done a DRY RUN of Scene 2 (the prompt walkthrough). Narration is comfortable.
- [ ] You have done a DRY RUN of Scene 3, specifically the "ternary action, binary log" kill shot
- [ ] You have a second monitor (or phone) with this plan open so you can reference the narration scripts without breaking flow

---

## 11. Upload target and email integration

When done, the email update is a single additional section at the bottom of the memo, inserted between Section 6 ("The ask") and Section 7 ("What I need from you"):

```markdown
---

**Recorded walkthrough (8 min).** I've recorded a walkthrough of the three identification failures I described in Section 2, with live code references and database queries against the actual CHW-Chat-Interface Postgres audit trail. It is at [LINK]. I'd prefer you watch this before the Friday meeting — showing is faster than arguing, and the evidence is stronger than my summary of it. If the link doesn't load, I'll bring my laptop to the meeting and walk through it live.
```

Renumber the existing Section 7 accordingly.

---

## 12. Notes and risks

- **Risk: DB has no historical runs.** Mitigate by running fresh data (Section 7) before recording. Budget 45 extra minutes.
- **Risk: Dev env won't spin up.** Mitigate with the static walkthrough (Section 8). Still record it — any evidence beats none.
- **Risk: You ramble in Scene 2 and go over 8 minutes.** Mitigate by recording in 6 separate takes (one per scene) and stitching in post. This also lets you redo individual scenes if you flub.
- **Risk: David watches it on his phone and can't read the code.** Mitigate by increasing VS Code zoom before recording so line numbers are readable at 720p playback.
- **Risk: The recording file is too large to upload.** Mitigate by re-exporting at 720p instead of 1080p, or trimming pauses more aggressively.
- **Risk: You feel defensive while narrating.** Re-read the "single sentence anchor" in Section 0. You are delivering a postmortem, not a vindication. The tone is "here is what we built, here is what we measured, here is what we learned." Do not say "I was right." Do not say "this proves the pivot." Let the evidence do the work.

---

## 13. After the meeting

If the video does its job and David signs off on Friday, delete this file and move on. If not, the three failure modes are already documented and the video can be re-shared or re-cut for any downstream audience (CHT developers, the broader Navigators team). Either way, the postmortem now exists as an artifact, which is the thing that should have existed on March 18.
