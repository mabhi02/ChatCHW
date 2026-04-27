# plan_RLM.md -- Claude Code Build Plan for CHW Navigator

> **What this document is:** A complete build plan designed to be passed to Claude Code alongside ORCHESTRATOR.md, the CHW Navigator v1 PDF, and the RLM library repo (https://github.com/alexzhang13/rlm). Claude Code should read all four and scaffold the full project.
>
> **What Claude Code should do:** Read this plan top to bottom, reference the companion docs at the sections indicated, and build each component in the order specified.

---

## 0. Companion Documents (Read These First)

Before writing any code, Claude Code must read:

1. **ORCHESTRATOR.md** -- Architecture spec. Section 0 has the 8-step pipeline overview. Section 2 has the full architecture diagram. Section 4 has the JSON schema. Section 5 has the system prompt structure. Section 6 has validator specs. Section 9 has failure modes.

2. **CHW_Navigator_v1.pdf** -- The professor's full spec. Key sections to study:
   - Pages 12-22: Variable naming codebook (ADOPT THIS, it's more complete than ORCHESTRATOR.md's version). Includes prefixes the orchestrator missed: `hx_`, `sys_`, `prev_`, `calc_`, `s_`, `sev_`, `need_`, `rx_`, `proc_`, `err_`.
   - Pages 24-45: Prompt fragments and master prompts (reusable clinical extraction logic for the system prompt).
   - Pages 46-52: Two-Table DMN architecture (Activator COLLECT + Router FIRST). This IS the Activator/Router pattern from ORCHESTRATOR.md.
   - Pages 89-94: Schema definitions (symbols, phrase bank, context, predicates, factsheet, synthetic patients).
   - Pages 112-142: Z3 verification code. Already written. Use directly for the `z3_check()` REPL function.

3. **RLM Library (https://github.com/alexzhang13/rlm)** -- The reference implementation:
   - `rlm/` directory: Core REPL loop, sandboxing, sub-call management. **Use this library (`pip install rlms`) instead of building the REPL from scratch.**
   - `rlm/clients/`: Backend adapters for OpenAI, Anthropic, etc. We use the Anthropic client.
   - `examples/`: Usage patterns for RLM completions.
   - `visualizer/`: Built-in trajectory visualizer (Node.js + shadcn/ui). Can be adapted for our web interface or used as-is for debugging.
   - `rlm/logger.py`: `RLMLogger` class that writes JSONL trajectory logs. Connect this to Neon for persistence.

---

## 1. The 8-Step Pipeline (from ORCHESTRATOR.md Section 0)

1. PDF --> JSON (external, assumed done)
2. REPL session: guide loaded as variable, Claude Opus writes code to extract
3. Model builds JSON output in REPL variables
4. Validators + Z3 run INSIDE the REPL (model calls validate() and z3_check())
5. Errors fed back in stdout, model fixes in same session
6. Model iterates until all checks pass
7. FINAL_VAR returns validated JSON
8. Deterministic converters: JSON --> DMN XML + Mermaid + XLSForm XLSX

---

## 2. Technology Stack

### Backend (Render)
- **Runtime:** Python 3.12, FastAPI
- **RLM Library:** `pip install rlms` -- provides REPL loop, sandboxing, sub-call management, logging
- **LLM:** Claude Opus (root), Claude Sonnet (sub-calls via llm_query), BYOK via Anthropic API
- **Sandbox:** Use RLM library's `DockerREPL` environment for isolation (or `LocalREPL` for dev). See `rlm/` source for implementation.
- **Database:** Neon Postgres (shared logging of trajectories, run metadata)
- **Queue:** QStash (Upstash) for long-running REPL sessions that exceed HTTP timeout
- **Converters:** openpyxl (XLSX), xml.etree (DMN XML), string templates (Mermaid)

### Frontend (Vercel)
- **Framework:** React + TypeScript
- **Styling:** Tailwind
- **Real-time:** SSE (Server-Sent Events) for REPL trajectory streaming
- **Session:** BYOK pattern from CRM-RecSys (API key in memory, never persisted)

### Infrastructure
- **Neon Postgres:** Run metadata + REPL step logs (shared across team)
- **Upstash Redis:** Session state for active REPL sessions (TTL-based expiry)
- **QStash:** Trigger long-running extraction jobs, handle Render timeout limits
- **Vercel:** Frontend deployment
- **Render:** Backend deployment (Docker for RLM sandboxing)

---

## 3. Project Structure

```
chw-navigator/
  README.md
  ORCHESTRATOR.md                    # Architecture spec (companion doc)
  plan_RLM.md                        # This file

  reference/
    CHW_Navigator_v1.pdf             # Professor's full spec
    rlm_paper.pdf                    # RLM paper
    who_2012_guide.json              # Test manual (PDF->JSON already done)
    grading_rubric.csv               # 16 clinical pathways ground truth
    patientbot_1000.csv              # 1000 test patients

  backend/
    pyproject.toml                   # Dependencies: rlms, fastapi, anthropic, openpyxl, z3-solver, asyncpg, httpx
    Dockerfile                       # For Render deployment (needs Docker for RLM sandbox)

    server.py                        # FastAPI app: /api/session/*, SSE streaming
    session_manager.py               # BYOK session lifecycle (create, stream, artifacts, cleanup)
    rlm_runner.py                    # Wraps rlm library: configures RLM with our system prompt, validators, z3

    schema.py                        # JSON schema for clinical_logic output
    system_prompt.py                 # Builds the system prompt from fragments

    prompts/
      system_generate.txt            # Full system prompt for REPL session
      fragments/                     # Reusable prompt pieces (adapted from CHW Navigator pp.24-31)
        role_knowledge_engineer.txt
        role_dmn_architect.txt
        std_logic_integrity.txt
        std_safe_endpoints.txt
        std_missingness_model.txt
        std_antigravity_data.txt
        std_queue_management.txt
        std_dmn_subset.txt
        footer_safety.txt

    validators/
      __init__.py                    # run_all_validators(logic: dict) -> list[str]
      architecture.py                # Activator/Router structure (see ORCHESTRATOR.md Section 6.2)
      completeness.py                # Cross-reference checks
      clinical.py                    # Boolean-only, default rows, fail-safe sanity
      naming.py                      # Variable prefix conventions (use CHW Navigator pp.12-22 codebook)

    z3_verifier/
      __init__.py                    # verify_clinical_logic(logic: dict) -> dict
      dmn_to_z3.py                   # Compile JSON decision tables to Z3 constraints
      evergreen_tests.py             # Adapted from CHW Navigator pp.112-128 (already written)
      patient_gen.py                 # Z3-generated synthetic patients (CHW Navigator pp.132-138)

    converters/
      json_to_dmn.py                 # JSON --> DMN 1.3 XML
      json_to_xlsx.py                # JSON --> CHT XLSForm (predicate compilation, state management)
      json_to_mermaid.py             # JSON --> Mermaid flowchart
      json_to_csv.py                 # JSON --> flat CSVs (predicates.csv, phrase_bank.csv) for review

    db/
      neon.py                        # Neon connection, logging functions
      schema.sql                     # Table definitions (extraction_runs, repl_steps)

  frontend/
    package.json
    tsconfig.json
    next.config.js                   # Next.js on Vercel

    src/
      app/
        page.tsx                     # Landing: paste API key, upload guide
        session/[id]/page.tsx        # Live REPL trajectory view

      components/
        ApiKeyInput.tsx              # BYOK field (memory only, never persisted)
        GuideUpload.tsx              # JSON file upload
        ReplStream.tsx               # SSE consumer: live code/stdout/validation display
        ValidationPanel.tsx          # Green/red per check
        Z3Panel.tsx                  # Exhaustiveness/reachability proof results
        ArtifactDownloads.tsx        # Download JSON, DMN XML, XLSX, Mermaid, CSVs
        CostTracker.tsx              # Token count, sub-call count, estimated cost

      hooks/
        useSession.ts                # BYOK session management (REFERENCE CRM-RecSys for this pattern)
        useSSE.ts                    # SSE stream consumption

      lib/
        api.ts                       # Backend API client (key in Authorization header, never stored)
```

---

## 4. Build Order (What to Build and When)

### Phase 1: Core Infrastructure (Days 1-3)

**Step 1: Schema + Validators**
- Build `schema.py` with the JSON schema from ORCHESTRATOR.md Section 4
- Build all 4 validators from ORCHESTRATOR.md Section 6
- Write unit tests with hand-crafted fixture JSON (one valid, several intentionally broken)
- Naming validator MUST use the full prefix taxonomy from CHW Navigator pp.12-22, not the abbreviated version

**Step 2: RLM Integration**
- `pip install rlms`
- Build `rlm_runner.py` that configures an RLM instance:
  - Backend: Anthropic (Claude Opus root, Claude Sonnet sub-calls)
  - Environment: `DockerREPL` for production, `LocalREPL` for dev
  - Logger: `RLMLogger` that also writes to Neon
- Expose `validate()` and `z3_check()` as callable functions in the REPL namespace
- Load the JSON guide as the `guide` variable in the REPL
- Reference the RLM library's `rlm/` source for how the REPL loop works, how `llm_query()` is implemented, how FINAL_VAR is handled
- Reference the RLM library's `examples/` for usage patterns

**Step 3: Neon Logging**
- Build `db/schema.sql` with two tables: extraction_runs and repl_steps (see ORCHESTRATOR.md Section 10)
- Build `db/neon.py` with async logging functions
- Connect RLMLogger output to Neon writes

### Phase 2: System Prompt (Days 4-7)

**Step 4: System Prompt Construction**
- Build `system_prompt.py` that assembles the prompt from fragments
- Adapt prompt fragments from CHW Navigator pp.24-31:
  - `role_knowledge_engineer.txt` (from role_knowledge_engineer fragment)
  - `role_dmn_architect.txt` (from role_dmn_architect fragment)
  - `std_logic_integrity.txt`, `std_safe_endpoints.txt`, `std_missingness_model.txt`
  - `std_antigravity_data.txt`, `std_queue_management.txt`, `std_dmn_subset.txt`
  - `footer_safety.txt` (from footer_safety_general and footer_safety_dmn)
- Add REPL-specific instructions from ORCHESTRATOR.md Section 5.1-5.4:
  - How to use the REPL environment (guide variable, llm_query, validate, z3_check)
  - The 7-phase extraction strategy (scan, schema, per-module, assemble, phrase bank, validate, return)
  - Boolean-only convention, fail-safe rules, provenance requirements
- **CRITICAL: The prompt must be fully manual-agnostic.** No WHO 2012 specifics. See ORCHESTRATOR.md Section 5 header.

**Step 5: Test on WHO 2012**
- Run the pipeline on who_2012_guide.json
- Iterate on the system prompt until validators pass
- Run 3 times for determinism check

### Phase 3: Z3 Verification (Days 6-8, parallel with prompt iteration)

**Step 6: Z3 Integration**
- Adapt the Z3 code from CHW Navigator pp.112-128 into `z3_verifier/`
- Key functions to implement:
  - `test_domain_satisfiable`: DOMAIN must be SAT
  - `test_rule_reachability`: every rule is reachable
  - `test_table_exhaustive`: no input gaps
  - `test_unique_mece`: mutual exclusivity for UNIQUE tables
  - `test_first_priority_shadowing`: no dead rules in FIRST tables
  - `build_router_graph` + `find_cycles_in_graph`: no routing loops
- Build `dmn_to_z3.py` that compiles JSON decision tables to Z3 constraints
- Build `patient_gen.py` using the approach from CHW Navigator pp.132-138 (Z3 generates synthetic patients, not humans)
- Expose as `z3_check()` in the REPL

### Phase 4: Converters (Days 7-10, parallel)

**Step 7: json_to_dmn.py**
- Map JSON modules to DMN 1.3 XML decision elements
- Activator gets hitPolicy="COLLECT", Router and modules get hitPolicy="FIRST"
- Provenance goes in rule description/annotation elements
- Use xml.etree.ElementTree

**Step 8: json_to_xlsx.py (HARDEST CONVERTER)**
- See ORCHESTRATOR.md Section 8.2 for XLSForm survey sheet structure
- Key challenges (from CHW Navigator pp.104-105):
  - The Boolean Trap: ODK evaluates string("false") as truthy. Use numeric 1/0, NEVER string true/false.
  - Predicate compilation with fail-safes: `if(measured, threshold_logic, fail_safe_value)`
  - Module state management: completed_modules bitmask with delimiter-safe set (avoid substring collisions)
  - Module-gated relevance: `relevant: contains(${required_modules}, 'mod_X') and not(contains(${completed_modules}, '|mod_X|'))`
- Use openpyxl

**Step 9: json_to_mermaid.py**
- Modules as subgraphs, rules as decision paths, emergency path in red
- See CHW Navigator pp.41-42 (D5_make prompt) for the Mermaid flowchart spec

**Step 10: json_to_csv.py**
- Trivial: flatten predicates and phrase_bank arrays to CSV for human review

### Phase 5: Web Interface (Days 9-13)

**Step 11: Backend Server**
- Build `server.py` with FastAPI endpoints:
  - `POST /api/session/start` -- receives API key (in Authorization header) + guide JSON, starts REPL session
  - `GET /api/session/{id}/stream` -- SSE endpoint streaming REPL steps
  - `GET /api/session/{id}/artifacts` -- download completed outputs
  - `GET /api/session/{id}/status` -- poll for completion
  - `POST /api/session/{id}/cancel` -- abort a running session
- The API key flows: frontend --> Authorization header --> session_manager --> RLM backend_kwargs --> REPL llm_query()
- The API key is NEVER written to Neon, Redis, or any persistent store
- For long-running sessions that exceed Render's HTTP timeout: use QStash to trigger the extraction job asynchronously, SSE streams from the job's progress stored in Redis

**Step 12: Session Manager**
- `session_manager.py` manages the lifecycle:
  - Create: generate session ID, store API key in Upstash Redis (TTL 2 hours, encrypted at rest)
  - Run: launch RLM session in background task, stream steps to SSE
  - Artifacts: after FINAL_VAR, run converters, store outputs temporarily
  - Cleanup: clear API key from Redis, clear temp files
- **REFERENCE CRM-RecSys for this pattern.** The CRM-RecSys project implements the same BYOK session lifecycle: key in memory, passed per-request, cleared on close. Study that codebase's session hook and key-in-header pattern.

**Step 13: Frontend**
- Next.js app on Vercel
- Landing page: API key input + guide upload
- Session page: live REPL trajectory (code blocks, stdout, validation results, Z3 proofs)
- Download panel: JSON, DMN XML, XLSX, Mermaid, CSVs
- Cost tracker: tokens used, sub-calls made, estimated cost
- **REFERENCE CRM-RecSys for `useSession.ts` hook** (key in React state, passed via Authorization header, cleared on unmount/tab close)

### Phase 6: End-to-End Testing (Days 12-14)

**Step 14: Run on WHO 2012**
- Full pipeline: upload guide JSON --> REPL extraction --> validation --> Z3 --> converters --> download
- Compare DMN output against hand-built DMNs from the medical team (if available)
- Determinism check: 3 runs, compare JSON hashes

**Step 15: Test Harness (External)**
- Feed form.xlsx + patientbot_1000.csv into the test harness
- Target: 1000/1000 correct diagnoses + treatments

---

## 5. How the RLM Library Changes the Build

The RLM library (`pip install rlms`) eliminates the need to build:
- The REPL execution loop (it's in `rlm/run_rlm.py`)
- The sandbox/security model (use `DockerREPL` or `LocalREPL`)
- Stdout truncation and history trimming (built into the library)
- Sub-call management and budgeting (built into the library)
- FINAL_VAR handling (built into the library)
- Trajectory logging (built into `RLMLogger`)

What we DO need to build:
- The system prompt (clinical extraction logic)
- The validators (exposed as REPL functions)
- The Z3 verifier (exposed as REPL function)
- The converters (post-REPL, deterministic Python)
- The web interface (BYOK session, SSE streaming, artifact downloads)
- The Neon logging bridge (connect RLMLogger output to Postgres)
- The QStash integration (for long-running sessions)

The RLM library's built-in visualizer (`visualizer/` directory, Node.js + shadcn/ui) can also serve as a reference or starting point for the trajectory view in our frontend.

---

## 6. Key Implementation Details

### 6.1 Configuring the RLM Instance

The core setup in `rlm_runner.py`:

- Create an `RLM` instance with `backend="anthropic"`, model set to Claude Opus
- Set `environment="docker"` for production (isolated) or `environment="local"` for dev
- Pass `logger=RLMLogger(log_dir="./logs")` for trajectory logging
- Before starting the REPL session, inject our custom functions into the REPL namespace:
  - `validate` = our validators wrapped as a callable
  - `z3_check` = our Z3 verifier wrapped as a callable
  - `guide` = the loaded JSON guide
- The system prompt goes in the RLM's system message
- The initial user message provides guide metadata (size, keys, section count, preview)

Study the RLM library's `rlm/` source code to understand:
- How `llm_query()` is exposed to the REPL (it's the sub-call mechanism)
- How FINAL_VAR is detected and returned
- How stdout is truncated
- How history is managed across iterations
- How the Anthropic client is configured in `rlm/clients/`

### 6.2 Connecting RLMLogger to Neon

The RLM library's `RLMLogger` writes JSONL files. We need to also push each log entry to Neon in real-time for team visibility and SSE streaming:
- Subclass or wrap `RLMLogger`
- On each log entry, async-write to Neon's `repl_steps` table
- On session completion, write to `extraction_runs` table
- The SSE endpoint reads from Neon (or from an in-memory buffer) to stream to the frontend

### 6.3 QStash for Long-Running Sessions

REPL extraction sessions can take 5-15 minutes. Render's HTTP timeout is typically 30s-5min depending on plan. Use QStash to decouple:
- Frontend calls `POST /api/session/start`
- Backend creates session in Redis + Neon, returns session ID immediately
- Backend publishes a QStash message to trigger the extraction job
- QStash calls a background endpoint `POST /api/internal/run-extraction`
- That endpoint runs the RLM session, writes progress to Neon
- Frontend polls via SSE from `GET /api/session/{id}/stream` which reads from Neon

### 6.4 The System Prompt Assembly

The system prompt is assembled from fragments (adapted from CHW Navigator pp.24-31) plus REPL-specific instructions (from ORCHESTRATOR.md Section 5). The assembly:

1. Role definition (adapted from `role_knowledge_engineer` + `role_dmn_architect` fragments)
2. REPL environment instructions (guide variable, llm_query, validate, z3_check, FINAL_VAR)
3. Output JSON schema description (from ORCHESTRATOR.md Section 4)
4. Clinical architecture (Activator/Router/Module/Integrative from ORCHESTRATOR.md Section 5.3)
5. Variable naming codebook (from CHW Navigator pp.12-22, the FULL version)
6. Predicate conventions (boolean-only, fail-safes, from ORCHESTRATOR.md Section 5.4)
7. Safety standards (from CHW Navigator fragments: std_logic_integrity, std_safe_endpoints, std_missingness_model, std_dmn_subset)
8. Extraction strategy (7-phase from ORCHESTRATOR.md Section 5.2)
9. Provenance requirements
10. Safety footer (from CHW Navigator footer_safety_general + footer_safety_dmn)

**ALL of this must be manual-agnostic.** The prompt defines conventions and architecture. The model discovers clinical content from the guide.

---

## 7. Environment Variables

```
# Backend (.env on Render)
NEON_DATABASE_URL=postgresql://...          # Shared team database
UPSTASH_REDIS_URL=https://...               # Session state
UPSTASH_REDIS_TOKEN=...
QSTASH_TOKEN=...                            # Background job queue
QSTASH_CURRENT_SIGNING_KEY=...
QSTASH_NEXT_SIGNING_KEY=...

# Frontend (.env on Vercel)
NEXT_PUBLIC_API_URL=https://chw-navigator-api.onrender.com

# NOT stored anywhere (BYOK, per-session):
# ANTHROPIC_API_KEY -- passed by user via Authorization header, held in memory only
```

---

## 8. Neon Schema

Two tables:

**extraction_runs:** id (UUID), run_id (TEXT), manual_name, model, started_at (TIMESTAMPTZ), completed_at, status (running/passed/failed/halted), total_iterations (INT), total_subcalls (INT), validation_errors (INT), final_json (JSONB), cost_estimate_usd (FLOAT).

**repl_steps:** id (UUID), run_id (TEXT FK), step_number (INT), step_type (exec/subcall/validate/z3/final), code (TEXT), stdout (TEXT), stderr (TEXT), prompt (TEXT), response (TEXT), execution_ms (INT), input_tokens (INT), output_tokens (INT), created_at (TIMESTAMPTZ). Index on run_id.

---

## 9. API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | /api/session/start | API key in Authorization header | Upload guide JSON, start extraction. Returns session_id. |
| GET | /api/session/{id}/stream | Session token | SSE stream of REPL steps (code, stdout, validation, z3) |
| GET | /api/session/{id}/status | Session token | Poll: running/passed/failed + progress counts |
| GET | /api/session/{id}/artifacts | Session token | Download links for all output files |
| GET | /api/session/{id}/artifacts/{type} | Session token | Download specific: json, dmn, xlsx, mermaid, predicates_csv, phrases_csv |
| POST | /api/session/{id}/cancel | Session token | Abort running session |
| POST | /api/internal/run-extraction | QStash signature | Internal: triggered by QStash to run the actual REPL session |

---

## 10. What This Does NOT Include

- PDF-to-JSON extraction (assumed done externally)
- User accounts or persistent auth (BYOK is the auth)
- Translation / localization of phrase bank
- CHT deployment integration (XLSX output feeds into CHT tooling)
- VLM extraction from PDF images/flowcharts
- The test harness itself (external consumer of form.xlsx)

---

## 11. Timeline Summary

| Phase | Days | What |
|-------|------|------|
| 1: Schema + Validators + RLM integration + Neon | 1-3 | Core infrastructure |
| 2: System prompt + WHO 2012 testing | 4-7 | The hard part (prompt iteration) |
| 3: Z3 verification | 6-8 | Parallel with prompt work |
| 4: Converters (DMN, XLSX, Mermaid) | 7-10 | Parallel |
| 5: Web interface (backend + frontend) | 9-13 | BYOK, SSE, downloads |
| 6: End-to-end testing | 12-14 | Full pipeline validation |
| **Total** | **~14-18 days** | |

The RLM library saves ~3-4 days vs building the REPL from scratch.
