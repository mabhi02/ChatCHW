# CHW Navigator — runbook

Fastest path to a successful local run with expected outputs.

For implementation details and caveats, use `Product/ARCHITECTURE.md`.

---

## 1) Prerequisites

- Clone with submodules:
  - `git submodule update --init --recursive`
- Backend env configured (`Product/backend/.env`) with required keys/services.
- Frontend env configured (`Product/frontend/.env.local`).

Reference: `Product/ARCHITECTURE.md` setup appendix.

---

## 2) Start services

Backend:

```bash
cd Product/backend
python main.py
```

Frontend:

```bash
cd Product/frontend
npm install
npm run dev
```

---

## 3) First successful run (happy path)

1. Open UI (`http://localhost:3000` unless overridden).
2. Provide Anthropic API key in UI (BYOK flow).
3. Upload a test guide PDF.
4. Run extraction.
5. Wait for completion; download artifacts.

Expected outputs:

- `clinical_logic.json`
- `clinical_logic.dmn`
- `form.xlsx`
- `flowchart.md`
- CSV exports (predicates/phrases)

---

## 4) Verification checklist

- Open `clinical_logic.dmn` in a DMN tool to confirm parse.
- Open `form.xlsx` and inspect survey/calculate sheets.
- Open `flowchart.md` to verify graph renders and is connected.
- Run backend tests:

```bash
cd Product/backend
pytest -v
```

- Run at least one cross-check harness in `Testing/`:
  - `Testing/Gigi/README.md` recommended first.

---

## 5) Failure triage order

1. Runtime exceptions in backend logs.
2. Missing/invalid env vars and credentials.
3. Submodule mismatch (`Product/` not initialized).
4. Ingestion failures (PDF/layout/vision path).
5. Extraction failures (labeling, REPL, validators).
6. Converter-specific failures (DMN/XLSForm/Mermaid).

Then review:

- `Product/issues.txt` (audit hypotheses; verify against current code),
- `handoff.md` open risks and known gaps.

---

## 6) Operational handoff checklist

- [ ] One successful local extraction and artifact download
- [ ] `pytest` green in `Product/backend`
- [ ] One `Testing/` harness run captured
- [ ] Known risks acknowledged (see `handoff.md`)
- [ ] Source-of-truth docs reviewed (`Product/ARCHITECTURE.md`, `Product/PIPELINE.md`)

