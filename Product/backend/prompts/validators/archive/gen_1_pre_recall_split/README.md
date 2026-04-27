# Generation 1 catcher prompts (archived 2026-04-11)

These are the original LLM catcher prompts used from project inception
through gate generation 1. They were replaced because they conflated four
different kinds of checks into one "critical_issues" list, which caused
runs to fail for reasons unrelated to the stated PASS criterion.

## What was wrong

The old prompts mixed:

1. **Real recall** — "guide mentions X, artifact lacks X" ← the only thing
   that SHOULD fail a run per the stated goal of "covering everything in
   the PDF"
2. **Style / naming conventions** — `fail_safe` defaults, `kind` field
   values, prefix rules (q_/ex_/v_/lab_/img_)
3. **Architecture rules** — router row 0 must reference danger signs,
   integrative must cover four categories, hit policies
4. **Hallucinated requirements** — "standard IMCI practice", "typical
   CHW setup", "WHO guidelines" citations that did not appear in the
   actual guide text

Run #8 (session 033248a7...) on the WHO 6-page sample failed 6 of 7
artifacts. Analysis of the 75 critical_issues found:

| Category | Count | Should be? |
|---|---|---|
| Real recall misses | ~15 | Critical (blocks run) |
| Style / convention / architecture | ~35 | Warning (does not block) |
| Hallucinated (no guide quote) | ~10 | Not flagged at all |
| Model correctly declined to invent | ~5 | Not flagged at all |
| Catcher infrastructure bug | ~5 | Not flagged at all |
| Genuine judgment calls | ~5 | Warning |

Only ~20% of the flagged criticals were legitimate recall failures. The
rest were noise that caused the model to thrash in repair iterations
trying to satisfy hallucinated requirements or style conventions.

## What replaced them

Generation 2 prompts in `backend/prompts/validators/completeness_*.txt`:

- **Narrower scope**: recall only, no style/architecture content
- **Hard hallucination guard**: every critical must cite a verbatim quote
  from the guide. If the catcher can't copy-paste the evidence, it cannot
  flag the item.
- **"When in doubt, pass"**: false positives are worse than false negatives
  because they make the model thrash. Recall failures are still blocking
  but only when the evidence is airtight.

Architecture and style checks moved to
`backend/validators/architecture.py` as **deterministic Python** — zero
LLM, zero variance, produces warnings by default (not criticals) so they
don't block recall-passing runs.

See INFRA.md section 9 for the gate generation log.
