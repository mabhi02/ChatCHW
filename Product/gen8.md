# gen8.md — comprehensive build plan

This file is the executable specification for building the gen8 / gen8.5 pipelines. It is designed to be run in bypass permissions mode without further user input. Every decision is pinned. Every file path is concrete. Stopping conditions are explicit — when one fires, halt and ask.

Background source documents (read at start, do not re-litigate):
- `C:\Users\athar\Documents\GitHub\CHW_RLM\4-22-notes.md` — design rationale, sections 1-7
- `C:\Users\athar\Documents\GitHub\CHW_RLM\professor artifacts\CHW Navigator hand-off.pdf` — scope brief
- `C:\Users\athar\Documents\GitHub\CHW_RLM\professor artifacts\Predicate table.pdf` — predicate spec
- `C:\Users\athar\Documents\GitHub\CHW_RLM\professor artifacts\Verification-First Clinical Engineering.pdf` — methodology
- `C:\Users\athar\Documents\GitHub\CHW_RLM\backend\output\run_8970cf65\FAILURES.md` — known issues to fix

---

## 0. Mission and constraints

### Mission
Stand up two new pipeline variants alongside the frozen gen7:
- **gen7**: frozen at git SHA `9000e86`. Opus mono. No further changes. Reference for "what we sent David Run 2."
- **gen8**: Opus mono labeler + full new architecture (Tier 0-3 from 4-22-notes section 7).
- **gen8.5**: 7-way Sonnet labeler + full new architecture. Shares ~95% of code with gen8; only `labeler.py` differs.

User selects pipeline at extraction time via `pipeline=gen7|gen8|gen8.5` query parameter on the existing extraction endpoint.

### Hard constraints (do not violate)
- **gen7 is frozen.** Never modify any file under `backend/gen7/`. If a Tier 0 hygiene fix is tempting to backport, do not.
- **No multi-agent generation chains.** Verifier is read-only — it emits `{agree, divergences[]}` and never edits artifact content. The 7-way Sonnet labeler is permitted because each pass is k=1 within itself and the merge is deterministic Python. (k=1 is set aside per user decision Q1, but the verifier-as-catcher discipline still applies.)
- **No same-pipeline fallback that hides failure.** If GPT is unavailable and verifier falls back to Anthropic-on-Anthropic, the verification sidecar must record `verifier_independence: "same-family-fallback"` and raise a `severity: warn` divergence on every artifact verified by fallback.
- **Default to maker output in rendered artifacts.** When verifier disagrees, the rendered DMN/XLSForm/flowchart still ships maker's output. The disagreement only appears in the verification sidecar and the divergence worklist.

### Stopping conditions (HALT and ask user)
1. Pre-flight: chunker does not preserve page numbers in Neon-cached guide JSON (see Section 2.1).
2. Pre-flight: concurrency probe shows Sonnet rate limit cannot accommodate 147 parallel calls even with throttling.
3. Tier 3: both OpenAI keys remain `insufficient_quota` AND user has not provided a Gemini key as alternative.
4. Any change requires editing `backend/gen7/`. (gen7 is frozen.)
5. Any architectural decision not pinned in this document.
6. Any test in Section 11 fails after a tier completes.

---

## 1. Architecture overview

```
backend/
  gen7/                         # FROZEN at SHA 9000e86. Do not touch.
    pipeline.py
    labeler.py
    chunker.py
    data_flow.py
    delivery_readme.py
    system_prompt_bundle.py

  gen8/                         # NEW. Both gen8 and gen8.5 live here.
    __init__.py
    pipeline.py                 # Orchestrator. Accepts `--labeler opus|sonnet7way`.
    chunker.py                  # Copy of gen7 chunker + page-number threading.
    labeler_opus.py             # gen8 labeler (Opus mono + Tier 0 prompt fixes).
    labeler_sonnet7way.py       # gen8.5 labeler (7 parallel Sonnet passes + reconciler).
    concurrency.py              # Throttle/queue/batcher (per Q6).
    data_flow.py                # Extended: universal/module/cross_module/derived/conditionally_collected.
    delivery_readme.py          # Updated for breakup + verification status callouts.
    system_prompt_bundle.py     # 3-stage bundle with new Stage 3 prompt rules.
    traffic_cops.py             # Cop 1 + Cop 2 emission and validation.

  provenance/                   # NEW shared module (used by gen8 and gen8.5).
    __init__.py
    sidecar.py                  # write_with_provenance, verify_sidecar.
    container.py                # compute_container_hash.
    schema.py                   # ProvenanceBlock, VerificationBlock pydantic models.

  verifier/                     # NEW shared module.
    __init__.py
    runner.py                   # verify_artifact(artifact_path, prompts, phase_outputs).
    models.py                   # Model selection: gpt -> anthropic same-family fallback.
    divergence.py               # Divergence schema + severity classification.
    worklist.py                 # divergence_worklist.json builder (signal-ranked).

  validators/
    referential_integrity.py    # EXISTS. Extend for predicate dedup.
    stockout_coverage.py        # NEW. Diagnosis x supply matrix.
    predicate_grammar.py        # NEW. Operator + missingness + units enforcement.

  converters/
    mermaid_to_png.py           # EXISTS.
    mermaid_split.py            # NEW. Walk module DAG, emit index + per-module sources.
    xlsform_split.py            # NEW. Aggregate workbook + per-module workbooks.

  server.py                     # MODIFY. Add `pipeline` query param to extraction endpoint.
  session_manager.py            # MODIFY. Route to gen7/gen8/gen8.5 based on query param.
```

### Pipeline selection plumbing
- Frontend: existing extraction widget. Add a `pipeline` query string. (No new UI element required for first delivery; the API supports it. Frontend dropdown can come later.)
- Backend `server.py`: extract `pipeline = request.query_params.get("pipeline", "gen7")`. Default is `gen7` to preserve current behavior.
- `session_manager.py`: dispatch to `backend.gen7.pipeline.run` or `backend.gen8.pipeline.run` based on flag. For `gen8.5`, pass `labeler="sonnet7way"` to the gen8 pipeline.

### Code-sharing principle
gen8.5 is gen8 with a different labeler. Do not duplicate `pipeline.py`, `data_flow.py`, etc. The labeler module is selected by string at runtime:

```python
# backend/gen8/pipeline.py
def run(..., labeler: str = "opus"):
    if labeler == "opus":
        from backend.gen8.labeler_opus import label_chunks
    elif labeler == "sonnet7way":
        from backend.gen8.labeler_sonnet7way import label_chunks
    else:
        raise ValueError(f"unknown labeler: {labeler}")
    labeled = label_chunks(chunks, ...)
    ...
```

---

## 2. Pre-flight checks (run before any code change)

### 2.1 Chunker page-number verification (BLOCKING)

User said: "Stop if you can't get that provenance."

Run this script to verify Neon-cached guide JSON preserves page numbers per block:

```bash
cd C:/Users/athar/Documents/GitHub/CHW_RLM && python -c "
import asyncio, os, json
from dotenv import load_dotenv
load_dotenv()

async def main():
    from prisma import Prisma
    db = Prisma()
    await db.connect()
    guides = await db.sourceguide.find_many(take=3, order={'ingestedAt':'desc'})
    print(f'Found {len(guides)} guides')
    for g in guides:
        print(f'  sha={g.sha256[:14]}... pages={g.pageCount} name={g.manualName!r}')
    if not guides:
        await db.disconnect()
        print('STOPPING_CONDITION: no guides in Neon')
        return
    g = guides[0]
    gj = g.guideJson if isinstance(g.guideJson, dict) else json.loads(g.guideJson)
    sections = gj.get('sections', {})
    pages = gj.get('pages', {})
    print(f'sections={len(sections)} pages={len(pages)}')
    sec_ids = list(sections.keys())[:3]
    page_check_passed = True
    for sid in sec_ids:
        sec = sections[sid]
        blocks = sec.get('blocks', [])
        block_pages = [b.get('page') for b in blocks[:5]]
        print(f'  {sid!r}: page_start={sec.get(\"page_start\")} page_end={sec.get(\"page_end\")} blocks_pages={block_pages}')
        if not all(p is not None for p in block_pages):
            page_check_passed = False
    if page_check_passed and pages:
        print('PRE_FLIGHT_OK: page numbers present per block AND per section AND in pages dict')
    else:
        print('STOPPING_CONDITION: blocks missing page numbers')
    await db.disconnect()

asyncio.run(main())
"
```

**Pass condition**: every sampled block has a non-None `page` field, every section has `page_start` and `page_end`, and `pages` dict is non-empty.

**Fail condition (HALT)**: any sampled block has `page: None` OR `pages` dict is empty. Stop and ask user.

### 2.2 Concurrency probe (BLOCKING for gen8.5 only — non-blocking for gen8)

7-way Sonnet wire-in requires 21 chunks × 7 passes = 147 parallel Sonnet calls. Anthropic Tier 4 limits on Sonnet are typically 4,000 RPM and 400K input TPM. Probe:

```bash
cd C:/Users/athar/Documents/GitHub/CHW_RLM && python -c "
import asyncio, os, time
from dotenv import load_dotenv
load_dotenv()
import anthropic
from anthropic import AsyncAnthropic

async def probe():
    client = AsyncAnthropic(api_key=os.environ['ANTHROPIC_KEY'])
    N = 14  # try 14 parallel calls (2x the per-chunk 7-way fanout)
    prompts = ['Reply with the single word OK.'] * N
    t0 = time.time()
    async def one(p):
        r = await client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=8,
            messages=[{'role':'user','content':p}],
        )
        return r.usage.output_tokens
    results = await asyncio.gather(*[one(p) for p in prompts], return_exceptions=True)
    elapsed = time.time() - t0
    fails = [r for r in results if isinstance(r, Exception)]
    print(f'{N} parallel calls in {elapsed:.1f}s; failures={len(fails)}')
    if fails:
        print(f'first failure: {fails[0]!r}')
        print('STOPPING_CONDITION: cannot probe at 14 parallel; concurrency strategy needs work')
    else:
        # Project: at ~elapsed for 14 calls, 147 calls = 147/14 * elapsed = ~10x
        projected = elapsed * (147/14)
        print(f'projected wall clock for 147 calls (no batching): {projected:.0f}s')
        print('PRE_FLIGHT_OK: concurrency feasible with throttling')

asyncio.run(probe())
"
```

**Pass condition**: zero failures at 14 parallel.
**Fail condition for gen8.5 only**: any rate-limit / 429 / throttle errors. gen8 (Opus mono) is unaffected — user can still extract via gen7 or gen8 while gen8.5 wait for rate-limit relief.

### 2.3 Verifier billing check (informational, non-blocking)

```bash
cd C:/Users/athar/Documents/GitHub/CHW_RLM && python -c "
import os, urllib.request, urllib.error, json
from dotenv import load_dotenv
load_dotenv()
for name in ['OPENAI_API_KEY','OPENAI_API_KEY_ALT']:
    key = os.environ.get(name,'')
    if not key:
        print(f'{name}: MISSING')
        continue
    body = json.dumps({'model':'gpt-5.4-mini','messages':[{'role':'user','content':'ok'}],'max_completion_tokens':4}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/chat/completions', data=body, headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f'{name}: BILLABLE')
    except urllib.error.HTTPError as e:
        print(f'{name}: HTTP {e.code}')
"
```

**Result handling**:
- If both keys BILLABLE → proceed normally with GPT verifier.
- If both keys 429 (`insufficient_quota`) → log warning. Do NOT halt. Tier 3 verifier wires in with Anthropic same-family fallback as primary; sidecars will record `verifier_independence: "same-family-fallback"` from day one. This is acceptable per user decision S2.

---

## 3. Repository layout decisions (locked)

- New directories: `backend/gen8/`, `backend/provenance/`, `backend/verifier/`.
- Existing directory edits: `backend/validators/`, `backend/converters/`, `backend/server.py`, `backend/session_manager.py`.
- Existing directory NO TOUCH: `backend/gen7/`, `backend/gen5/`, `backend/gen6/`.
- Output directory naming: gen8/8.5 runs go under `backend/output/run_<id>/` exactly as gen7 does. The `manifest.json` at the run root records which pipeline produced the run.

---

## 4. Tier 0 — Predicate hygiene (Stage 3 prompt fixes)

These are pure prompt edits in `backend/gen8/system_prompt_bundle.py`. The Stage 3 system prompt is what Opus reads in the REPL session.

### 4.0 Acceptance criteria

After Tier 0 lands, a fresh gen8 run on WHO 2012 must produce a `predicates.json` where:
- All operators in `threshold_expression` are lowercase: `=`, `>`, `>=`, `<`, `<=`, `and`, `or`, `not`, parentheses only. No `==`, no uppercase `AND`/`OR`, no quoted string literals. (Strings → enum predicates instead.)
- No `p_X` / `p_has_X` / `p_X_days` triplets. Each clinical concept has exactly one canonical predicate.
- Every entry has 4 new fields: `units` (string, may be empty for booleans), `missingness_rule` (one of `FALSE_IF_MISSING` / `TRIGGER_REFERRAL` / `BLOCK_DECISION` / `ALERT_NO_RULE_SPECIFIED`), `allowed_input_domain` (string range or set), `rounding_parsing_rule` (string, default `"parse as float; no rounding before comparison"` for numeric).
- Every entry's `provenance` field reads `"WHO 2012 page <N>"` not the chunker section slug.

### 4.1 Stage 3 prompt edits

Locate the Stage 3 prompt block in `system_prompt_bundle.py` (function `build_sendable_system_prompt`). Append this section before the existing artifact-emission instructions:

```
## Predicate-table requirements (gen8 strict)

When emitting `predicates`, every entry MUST have ALL of:
- `id` — `p_<concept>` lowercase. Examples: `p_fever`, `p_fast_breathing_2_to_12mo`, `p_chest_indrawing`. Stable across runs.
- `description_clinical` — plain-language clinician-readable, e.g. "Axillary temperature above 37.5°C indicating fever".
- `inputs_used` — list of canonical raw variable ids referenced.
- `formal_definition` — the boolean expression. Allowed operators ONLY: `>`, `<`, `>=`, `<=`, `=`, `and`, `or`, `not`, parentheses. No `==`, no uppercase logic ops, no quoted string literals. For categorical values use enum predicates: instead of `v_muac_color = "red"` define `p_muac_red` whose `formal_definition` is `v_muac_color_is_red` and have a separate computed boolean.
- `units` — REQUIRED even when obvious. Examples: "°C", "breaths/min", "months", "(boolean)".
- `missingness_rule` — exactly one of: `FALSE_IF_MISSING`, `TRIGGER_REFERRAL`, `BLOCK_DECISION`, `ALERT_NO_RULE_SPECIFIED`. Use `ALERT_NO_RULE_SPECIFIED` only when the source manual gives no guidance.
- `allowed_input_domain` — explicit range or set. Examples: "34 to 43 °C", "0 to 200 breaths/min", "{red, green, yellow}".
- `rounding_parsing_rule` — explicit. Default for numerics: "parse as float; no rounding before comparison". Default for booleans: "parse as boolean".
- `provenance` — citation in form "WHO 2012 page <N>" using the page number from the source chunk (provided in the chunk metadata as `manual_page_start` and `manual_page_end`).

DO NOT emit duplicate predicates. If two predicates would name the same clinical concept (e.g., `p_fever` and `p_has_fever`), emit only ONE. Prefer the shorter id. The pipeline will reject duplicates downstream.

DO NOT emit predicates with empty `formal_definition`. Every predicate MUST be computable.
```

### 4.2 Post-emission deterministic validation (Tier 0 deps)

Add `backend/validators/predicate_grammar.py`:

```python
"""Validates Stage 3 predicate output conforms to gen8 strict schema."""
import re
from typing import Any

ALLOWED_OPS = re.compile(r'^[a-zA-Z0-9_\s\(\)\.\,><=]+$')
BANNED_TOKENS = ['==', ' AND ', ' OR ', ' NOT ', '"', "'"]
REQUIRED_FIELDS = [
    'id', 'description_clinical', 'inputs_used', 'formal_definition',
    'units', 'missingness_rule', 'allowed_input_domain',
    'rounding_parsing_rule', 'provenance',
]
VALID_MISSINGNESS = {
    'FALSE_IF_MISSING', 'TRIGGER_REFERRAL', 'BLOCK_DECISION', 'ALERT_NO_RULE_SPECIFIED',
}


def validate_predicate(p: dict) -> list[str]:
    """Returns list of validation errors for one predicate. Empty = valid."""
    errors = []
    for f in REQUIRED_FIELDS:
        if f not in p or p.get(f) in (None, ""):
            errors.append(f"missing required field: {f}")
    if 'formal_definition' in p:
        expr = p['formal_definition']
        for tok in BANNED_TOKENS:
            if tok in expr:
                errors.append(f"banned token in formal_definition: {tok!r}")
    if p.get('missingness_rule') not in VALID_MISSINGNESS:
        errors.append(f"invalid missingness_rule: {p.get('missingness_rule')!r}")
    if not p.get('id', '').startswith('p_'):
        errors.append(f"id must start with 'p_': {p.get('id')!r}")
    return errors


def dedup_predicate_triplets(predicates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Collapse p_X / p_has_X / p_X_days triplets to canonical (shortest id).
    Returns (kept, dropped).
    """
    by_concept: dict[str, list[dict]] = {}
    for p in predicates:
        pid = p.get('id', '')
        # Strip _days, _has_, etc to find concept
        concept = pid
        for prefix in ('p_has_', 'p_'):
            if concept.startswith(prefix):
                concept = concept[len(prefix):]
                break
        for suffix in ('_days', '_count'):
            if concept.endswith(suffix):
                concept = concept[:-len(suffix)]
        by_concept.setdefault(concept, []).append(p)
    kept, dropped = [], []
    for concept, group in by_concept.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            # Prefer the shortest id (most canonical)
            group.sort(key=lambda p: len(p.get('id', '')))
            kept.append(group[0])
            for d in group[1:]:
                d['_dedup_reason'] = f'duplicate of {group[0]["id"]}'
                dropped.append(d)
    return kept, dropped


def validate_all(predicates: list[dict]) -> dict:
    """Validate every predicate. Returns report dict."""
    per_pred = []
    for p in predicates:
        errs = validate_predicate(p)
        per_pred.append({'id': p.get('id', '?'), 'errors': errs})
    total_errors = sum(len(x['errors']) for x in per_pred)
    return {
        'total': len(predicates),
        'with_errors': sum(1 for x in per_pred if x['errors']),
        'total_errors': total_errors,
        'per_predicate': per_pred,
    }
```

Wire into `backend/gen8/pipeline.py` immediately after Stage 3 emits `predicates.json`:

```python
from backend.validators.predicate_grammar import validate_all, dedup_predicate_triplets
preds = clinical_logic.get('predicates', [])
preds, dropped = dedup_predicate_triplets(preds)
clinical_logic['predicates'] = preds
report = validate_all(preds)
(output_dir / "artifacts" / "predicates_validation.json").write_text(
    json.dumps({"report": report, "dropped_dups": dropped}, indent=2)
)
if report['with_errors'] > 0:
    logger.warning("gen8 Tier 0: %d predicates have schema errors", report['with_errors'])
```

### 4.3 Page-number threading

Modify `backend/gen8/chunker.py` (copy gen7's, then extend):

In `_make_chunk`, add to the returned dict:
```python
"manual_page_start": min((b.get("page") for b in blocks if b.get("page") is not None), default=None),
"manual_page_end": max((b.get("page") for b in blocks if b.get("page") is not None), default=None),
```

In `backend/gen8/labeler_opus.py` and `backend/gen8/labeler_sonnet7way.py`, pass `manual_page_start` and `manual_page_end` through `_labeling_meta`. In Stage 3 prompt, the chunk metadata block must include `pages: <start>-<end>` so Opus can cite "WHO 2012 page N" correctly.

---

## 5. Tier 1 — Provenance + xlsx/flowchart breakup

### 5.1 Provenance sidecar module (`backend/provenance/`)

#### `backend/provenance/schema.py`

```python
"""Pydantic schemas for provenance and verification sidecars."""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ParentRef(BaseModel):
    kind: str  # e.g. "deduped_labels", "reconstructed_guide", "chunk_001_labels"
    content_sha256: str


class SourceManual(BaseModel):
    name: str  # "Caring for the sick child in the community"
    publisher: str  # "WHO, Geneva, 2012"
    pdf_sha256: str  # 64 hex chars
    page_count: int
    neon_guide_id: Optional[str] = None  # uuid from SourceGuide


class ProvenanceBlock(BaseModel):
    artifact_kind: str  # "supply_list", "predicates", etc
    content_sha256: str  # SHA-256 of the artifact's bytes (excluding sidecar itself)
    run_id: str
    pipeline: str  # "gen8" or "gen8.5"
    labeler: str  # "opus" or "sonnet7way"
    created_at: str  # ISO 8601
    pipeline_git_sha: str
    model: str  # generator model
    source_manual: SourceManual
    parents: list[ParentRef] = Field(default_factory=list)
    schema_version: int = 1
    status: Literal["unverified", "verified", "verification_failed"] = "unverified"


class Divergence(BaseModel):
    type: str  # "missing_item", "spurious_item", "value_mismatch", "ordering_issue", etc
    severity: Literal["info", "warn", "error"] = "warn"
    detail: str
    evidence: dict = Field(default_factory=dict)  # chunk_index, quote, etc


class VerificationBlock(BaseModel):
    artifact_kind: str
    artifact_content_sha256: str
    verifier_model: str  # "gpt-5.4" or "claude-sonnet-4-6" etc
    verifier_independence: Literal["different-family", "same-family-fallback"]
    verifier_run_at: str
    agree: bool
    divergences: list[Divergence] = Field(default_factory=list)
    schema_version: int = 1
```

#### `backend/provenance/sidecar.py`

```python
"""Write artifacts with provenance sidecars. Sidecar is a separate file."""
import json
import hashlib
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from backend.provenance.schema import ProvenanceBlock, ParentRef, SourceManual


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_artifact(data: Any) -> bytes:
    """Canonical JSON serialization for hashing."""
    return json.dumps(data, indent=2, sort_keys=True, default=str).encode("utf-8")


def write_with_provenance(
    path: Path,
    data: Any,
    *,
    artifact_kind: str,
    run_id: str,
    pipeline: str,
    labeler: str,
    pipeline_git_sha: str,
    model: str,
    source_manual: SourceManual,
    parents: list[ParentRef] | None = None,
) -> str:
    """Write data to path. Compute SHA. Emit sidecar at path.with_suffix('.provenance.json').
    Returns content_sha256.
    """
    serialized = serialize_artifact(data)
    sha = compute_sha256(serialized)
    path.write_bytes(serialized)
    prov = ProvenanceBlock(
        artifact_kind=artifact_kind,
        content_sha256=sha,
        run_id=run_id,
        pipeline=pipeline,
        labeler=labeler,
        created_at=datetime.now(timezone.utc).isoformat(),
        pipeline_git_sha=pipeline_git_sha,
        model=model,
        source_manual=source_manual,
        parents=parents or [],
    )
    sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
    sidecar_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")
    return sha


def verify_sidecar(path: Path) -> bool:
    """Check that artifact at path matches the SHA in its sidecar."""
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    if not sidecar.exists():
        return False
    prov = json.loads(sidecar.read_text())
    actual = compute_sha256(path.read_bytes())
    return actual == prov["content_sha256"]
```

#### `backend/provenance/container.py`

```python
"""Container hash: rolls up all artifact SHAs + source PDF SHA + git SHA + verifier-model identifier."""
import hashlib
import json
import subprocess
from pathlib import Path


def get_git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent.parent.parent).decode().strip()
    except Exception:
        return "unknown"


def compute_container_hash(
    artifact_shas: dict[str, str],  # {artifact_kind: content_sha256}
    source_pdf_sha: str,
    git_sha: str,
    verifier_model: str,
) -> str:
    """Roll up all inputs into one canonical hash. Order is alphabetical for stability."""
    payload = {
        "artifacts": dict(sorted(artifact_shas.items())),
        "source_pdf_sha256": source_pdf_sha,
        "pipeline_git_sha": git_sha,
        "verifier_model": verifier_model,
    }
    canonical = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_manifest(output_dir: Path, **kwargs) -> str:
    """Write the run-level manifest.json with container_sha256."""
    container_sha = compute_container_hash(**kwargs)
    manifest = {
        "container_sha256": container_sha,
        **kwargs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return container_sha
```

### 5.2 Wire provenance into pipeline write sites

Replace every `(path).write_text(json.dumps(data))` in `backend/gen8/pipeline.py` with `write_with_provenance(path, data, kind=..., parents=[...])`. Specifically:
- `labeled_chunks.json` — kind=`labeled_chunks`, parents=`[reconstructed_guide]`
- `deduped_labels.json` — kind=`deduped_labels`, parents=`[labeled_chunks]`
- `chunk_difficulty.json` — kind=`chunk_difficulty`, parents=`[reconstructed_guide]`
- `stage3_repl.json` — kind=`stage3_repl`, parents=`[deduped_labels, reconstructed_guide]`
- `referential_integrity.json` — kind=`referential_integrity`, parents=`[clinical_logic]`
- `clinical_logic.json` — kind=`clinical_logic`, parents=`[stage3_repl]`
- Each per-artifact `artifacts/<kind>.json` — kind=`<kind>`, parents=`[clinical_logic]`
- `artifacts/data_flow.json` — kind=`data_flow`, parents=`[variables, modules, predicates]`
- `test_suite.json` — kind=`test_suite`, parents=`[clinical_logic]`

After all writes, call `write_manifest()` to roll everything up into `manifest.json` at the run root.

### 5.3 Backfill Run 2 (one-shot)

Create `scripts/backfill_provenance_run2.py`:

```python
"""One-shot: walk run_8970cf65, compute SHAs, emit provenance sidecars."""
from pathlib import Path
import json
from backend.provenance.sidecar import write_with_provenance, compute_sha256
from backend.provenance.container import write_manifest, get_git_sha
from backend.provenance.schema import SourceManual

RUN_DIR = Path("backend/output/run_8970cf65")
SOURCE_MANUAL = SourceManual(
    name="Caring for the sick child in the community",
    publisher="WHO, Geneva, 2012",
    pdf_sha256="1d2708940727" + "0" * 52,  # TODO: fetch real SHA from Neon SourceGuide
    page_count=140,
)

# For each json file under RUN_DIR (excluding sidecars), emit a sidecar.
artifact_shas = {}
for path in RUN_DIR.rglob("*.json"):
    if path.name.endswith(".provenance.json"):
        continue
    data = json.loads(path.read_bytes())
    sha = write_with_provenance(
        path, data,
        artifact_kind=path.stem,
        run_id="run_8970cf65",
        pipeline="gen7",
        labeler="opus",
        pipeline_git_sha="9000e86",
        model="claude-opus-4-6",
        source_manual=SOURCE_MANUAL,
        parents=[],  # parent edges unknown for backfill; leave empty
    )
    artifact_shas[path.stem] = sha

write_manifest(
    RUN_DIR,
    artifact_shas=artifact_shas,
    source_pdf_sha=SOURCE_MANUAL.pdf_sha256,
    git_sha="9000e86",
    verifier_model="none",
)
print(f"Backfilled {len(artifact_shas)} artifacts in {RUN_DIR}")
```

Run this script once gen8 Tier 1 lands. It emits sidecars for the existing Run 2 artifacts so Run 2's zip can be re-issued with provenance.

### 5.4 xlsx breakup (`backend/converters/xlsform_split.py`)

Stub:

```python
"""Emit aggregate XLSForm + per-module XLSForm workbooks.
Aggregate is the deployable file CHT consumes. Per-module are review-only.
"""
from pathlib import Path
import openpyxl
from openpyxl import Workbook


def build_aggregate_workbook(clinical_logic: dict, out: Path) -> None:
    """One workbook with TOC + survey + choices + settings + routing + glossary sheets."""
    wb = Workbook()
    # TOC
    ws = wb.active
    ws.title = "TOC"
    ws.append(["module_id", "title", "symptom", "page_in_workbook"])
    for m in clinical_logic.get("modules", {}).values():
        ws.append([m.get("id"), m.get("title"), m.get("symptom_trigger"), ""])  # page filled later
    # survey + choices + settings (CHT-required sheets)
    _emit_survey_sheet(wb, clinical_logic)
    _emit_choices_sheet(wb, clinical_logic)
    _emit_settings_sheet(wb, clinical_logic)
    # routing
    ws = wb.create_sheet("routing")
    ws.append(["from_module", "to_module", "condition"])
    for m in clinical_logic.get("modules", {}).values():
        for route in m.get("routes_to", []):
            ws.append([m["id"], route.get("target"), route.get("condition")])
    # glossary
    ws = wb.create_sheet("glossary")
    ws.append(["variable_id", "human_label", "type", "units"])
    for v in sorted(clinical_logic.get("variables", []), key=lambda x: x.get("id", "")):
        ws.append([v.get("id"), v.get("human_label"), v.get("type"), v.get("units")])
    wb.save(out)


def build_per_module_workbook(module: dict, clinical_logic: dict, out: Path) -> None:
    """One workbook per module, scoped to that module's variables, predicates, rules, routes."""
    wb = Workbook()
    ws = wb.active
    ws.title = f"mod_{module['id']}"
    ws.append(["field", "value", "source_quote", "source_section"])
    # ... emit per-module survey + rules + routes with traces
    wb.save(out)


def emit_all(clinical_logic: dict, output_dir: Path) -> dict:
    """Write aggregate and per-module workbooks. Returns paths."""
    paths = {}
    aggregate = output_dir / "form.xlsx"
    build_aggregate_workbook(clinical_logic, aggregate)
    paths["aggregate"] = aggregate
    modules_dir = output_dir / "form_per_module"
    modules_dir.mkdir(exist_ok=True)
    for m in clinical_logic.get("modules", {}).values():
        per = modules_dir / f"form_mod_{m['id']}.xlsx"
        build_per_module_workbook(m, clinical_logic, per)
        paths[f"mod_{m['id']}"] = per
    return paths
```

Wire into pipeline after `clinical_logic.json` is written. Both files get provenance sidecars (one for `form.xlsx`, one per per-module workbook).

### 5.5 Flowchart breakup (`backend/converters/mermaid_split.py`)

Stub:

```python
"""Emit index Mermaid + per-module Mermaid sources.
Index shows only module boxes + routing edges between them.
Each per-module flowchart has explicit Entry/Exit nodes naming the next module.
"""
from pathlib import Path


def build_index_mermaid(clinical_logic: dict) -> str:
    lines = ["flowchart TD"]
    modules = clinical_logic.get("modules", {})
    for mid, m in modules.items():
        lines.append(f'  {mid}["{m.get("title", mid)}"]')
        lines.append(f'  click {mid} "flowchart_mod_{mid}.png"')
    for mid, m in modules.items():
        for route in m.get("routes_to", []):
            tgt = route.get("target")
            cond = route.get("condition", "")
            lines.append(f'  {mid} -->|{cond}| {tgt}')
    return "\n".join(lines)


def build_module_mermaid(module: dict) -> str:
    """Mermaid source for one module, with explicit Entry / Exit nodes naming neighbors."""
    lines = ["flowchart TD"]
    lines.append(f'  Entry["Enter from index"]')
    # ... emit decision nodes from module.rules
    for route in module.get("routes_to", []):
        tgt = route.get("target")
        cond = route.get("condition", "")
        lines.append(f'  Exit_{tgt}["Exit to flowchart_mod_{tgt}<br>(if {cond})"]')
    return "\n".join(lines)


def emit_all(clinical_logic: dict, output_dir: Path) -> dict:
    """Write index + per-module .md files. Render via existing mermaid_to_png.py."""
    from backend.converters.mermaid_to_png import render_to_png
    paths = {}
    idx_md = output_dir / "flowchart_index.md"
    idx_md.write_text(build_index_mermaid(clinical_logic), encoding="utf-8")
    idx_png = output_dir / "flowchart_index.png"
    render_to_png(idx_md, idx_png)
    paths["index"] = (idx_md, idx_png)
    modules_dir = output_dir / "flowcharts_per_module"
    modules_dir.mkdir(exist_ok=True)
    for mid, m in clinical_logic.get("modules", {}).items():
        md = modules_dir / f"flowchart_mod_{mid}.md"
        md.write_text(build_module_mermaid(m), encoding="utf-8")
        png = modules_dir / f"flowchart_mod_{mid}.png"
        render_to_png(md, png)
        paths[f"mod_{mid}"] = (md, png)
    return paths
```

The existing `flowchart.png` (monolithic) stays as a fallback, but the README points to `flowchart_index.png` first.

---

## 6. Tier 2 — New artifacts and traffic cops

### 6.1 Manual-order threading

In `backend/gen8/chunker.py`, every chunk already has `manual_page_start` and `manual_page_end` from Tier 0.4.

In Stage 3 prompt (`system_prompt_bundle.py`), add:

```
## Ordering rule (MANDATORY)

Every list-shaped artifact (`variables`, `predicates`, `modules`, `phrase_bank`, `supply_list`) must order items by manual position. Each item MUST carry `_order_source` set to one of:
- `"manual"`: ordering follows source manual page sequence. Include `_manual_page` field.
- `"explicit_hint"`: source manual gives explicit ordering instruction (e.g., "ask cough before fever"). Include `_order_quote` field with the source quote.
- `"alphabetical_fallback"`: only when (1) and (2) provide no signal.

Within `modules`, `routes_to`, `rules`, and `inputs` lists, ordering follows manual sequence. Sort items by `_manual_page` ascending; ties broken by `_order_source == "explicit_hint" > manual > alphabetical_fallback`.
```

In `backend/gen8/pipeline.py` post-Stage-3, add a deterministic validator that asserts every item in every list-shaped artifact has `_order_source` set. If any item is missing it, log a warning but do not halt (the verifier will catch this as a divergence).

### 6.2 Source-of-each-variable artifact

Extend `backend/gen8/data_flow.py` from gen7's version. New scopes:
- `universal` — collected at session start, used by multiple modules
- `module_local` — collected only when one module is active
- `cross_module` — collected by one module, used by another
- `derived` — computed from other variables (no CHW input)
- `conditionally_collected` — collected only when an upstream condition fires (e.g., "if temp missing → measure axillary temp")
- `orphan` — referenced by predicates/rules but no module collects (THIS IS A BUG; should be 0 in gen8 output)

Add `data_flow_classify(variables, modules, predicates) -> dict` that returns:

```json
{
  "by_variable_id": {
    "v_temp_c": {"scope": "universal", "collected_by": ["mod_startup"], "used_by_modules": ["mod_fever"], "used_by_predicates": ["p_fever"], "ordering": {"_manual_page": 12}},
    ...
  },
  "summary": {
    "universal": 15, "module_local": 42, "cross_module": 8, "derived": 5, "conditionally_collected": 3, "orphan": 0
  }
}
```

Acceptance criterion: a clean gen8 run produces `summary.orphan == 0`. If non-zero, the verifier flags each as a `severity: error` divergence.

### 6.3 Two-traffic-cop split

Modify Stage 3 prompt to emit `router.json` as TWO distinct DMN tables:

```json
{
  "cop1_queue_builder": {
    "hit_policy": "COLLECT",
    "description": "Add modules to queue based on symptom flags",
    "rules": [
      {"id": "r_at_start", "conditions": {"at_start": true}, "actions": {"add_to_queue": ["mod_startup", "mod_integrative"]}},
      {"id": "r_fever", "conditions": {"q_has_fever": true}, "actions": {"add_to_queue": ["mod_fever"]}},
      ...
    ]
  },
  "cop2_next_module": {
    "hit_policy": "UNIQUE",
    "description": "Pick next module to run from queue",
    "rules": [
      {"id": "r_queue_empty", "conditions": {"queue_empty": true}, "actions": {"end_visit": true}},
      {"id": "r_urgent_referral", "conditions": {"is_urgent_referral": true}, "actions": {"next_module": "mod_integrative"}},
      ...
    ]
  }
}
```

Wire into `backend/gen8/traffic_cops.py`:

```python
"""Validate and emit two-traffic-cop router structure."""

def validate_router(router: dict) -> list[str]:
    errors = []
    if "cop1_queue_builder" not in router:
        errors.append("missing cop1_queue_builder")
    if "cop2_next_module" not in router:
        errors.append("missing cop2_next_module")
    if router.get("cop1_queue_builder", {}).get("hit_policy") != "COLLECT":
        errors.append("cop1 hit_policy must be COLLECT (multi-hit)")
    if router.get("cop2_next_module", {}).get("hit_policy") != "UNIQUE":
        errors.append("cop2 hit_policy must be UNIQUE (mutually exclusive)")
    return errors
```

DMN converter (downstream) emits these as two `<decision>` blocks within `clinical_logic.dmn`.

### 6.4 Stockout coverage enumeration

Add `backend/validators/stockout_coverage.py`:

```python
"""Verify every diagnosis x supply pair has explicit coverage or ALERT row."""

def check_stockout_coverage(clinical_logic: dict) -> list[dict]:
    """Returns list of gaps: diagnoses where some referenced supply has no stockout fallback."""
    supplies = {s["id"]: s for s in clinical_logic.get("supply_list", [])}
    gaps = []
    for module in clinical_logic.get("modules", {}).values():
        for rule in module.get("rules", []):
            for supply_ref in rule.get("treatment_supplies", []):
                # Check if rule has a stockout fallback
                has_fallback = any(
                    r.get("conditions", {}).get(f"{supply_ref}_available") is False
                    for r in module.get("rules", [])
                )
                if not has_fallback:
                    gaps.append({
                        "module": module["id"],
                        "rule": rule["id"],
                        "supply": supply_ref,
                        "alert_message": f"ALERT: Manual does not cover stockout for {supply_ref} in {module['id']}/{rule['id']}",
                    })
    return gaps
```

Wire into pipeline post-Stage-3. Result goes to `artifacts/stockout_coverage.json` (with provenance sidecar). Each gap is added to the verifier's input as a known issue.

---

## 7. Tier 3 — Verifier harness

### 7.1 Verifier model selection (`backend/verifier/models.py`)

```python
"""Model selection: GPT primary, Anthropic same-family fallback."""
import os
import logging

logger = logging.getLogger(__name__)


def select_verifier_model(generator_model: str) -> tuple[str, str]:
    """Returns (model_id, independence_label).
    independence_label: 'different-family' or 'same-family-fallback'.
    """
    # Probe GPT availability with a 1-token call
    if _gpt_available():
        return ("gpt-5.4-mini", "different-family")
    logger.warning("GPT unavailable; falling back to Anthropic same-family verifier")
    if generator_model.startswith("claude-opus"):
        return ("claude-sonnet-4-6", "same-family-fallback")
    elif generator_model.startswith("claude-sonnet"):
        return ("claude-opus-4-6", "same-family-fallback")
    return ("claude-sonnet-4-6", "same-family-fallback")


def _gpt_available() -> bool:
    """One-shot health check."""
    import urllib.request, urllib.error, json
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return False
    try:
        body = json.dumps({"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "ok"}], "max_completion_tokens": 4}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception:
        return False
```

### 7.2 Verifier runner (`backend/verifier/runner.py`)

```python
"""Per-artifact verifier. Reads artifact + relevant prompts + relevant phase outputs.
Emits {agree: bool, divergences: list[Divergence]}.
"""
import json
from pathlib import Path
from datetime import datetime, timezone
from backend.verifier.models import select_verifier_model
from backend.provenance.schema import VerificationBlock, Divergence


VERIFIER_PROMPT = """You are an independent verifier of a medical decision-logic extraction pipeline.
You will receive (a) the source clinical guideline, (b) the artifact produced by another AI, and (c) the prompts that produced it.

Your job:
1. Read the source.
2. Read the artifact.
3. Determine whether the artifact correctly represents the source.
4. If you disagree, emit a divergence object: {"type": "...", "severity": "info|warn|error", "detail": "...", "evidence": {...}}.

Severity guide:
- "error": clinical correctness compromised (wrong threshold, missing danger sign, etc.)
- "warn": structural issue that could mislead reviewers (duplicate predicate, missing units, ambiguous wording).
- "info": cosmetic or stylistic.

Reply ONLY with a JSON object: {"agree": true|false, "divergences": [...]}.
DO NOT edit the artifact. DO NOT propose corrections. Your job is to flag, not fix.
"""


def verify_artifact(
    artifact_path: Path,
    artifact_kind: str,
    artifact_content_sha: str,
    source_manual_text: str,
    generator_prompts: dict,
    generator_model: str,
) -> VerificationBlock:
    artifact_data = json.loads(artifact_path.read_text())
    verifier_model, independence = select_verifier_model(generator_model)
    user_msg = (
        f"=== SOURCE MANUAL ===\n{source_manual_text[:50000]}\n\n"
        f"=== ARTIFACT ({artifact_kind}) ===\n{json.dumps(artifact_data, indent=2)[:30000]}\n\n"
        f"=== PROMPTS THAT PRODUCED THIS ARTIFACT ===\n{json.dumps(generator_prompts, indent=2)[:10000]}\n\n"
        f"Verify."
    )
    response_json = _call_verifier(verifier_model, VERIFIER_PROMPT, user_msg)
    divergences = [Divergence(**d) for d in response_json.get("divergences", [])]
    if independence == "same-family-fallback":
        divergences.append(Divergence(
            type="same_family_fallback",
            severity="warn",
            detail=f"Verifier {verifier_model} is same model family as generator {generator_model}; independence weakened.",
        ))
    return VerificationBlock(
        artifact_kind=artifact_kind,
        artifact_content_sha256=artifact_content_sha,
        verifier_model=verifier_model,
        verifier_independence=independence,
        verifier_run_at=datetime.now(timezone.utc).isoformat(),
        agree=response_json.get("agree", False) and len([d for d in divergences if d.severity == "error"]) == 0,
        divergences=divergences,
    )


def _call_verifier(model_id: str, system_prompt: str, user_msg: str) -> dict:
    """Dispatch to GPT or Anthropic. Both return parsed JSON."""
    if model_id.startswith("gpt"):
        return _call_openai(model_id, system_prompt, user_msg)
    return _call_anthropic(model_id, system_prompt, user_msg)


def _call_openai(model_id, system_prompt, user_msg):
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _call_anthropic(model_id, system_prompt, user_msg):
    from anthropic import Anthropic
    client = Anthropic()
    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text
    # Strip code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)
```

### 7.3 Verifier sidecar emission (`backend/verifier/__init__.py`)

```python
"""Write verification sidecar separate from provenance sidecar (per FU5 = option B)."""
from pathlib import Path
from backend.provenance.schema import VerificationBlock


def write_verification_sidecar(artifact_path: Path, vblock: VerificationBlock) -> None:
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".verification.json")
    sidecar.write_text(vblock.model_dump_json(indent=2), encoding="utf-8")


def update_provenance_status(artifact_path: Path, vblock: VerificationBlock) -> None:
    """Update the .provenance.json `status` field based on verification result."""
    import json
    prov_path = artifact_path.with_suffix(artifact_path.suffix + ".provenance.json")
    prov = json.loads(prov_path.read_text())
    has_blocking = any(d.severity in ("error",) for d in vblock.divergences)
    if vblock.agree and not has_blocking:
        prov["status"] = "verified"
    elif vblock.agree:
        prov["status"] = "verified"  # warn divergences don't block
    else:
        prov["status"] = "verification_failed"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
```

### 7.4 Divergence worklist (`backend/verifier/worklist.py`)

```python
"""Build divergence_worklist.json — the MD reviewer's starting point.
Per FU + Q11: per-artifact divergence, signal-ranked. Default to maker output, raise the flag.
"""
import json
from pathlib import Path
from collections import Counter


def build_worklist(output_dir: Path) -> Path:
    """Walk all *.verification.json files. Build a signal-ranked worklist."""
    rows = []
    for vpath in output_dir.rglob("*.verification.json"):
        v = json.loads(vpath.read_text())
        for d in v.get("divergences", []):
            rows.append({
                "artifact_kind": v["artifact_kind"],
                "artifact_path": str(vpath).replace(".verification.json", ""),
                "type": d["type"],
                "severity": d["severity"],
                "detail": d["detail"],
                "evidence": d.get("evidence", {}),
            })
    # Signal ranking: severity (error > warn > info), then frequency of (artifact_kind, type) across all rows
    sev_rank = {"error": 0, "warn": 1, "info": 2}
    freq = Counter((r["artifact_kind"], r["type"]) for r in rows)
    rows.sort(key=lambda r: (sev_rank.get(r["severity"], 99), -freq[(r["artifact_kind"], r["type"])]))
    out = output_dir / "divergence_worklist.json"
    out.write_text(json.dumps({
        "summary": {
            "total_divergences": len(rows),
            "error_count": sum(1 for r in rows if r["severity"] == "error"),
            "warn_count": sum(1 for r in rows if r["severity"] == "warn"),
            "info_count": sum(1 for r in rows if r["severity"] == "info"),
        },
        "ordered_worklist": rows,
    }, indent=2), encoding="utf-8")
    return out
```

### 7.5 Wire verifier into pipeline

In `backend/gen8/pipeline.py`, after all artifacts are written and provenance sidecars exist, call:

```python
from backend.verifier.runner import verify_artifact
from backend.verifier import write_verification_sidecar, update_provenance_status
from backend.verifier.worklist import build_worklist

verified_artifacts = []
for artifact_kind in ["supply_list", "variables", "predicates", "modules", "router", "integrative", "phrase_bank"]:
    apath = output_dir / "artifacts" / f"{artifact_kind}.json"
    prov = json.loads(apath.with_suffix(".json.provenance.json").read_text())
    vblock = verify_artifact(
        apath,
        artifact_kind,
        prov["content_sha256"],
        source_manual_text=reconstructed_guide,
        generator_prompts={"stage3_prompt": stage3_system_prompt},
        generator_model="claude-opus-4-6",
    )
    write_verification_sidecar(apath, vblock)
    update_provenance_status(apath, vblock)
    verified_artifacts.append(vblock)

worklist_path = build_worklist(output_dir)
logger.info("gen8: verifier complete, worklist at %s", worklist_path)
```

---

## 8. gen8.5 specifics: 7-way Sonnet labeler

### 8.1 Labeler structure (`backend/gen8/labeler_sonnet7way.py`)

7 narrow-prompt Sonnet passes per chunk, one per artifact type:

```python
"""gen8.5 labeler: 7 parallel narrow-prompt Sonnet calls per chunk + deterministic merge.
See labeling_7way_demo/chunk_enriched.json for the validated output shape.
"""
import asyncio
from anthropic import AsyncAnthropic
from backend.gen8.concurrency import throttled_gather

SONNET_MODEL = "claude-sonnet-4-6"

# Narrow prompts pinned to exact type strings. See 4-22-notes section 6 for rationale.
PASS_PROMPTS = {
    "supply_list": "Label only supply_list items in this chunk. Type: ALWAYS use exactly 'supply_list'. ...",
    "variables": "Label only variables in this chunk. Type: ALWAYS use exactly 'variables'. ...",
    "predicates": "Label only predicates in this chunk. Type: ALWAYS use exactly 'predicates'. ...",
    "modules": "Label only modules in this chunk. Type: ALWAYS use exactly 'modules'. ...",
    "phrase_bank": "Label only phrase_bank items in this chunk. Type: ALWAYS use exactly 'phrase_bank'. ...",
    "router": "Label only router rules in this chunk. Type: ALWAYS use exactly 'router'. ...",
    "integrative": "Label only integrative rules in this chunk. Type: ALWAYS use exactly 'integrative'. ...",
}


async def label_one_chunk(chunk: dict, client: AsyncAnthropic) -> dict:
    """7 parallel passes + deterministic reconciliation."""
    pass_tasks = [
        _run_pass(client, chunk, pass_name, prompt)
        for pass_name, prompt in PASS_PROMPTS.items()
    ]
    pass_results = await throttled_gather(pass_tasks, max_concurrent=7)
    raw_labels = []
    for res in pass_results:
        raw_labels.extend(res)
    final, dropped, cross_type = _reconcile(raw_labels, chunk["text"])
    return {
        **chunk,
        "labels": final,
        "_labeling_meta": {
            "pipeline": "gen8.5",
            "model": SONNET_MODEL,
            "method": "7-way-sonnet",
            "passes": list(PASS_PROMPTS.keys()),
            "raw_count": len(raw_labels),
            "final_count": len(final),
            "dropped_count": len(dropped),
            "cross_type_count": len(cross_type),
        },
        "_dropped_labels": dropped,
        "_cross_type_spans": cross_type,
    }


def _reconcile(raw_labels: list[dict], chunk_text: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Span-grep grounding + (id, type) dedup + shorter-id tiebreaker + cross-type preservation."""
    # ... 60 lines per memory project_7way_labeling_experiment.md
    pass


async def label_chunks(chunks: list[dict]) -> list[dict]:
    client = AsyncAnthropic()
    return await throttled_gather([label_one_chunk(c, client) for c in chunks], max_concurrent=3)
```

### 8.2 Concurrency control (`backend/gen8/concurrency.py`)

Per Q6: queue + throttle + batch.

```python
"""Throttled parallel execution.
Anthropic Tier 4 limits: 4000 RPM, 400K input TPM, 80K output TPM (approx).
For 7-way Sonnet at 21 chunks: 147 calls. Keep RPS under 30 to stay safe.
"""
import asyncio
from typing import Awaitable


async def throttled_gather(coros: list[Awaitable], max_concurrent: int = 5) -> list:
    """Run coroutines with bounded concurrency."""
    sem = asyncio.Semaphore(max_concurrent)
    async def _wrap(coro):
        async with sem:
            return await coro
    return await asyncio.gather(*[_wrap(c) for c in coros], return_exceptions=False)
```

For gen8.5 the caller uses `max_concurrent=7` for the inner pass-fanout and `max_concurrent=3` for the outer chunk-fanout, capping total in-flight at 21 calls — well under Tier 4 RPM.

### 8.3 gen8.5 dispatch

In `backend/gen8/pipeline.py`:

```python
def run(..., labeler: str = "opus"):
    if labeler == "sonnet7way":
        from backend.gen8.labeler_sonnet7way import label_chunks
        labeled = asyncio.run(label_chunks(chunks))
    else:
        from backend.gen8.labeler_opus import label_chunks
        labeled = label_chunks(chunks)
```

---

## 9. Server / session_manager wiring

### `backend/server.py`

Find the extraction endpoint (likely `@app.post("/extract")` or similar). Add a query parameter:

```python
@app.post("/extract")
async def extract(request: Request, ...):
    pipeline = request.query_params.get("pipeline", "gen7")
    if pipeline not in ("gen7", "gen8", "gen8.5"):
        raise HTTPException(400, f"unknown pipeline: {pipeline}")
    ...
    await session_manager.run_extraction(..., pipeline=pipeline)
```

### `backend/session_manager.py`

```python
async def run_extraction(..., pipeline: str = "gen7"):
    if pipeline == "gen7":
        from backend.gen7.pipeline import run_gen7_extraction
        return await run_gen7_extraction(...)
    elif pipeline == "gen8":
        from backend.gen8.pipeline import run as run_gen8
        return await run_gen8(..., labeler="opus")
    elif pipeline == "gen8.5":
        from backend.gen8.pipeline import run as run_gen8
        return await run_gen8(..., labeler="sonnet7way")
```

---

## 10. README update for gen8 / gen8.5 outputs

Update `backend/gen8/delivery_readme.py` (copy from gen7 and modify):

The header table gains:
- `Pipeline: gen8` or `gen8.5`
- `Container hash: <full sha>`
- `Verifier model: gpt-5.4-mini` (or `claude-sonnet-4-6 (same-family-fallback)`)
- `Verification status: <N> verified / <M> verification_failed / <K> unverified`

A new section "Verification" lists each artifact's status. If any are `verification_failed` or `same-family-fallback`, a callout box says "Reviewers should start with `divergence_worklist.json`."

The "Quick start" section adds:
- "If you want to **see what the verifier flagged**, open `divergence_worklist.json`. Items are signal-ranked: errors first, then high-frequency divergences."

---

## 11. Test/verify checkpoints (run after each tier)

### After Tier 0
1. Read `backend/output/run_<gen8 run id>/artifacts/predicates.json`. Assert no `==`, no uppercase `AND`/`OR`, no quoted strings.
2. Assert no duplicate predicates (no `p_X` and `p_has_X` both present).
3. Assert every predicate has all 9 required fields.
4. Assert every predicate's `provenance` matches `^WHO 2012 page \d+$`.

### After Tier 1
1. For every `*.json` under the run dir, assert a `*.provenance.json` sidecar exists.
2. Run `verify_sidecar` on each pair; assert all return True.
3. Assert `manifest.json` exists at run root with `container_sha256`.
4. Assert `form.xlsx` exists (aggregate) AND `form_per_module/` directory exists with one xlsx per module.
5. Assert `flowchart_index.png` exists AND `flowcharts_per_module/` directory exists with one PNG per module.

### After Tier 2
1. Read `artifacts/data_flow.json`. Assert `summary.orphan == 0`.
2. Read `artifacts/router.json`. Assert it has `cop1_queue_builder` AND `cop2_next_module` keys.
3. Run `validate_router`; assert empty error list.
4. Read `artifacts/stockout_coverage.json`. Log gap count (informational).
5. For every list-shaped artifact, sample 5 items and assert each has `_order_source` field.

### After Tier 3
1. For every artifact, assert `*.verification.json` sidecar exists.
2. Read `divergence_worklist.json`. Assert structure matches schema.
3. Read each provenance sidecar. Assert `status` is set to one of `verified | unverified | verification_failed`.
4. If verifier ran in `same-family-fallback` mode, assert every verification sidecar has at least one `severity: warn` divergence with `type: same_family_fallback`.

### Final run-3 acceptance
Run gen8 on WHO 2012. Compare to Run 2 (`run_8970cf65`):
- Predicate count: should be ≤ Run 2's 44 (after dedup).
- `_auto_registered` count: should be 0 (replaced by Tier 0 dedup + missingness rule).
- Orphan variable count: should be 0.
- Container hash: should be present in `manifest.json`.
- Every artifact: should have provenance + verification sidecar.

---

## 12. Decision rules for ambiguity (apply in bypass mode)

When a non-blocking ambiguity arises during execution, apply these rules instead of asking:

1. **If a function signature in this plan doesn't match an existing function**: Read the existing function with the Read tool, infer the actual signature, adapt accordingly. Log the adaptation in commit message.
2. **If an import path doesn't exist**: Use Grep to find the actual location. If still missing, create the module per Section 1's repo layout.
3. **If a Stage 3 prompt edit conflicts with an existing prompt section**: Append the new section after existing content. Do not delete or restructure existing prompt content unless it directly contradicts a Tier 0 rule.
4. **If a test in Section 11 fails**: HALT. Do not paper over with try/except. Surface the failure and ask user.
5. **If gen7 needs a hotfix to make gen8 work**: HALT. gen7 is frozen.
6. **If a Python import fails at runtime due to missing dep**: install via pip (`backend/.venv/Scripts/pip install <pkg>`), add to `requirements.txt`, continue.
7. **If a shell command fails on Windows path quoting**: try forward slashes; never silently degrade.

---

## 13. Order of work (strict — do in this order)

1. **Pre-flight 2.1** (chunker page-number check). HALT if fails.
2. **Pre-flight 2.2** (concurrency probe). HALT for gen8.5 only if fails; gen8 proceeds.
3. **Pre-flight 2.3** (verifier billing). Log result; do not halt.
4. **Repo scaffold** (Section 3): create `backend/gen8/`, `backend/provenance/`, `backend/verifier/` directories. Add `__init__.py` to each.
5. **Tier 0** (Section 4). After: run Section 11 Tier 0 tests.
6. **Tier 1** (Section 5). After: run Section 11 Tier 1 tests.
7. **Backfill Run 2** (Section 5.3). After: assert sidecars exist for `run_8970cf65`.
8. **Tier 2** (Section 6). After: run Section 11 Tier 2 tests.
9. **Tier 3** (Section 7). After: run Section 11 Tier 3 tests.
10. **Server wiring** (Section 9). Restart backend. Verify endpoint accepts `pipeline=gen8`.
11. **Smoke test gen8**: extract WHO 2012 via `pipeline=gen8`. Assert manifest.json present, all sidecars present, no Tier 0 violations.
12. **gen8.5 labeler** (Section 8). After: run a single-chunk gen8.5 dry run; verify reconciliation produces non-empty labels.
13. **Smoke test gen8.5**: extract WHO 2012 via `pipeline=gen8.5`. Assert wall clock < 1500s, cost < $10, container hash present.
14. **Final acceptance run** (Section 11 final). Document results in `backend/output/run_<id>/README.md`.

---

## 14. Final deliverable checklist

When the plan is complete, the user can run:

- `pipeline=gen7` → frozen Run 2 behavior, Opus mono, no provenance, no verifier.
- `pipeline=gen8` → Opus mono labeler + provenance + container hash + xlsx breakup + flowchart breakup + traffic cops + manual ordering + source-of-each-variable + stockout coverage + verifier with GPT primary / Anthropic fallback.
- `pipeline=gen8.5` → 7-way Sonnet labeler + everything from gen8.

Each run produces:
- `manifest.json` at run root (container hash + all artifact SHAs)
- `*.provenance.json` sidecar next to every artifact
- `*.verification.json` sidecar next to every artifact (when Tier 3 ran)
- `divergence_worklist.json` (signal-ranked review queue)
- `form.xlsx` (aggregate, deployable to CHT)
- `form_per_module/` directory (review-only XLSForm workbooks)
- `flowchart_index.png` + `flowcharts_per_module/` directory
- `clinical_logic.dmn`, `clinical_logic.json` (provenance-tagged)
- `artifacts/` directory with all 7 core artifacts + `data_flow.json` + `referential_integrity.json` + `stockout_coverage.json`
- `README.md` (navigation file with verification status callouts)
- `FAILURES.md` is no longer hand-written — it is auto-generated from `divergence_worklist.json` (`severity: error` and `severity: warn` items)

---

## 15. Open architectural decisions deferred (not in scope for this build)

These are NOT part of Tier 0-3 and should NOT be implemented during this run:
- D1: Repair loop yes/no (currently: NO — verifier is read-only)
- D2: Two-pipeline consensus (deferred; gen8.5 is already 7-way internal consensus)
- Z3 formal verification (Tier 4 — separate future build)
- MOH human-sovereignty gates (Tier 5 — requires stateful pipeline, separate build)
- Translation pipeline (Tier 5)
- Synthetic-patient differential testing (Tier 4)
- ICD-10 / SNOMED dictionary (skipped per user Q7)

If any of these come up during the build, do not implement them. Note in commit messages that they are deferred.
