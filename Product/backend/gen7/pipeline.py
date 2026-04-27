"""Gen 7 pipeline: Opus-only. micro-chunk -> Opus label -> Opus REPL compile.

Phase 0: Re-chunk to ~1-1.5K tokens (reuses Gen 6 chunker)
Phase 1: 1x Opus per chunk (temp=0, prompt-cached system block)
Phase 2: Opus REPL loads all labeled chunks, writes Python to compile 7 artifacts

Single Anthropic key throughout. No OpenAI dependency.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from backend.gen7.chunker import micro_chunk_guide
from backend.gen7.labeler import label_all_chunks

logger = logging.getLogger(__name__)


def _build_repl_system_prompt(codebook: str) -> str:
    """Build the system prompt for the Opus REPL compilation phase.

    Gen 7 v2 uses a Module Maker + Dispatcher pattern (professor's design).
    Instead of one-shot compiling all 7 artifacts, the REPL:
      Phase A: scan deduped labels, identify modules and their trigger flags
      Phase B: use llm_query_batched() to generate each module's DMN in
               parallel (one sub-call per module, sees cached guide+labels)
      Phase C: programmatically assemble the dispatcher from the module
               list using priority metadata + alphabetical tie-breaking
      Phase D: assemble the other 6 artifacts + FINAL_VAR

    Two blocks are cached in the system prompt (invisible here; attached
    by the rlm_runner monkey-patch via _set_gen7_cached_context):
      - FULL SOURCE GUIDE: reconstructed guide text for grounding
      - DEDUPED LABEL INVENTORY: clean list of unique (id, type) entries
    Every REPL turn AND every llm_query_batched sub-call reads both at
    cache-read rates ($0.50/M instead of $5/M) after the first call.
    """
    return (
        "You are compiling clinical decision logic from a PRE-LABELED, PRE-DEDUPED "
        "inventory. Two cached blocks are attached to this system prompt:\n"
        "  - FULL SOURCE GUIDE (reconstructed text) -- ground truth for Module Maker\n"
        "  - DEDUPED LABEL INVENTORY -- unique (id, type) entries with quote_context\n"
        "You see both cache blocks on every turn AT CACHE-READ RATES (10x cheaper\n"
        "than fresh input). Use them freely. Do NOT paste them into code blocks\n"
        "-- reference them by content in your Python decisions.\n\n"
        "The `context` Python variable in the REPL also contains the per-chunk\n"
        "labels (nested structure) for cross-referencing section_ids if needed.\n\n"
        "AVAILABLE REPL FUNCTIONS:\n"
        "  - print(...)               -- output echoed back on next turn\n"
        "  - FINAL_VAR(value)         -- returns `value` as final answer. Call ONCE at end.\n"
        "  - llm_query(prompt)        -- invoke a sub-model on ONE prompt (rare)\n"
        "  - llm_query_batched(prompts) -- PREFERRED for per-module extraction.\n"
        "                                Takes list[str], returns list[str], up to 10 parallel.\n"
        "\n"
        "There is NO emit_artifact(), NO validate(), NO z3_check(). You build\n"
        "one `clinical_logic` dict and FINAL_VAR it. Nothing else.\n\n"
        "================================================================\n"
        "ARCHITECTURE: Module Maker + Dispatcher (4 phases, multi-turn)\n"
        "================================================================\n"
        "Modules are the hard part. Supply/variables/phrase_bank are easy\n"
        "because they're flat lists. The dispatcher (formerly 'router') is\n"
        "built programmatically from module metadata. So the flow is:\n\n"
        "PHASE A -- Scan & plan (1 turn):\n"
        "  From the DEDUPED LABEL INVENTORY (cached block), extract the module\n"
        "  list. For each `mod_X` label, record:\n"
        "    - module_id\n"
        "    - display_name (from span)\n"
        "    - trigger flag: the symptom/condition that activates this module.\n"
        "      Often one of the variable labels (q_has_X, ex_X). Infer from\n"
        "      the module's quote_context and nearby variables.\n"
        "    - priority tier (int):\n"
        "         0 = emergency/priority_exit (danger signs, severe)\n"
        "         1 = startup (first module of the flow, demographic setup)\n"
        "       100 = regular clinical modules (default)\n"
        "       999 = closing module (fires only after all required done)\n"
        "    - done_flag name: e.g. `mod_X_done`\n"
        "  Print the plan with module_ids and priorities.\n\n"
        "PHASE B -- Module Maker (1 turn, parallel sub-calls):\n"
        "  Use llm_query_batched(prompts) with one prompt per module. Each\n"
        "  sub-call sees the SAME cached guide + labels (cheap) plus a\n"
        "  focused instruction: 'Generate a DMN for mod_X using the\n"
        "  predicates and phrase_bank entries that describe its rules.'\n"
        "  Each sub-response returns a JSON object: rules, inputs, outputs,\n"
        "  done_flag output, has_Y cross-trigger outputs, is_priority_exit\n"
        "  output where applicable. Parse each response into a module dict.\n\n"
        "  ALL prompts must be constructed BEFORE calling llm_query_batched\n"
        "  (it runs all in parallel). Each prompt is a string that includes:\n"
        "    - module_id and display_name\n"
        "    - the module's trigger flag and done_flag\n"
        "    - a pointer to the cached guide + labels ('refer to the\n"
        "      DEDUPED LABEL INVENTORY above')\n"
        "    - the module schema: {'module_id': ..., 'rules': [...]}\n"
        "    - explicit JSON-only output instruction\n\n"
        "PHASE C -- Programmatic Dispatcher (0 LLM calls):\n"
        "  Build the dispatcher dict from the module list. This is PURE CODE,\n"
        "  no LLM variance:\n"
        "    rows = []\n"
        "    # Priority 0: is_priority_exit short-circuit\n"
        "    rows.append({'priority': 0, 'condition': 'is_priority_exit == true',\n"
        "                 'output_module': 'PRIORITY_EXIT', 'description': '...'})\n"
        "    # Priority 1: startup if not done\n"
        "    startup_mod = [m for m in modules if m['priority'] == 1][0]\n"
        "    rows.append({'priority': 1,\n"
        "                 'condition': f'{startup_mod[\"done_flag\"]} == false',\n"
        "                 'output_module': startup_mod['module_id'], ...})\n"
        "    # Priority 100 (sorted alphabetically by module_id for determinism):\n"
        "    for m in sorted([m for m in modules if m['priority'] == 100],\n"
        "                    key=lambda m: m['module_id']):\n"
        "        rows.append({'priority': 100,\n"
        "                     'condition': f'{m[\"trigger_flag\"]} == true AND '\n"
        "                                  f'{m[\"done_flag\"]} == false',\n"
        "                     'output_module': m['module_id'], ...})\n"
        "    # Priority 999: closing\n"
        "    closing_mod = [m for m in modules if m['priority'] == 999][0]\n"
        "    rows.append({'priority': 999, 'condition': 'true',\n"
        "                 'output_module': closing_mod['module_id'], ...})\n"
        "    dispatcher = {'hit_policy': 'priority', 'rows': rows}\n\n"
        "PHASE D -- Flat artifacts + final assembly (1 turn):\n"
        "  Build the easy artifacts from the deduped labels:\n"
        "    - supply_list: filter type==supply_list, one entry per label\n"
        "    - variables: filter type==variables, add per-module has_X /\n"
        "      mod_X_done booleans + is_priority_exit boolean\n"
        "    - predicates: filter type==predicates, derive threshold_expression\n"
        "      from span + quote_context. fail_safe=1 for equipment-dependent,\n"
        "      0 for self-reported.\n"
        "    - phrase_bank: filter type==phrase_bank, one entry per label\n"
        "    - integrative: leave minimal (or empty) since the dispatcher now\n"
        "      handles cross-module flow via flags\n"
        "  Assemble clinical_logic with all 7 keys: supply_list, variables,\n"
        "  predicates, modules, router (= dispatcher), phrase_bank, integrative.\n"
        "  Call FINAL_VAR(clinical_logic).\n\n"
        "================================================================\n"
        "ARTIFACT SCHEMAS\n"
        "================================================================\n"
        "  supply_list: list of dicts with\n"
        "    (id, display_name, kind='equipment'|'consumable', source_quote, source_section_id)\n"
        "  variables: list of dicts with\n"
        "    (id, display_name, prefix, data_type, unit, depends_on_supply,\n"
        "     source_quote, source_section_id)\n"
        "  predicates: list of dicts with\n"
        "    (id, threshold_expression, fail_safe=0|1, source_vars=list,\n"
        "     human_label, source_quote, source_section_id)\n"
        "  modules: dict keyed by module_id, each with\n"
        "    (module_id, display_name, hit_policy='unique', priority=int,\n"
        "     trigger_flag='has_X', done_flag='mod_X_done', inputs=list,\n"
        "     outputs=list, rules=list of dicts with condition + outputs dict)\n"
        "    Each rule's outputs dict includes module-state changes:\n"
        "     { 'treatment_code': '...', 'is_priority_exit': true|false,\n"
        "       'priority_exit_destination': 'hospital'|'clinic'|null,\n"
        "       '<done_flag>': true, 'has_Y': true (if discovers another symptom) }\n"
        "  router: dict with\n"
        "    (hit_policy='priority', rows=list of dicts with\n"
        "     priority, condition, output_module, description)\n"
        "    Built programmatically in Phase C, NOT by an LLM call.\n"
        "  phrase_bank: list of dicts with\n"
        "    (id, category, text, module_context, source_quote, source_section_id)\n"
        "  integrative: dict with (rules=[]) -- can be empty since dispatcher\n"
        "    handles cross-module flow via flags.\n\n"
        f"NAMING CODEBOOK (for reference):\n{codebook[:2500]}\n\n"
        "REPRODUCIBILITY (non-negotiable):\n"
        "  - Sort all output lists/dicts alphabetically by id.\n"
        "  - Use content-derived IDs (never random UUIDs).\n"
        "  - Phase C dispatcher assembly is deterministic (sorted alphabetically).\n"
        "  - If you use llm_query_batched, construct the prompt list SORTED by\n"
        "    module_id so the batch order is deterministic.\n"
        "  - Pick canonical forms over shortcuts. Pick simple over clever.\n"
    )


async def run_gen7_extraction(
    guide_json: dict,
    anthropic_key: str,
    naming_codebook: str = "",
    output_dir: Optional[Path] = None,
    on_progress: Any = None,
    manual_name_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full Gen 7 pipeline.

    Only requires an Anthropic API key. No OpenAI dependency.

    Args:
        guide_json: Decomposed guide JSON with sections.
        anthropic_key: Anthropic API key (user BYOK).
        naming_codebook: Variable naming codebook string.
        output_dir: Where to save output files.
        on_progress: Async callback for progress events.

    Returns:
        Dict with clinical_logic, labeled_chunks, stats, status.
    """
    start = time.time()

    # Reset _run_usage so this run's accumulator starts at zero (otherwise
    # consecutive Gen 7 runs in the same backend process accumulate cross-run).
    try:
        from backend.rlm_runner import _reset_run_usage
        _reset_run_usage()
    except Exception:
        pass


    # -- Phase 0: Micro-chunk --
    logger.info("Gen7: Phase 0 -- micro-chunking guide")
    if on_progress:
        await on_progress({"phase": "chunk", "status": "starting"})

    chunks = micro_chunk_guide(guide_json)
    logger.info("Gen7: %d micro-chunks created", len(chunks))

    if on_progress:
        await on_progress({"phase": "chunk", "status": "done", "count": len(chunks)})

    # -- Phase 1: Opus labeling --
    logger.info("Gen7: Phase 1 -- Opus labeling (%d chunks)", len(chunks))
    if on_progress:
        await on_progress({"phase": "label", "status": "starting", "model": "opus"})

    # Forward per-chunk labeling events to the progress callback
    async def _on_chunk_labeled(event: dict) -> None:
        if on_progress:
            await on_progress({**event, "phase": "label"})

    labeled_chunks = await label_all_chunks(
        chunks, naming_codebook, anthropic_key,
        on_chunk_labeled=_on_chunk_labeled,
    )

    total_labels = sum(len(c.get("labels", [])) for c in labeled_chunks)
    label_errors = sum(
        1 for c in labeled_chunks
        if c.get("_labeling_meta", {}).get("error")
    )

    logger.info(
        "Gen7: labeling complete -- %d labels across %d chunks, %d errors",
        total_labels, len(labeled_chunks), label_errors,
    )

    if on_progress:
        await on_progress({
            "phase": "label", "status": "done",
            "total_labels": total_labels, "errors": label_errors,
        })

    # -- Post-labeling: dedupe labels and reconstruct full guide text --
    # Algorithmic exact-string dedup on (id, type) produces the 26th artifact.
    # No semantic matching -- the REPL handles semantic merging with reading
    # comprehension using the cached guide text as ground truth.
    from backend.gen7.labeler import deduplicate_labels
    deduped_labels = deduplicate_labels(labeled_chunks)
    logger.info(
        "Gen7: dedup produced %d unique labels from %d raw",
        len(deduped_labels), total_labels,
    )

    # Reconstruct the full source guide text by concatenating chunk.text in
    # chunk_index order. This becomes one of the cached system blocks so the
    # Module Maker phase can ground each module's DMN against source text
    # (not just the quote_context fragments stored per-label).
    # Filter out chunks whose text is an ingestion placeholder (e.g.
    # "(original text as provided)") or empty -- these are PDF extraction
    # failures that should not contribute to the cached source.
    _PLACEHOLDER_PATTERNS = (
        "(original text as provided)",
        "(content unavailable)",
        "(no text extracted)",
    )

    def _is_real_chunk(text: str) -> bool:
        """Drop only chunks whose ENTIRE content is a placeholder or empty.
        A chunk that CONTAINS a placeholder substring but also has real text
        (e.g., as a PDF watermark mid-chunk) stays in.
        """
        t = (text or "").strip()
        if not t or len(t) < 20:
            return False
        # Strip all known placeholders and see if anything meaningful remains
        cleaned = t
        for p in _PLACEHOLDER_PATTERNS:
            cleaned = cleaned.replace(p, "").replace(p.lower(), "").strip()
        return len(cleaned) >= 50

    # Build from the chunker's output, not from labeled_chunks. The chunker's
    # text field is always the real source text Opus received; labeled_chunks'
    # text field is Opus's echo, which sometimes collapses 8K chars of input
    # into the shorthand "(original text as provided)" to save output tokens.
    # Filtering on labeled_chunks dropped ~57% of the guide on the 2026-04-14
    # run even though the labels themselves were grounded in real input.
    reconstructed_guide = "\n\n".join(
        c.get("text", "") for c in sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        if _is_real_chunk(c.get("text", ""))
    )
    logger.info(
        "Gen7: reconstructed guide text = %d chars (~%d tokens) from %d chunker chunks",
        len(reconstructed_guide), len(reconstructed_guide) // 4, len(chunks),
    )

    # Save labeled chunks early (before REPL, in case REPL fails)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "labeled_chunks.json").write_text(
            json.dumps(labeled_chunks, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "deduped_labels.json").write_text(
            json.dumps(deduped_labels, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "reconstructed_guide.txt").write_text(
            reconstructed_guide, encoding="utf-8"
        )
        (output_dir / "chunk_difficulty.json").write_text(json.dumps({
            "total_chunks": len(chunks),
            "distribution": {
                d: sum(1 for c in chunks if c.get("difficulty") == d)
                for d in ["trivial", "easy", "medium", "hard", "extreme"]
            },
            "chunks": [
                {"index": c["chunk_index"], "section": c["section_id"],
                 "difficulty": c["difficulty"], "score": c.get("difficulty_score", 0)}
                for c in chunks
            ],
        }, indent=2), encoding="utf-8")

    # -- Phase 2: Opus REPL compilation --
    logger.info("Gen7: Phase 2 -- Opus REPL compilation from %d labeled chunks", len(labeled_chunks))
    if on_progress:
        await on_progress({"phase": "compile", "status": "starting"})

    # Build the context payload for the REPL: stripped labeled chunks.
    # The REPL accesses these via the `context` Python variable in its local
    # namespace. The FULL guide text and DEDUPED label list are separately
    # attached to the cached system prompt (not in `context`) so they're
    # available to both REPL turns and llm_query_batched sub-calls at
    # cache-read rates.
    repl_context = [
        {
            "chunk_index": c.get("chunk_index", -1),
            "section_id": c.get("section_id", ""),
            "section_title": c.get("section_title", ""),
            "text": c.get("text", ""),
            "labels": c.get("labels", []),
        }
        for c in labeled_chunks
    ]

    system_prompt = _build_repl_system_prompt(naming_codebook)

    # The rlm library's build_rlm_system_prompt() calls
    # custom_system_prompt.format(custom_tools_section=...) which crashes on
    # any literal curly brace (JSON examples, dict shapes, f-string syntax,
    # etc.). Escape all literal braces so str.format() treats them as text,
    # then append the one placeholder the library needs.
    system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
    system_prompt += "\n\n{custom_tools_section}"

    # rlm package lives at project_root/rlm/rlm/; ensure it's importable
    import sys
    rlm_path = str(Path(__file__).parent.parent.parent / "rlm")
    if rlm_path not in sys.path:
        sys.path.insert(0, rlm_path)

    from rlm import RLM
    from rlm.logger import RLMLogger

    # Bridge for REPL iteration events: RLM runs in a worker thread, but
    # on_progress is async. Use run_coroutine_threadsafe to cross the boundary.
    loop = asyncio.get_running_loop()
    repl_iter_count = {"n": 0}
    # Running token totals across REPL iterations for live cost tracking.
    # We read AUTHORITATIVE token counts from rlm_runner._run_usage, which is
    # a global accumulator populated by the monkey-patched Anthropic client's
    # _track_cost() method. It reads response.usage.input_tokens directly from
    # each Anthropic API response, so these are exactly what Anthropic bills.
    # Snapshot baseline BEFORE the REPL starts so we only count REPL tokens
    # (not labeling tokens which already went through _run_usage).
    try:
        from backend.rlm_runner import _run_usage
        _repl_baseline = {
            "input": _run_usage.get("input_tokens", 0),
            "output": _run_usage.get("output_tokens", 0),
        }
    except Exception:
        _run_usage = None
        _repl_baseline = {"input": 0, "output": 0}

    repl_running = {"input_tokens": 0, "output_tokens": 0, "last_in": 0, "last_out": 0}

    def _on_repl_iteration_complete(depth: int, iteration: int, duration: float) -> None:
        """Called from the RLM worker thread after each REPL turn."""
        repl_iter_count["n"] = iteration + 1
        if on_progress:
            code_text = ""
            stdout_text = ""
            iter_in_tokens = 0
            iter_out_tokens = 0

            # Extract code/stdout from trajectory for display
            try:
                trajectory = rlm_logger.get_trajectory()
                if trajectory:
                    iterations_list = trajectory.get("iterations", [])
                    if len(iterations_list) > iteration:
                        it = iterations_list[iteration]
                        code_blocks = it.get("code_blocks", [])
                        if code_blocks:
                            cb = code_blocks[-1]
                            code_text = cb.get("code", "")[:2000]
                            result_obj = cb.get("result", {})
                            if result_obj:
                                stdout_text = result_obj.get("stdout", "")[:3000]
            except Exception:
                pass

            # AUTHORITATIVE token counts from _run_usage (populated by
            # monkey-patched AnthropicClient._track_cost reading response.usage
            # directly). Snapshot-delta pattern: compute what was added since
            # the last iteration. Falls back to tiktoken estimate if unavailable.
            if _run_usage is not None:
                try:
                    current_in = _run_usage.get("input_tokens", 0) - _repl_baseline["input"]
                    current_out = _run_usage.get("output_tokens", 0) - _repl_baseline["output"]
                    iter_in_tokens = current_in - repl_running["last_in"]
                    iter_out_tokens = current_out - repl_running["last_out"]
                    repl_running["last_in"] = current_in
                    repl_running["last_out"] = current_out
                except Exception:
                    pass

            # Fallback: tiktoken estimate (undercounts Claude tokenization by 30-50%)
            if iter_in_tokens == 0 and iter_out_tokens == 0:
                try:
                    from rlm.utils.token_utils import count_tokens
                    trajectory = rlm_logger.get_trajectory()
                    if trajectory:
                        iterations_list = trajectory.get("iterations", [])
                        if len(iterations_list) > iteration:
                            it = iterations_list[iteration]
                            prompt_msgs = it.get("prompt", [])
                            if isinstance(prompt_msgs, list):
                                iter_in_tokens = count_tokens(prompt_msgs, "claude-opus-4-6")
                            elif isinstance(prompt_msgs, str):
                                iter_in_tokens = count_tokens(
                                    [{"role": "user", "content": prompt_msgs}],
                                    "claude-opus-4-6",
                                )
                            resp_text = it.get("response", "") or ""
                            iter_out_tokens = count_tokens(
                                [{"role": "assistant", "content": resp_text}],
                                "claude-opus-4-6",
                            )
                except Exception:
                    pass

            repl_running["input_tokens"] += iter_in_tokens
            repl_running["output_tokens"] += iter_out_tokens

            future = asyncio.run_coroutine_threadsafe(
                on_progress({
                    "phase": "compile",
                    "event": "repl_iteration",
                    "iteration": iteration + 1,
                    "duration_sec": round(duration, 2),
                    "code": code_text,
                    "stdout": stdout_text,
                    "input_tokens": iter_in_tokens,
                    "output_tokens": iter_out_tokens,
                    "repl_total_input_tokens": repl_running["input_tokens"],
                    "repl_total_output_tokens": repl_running["output_tokens"],
                }),
                loop,
            )
            try:
                future.result(timeout=5)
            except Exception:
                pass

    rlm_logger = RLMLogger()
    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": "claude-opus-4-6",
            "api_key": anthropic_key,
            "max_tokens": 16384,
        },
        environment="local",
        environment_kwargs={
            "context_payload": repl_context,
        },
        max_iterations=30,
        custom_system_prompt=system_prompt,
        compaction=False,
        max_budget=30.0,
        max_errors=5,
        max_timeout=1800.0,
        logger=rlm_logger,
        verbose=False,
        on_iteration_complete=_on_repl_iteration_complete,
    )

    # Build the initial message that kickstarts the REPL
    initial_message = _build_initial_repl_message(repl_context, total_labels, len(deduped_labels))

    # Install Gen 7 cached system blocks (guide text + deduped labels) via
    # the rlm_runner monkey-patch. Both blocks get cache_control so every
    # REPL turn AND every llm_query_batched sub-call reads them at 0.1x
    # Opus input rate after the first call. Cleared in a finally clause so
    # a later legacy-pipeline run doesn't inherit stale cache state.
    from backend.rlm_runner import _set_gen7_cached_context
    _set_gen7_cached_context(reconstructed_guide, deduped_labels)

    # Capture Stage 3 REPL input BEFORE the call so even a crash leaves a
    # record. The final response + iteration count are filled in after.
    _stage3_artifact = {
        "phase": "stage3_repl",
        "model": "claude-opus-4-6",
        "system_prompt": system_prompt,
        "cached_source_guide": reconstructed_guide,
        "cached_deduped_labels": deduped_labels,
        "initial_user_message": initial_message,
        "input_chars_total": (
            len(system_prompt) + len(reconstructed_guide)
            + len(json.dumps(deduped_labels)) + len(initial_message)
        ),
        "reconstructed_guide_chars": len(reconstructed_guide),
        "deduped_label_count": len(deduped_labels),
        "response_text": None,
        "total_iterations": None,
    }

    try:
        result = await asyncio.to_thread(
            rlm.completion,
            initial_message,
        )
    finally:
        _set_gen7_cached_context(None, None)

    _stage3_artifact["response_text"] = str(getattr(result, "response", "") or "")
    _stage3_artifact["response_chars"] = len(_stage3_artifact["response_text"])

    # Iteration count: prefer the per-callback counter, but fall back to the
    # rlm logger's trajectory length if the callback never fired (silent
    # callback failures are swallowed by the rlm library's try/except, which
    # leaves repl_iter_count["n"] at zero even after a multi-turn session).
    iter_count_from_callback = repl_iter_count["n"]
    iter_count_from_trajectory = 0
    try:
        traj = rlm_logger.get_trajectory()
        if traj:
            iter_count_from_trajectory = len(traj.get("iterations", []) or [])
    except Exception:
        pass
    _stage3_artifact["total_iterations"] = max(
        iter_count_from_callback, iter_count_from_trajectory
    )

    elapsed = time.time() - start

    if on_progress:
        await on_progress({
            "phase": "compile", "status": "done",
            "elapsed": round(elapsed, 1),
            "total_iterations": repl_iter_count["n"],
        })

    # -- Parse REPL output --
    response_text = getattr(result, "response", None)
    clinical_logic = _parse_repl_result(response_text)

    status = "passed" if clinical_logic and isinstance(clinical_logic, dict) else "failed"

    # -- Save all outputs --
    if output_dir:
        _save_outputs(output_dir, clinical_logic, labeled_chunks, system_prompt)

    logger.info(
        "Gen7: complete in %.0fs -- status=%s chunks=%d labels=%d",
        elapsed, status, len(chunks), total_labels,
    )

    # -- AUTHORITATIVE cost snapshot from rlm_runner._run_usage --
    # _run_usage is populated by BOTH code paths:
    #   1. Labeler Call 1 + Call 2 -> accumulate_catcher_usage() -> _run_usage
    #   2. REPL main + llm_query_batched sub-calls -> monkey-patched
    #      _acompletion_with_cache -> _accumulate_run_usage() -> _run_usage
    # This captures EVERY Anthropic API call in the run, including cache
    # creation (2x) and cache read (0.1x) at the correct rates.
    from backend.rlm_runner import _snapshot_run_usage
    run_snap = _snapshot_run_usage()
    total_in = run_snap.get("input_tokens", 0)
    total_out = run_snap.get("output_tokens", 0)
    total_cached_read = run_snap.get("cached_tokens", 0)
    total_cache_write = run_snap.get("cache_write_tokens", 0)
    total_cost = run_snap.get("cost_usd", 0.0)
    calls_opus = run_snap.get("calls_opus", 0)
    calls_sonnet = run_snap.get("calls_sonnet", 0)
    calls_haiku = run_snap.get("calls_haiku", 0)

    # Per-phase breakdowns for telemetry (not authoritative, just informative)
    label_input_tokens = sum(
        c.get("_labeling_meta", {}).get("input_tokens", 0) for c in labeled_chunks
    )
    label_output_tokens = sum(
        c.get("_labeling_meta", {}).get("output_tokens", 0) for c in labeled_chunks
    )
    label_cache_read = sum(
        c.get("_labeling_meta", {}).get("cache_read_tokens", 0) for c in labeled_chunks
    )
    label_cost = label_input_tokens * 5.0 / 1e6 + label_output_tokens * 25.0 / 1e6
    repl_input_tokens = max(0, total_in - label_input_tokens)
    repl_output_tokens = max(0, total_out - label_output_tokens)
    repl_cost = max(0.0, total_cost - label_cost)
    # Same fallback as stage3_artifact: prefer callback, fall back to logger.
    repl_iterations = max(
        repl_iter_count["n"],
        _stage3_artifact.get("total_iterations", 0) or 0,
    )

    logger.info("=" * 60)
    logger.info("Gen7 COST SUMMARY (from _run_usage, reconcile with Anthropic billing):")
    logger.info("  Labeling (Call 1 + Call 2 per chunk): %d in / %d out tokens ~= $%.4f",
                label_input_tokens, label_output_tokens, label_cost)
    logger.info("  REPL (main + llm_query_batched): %d in / %d out tokens ~= $%.4f",
                repl_input_tokens, repl_output_tokens, repl_cost)
    logger.info("  TOTAL (authoritative): %d in / %d out / %d cached-read / %d cache-write = $%.4f",
                total_in, total_out, total_cached_read, total_cache_write, total_cost)
    logger.info("  Calls: Opus=%d Sonnet=%d Haiku=%d", calls_opus, calls_sonnet, calls_haiku)
    logger.info("  Rates: Opus 4.6 @ $5/M in, $25/M out, $0.50/M cache-read, $10/M cache-write")
    logger.info("=" * 60)

    # -- Bundle-meta writers: sendable system prompt, stage3_repl.json,
    # referential_integrity.json, data_flow.json, and README.md. All
    # deterministic, no LLM calls. Runs last so stats are populated.
    if output_dir and clinical_logic:
        try:
            _write_bundle_meta(
                output_dir=output_dir,
                clinical_logic=clinical_logic,
                labeled_chunks=labeled_chunks,
                repl_system_prompt=system_prompt,
                naming_codebook=naming_codebook,
                stage3_artifact=_stage3_artifact,
                run_id=output_dir.name,
                manual_name=(
                    manual_name_hint
                    or (guide_json.get("metadata", {}) or {}).get("manual_name", "")
                    or (guide_json.get("metadata", {}) or {}).get("filename", "")
                    or "unknown"
                ),
                started_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start)),
                wall_clock_sec=elapsed,
                stats={
                    "total_chunks": len(chunks),
                    "total_labels": total_labels,
                    "label_errors": label_errors,
                    "label_cost_usd": round(label_cost, 4),
                    "label_input_tokens": label_input_tokens,
                    "label_output_tokens": label_output_tokens,
                    "label_cache_read_tokens": label_cache_read,
                    "repl_cost_usd": round(repl_cost, 4),
                    "repl_input_tokens": repl_input_tokens,
                    "repl_output_tokens": repl_output_tokens,
                    "repl_iterations": repl_iterations,
                    "total_cost_usd": round(label_cost + repl_cost, 4),
                    "elapsed_sec": round(elapsed, 1),
                    "model": "claude-opus-4-6",
                    "provider": "anthropic-only",
                },
            )
        except Exception as exc:
            logger.warning("Gen7: bundle-meta writer failed: %s", exc, exc_info=True)

    return {
        "clinical_logic": clinical_logic,
        "labeled_chunks": labeled_chunks,
        "chunk_difficulty": [
            {"index": c["chunk_index"], "section": c["section_id"],
             "difficulty": c["difficulty"]}
            for c in chunks
        ],
        "stats": {
            "total_chunks": len(chunks),
            "total_labels": total_labels,
            "label_errors": label_errors,
            "label_cost_usd": round(label_cost, 4),
            "label_input_tokens": label_input_tokens,
            "label_output_tokens": label_output_tokens,
            "label_cache_read_tokens": label_cache_read,
            "repl_cost_usd": round(repl_cost, 4),
            "repl_input_tokens": repl_input_tokens,
            "repl_output_tokens": repl_output_tokens,
            "repl_iterations": repl_iterations,
            "total_cost_usd": round(label_cost + repl_cost, 4),
            "elapsed_sec": round(elapsed, 1),
            "model": "claude-opus-4-6",
            "provider": "anthropic-only",
        },
        "status": status,
    }


def _build_initial_repl_message(
    repl_context: list[dict],
    total_labels: int,
    deduped_label_count: int,
) -> str:
    """Build the first message sent to the Opus REPL.

    Walks the model through the 4-phase Module Maker + Dispatcher flow:
    Phase A scan, Phase B llm_query_batched module extraction, Phase C
    programmatic dispatcher, Phase D flat artifacts + FINAL_VAR.
    """
    return (
        f"The labeled guide has {len(repl_context)} chunks with {total_labels} raw labels "
        f"and {deduped_label_count} DEDUPED unique (id, type) entries.\n\n"
        "Two cached blocks are attached to your system prompt:\n"
        "  - FULL SOURCE GUIDE (reconstructed)\n"
        "  - DEDUPED LABEL INVENTORY (the clean list you will compile from)\n"
        "Every REPL turn and every llm_query_batched sub-call reads both at\n"
        "cache-read rates. Use them freely.\n\n"
        f"The `context` Python variable also contains the per-chunk labels "
        f"(list of {len(repl_context)} chunks with 'labels' arrays) for when "
        f"you need per-section grounding.\n\n"
        "Start with PHASE A -- scan and plan. In one code block:\n"
        "```repl\n"
        "# Phase A: gather all deduped labels, group by type, identify modules.\n"
        "# The deduped inventory lives in context; flatten chunks to get labels.\n"
        "all_labels = []\n"
        "for c in context:\n"
        "    for l in c.get('labels', []):\n"
        "        all_labels.append(l)\n"
        "print(f'Total raw labels across chunks: {len(all_labels)}')\n"
        "\n"
        "by_type = {}\n"
        "for l in all_labels:\n"
        "    t = l.get('type', 'unknown')\n"
        "    by_type.setdefault(t, []).append(l)\n"
        "for t in sorted(by_type):\n"
        "    print(f'  {t}: {len(by_type[t])} labels')\n"
        "\n"
        "# Build the module plan: list of {module_id, display_name, priority,\n"
        "# trigger_flag, done_flag}. Infer priority from module_id semantics:\n"
        "#   contains 'danger', 'severe', 'emergency' -> priority 0 (priority_exit)\n"
        "#   'intro', 'demographics', 'startup', 'initial' -> priority 1 (startup)\n"
        "#   'closing', 'finalize', 'wrap' -> priority 999 (closing)\n"
        "#   everything else -> priority 100 (regular)\n"
        "module_labels = by_type.get('modules', [])\n"
        "seen = set()\n"
        "module_plan = []\n"
        "for m in sorted(module_labels, key=lambda x: x.get('id', '')):\n"
        "    mid = m.get('id', '').strip()\n"
        "    if not mid or mid in seen:\n"
        "        continue\n"
        "    seen.add(mid)\n"
        "    name = m.get('span', mid).strip()\n"
        "    lid_lower = mid.lower()\n"
        "    if any(k in lid_lower for k in ['danger', 'severe', 'emergency', 'critical']):\n"
        "        priority = 0\n"
        "    elif any(k in lid_lower for k in ['intro', 'startup', 'initial', 'demographics', 'ask_first']):\n"
        "        priority = 1\n"
        "    elif any(k in lid_lower for k in ['closing', 'finalize', 'wrap', 'summary']):\n"
        "        priority = 999\n"
        "    else:\n"
        "        priority = 100\n"
        "    topic = mid.replace('mod_', '', 1)\n"
        "    trigger_flag = f'has_{topic}'\n"
        "    done_flag = f'{mid}_done'\n"
        "    module_plan.append({\n"
        "        'module_id': mid,\n"
        "        'display_name': name,\n"
        "        'priority': priority,\n"
        "        'trigger_flag': trigger_flag,\n"
        "        'done_flag': done_flag,\n"
        "        'quote_context': m.get('quote_context', ''),\n"
        "    })\n"
        "print(f'\\nModule plan ({len(module_plan)} modules):')\n"
        "for mp in module_plan:\n"
        "    print(f'  p={mp[\"priority\"]:3d} {mp[\"module_id\"]} (trigger={mp[\"trigger_flag\"]}, done={mp[\"done_flag\"]})')\n"
        "```\n\n"
        "After Phase A prints, proceed to Phase B: construct a list of prompts "
        "(one per module), call llm_query_batched() to generate all module DMNs "
        "in parallel, parse responses into module dicts.\n\n"
        "Then Phase C: programmatically assemble the dispatcher from module_plan.\n\n"
        "Then Phase D: build supply_list, variables, predicates, phrase_bank from "
        "the deduped label types. Assemble clinical_logic with all 7 keys and "
        "call FINAL_VAR(clinical_logic). The integrative artifact can be "
        "{'rules': []} since the dispatcher handles cross-module flow via flags.\n\n"
        "IMPORTANT: Build each artifact into a variable first, then use variable "
        "references when assembling clinical_logic. Do NOT put empty literal lists "
        "`[]` in the final dict -- that defeats the whole extraction."
    )


def _parse_repl_result(response_text: Any) -> Optional[dict]:
    """Parse the FINAL_VAR result from the REPL."""
    if isinstance(response_text, dict):
        return response_text
    if isinstance(response_text, str):
        try:
            return json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            pass
        # ast.literal_eval safely parses Python literals (dict, list, str,
        # int, float, bool, None). No arbitrary code execution. Used because
        # the REPL returns Python dict repr with single quotes and True/False.
        import ast
        try:
            result = ast.literal_eval(response_text)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
    return None


def _write_bundle_meta(
    output_dir: Path,
    clinical_logic: dict,
    labeled_chunks: list[dict],
    repl_system_prompt: str,
    naming_codebook: str,
    stage3_artifact: dict,
    run_id: str,
    manual_name: str,
    started_at: str,
    wall_clock_sec: float,
    stats: dict,
) -> None:
    """Write the post-cost-stats bundle metadata:
      - system_prompt.md (full sendable multi-stage bundle + codebook)
      - stage3_repl.json (full Stage 3 input + output; one artifact per run)
      - artifacts/referential_integrity.json (cross-ref audit + auto-reconcile)
      - artifacts/data_flow.json (per-variable scope + lifecycle)
      - README.md (file map + red-flag surfacer)

    All deterministic, no LLM calls. Mutates `clinical_logic` in place by
    adding auto-registered ids to the registries (so the downstream
    clinical_logic.dmn reflects the reconciled shape).

    Per-chunk Opus call info lives in labeled_chunks.json::_labeling_meta;
    see stage1a.diff_chars / stage1a.underproduced for the Stage 1A red flag.
    """
    import subprocess

    # 1. Full sendable system prompt bundle
    try:
        from backend.gen7.labeler import (
            _build_labeling_system_prompt,
            _build_distillation_system_prompt,
        )
        from backend.gen7.system_prompt_bundle import build_sendable_system_prompt

        labeling_prompt = _build_labeling_system_prompt(
            "<see 'Naming codebook' section below>"
        )
        distill_prompt = _build_distillation_system_prompt(
            "<see 'Naming codebook' section below>"
        )
        sendable = build_sendable_system_prompt(
            labeling_prompt=labeling_prompt,
            distillation_prompt=distill_prompt,
            repl_prompt=repl_system_prompt,
            naming_codebook=naming_codebook,
            run_id=run_id,
            manual_name=manual_name,
        )
        (output_dir / "system_prompt.md").write_text(sendable, encoding="utf-8")
        logger.info("Gen7: wrote sendable system_prompt.md (%d chars)", len(sendable))
    except Exception as exc:
        logger.warning("Gen7: sendable system prompt failed: %s", exc)

    # 2. Stage 3 REPL artifact (one record: input + output for the session)
    try:
        (output_dir / "stage3_repl.json").write_text(
            json.dumps(stage3_artifact, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Gen7: wrote stage3_repl.json (input=%d chars, output=%d chars, iterations=%s)",
            stage3_artifact.get("input_chars_total", 0),
            stage3_artifact.get("response_chars", 0),
            stage3_artifact.get("total_iterations"),
        )
    except Exception as exc:
        logger.warning("Gen7: stage3_repl.json write failed: %s", exc)

    # 3. Referential integrity audit + auto-reconciliation
    audit_report = {}
    try:
        from backend.validators.referential_integrity import audit_and_reconcile
        audit_report = audit_and_reconcile(clinical_logic)
        # Overwrite the artifacts with the reconciled versions so the DMN + XLSX
        # + CSV downstream see complete registries.
        patched = audit_report.get("patched", {})
        for key in ("variables", "predicates", "supply_list", "phrase_bank"):
            if key in patched:
                clinical_logic[key] = patched[key]
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Persist the report itself (without the patched copy to keep file size down)
        report_for_disk = {
            "audit_before": audit_report["audit_before"],
            "reconciliation": audit_report["reconciliation"],
            "audit_after": audit_report["audit_after"],
        }
        (artifacts_dir / "referential_integrity.json").write_text(
            json.dumps(report_for_disk, indent=2, default=str), encoding="utf-8"
        )
        # Rewrite the per-artifact JSONs and clinical_logic.json with the
        # reconciled (patched) content so downstream consumers see the
        # complete registries.
        (output_dir / "clinical_logic.json").write_text(
            json.dumps(clinical_logic, indent=2, default=str), encoding="utf-8"
        )
        for key in ("variables", "predicates", "supply_list", "phrase_bank"):
            if key in clinical_logic:
                (artifacts_dir / f"{key}.json").write_text(
                    json.dumps(clinical_logic[key], indent=2, default=str),
                    encoding="utf-8",
                )
        logger.info(
            "Gen7: referential_integrity reconciled (before_missing=%s after_missing=%s)",
            {k: v["missing"] for k, v in audit_report["audit_before"]["stats"].items() if "missing" in v},
            {k: v["missing"] for k, v in audit_report["audit_after"]["stats"].items() if "missing" in v},
        )
    except Exception as exc:
        logger.warning("Gen7: referential_integrity failed: %s", exc, exc_info=True)

    # 4. Data-flow derivation
    data_flow = {}
    try:
        from backend.gen7.data_flow import derive as _derive_data_flow
        data_flow = _derive_data_flow(clinical_logic)
        (output_dir / "artifacts" / "data_flow.json").write_text(
            json.dumps(data_flow, indent=2, default=str), encoding="utf-8"
        )
        logger.info(
            "Gen7: data_flow derived (vars=%d preds=%d orphans=%d)",
            data_flow.get("summary", {}).get("total_variables", 0),
            data_flow.get("summary", {}).get("total_predicates", 0),
            len(data_flow.get("summary", {}).get("orphan_variables", [])),
        )
    except Exception as exc:
        logger.warning("Gen7: data_flow derivation failed: %s", exc, exc_info=True)

    # 5. Git SHA for README traceability
    git_sha = "(unknown)"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(output_dir.parent.parent.parent if (output_dir.parent.parent.parent / ".git").exists() else output_dir),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    # 6. README.md (bundle file map + red flags)
    try:
        from backend.gen7.delivery_readme import build_readme
        readme = build_readme(
            output_dir=output_dir,
            run_id=run_id,
            manual_name=manual_name,
            git_sha=git_sha,
            started_at=started_at,
            wall_clock_sec=wall_clock_sec,
            labeled_chunks=labeled_chunks,
            stats=stats,
            audit_report=audit_report,
            data_flow=data_flow,
            stage3_artifact=stage3_artifact,
        )
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
        logger.info("Gen7: wrote bundle README.md (%d chars)", len(readme))
    except Exception as exc:
        logger.warning("Gen7: README build failed: %s", exc, exc_info=True)


def _save_outputs(
    output_dir: Path,
    clinical_logic: Optional[dict],
    labeled_chunks: list[dict],
    system_prompt: str,
) -> None:
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save system prompt
    (output_dir / "system_prompt.md").write_text(system_prompt, encoding="utf-8")

    if not clinical_logic or not isinstance(clinical_logic, dict):
        logger.warning("Gen7: no clinical_logic to save")
        return

    # 2. Save clinical_logic.json (combined)
    (output_dir / "clinical_logic.json").write_text(
        json.dumps(clinical_logic, indent=2, default=str), encoding="utf-8"
    )

    # 3. Save individual artifact JSONs
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name, artifact_data in clinical_logic.items():
        (artifacts_dir / f"{artifact_name}.json").write_text(
            json.dumps(artifact_data, indent=2, default=str), encoding="utf-8"
        )

    # 4. Run converters: DMN XML, XLSX, Mermaid, CSV
    try:
        from backend.converters import convert_to_dmn, convert_to_xlsx, convert_to_mermaid, convert_to_csv
    except ImportError:
        logger.warning("Gen7: converters not available, skipping DMN/XLSX/Mermaid/CSV")
        convert_to_dmn = convert_to_xlsx = convert_to_mermaid = convert_to_csv = None

    if convert_to_dmn:
        try:
            dmn_xml = convert_to_dmn(clinical_logic)
            (output_dir / "clinical_logic.dmn").write_text(dmn_xml, encoding="utf-8")
            logger.info("Gen7: saved DMN XML")
        except Exception as exc:
            logger.warning("Gen7: DMN conversion failed: %s", exc)

    if convert_to_xlsx:
        try:
            xlsx_path = str(output_dir / "form.xlsx")
            convert_to_xlsx(clinical_logic, xlsx_path)
            logger.info("Gen7: saved XLSX")
        except Exception as exc:
            logger.warning("Gen7: XLSX conversion failed: %s", exc)

    if convert_to_mermaid:
        try:
            mermaid_src = convert_to_mermaid(clinical_logic)
            (output_dir / "flowchart.md").write_text(mermaid_src, encoding="utf-8")
            logger.info("Gen7: saved Mermaid flowchart")
        except Exception as exc:
            logger.warning("Gen7: Mermaid conversion failed: %s", exc)

    if convert_to_csv:
        try:
            csvs = convert_to_csv(clinical_logic)
            for csv_name, csv_content in csvs.items():
                (output_dir / f"{csv_name}.csv").write_text(csv_content, encoding="utf-8")
            logger.info("Gen7: saved %d CSVs", len(csvs))
        except Exception as exc:
            logger.warning("Gen7: CSV conversion failed: %s", exc)

    # 5. Save test suite (frozen labels for downstream validation)
    all_labels = []
    for c in labeled_chunks:
        for label in c.get("labels", []):
            label_copy = dict(label)
            label_copy["_chunk_index"] = c.get("chunk_index", -1)
            label_copy["_section_id"] = c.get("section_id", "")
            all_labels.append(label_copy)

    (output_dir / "test_suite.json").write_text(json.dumps({
        "total_labels": len(all_labels),
        "labels_by_type": {
            t: [l for l in all_labels if l.get("type") == t]
            for t in ["supply_list", "variables", "predicates", "modules",
                      "phrase_bank", "router", "integrative"]
        },
    }, indent=2, default=str), encoding="utf-8")

    # 6. List all output files
    logger.info("Gen7: output files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            logger.info("  %s (%d bytes)", f.relative_to(output_dir), f.stat().st_size)
