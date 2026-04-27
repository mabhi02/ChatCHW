"""Assemble the full multi-stage system prompt for a run.

Replaces the old behavior where `system_prompt.md` in the output bundle
held only the Stage 3 REPL compile prompt. The sendable version contains
ALL three prompts Opus sees in the pipeline:

    Stage 1 (per-chunk labeling)     -> labeler._build_labeling_system_prompt
    Stage 2 (per-chunk QC pass)      -> labeler._build_distillation_system_prompt
    Stage 3 (REPL compilation)       -> pipeline._build_repl_system_prompt

Output is a single markdown document with one H2 per stage and the full
prompt text verbatim. Curly-brace escaping (the {{ }} doubling) from the
rlm library is unescaped for readability: what the model actually sees
at inference time, not the pre-format-template artifact.
"""

from __future__ import annotations


def build_sendable_system_prompt(
    labeling_prompt: str,
    distillation_prompt: str,
    repl_prompt: str,
    naming_codebook: str = "",
    run_id: str = "",
    manual_name: str = "",
) -> str:
    # Unescape the {{ }} doubling from the rlm library's str.format call so
    # readers see what Opus actually sees, not the pre-format template.
    repl_clean = repl_prompt.replace("{{", "{").replace("}}", "}")
    # Drop the trailing {custom_tools_section} placeholder: it's a runtime
    # injection point, not a prompt the model reads in that form.
    repl_clean = repl_clean.replace("{custom_tools_section}", "").rstrip()

    header = [
        "# CHW Navigator System Prompts",
        "",
        "_Full multi-stage prompt bundle: every prompt Opus sees in this pipeline._",
    ]
    if run_id:
        header.append(f"_Run ID: `{run_id}`_")
    if manual_name:
        header.append(f"_Manual: {manual_name}_")
    header += ["_Model: Claude Opus 4.6 at temperature 0 for every stage._", ""]

    overview = [
        "## Pipeline overview",
        "",
        "Three Opus calls per chunk-or-session against the same content-hashed manual. "
        "The identification claim is that the system prompts below are the only experimental variable: "
        "chunking is deterministic Python, label dedup is exact-string, validators are Python, "
        "XLSForm / DMN / flowchart assembly is deterministic. Any run-to-run diff is attributable "
        "to Opus's own stochasticity, nothing else.",
        "",
        "| Stage | Phase | Input | Output |",
        "|-------|-------|-------|--------|",
        "| 1 | Per-chunk labeling (Call 1) | one ~2K-token chunk | labels array |",
        "| 2 | Per-chunk QC (Call 2) | Stage 1 candidate labels | verified labels |",
        "| 3 | REPL compilation | deduped labels + reconstructed guide | 7 DMN artifacts |",
        "",
        "The naming codebook is injected into all three stages so ids are compatible across stages.",
        "",
    ]

    stage1 = [
        "## Stage 1: Chunk labeling system prompt",
        "",
        "Used once per micro-chunk. Anthropic prompt caching is set on the system block so the codebook "
        "and instructions are read at cache-read rates ($0.50/M) for every chunk after the first.",
        "",
        "```",
        labeling_prompt.strip(),
        "```",
        "",
    ]

    stage2 = [
        "## Stage 2: Label QC / distillation system prompt",
        "",
        "Second pass per chunk. Verifies each Stage 1 candidate label is grounded in the input text "
        "and has a codebook-compliant ID. Drops hallucinations. NEVER adds new labels: only verifies "
        "or drops the candidates from Call 1.",
        "",
        "```",
        distillation_prompt.strip(),
        "```",
        "",
    ]

    stage3 = [
        "## Stage 3: REPL compilation system prompt",
        "",
        "The prompt that drives the Python REPL for the final compile session. Two blocks attach to "
        "this prompt at runtime (the full reconstructed guide text, and the deduped label inventory) "
        "so every REPL turn and every `llm_query_batched` sub-call reads them at cache-read rates. "
        "The `{custom_tools_section}` placeholder at the end is filled in by the rlm library with the "
        "tool signatures (`emit_artifact`, `validate`, `z3_check`, `FINAL_VAR`, `llm_query`, "
        "`llm_query_batched`, `rlm_query_batched`).",
        "",
        "```",
        repl_clean.strip(),
        "```",
        "",
    ]

    codebook_section: list[str] = []
    if naming_codebook:
        codebook_section = [
            "## Naming codebook (injected into all three stages)",
            "",
            "This block is pasted verbatim into Stage 1, Stage 2, and Stage 3 so ids are compatible "
            "across stages. It is the only place prefix conventions are defined; all three prompts "
            "reference this codebook by inclusion.",
            "",
            "```",
            naming_codebook.strip(),
            "```",
            "",
        ]

    return "\n".join(header + overview + codebook_section + stage1 + stage2 + stage3)
