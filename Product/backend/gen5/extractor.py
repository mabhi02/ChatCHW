"""Gen 5 Phase 1: Multi-model parallel extraction.

Extracts raw items from guide chunks using the optimal model per artifact type.
All extraction calls are independent and run in parallel.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Optimal model per artifact type (from v7 benchmark)
MODEL_ROUTING = {
    "supply_list":  [("openai", "gpt-5.4")],
    "variables":    [("openai", "gpt-5.4"), ("anthropic", "claude-opus-4-6")],
    "predicates":   [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "modules":      [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "phrase_bank":  [("anthropic", "claude-sonnet-4-6")],
    "router":       [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "integrative":  [("openai", "gpt-5.4")],
}

# Fallback: Anthropic-only routing (no OpenAI key)
MODEL_ROUTING_ANTHROPIC_ONLY = {
    "supply_list":  [("anthropic", "claude-opus-4-6")],
    "variables":    [("anthropic", "claude-opus-4-6")],
    "predicates":   [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "modules":      [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "phrase_bank":  [("anthropic", "claude-sonnet-4-6")],
    "router":       [("anthropic", "claude-opus-4-6"), ("anthropic", "claude-haiku-4-5")],
    "integrative":  [("anthropic", "claude-opus-4-6")],
}

ARTIFACT_DESCRIPTIONS = {
    "supply_list": "supply items (consumable medications, test kits, disposables) and equipment (durable tools, forms, devices)",
    "variables": "clinical variables (data fields collected: questions asked, examination findings, vital measurements, lab tests, demographics, history)",
    "predicates": "boolean thresholds and clinical conditions (age cutoffs, measurement thresholds, test result interpretations, danger sign definitions)",
    "modules": "clinical decision modules (distinct assessment topics with classification and treatment logic)",
    "phrase_bank": "clinician-facing phrases (questions to ask, diagnosis announcements, treatment instructions, advice, referral messages, follow-up reminders)",
    "router": "routing decisions (which clinical module to activate based on symptoms and findings)",
    "integrative": "cross-module comorbidity rules (what happens when multiple conditions co-occur)",
}


def _make_anthropic_prompt(artifact_type: str) -> str:
    desc = ARTIFACT_DESCRIPTIONS[artifact_type]
    return (
        f"You are extracting {desc} from a clinical guide chunk. "
        f"Read EVERY sentence carefully. List EVERY item you find. "
        f"For each item return a JSON object with: "
        f"id (descriptive name), quote (verbatim text from the guide), "
        f"section (which section it came from). "
        f"Return ONLY a JSON array. No markdown fences. No explanation."
    )


def _make_openai_prompt(artifact_type: str) -> str:
    desc = ARTIFACT_DESCRIPTIONS[artifact_type]
    return (
        f"You are extracting {desc} from a clinical guide chunk. "
        f"Read EVERY sentence carefully. List EVERY item you find. "
        f"For each item return: id (descriptive name), quote (verbatim text), "
        f"section (which section it came from)."
        ' Respond with JSON: {"items": [{"id": "x", "quote": "verbatim", "section": "sec"}]}'
    )


async def _call_anthropic(
    chunk_text: str,
    prompt: str,
    api_key: str,
    model: str,
    temp: float = 0.0,
) -> list[dict]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        r = await client.messages.create(
            model=model, max_tokens=8000, temperature=temp,
            system=prompt,
            messages=[{"role": "user", "content": chunk_text}],
        )
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        if text.startswith("["):
            items = json.loads(text)
            return [i for i in items if isinstance(i, dict)]
        return []
    except Exception as exc:
        logger.warning("Anthropic extraction error (%s): %s", model, exc)
        return []


async def _call_openai(
    chunk_text: str,
    prompt: str,
    api_key: str,
    model: str = "gpt-5.4",
    temp: float = 0.0,
) -> list[dict]:
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)
    try:
        r = await client.chat.completions.create(
            model=model, max_completion_tokens=8000, temperature=temp,
            response_format={"type": "json_object"}, timeout=180,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk_text},
            ],
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            items = parsed.get("items", [])
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []
        return [i for i in items if isinstance(i, dict)]
    except Exception as exc:
        logger.warning("OpenAI extraction error (%s): %s", model, exc)
        return []


async def extract_chunk(
    chunk: dict,
    chunk_idx: int,
    artifact_type: str,
    anthropic_key: str,
    openai_key: str | None = None,
) -> list[dict]:
    """Extract items of one artifact type from one chunk.

    Uses the optimal model routing. Returns list of raw item dicts.
    """
    # Pick routing based on available keys
    if openai_key:
        routing = MODEL_ROUTING.get(artifact_type, [("anthropic", "claude-opus-4-6")])
    else:
        routing = MODEL_ROUTING_ANTHROPIC_ONLY.get(artifact_type, [("anthropic", "claude-opus-4-6")])

    chunk_text = "GUIDE CHUNK:\n" + json.dumps(chunk.get("sections", {}), indent=2)[:40000]

    # Run all models for this artifact type in parallel
    coros = []
    for provider, model in routing:
        if provider == "openai" and openai_key:
            prompt = _make_openai_prompt(artifact_type)
            coros.append(_call_openai(chunk_text, prompt, openai_key, model))
        elif provider == "anthropic":
            prompt = _make_anthropic_prompt(artifact_type)
            coros.append(_call_anthropic(chunk_text, prompt, anthropic_key, model))

    results = await asyncio.gather(*coros, return_exceptions=True)

    # Union all items, tag with chunk_idx and artifact_type
    all_items = []
    for r in results:
        if isinstance(r, list):
            for item in r:
                item["_chunk"] = chunk_idx
                item["_artifact_type"] = artifact_type
                all_items.append(item)

    logger.info(
        "Extracted %d raw items: chunk=%d type=%s models=%d",
        len(all_items), chunk_idx, artifact_type, len(coros),
    )
    return all_items


async def extract_all(
    guide_json: dict,
    anthropic_key: str,
    openai_key: str | None = None,
    on_progress: Any = None,
) -> dict[str, list[dict]]:
    """Phase 1: Extract ALL artifact types from ALL chunks in parallel.

    Returns {artifact_type: [raw_items]} with items tagged by chunk.
    """
    from backend.validators.phases import chunk_guide_for_catcher

    chunks = chunk_guide_for_catcher(guide_json)
    logger.info("Gen5 extraction: %d chunks, 7 artifact types", len(chunks))

    artifact_types = list(ARTIFACT_DESCRIPTIONS.keys())

    # Launch ALL extraction calls in parallel
    tasks = []
    task_keys = []
    for at in artifact_types:
        for i, chunk in enumerate(chunks):
            tasks.append(extract_chunk(chunk, i, at, anthropic_key, openai_key))
            task_keys.append((at, i))

    total = len(tasks)
    logger.info("Gen5 extraction: launching %d parallel calls", total)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Organize by artifact type
    raw_items: dict[str, list[dict]] = {at: [] for at in artifact_types}
    for (at, chunk_idx), result in zip(task_keys, results):
        if isinstance(result, list):
            raw_items[at].extend(result)
        else:
            logger.warning("Extraction failed: type=%s chunk=%d error=%s", at, chunk_idx, result)

    for at in artifact_types:
        logger.info("Gen5 raw: %s = %d items", at, len(raw_items[at]))

    return raw_items
