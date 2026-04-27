"""Gen 6 Phase 1+2: Codebook labeling with GPT-4o-mini swarm + Sonnet resolution.

Phase 1: 3x GPT-4o-mini (temp=0, strict JSON) labels each chunk inline
Phase 2: Compare 3 outputs. If agree: use directly. If disagree: Sonnet resolves.
         Extreme difficulty chunks: auto-escalate to Sonnet.
"""

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

GPT_MINI = "gpt-4o-mini"
SONNET = "claude-sonnet-4-6"

# Schema for labeled chunk output (strict JSON)
LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "section_id": {"type": "string"},
        "section_title": {"type": "string"},
        "text": {"type": "string"},
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "span": {"type": "string"},
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["supply_list", "variables", "predicates",
                                 "modules", "phrase_bank", "router", "integrative"]
                    },
                    "subtype": {"type": "string"},
                    "quote_context": {"type": "string"},
                },
                "required": ["span", "id", "type"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["section_id", "section_title", "text", "labels"],
    "additionalProperties": False,
}


def _build_labeling_prompt(codebook: str) -> str:
    return (
        "You are a clinical document labeler. Label every clinical item in the text.\n\n"
        "For EVERY item, add a label with:\n"
        "- span: exact verbatim text\n"
        "- id: canonical ID with prefix:\n"
        "  supply_ = consumable medication/supply, equip_ = durable equipment,\n"
        "  q_ = question asked, ex_ = examination finding, v_ = measurement,\n"
        "  lab_ = lab test, demo_ = demographic, hx_ = history,\n"
        "  p_ = boolean threshold/predicate, mod_ = decision module,\n"
        "  m_ = clinician phrase, dx_ = diagnosis, tx_ = treatment, ref_ = referral\n"
        "- type: one of: supply_list, variables, predicates, modules, "
        "phrase_bank, router, integrative\n"
        "  supply_list = medications + equipment. variables = q_ ex_ v_ lab_ demo_ hx_ items.\n"
        "  predicates = p_ thresholds. modules = mod_ decision topics.\n"
        "  phrase_bank = m_ things CHW says. router = routing decisions. "
        "integrative = comorbidity rules.\n"
        "- quote_context: surrounding sentence\n\n"
        "Label EVERYTHING: medications, equipment, questions, thresholds, "
        "danger signs, dosages, treatments, advice, referrals, age cutoffs.\n\n"
        "CRITICAL: IDs must be DESCRIPTIVE, not numbered. "
        "GOOD: q_has_cough, p_fast_breathing, supply_amoxicillin_250mg, equip_thermometer. "
        "BAD: q_1, p_2, supply_3. The ID should describe WHAT the item is.\n\n"
        "Return JSON: {\"section_id\": \"...\", \"section_title\": \"...\", "
        "\"text\": \"(original text)\", \"labels\": [{\"span\": \"...\", "
        "\"id\": \"prefix_descriptive_name\", \"type\": \"...\", \"quote_context\": \"...\"}]}"
    )


async def _call_gpt_mini(
    chunk: dict,
    codebook: str,
    openai_key: str,
    replica_idx: int,
) -> dict:
    """One GPT-4o-mini labeling call. Strict JSON output."""
    import openai
    client = openai.AsyncOpenAI(api_key=openai_key)

    prompt = _build_labeling_prompt(codebook)

    # Send clean text with section header, not raw JSON structure
    clean_input = (
        f"Section: {chunk['section_title']} ({chunk['section_id']})\n\n"
        f"{chunk['text']}"
    )

    try:
        r = await client.chat.completions.create(
            model=GPT_MINI,
            max_completion_tokens=4000,
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=60,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"CHUNK TO LABEL:\n{clean_input}"},
            ],
        )
        text = (r.choices[0].message.content or "").strip()
        if not text:
            return {"section_id": chunk["section_id"], "labels": [], "_error": "empty"}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {"section_id": chunk["section_id"], "labels": [], "_error": "not dict"}
        # Ensure labels field exists
        if "labels" not in parsed:
            parsed["labels"] = []
        return parsed
    except Exception as exc:
        logger.warning("GPT-mini labeling error (chunk %d, replica %d): %s",
                       chunk.get("chunk_index", -1), replica_idx, exc)
        return {"section_id": chunk["section_id"], "labels": [], "_error": str(exc)}


async def _call_sonnet_resolve(
    chunk: dict,
    outputs: list[dict],
    codebook: str,
    anthropic_key: str,
) -> dict:
    """Sonnet resolves disagreement between 3 GPT-4o-mini outputs."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=anthropic_key)

    outputs_json = json.dumps(outputs, indent=1)[:12000]
    chunk_json = json.dumps({
        "section_id": chunk["section_id"],
        "section_title": chunk["section_title"],
        "text": chunk["text"],
    }, indent=1)

    prompt = (
        "Three labeling passes produced different results for this clinical guide chunk. "
        "Review all three outputs against the original text and codebook. "
        "Produce the DEFINITIVE labeled version.\n\n"
        "Rules:\n"
        "- Include every item that at least 2 of 3 passes found (majority)\n"
        "- For items where IDs disagree, pick the one that best matches the codebook\n"
        "- Add any items ALL passes missed but are clearly present in the text\n"
        "- Remove items that are clearly wrong (hallucinated, wrong prefix)\n\n"
        f"CODEBOOK:\n{codebook[:2000]}\n\n"
        f"ORIGINAL CHUNK:\n{chunk_json}\n\n"
        f"THREE LABELING OUTPUTS:\n{outputs_json}\n\n"
        "Return the definitive labeled chunk as JSON with section_id, section_title, "
        "text, and labels array. No markdown fences."
    )

    try:
        r = await client.messages.create(
            model=SONNET, max_tokens=4000, temperature=0.0,
            system="Resolve labeling disagreements. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        import re
        text = r.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed.setdefault("labels", [])
            return parsed
        return outputs[0]  # fallback
    except Exception as exc:
        logger.warning("Sonnet resolve error (chunk %d): %s",
                       chunk.get("chunk_index", -1), exc)
        return outputs[0]  # fallback to first GPT output


def _check_agreement(outputs: list[dict]) -> bool:
    """Check if 3 GPT outputs agree on labels.

    Agreement = same set of IDs (order doesn't matter).
    """
    label_sets = []
    for out in outputs:
        labels = out.get("labels", [])
        ids = frozenset(l.get("id", "") for l in labels if isinstance(l, dict))
        label_sets.append(ids)

    # All 3 must have the same ID set
    if len(label_sets) >= 2:
        return all(s == label_sets[0] for s in label_sets[1:])
    return True


async def label_chunk(
    chunk: dict,
    codebook: str,
    openai_key: str,
    anthropic_key: str,
) -> dict:
    """Label one micro-chunk with codebook annotations.

    Runs 3x GPT-4o-mini. If they agree, returns directly.
    If they disagree (or chunk is extreme difficulty), Sonnet resolves.

    Returns the labeled chunk dict with 'labels' array + '_labeling_meta'.
    """
    chunk_idx = chunk.get("chunk_index", -1)
    difficulty = chunk.get("difficulty", "medium")
    auto_sonnet = difficulty == "extreme"

    # Phase 1: 3x GPT-4o-mini in parallel
    gpt_coros = [
        _call_gpt_mini(chunk, codebook, openai_key, i)
        for i in range(3)
    ]
    gpt_outputs = await asyncio.gather(*gpt_coros)
    gpt_outputs = [o for o in gpt_outputs if isinstance(o, dict)]

    # Check for errors
    valid_outputs = [o for o in gpt_outputs if "_error" not in o]
    if not valid_outputs:
        valid_outputs = gpt_outputs  # use error outputs as fallback

    # Phase 2: Check agreement or escalate
    agreed = _check_agreement(valid_outputs) if len(valid_outputs) >= 2 else False

    meta = {
        "gpt_replicas": len(gpt_outputs),
        "valid_replicas": len(valid_outputs),
        "agreed": agreed,
        "auto_sonnet": auto_sonnet,
        "sonnet_escalated": False,
        "difficulty": difficulty,
    }

    if agreed and not auto_sonnet:
        # All 3 agree — use the first valid output
        result = valid_outputs[0]
        label_count = len(result.get("labels", []))
        logger.info("Gen6 label: chunk %d (%s) — 3 GPT agree, %d labels",
                    chunk_idx, difficulty, label_count)
    else:
        # Disagreement or extreme — Sonnet resolves
        result = await _call_sonnet_resolve(chunk, valid_outputs, codebook, anthropic_key)
        meta["sonnet_escalated"] = True
        label_count = len(result.get("labels", []))
        reason = "auto-extreme" if auto_sonnet else "disagreement"
        logger.info("Gen6 label: chunk %d (%s) — Sonnet resolved (%s), %d labels",
                    chunk_idx, difficulty, reason, label_count)

    # Attach metadata
    result["_labeling_meta"] = meta
    result["chunk_index"] = chunk_idx
    result["difficulty"] = difficulty

    return result


async def label_all_chunks(
    chunks: list[dict],
    codebook: str,
    openai_key: str,
    anthropic_key: str,
) -> list[dict]:
    """Label all micro-chunks in parallel.

    Returns list of labeled chunk dicts.
    """
    logger.info("Gen6 labeler: %d chunks to label", len(chunks))

    # Run all chunks in parallel (each chunk = 3 GPT calls + optional Sonnet)
    coros = [
        label_chunk(chunk, codebook, openai_key, anthropic_key)
        for chunk in chunks
    ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    labeled = []
    errors = 0
    sonnet_count = 0
    for i, r in enumerate(results):
        if isinstance(r, dict):
            labeled.append(r)
            if r.get("_labeling_meta", {}).get("sonnet_escalated"):
                sonnet_count += 1
        else:
            errors += 1
            logger.warning("Gen6 label: chunk %d failed: %s", i, r)
            # Fallback: return unlabeled chunk
            labeled.append({**chunks[i], "labels": [], "_labeling_meta": {"error": str(r)}})

    logger.info(
        "Gen6 labeler: done — %d labeled, %d errors, %d Sonnet escalations (%.0f%%)",
        len(labeled), errors, sonnet_count,
        sonnet_count / len(chunks) * 100 if chunks else 0,
    )
    return labeled
