"""Gen 4: Stationary test suite generation and matching.

Replaces per-round guide-dependent catcher enumeration with a frozen test
suite generated once per guide. The test suite is a list of items the guide
requires, each with a suggested_id, guide_quote, section, and artifact_type.

Generation: 7 Haiku voters at wide temperatures + 1 Opus oversight per chunk.
Matching: pure Python, no LLM calls. Check each test item against the artifact.

The generators NEVER see the tests. The tests are a frozen measurement
instrument; the meaningful variance is whether the generator passes or fails.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import re as _re_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk difficulty classifier (deterministic NLP, manual-agnostic)
# ---------------------------------------------------------------------------
# Predicts which chunks need more enumeration passes. Uses features that
# correlate with GPT/Opus recall from the benchmark (n=7 chunks):
#   list_ratio (-0.77), imperative_ratio (+0.65), TTR x sections (-0.64)

def classify_chunk_difficulty(chunk_json: dict) -> dict:
    """Classify a guide chunk's extraction difficulty using NLP heuristics.

    Returns dict with:
      - difficulty: "easy" | "medium" | "hard"
      - score: float (higher = harder)
      - recommended_passes: int (1, 2, or 3)
      - features: dict of computed NLP features
    """
    sections = chunk_json.get("sections", {})
    num_sections = len(sections)

    # Extract full text
    full_text = ""
    for sec in sections.values():
        full_text += (sec.get("raw_text", "") or "") + " "
        for block in sec.get("blocks", []):
            full_text += (block.get("text", "") or "") + " "

    chars = len(full_text)
    if chars < 100:
        return {
            "difficulty": "easy", "score": 0.0,
            "recommended_passes": 1, "features": {},
        }

    # Tokenize
    tokens = _re_module.findall(r"[a-z]+", full_text.lower())
    types = set(tokens)

    # 1. Type-Token Ratio (lexical diversity)
    ttr = len(types) / len(tokens) if tokens else 0

    # 2. List ratio (bulleted/numbered lines vs total)
    lines = full_text.split("\n")
    list_lines = sum(
        1 for l in lines
        if _re_module.match(r"\s*[-*\u2022]\s|\s*\d+[.)]\s", l)
    )
    total_lines = len([l for l in lines if len(l.strip()) > 5])
    list_ratio = list_lines / total_lines if total_lines else 0

    # 3. Imperative sentence ratio
    sentences = [
        s.strip() for s in _re_module.split(r"[.!?]+", full_text)
        if len(s.strip()) > 5
    ]
    imp_pattern = r"^(give|use|prepare|check|count|look|ask|advise|help|wash|open|put|pour|measure|record|write|refer|tell|show|dispose|discard|clean|prick|remove|cut|crush|mix)"
    imp_count = sum(
        1 for s in sentences
        if _re_module.match(imp_pattern, s.strip().lower())
    )
    imperative_ratio = imp_count / len(sentences) if sentences else 0

    # 4. Repetition rate (content word avg frequency)
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "with", "this", "that", "it",
        "be", "as", "by", "from", "has", "have", "had", "not", "but",
        "will", "can", "do", "does", "did", "if", "her", "his", "she",
        "he", "they", "you", "your", "we", "our", "may",
    }
    from collections import Counter
    content_words = [t for t in tokens if t not in stopwords and len(t) > 2]
    content_freq = Counter(content_words)
    rep_rate = (
        sum(content_freq.values()) / len(content_freq)
        if content_freq else 0
    )

    # 5. Sections per 10K chars
    secs_per_10k = num_sections / (chars / 10000) if chars > 0 else 0

    # Composite difficulty score (weighted by correlation strength)
    # list_ratio: -0.77, imperative: +0.65, ttr*secs: -0.64, rep_rate: +0.50
    score = (
        list_ratio * 0.4
        + (ttr * num_sections / 10) * 0.3
        + (1.0 - imperative_ratio) * 0.15
        + (1.0 / max(rep_rate, 1.0)) * 0.15
    )

    # 5-level taxonomy based on natural score gaps in WHO 2012 guide.
    # Each level gets a specific model allocation for test suite generation:
    #   trivial: 1 GPT call
    #   easy:    1 GPT + 1 Opus
    #   medium:  1 GPT + 1 Opus + 1 GPT targeted
    #   hard:    1 GPT + 1 Opus + 1 GPT targeted + 1 Opus targeted
    #   extreme: 1 GPT + 1 Opus + 2 GPT targeted + 1 Opus targeted
    if score > 0.32:
        difficulty = "extreme"
        passes = 5
    elif score > 0.27:
        difficulty = "hard"
        passes = 4
    elif score > 0.22:
        difficulty = "medium"
        passes = 3
    elif score > 0.17:
        difficulty = "easy"
        passes = 2
    else:
        difficulty = "trivial"
        passes = 1

    return {
        "difficulty": difficulty,
        "score": round(score, 4),
        "recommended_passes": passes,
        "features": {
            "ttr": round(ttr, 4),
            "list_ratio": round(list_ratio, 4),
            "imperative_ratio": round(imperative_ratio, 4),
            "rep_rate": round(rep_rate, 2),
            "secs_per_10k": round(secs_per_10k, 2),
            "num_sections": num_sections,
            "chars": chars,
        },
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 7 voters at wide temperatures for enumeration.
# Higher temps (0.5-0.8) are safe for enumeration (post-filtered by quote
# verification) but unsafe for validation (JSON structure breaks).
_TEST_GEN_TEMPERATURES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8)
_TEST_GEN_VOTER_COUNT = len(_TEST_GEN_TEMPERATURES)

_HAIKU_MODEL = "claude-haiku-4-5"
_OPUS_MODEL = "claude-opus-4-6"

# Cache directory for persisted test suites
_SUITE_CACHE_DIR = Path(os.environ.get(
    "TEST_SUITE_CACHE_DIR",
    str(Path(__file__).parent.parent / "output" / "test_suites"),
))


def _guide_content_hash(guide_json: dict) -> str:
    """SHA-256 of the guide's sections content for cache keying."""
    raw = json.dumps(guide_json.get("sections", {}), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Test item schema
# ---------------------------------------------------------------------------

class TestItem:
    """One frozen test: the guide requires this item in the artifact."""

    __slots__ = ("suggested_id", "guide_quote", "section",
                 "artifact_type", "description", "repair_instruction",
                 "voter_count")

    def __init__(self, **kwargs):
        self.suggested_id: str = kwargs.get("suggested_id", "")
        self.guide_quote: str = kwargs.get("guide_quote", "")
        self.section: str = kwargs.get("section", "")
        self.artifact_type: str = kwargs.get("artifact_type", "")
        self.description: str = kwargs.get("description", "")
        self.repair_instruction: str = kwargs.get("repair_instruction", "")
        self.voter_count: int = kwargs.get("voter_count", 1)

    def to_dict(self) -> dict:
        return {
            "suggested_id": self.suggested_id,
            "guide_quote": self.guide_quote,
            "section": self.section,
            "artifact_type": self.artifact_type,
            "description": self.description,
            "repair_instruction": self.repair_instruction,
            "voter_count": self.voter_count,
        }

    @staticmethod
    def from_dict(d: dict) -> "TestItem":
        return TestItem(**d)

    def dedupe_key(self) -> tuple:
        return (
            self.suggested_id.strip().lower(),
            self.section.strip().lower(),
        )


class TestSuite:
    """Frozen test suite for one guide."""

    def __init__(self, guide_hash: str, items: list[TestItem]):
        self.guide_hash = guide_hash
        self.items = items

    def for_artifact(self, artifact_type: str) -> list[TestItem]:
        """Get test items for a specific artifact type."""
        return [t for t in self.items if t.artifact_type == artifact_type]

    def to_dict(self) -> dict:
        return {
            "guide_hash": self.guide_hash,
            "item_count": len(self.items),
            "items": [t.to_dict() for t in self.items],
        }

    @staticmethod
    def from_dict(d: dict) -> "TestSuite":
        return TestSuite(
            guide_hash=d["guide_hash"],
            items=[TestItem.from_dict(i) for i in d["items"]],
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist to disk."""
        if path is None:
            _SUITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = _SUITE_CACHE_DIR / f"suite_{self.guide_hash}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Test suite saved: %s (%d items)", path, len(self.items))
        return path

    @staticmethod
    def load(guide_hash: str) -> Optional["TestSuite"]:
        """Load from cache if exists."""
        path = _SUITE_CACHE_DIR / f"suite_{guide_hash}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                suite = TestSuite.from_dict(d)
                logger.info("Test suite loaded from cache: %s (%d items)", path, len(suite.items))
                return suite
            except Exception as exc:
                logger.warning("Failed to load cached test suite: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Artifact type detection from catcher name
# ---------------------------------------------------------------------------

_CATCHER_TO_ARTIFACT = {
    "completeness_supply_list": "supply_list",
    "completeness_variables": "variables",
    "completeness_predicates": "predicates",
    "completeness_modules": "modules",
    "completeness_router": "router",
    "completeness_integrative": "integrative",
    "completeness_phrase_bank": "phrase_bank",
}


# ---------------------------------------------------------------------------
# Phase 1: Haiku enumeration
# ---------------------------------------------------------------------------

async def _enumerate_chunk(
    chunk: dict,
    artifact_type: str,
    api_key: str,
    catcher_name: str,
) -> list[dict]:
    """Run 7 Haiku voters on one chunk, return union of structured criticals.

    Uses the existing catcher infrastructure but with wide temperatures
    for maximum recall. Returns raw dicts from the catcher verdict.
    """
    from backend.validators.phases import _call_catcher, _get_catcher_semaphore

    sem = _get_catcher_semaphore()

    async def one_vote(temp: float) -> dict:
        async with sem:
            return await _call_catcher(
                catcher_name=catcher_name,
                artifact=[],  # empty artifact — find EVERYTHING the guide has
                guide_json=chunk,
                api_key=api_key,
                extra_context=None,
                temperature=temp,
            )

    # Voter 1 alone (writes cache), then rest in parallel
    first = await one_vote(_TEST_GEN_TEMPERATURES[0])
    await asyncio.sleep(0.5)  # cache registration
    rest = await asyncio.gather(*[
        one_vote(temp) for temp in _TEST_GEN_TEMPERATURES[1:]
    ])

    results = [first] + list(rest)

    # Union all structured criticals with vote counting
    seen: dict[tuple, dict] = {}  # dedupe_key -> item_dict
    vote_counts: dict[tuple, int] = {}

    for r in results:
        voter_seen: set = set()
        for sc in r.get("_structured_criticals", []):
            if not isinstance(sc, dict):
                continue
            sid = sc.get("suggested_id", "")
            quote = sc.get("guide_quote", "")
            if not sid or not quote:
                continue
            key = (sid.strip().lower(), sc.get("section", "").strip().lower())
            if key not in voter_seen:
                voter_seen.add(key)
                vote_counts[key] = vote_counts.get(key, 0) + 1
                if key not in seen:
                    seen[key] = {
                        "suggested_id": str(sid),
                        "guide_quote": str(quote),
                        "section": str(sc.get("section", "")),
                        "description": str(sc.get("description", "")),
                        "repair_instruction": str(sc.get("repair_instruction", "")),
                        "artifact_type": artifact_type,
                    }

    # Attach vote counts
    items = []
    for key, item in seen.items():
        item["voter_count"] = vote_counts.get(key, 1)
        items.append(item)

    logger.info(
        "Test gen: %s chunk — %d unique items from %d voters (max votes: %d)",
        catcher_name,
        len(items),
        len(results),
        max(vote_counts.values()) if vote_counts else 0,
    )
    return items


# ---------------------------------------------------------------------------
# Phase 1b: Quote verification (deterministic filter)
# ---------------------------------------------------------------------------

def _verify_quotes(items: list[dict], guide_json: dict) -> list[dict]:
    """Filter items whose guide_quote is not found in the guide text."""
    # Build searchable text from all sections
    full_text = ""
    for sec_id, sec in guide_json.get("sections", {}).items():
        raw = sec.get("raw_text", "")
        if raw:
            full_text += " " + raw
        for block in sec.get("blocks", []):
            text = block.get("text", "")
            if text:
                full_text += " " + text

    full_text_lower = full_text.lower()

    verified = []
    dropped = 0
    for item in items:
        quote = item.get("guide_quote", "").strip()
        if not quote:
            dropped += 1
            continue
        # Normalize whitespace for matching
        quote_normalized = " ".join(quote.lower().split())
        if len(quote_normalized) > 10 and quote_normalized in full_text_lower:
            verified.append(item)
        else:
            # Try shorter substring (first 50 chars)
            short = quote_normalized[:50]
            if len(short) > 10 and short in full_text_lower:
                verified.append(item)
            else:
                dropped += 1

    if dropped:
        logger.info("Test gen: quote verification dropped %d/%d items", dropped, dropped + len(verified))
    return verified


# ---------------------------------------------------------------------------
# Phase 2: Opus oversight
# ---------------------------------------------------------------------------

async def _opus_oversight(
    chunk: dict,
    haiku_items: list[dict],
    api_key: str,
) -> list[dict]:
    """Opus reviews Haiku enumeration for completeness.

    Returns additional items Opus found that Haiku missed.
    """
    import anthropic as _sdk

    # Build the items summary for Opus
    items_summary = "\n".join(
        f"  - {item['suggested_id']}: {item['description'][:100]}"
        for item in haiku_items[:50]  # cap to keep prompt reasonable
    )

    chunk_text = json.dumps(chunk.get("sections", {}), indent=2)[:30000]

    prompt = (
        f"You are reviewing an enumeration of clinical items found in a guide chunk.\n\n"
        f"GUIDE CHUNK (abbreviated):\n{chunk_text}\n\n"
        f"ITEMS ALREADY FOUND ({len(haiku_items)} total):\n{items_summary}\n\n"
        f"YOUR TASK:\n"
        f"1. Read the guide chunk carefully.\n"
        f"2. List any clinical items (variables, predicates, modules, supply items, "
        f"phrases, thresholds, treatments, referral criteria) that SHOULD be in the "
        f"list above but are NOT.\n"
        f"3. For each missing item, provide: suggested_id, guide_quote (verbatim), "
        f"section, description, artifact_type (supply_list/variables/predicates/"
        f"modules/phrase_bank).\n\n"
        f"Return a JSON array of missing items. If the list is complete, return [].\n"
        f"No markdown fences. No explanation."
    )

    client = _sdk.AsyncAnthropic(api_key=api_key)
    try:
        response = await client.messages.create(
            model=_OPUS_MODEL,
            max_tokens=4000,
            system="Return ONLY a valid JSON array. No markdown, no prose.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        import re
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        additional = json.loads(text)
        if not isinstance(additional, list):
            additional = []

        logger.info("Test gen: Opus oversight found %d additional items", len(additional))
        return [item for item in additional if isinstance(item, dict) and item.get("suggested_id")]
    except Exception as exc:
        logger.warning("Test gen: Opus oversight failed (non-fatal): %s", exc)
        return []


# ---------------------------------------------------------------------------
# Main: generate_test_suite
# ---------------------------------------------------------------------------

async def generate_test_suite(
    guide_json: dict,
    api_key: str,
    force_regenerate: bool = False,
) -> TestSuite:
    """Generate a frozen test suite for a guide.

    Uses cache: if a suite exists for this guide hash, returns it.
    Otherwise generates via Haiku enumeration + Opus oversight.
    """
    from backend.validators.phases import chunk_guide_for_catcher

    guide_hash = _guide_content_hash(guide_json)

    # Check cache
    if not force_regenerate:
        cached = TestSuite.load(guide_hash)
        if cached:
            return cached

    logger.info("Test gen: generating test suite for guide hash=%s", guide_hash)

    # Chunk the guide
    chunks = chunk_guide_for_catcher(guide_json)
    logger.info("Test gen: %d chunks to process", len(chunks))

    # For each completeness catcher, enumerate across all chunks
    all_items: list[dict] = []

    for catcher_name, artifact_type in _CATCHER_TO_ARTIFACT.items():
        if catcher_name in ("completeness_router", "completeness_integrative"):
            # Router/integrative are structural, not guide-content-dependent
            # Their tests come from the modules artifact, not the guide
            continue

        for chunk_idx, chunk in enumerate(chunks):
            logger.info("Test gen: %s chunk %d/%d", catcher_name, chunk_idx + 1, len(chunks))

            # Tier 1: Haiku enumeration
            chunk_items = await _enumerate_chunk(chunk, artifact_type, api_key, catcher_name)

            # Quote verification
            chunk_items = _verify_quotes(chunk_items, guide_json)

            # Tier 2: Opus oversight
            additional = await _opus_oversight(chunk, chunk_items, api_key)
            additional = _verify_quotes(additional, guide_json)

            all_items.extend(chunk_items)
            all_items.extend(additional)

    # Deduplicate across chunks and catchers
    seen_keys: set = set()
    deduped: list[TestItem] = []
    for item in all_items:
        ti = TestItem.from_dict(item)
        key = ti.dedupe_key()
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(ti)

    suite = TestSuite(guide_hash=guide_hash, items=deduped)
    suite.save()

    logger.info(
        "Test gen: suite complete — %d items across %d artifact types",
        len(deduped),
        len({t.artifact_type for t in deduped}),
    )
    return suite


# ---------------------------------------------------------------------------
# Phase 2 validation: deterministic matching
# ---------------------------------------------------------------------------

def validate_against_test_suite(
    artifact_name: str,
    artifact: Any,
    suite: TestSuite,
) -> dict:
    """Pure Python validation: check each test item against the artifact.

    Returns a dict shaped like a catcher result:
    {"passed": bool, "critical_issues": [...], "warnings": [...],
     "_structured_criticals": [...]}
    """
    tests = suite.for_artifact(artifact_name)
    if not tests:
        return {
            "passed": True,
            "critical_issues": [],
            "warnings": [],
            "_structured_criticals": [],
            "_source": "test_suite_no_tests",
        }

    # Build searchable index from artifact entries
    artifact_ids: set[str] = set()
    artifact_quotes: list[str] = []

    if isinstance(artifact, list):
        for entry in artifact:
            if isinstance(entry, dict):
                eid = entry.get("id", "")
                if eid:
                    artifact_ids.add(eid.strip().lower())
                quote = entry.get("source_quote", "")
                if quote:
                    artifact_quotes.append(quote.lower())
    elif isinstance(artifact, dict):
        for key, entry in artifact.items():
            artifact_ids.add(key.strip().lower())
            if isinstance(entry, dict):
                eid = entry.get("id", entry.get("module_id", ""))
                if eid:
                    artifact_ids.add(eid.strip().lower())
                quote = entry.get("source_quote", "")
                if quote:
                    artifact_quotes.append(quote.lower())
                # For modules: also index rule_ids
                for rule in entry.get("rules", []):
                    if isinstance(rule, dict):
                        rid = rule.get("rule_id", "")
                        if rid:
                            artifact_ids.add(rid.strip().lower())

    # Match each test item
    missing: list[TestItem] = []
    for test in tests:
        sid_lower = test.suggested_id.strip().lower()

        # Match 1: exact id match
        if sid_lower in artifact_ids:
            continue

        # Match 2: guide_quote substring in any artifact entry's source_quote
        quote_lower = test.guide_quote.strip().lower()
        quote_short = " ".join(quote_lower.split())[:50]
        if any(quote_short in aq for aq in artifact_quotes):
            continue

        # No match — this test item is missing from the artifact
        missing.append(test)

    # Build result
    criticals = [
        f"[test_suite] MISSING {t.suggested_id}: {t.description}. "
        f"GUIDE QUOTE: '{t.guide_quote[:200]}'. SECTION: {t.section}."
        for t in missing
    ]

    structured = [
        {
            "suggested_id": t.suggested_id,
            "description": t.description,
            "guide_quote": t.guide_quote,
            "section": t.section,
            "repair_instruction": t.repair_instruction,
            "catcher": "test_suite",
        }
        for t in missing
    ]

    return {
        "passed": len(criticals) == 0,
        "critical_issues": criticals,
        "warnings": [],
        "_structured_criticals": structured,
        "_source": f"test_suite_{len(tests)}_tests_{len(missing)}_missing",
        "_test_count": len(tests),
        "_missing_count": len(missing),
    }
