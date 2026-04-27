"""Assembler: converts Unstructured pages + vision enrichments into the RLM guide.

Produces both hierarchical sections (keyed by slugified title) and a flat pages
dict (keyed by page number string) so the RLM can navigate either way. Scores
the hierarchy quality and annotates the metadata so downstream consumers can
detect fallback mode.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from backend.ingestion.schema import (
    UnstructuredElement,
    UnstructuredPage,
    TableRefinement,
    ImageDescription,
    FlowchartStructure,
    RLMBlock,
    RLMSection,
    RLMPage,
    RLMGuideMetadata,
    RLMGuide,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Element types that Unstructured emits but that we still keep as paragraphs so
# the RLM has full raw context (it can decide to filter them itself).
_PARAGRAPH_PASSTHROUGH = {
    "NarrativeText",
    "Header",
    "Footer",
    "PageBreak",
    "UncategorizedText",
}

# Heuristic: a Title shorter than this is considered a "top-level" candidate.
_TITLE_MAX_CHARS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slugify(title: str) -> str:
    """Convert a section title to a dict-safe key.

    'General Danger Signs' -> 'general_danger_signs'
    Lowercase, replace non-alnum with underscore, collapse multiple underscores,
    strip leading/trailing underscores. Returns 'unnamed' for empty input.
    """
    s = (title or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "unnamed"


def _disambiguate_key(key: str, page_num: int, existing: dict) -> str:
    """Return a unique key, appending _p{page} then _2, _3, ... as needed."""
    if key not in existing:
        return key
    candidate = f"{key}_p{page_num}"
    if candidate not in existing:
        return candidate
    i = 2
    while f"{candidate}_{i}" in existing:
        i += 1
    return f"{candidate}_{i}"


def _is_top_level_title(element: UnstructuredElement, first_title_on_page: bool) -> bool:
    """Heuristic for whether a Title element should start a new top-level section."""
    if element.element_type != "Title":
        return False
    text = (element.text or "").strip()
    if not text:
        return False
    if first_title_on_page:
        return True
    if len(text) <= _TITLE_MAX_CHARS and text[:1].isalpha() and text[:1].isupper():
        return True
    return False


# ---------------------------------------------------------------------------
# Element -> Block conversion
# ---------------------------------------------------------------------------


def _make_paragraph_block(element: UnstructuredElement) -> RLMBlock:
    return RLMBlock(
        type="paragraph",
        page=element.page_number,
        text=element.text or "",
        element_id=element.element_id,
    )


def _make_heading_block(element: UnstructuredElement) -> RLMBlock:
    return RLMBlock(
        type="heading",
        page=element.page_number,
        text=element.text or "",
        element_id=element.element_id,
    )


def _make_list_block(items: list[UnstructuredElement]) -> RLMBlock:
    """Merge a run of consecutive ListItem elements into one list block."""
    first = items[0]
    return RLMBlock(
        type="list",
        page=first.page_number,
        items=[(e.text or "") for e in items],
        element_id=first.element_id,
    )


def _make_table_block(
    element: UnstructuredElement,
    refinement: Optional[TableRefinement],
) -> RLMBlock:
    """Build a table block, preferring refined output when available."""
    html: Optional[str] = element.text_as_html
    rows: Optional[list[list[str]]] = None
    english_text: Optional[str] = None

    if refinement is not None:
        rows = refinement.rows
        if refinement.was_translated:
            english_text = refinement.english_text
        # If vision refinement succeeded, prefer its structured rows as the
        # canonical representation. Keep original HTML too for debugging.
    return RLMBlock(
        type="table",
        page=element.page_number,
        text=element.text or "",
        html=html,
        rows=rows,
        english_text=english_text,
        element_id=element.element_id,
    )


def _make_image_or_flowchart_block(
    element: UnstructuredElement,
    description: Optional[ImageDescription],
    flowchart: Optional[FlowchartStructure],
) -> RLMBlock:
    """Build either a flowchart or image block depending on enrichment results."""
    # Flowchart path: description flagged is_flowchart AND we have structure.
    if description is not None and description.is_flowchart and flowchart is not None:
        return RLMBlock(
            type="flowchart",
            page=element.page_number,
            image_description=flowchart.english_text,
            flowchart_nodes=flowchart.nodes,
            flowchart_edges=flowchart.edges,
            element_id=element.element_id,
        )
    # Plain image path.
    image_description_text: Optional[str] = None
    if description is not None:
        image_description_text = description.description
    return RLMBlock(
        type="image",
        page=element.page_number,
        text=element.text or None,
        image_description=image_description_text,
        element_id=element.element_id,
    )


def _elements_to_blocks(
    elements: list[UnstructuredElement],
    table_refinements: dict[str, TableRefinement],
    image_descriptions: dict[str, ImageDescription],
    flowchart_structures: dict[str, FlowchartStructure],
) -> list[RLMBlock]:
    """Convert a list of Unstructured elements to RLMBlocks, merging list runs."""
    blocks: list[RLMBlock] = []
    i = 0
    n = len(elements)
    while i < n:
        el = elements[i]
        etype = el.element_type

        if etype == "ListItem":
            run = [el]
            j = i + 1
            while j < n and elements[j].element_type == "ListItem":
                run.append(elements[j])
                j += 1
            blocks.append(_make_list_block(run))
            i = j
            continue

        if etype == "Title":
            blocks.append(_make_heading_block(el))
        elif etype == "Table":
            refinement = table_refinements.get(el.element_id)
            blocks.append(_make_table_block(el, refinement))
        elif etype == "Image":
            description = image_descriptions.get(el.element_id)
            flowchart = flowchart_structures.get(el.element_id)
            blocks.append(_make_image_or_flowchart_block(el, description, flowchart))
        elif etype in _PARAGRAPH_PASSTHROUGH:
            blocks.append(_make_paragraph_block(el))
        else:
            # Unknown element type: keep the text as a paragraph so nothing is
            # silently dropped.
            logger.debug("Unknown element_type %r, keeping as paragraph", etype)
            blocks.append(_make_paragraph_block(el))
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# Raw-text helpers
# ---------------------------------------------------------------------------


def _block_text(block: RLMBlock) -> str:
    """Return a plain-text rendering of a block for section.raw_text."""
    btype = block.type
    if btype in ("paragraph", "heading"):
        return block.text or ""
    if btype == "list":
        return "\n".join(block.items or [])
    if btype == "table":
        return block.english_text or block.text or ""
    if btype in ("image", "flowchart", "image_description"):
        return block.image_description or ""
    return block.text or ""


def _sections_raw_text(blocks: list[RLMBlock]) -> str:
    parts = [_block_text(b) for b in blocks]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Hierarchy scoring
# ---------------------------------------------------------------------------


def score_hierarchy(sections: dict, total_pages: int) -> tuple[str, str]:
    """Return (quality_label, explanation).

    quality_label in {'good', 'sparse', 'fallback', 'noisy'}.

    Rules:
      - sections empty -> 'fallback', 'no titles found'
      - 1 section covering >10 pages -> 'sparse'
      - more than total_pages * 0.5 sections -> 'noisy'
      - average pages/section between 1.5 and 30 -> 'good'
      - otherwise -> 'sparse'
    """
    if not sections:
        return "fallback", "no titles found"

    section_count = len(sections)

    if section_count == 1:
        only = next(iter(sections.values()))
        span = max(1, (only.page_end - only.page_start + 1))
        if span > 10:
            return "sparse", f"only 1 section covering {span} pages"

    if total_pages > 0 and section_count > total_pages * 0.5:
        return (
            "noisy",
            f"{section_count} sections over {total_pages} pages (ratio > 0.5)",
        )

    if total_pages > 0:
        avg = total_pages / section_count
        if 1.5 <= avg <= 30:
            return (
                "good",
                f"{section_count} sections, avg {avg:.1f} pages/section",
            )
        return (
            "sparse",
            f"{section_count} sections, avg {avg:.1f} pages/section outside [1.5, 30]",
        )

    return "sparse", f"{section_count} sections, unknown page count"


# ---------------------------------------------------------------------------
# Section building
# ---------------------------------------------------------------------------


def _build_sections(
    pages: list[UnstructuredPage],
    blocks_by_element_id: dict[str, RLMBlock],
    all_elements: list[UnstructuredElement],
) -> dict[str, RLMSection]:
    """Walk the document in element order and build sections keyed by slug.

    Strategy:
      - Walk elements in document order.
      - Detect top-level Titles (first Title on a page, or short capitalized Title).
      - Everything from one top-level Title up to (but not including) the next
        belongs to that section.
      - Elements before the first top-level Title become a 'preamble' section.
    """
    sections: dict[str, RLMSection] = {}

    # Precompute which elements are top-level titles.
    seen_title_on_page: set[int] = set()
    top_level_flags: list[bool] = []
    for el in all_elements:
        first_on_page = (
            el.element_type == "Title" and el.page_number not in seen_title_on_page
        )
        if el.element_type == "Title":
            seen_title_on_page.add(el.page_number)
        top_level_flags.append(_is_top_level_title(el, first_on_page))

    # Walk elements and group into section buckets.
    current_title: Optional[UnstructuredElement] = None
    current_blocks: list[RLMBlock] = []
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None
    preamble_blocks: list[RLMBlock] = []
    preamble_page_start: Optional[int] = None
    preamble_page_end: Optional[int] = None

    def flush_current() -> None:
        nonlocal current_title, current_blocks, current_page_start, current_page_end
        if current_title is None:
            return
        key = slugify(current_title.text)
        key = _disambiguate_key(key, current_title.page_number, sections)
        raw_text = _sections_raw_text(current_blocks)
        sections[key] = RLMSection(
            title=(current_title.text or "").strip() or "Untitled",
            level=1,
            page_start=current_page_start or current_title.page_number,
            page_end=current_page_end or current_title.page_number,
            blocks=current_blocks,
            raw_text=raw_text,
        )
        current_title = None
        current_blocks = []
        current_page_start = None
        current_page_end = None

    for el, is_top in zip(all_elements, top_level_flags):
        block = blocks_by_element_id.get(el.element_id)

        if is_top:
            # Close out the previous section (if any) before opening a new one.
            flush_current()
            current_title = el
            current_page_start = el.page_number
            current_page_end = el.page_number
            # Include the heading block as the first block of the section.
            if block is not None:
                current_blocks.append(block)
            continue

        if current_title is not None:
            if block is not None:
                # A list-run block is registered by its first element_id; skip
                # duplicate additions for subsequent list-run members.
                if block not in current_blocks:
                    current_blocks.append(block)
            # CRITICAL: page_end advances on every element in the section,
            # not just ones that produced a new block. A ListItem that was
            # merged into a list-run block still counts for page range — the
            # physical content is on that page even though no new block is
            # emitted. Do NOT indent this under `if block is not None:`.
            if current_page_end is None or el.page_number > current_page_end:
                current_page_end = el.page_number
        else:
            # Pre-title content becomes the preamble.
            if block is not None and block not in preamble_blocks:
                preamble_blocks.append(block)
            if preamble_page_start is None:
                preamble_page_start = el.page_number
            if preamble_page_end is None or el.page_number > preamble_page_end:
                preamble_page_end = el.page_number

    # Flush whichever section is still open.
    flush_current()

    # Emit the preamble (if we collected any content) as its own section.
    # Python dicts preserve insertion order, and the RLM iterates sections
    # by that order. Front-matter should appear FIRST, not last. We build
    # a fresh dict with the preamble at the front, then merge in all the
    # titled sections we just flushed.
    if preamble_blocks:
        pre_key = _disambiguate_key("preamble", preamble_page_start or 1, sections)
        preamble_section = RLMSection(
            title="Preamble",
            level=1,
            page_start=preamble_page_start or 1,
            page_end=preamble_page_end or (preamble_page_start or 1),
            blocks=preamble_blocks,
            raw_text=_sections_raw_text(preamble_blocks),
        )
        reordered: dict[str, RLMSection] = {pre_key: preamble_section}
        for k, v in sections.items():
            reordered[k] = v
        sections = reordered

    return sections


def _build_fallback_sections(pages_dict: dict[str, RLMPage]) -> dict[str, RLMSection]:
    """Build per-page sections when no titles were detected."""
    sections: dict[str, RLMSection] = {}

    # Sort by numeric page number for stable ordering.
    # Defensive: if a key can't be parsed as int (shouldn't happen — upstream
    # we key by str(page_number) — but a malformed upstream state shouldn't
    # crash the whole assembler), fall back to lexicographic order for those
    # keys by assigning them a high sort value.
    def _page_sort_key(k: str) -> int:
        try:
            return int(k)
        except (TypeError, ValueError):
            logger.warning(
                "Non-integer page key %r in fallback sections; sorting to end",
                k,
            )
            return 10**9  # push to end without crashing

    for key in sorted(pages_dict.keys(), key=_page_sort_key):
        rlm_page = pages_dict[key]
        page_num = rlm_page.page_number
        # Use first heading block's text as the title if present.
        title: Optional[str] = None
        for b in rlm_page.blocks:
            if b.type == "heading" and (b.text or "").strip():
                title = b.text.strip()
                break
        if not title:
            title = f"Page {page_num}"
        section_key = f"page_{page_num:04d}"
        sections[section_key] = RLMSection(
            title=title,
            level=1,
            page_start=page_num,
            page_end=page_num,
            blocks=rlm_page.blocks,
            raw_text=rlm_page.raw_text,
        )
    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assemble_guide(
    pages: list[UnstructuredPage],
    table_refinements: dict[str, TableRefinement],
    image_descriptions: dict[str, ImageDescription],
    flowchart_structures: dict[str, FlowchartStructure],
    metadata: RLMGuideMetadata,
) -> RLMGuide:
    """Build the final RLMGuide from parsed pages + vision enrichment results.

    Produces BOTH views:
      - sections: dict[slug_key, RLMSection] - hierarchical, keyed by slugified Title
      - pages: dict[str(page_num), RLMPage] - flat per-page fallback

    Scores hierarchy quality and writes it to metadata.ingestion_meta['hierarchy_quality'].
    """
    logger.info(
        "Assembling guide: %d pages, %d table refinements, %d image descriptions, %d flowcharts",
        len(pages),
        len(table_refinements),
        len(image_descriptions),
        len(flowchart_structures),
    )

    # ---- Step 1: Build pages_dict ---------------------------------------
    pages_dict: dict[str, RLMPage] = {}
    all_elements: list[UnstructuredElement] = []
    blocks_by_element_id: dict[str, RLMBlock] = {}

    for page in pages:
        page_blocks = _elements_to_blocks(
            page.elements,
            table_refinements,
            image_descriptions,
            flowchart_structures,
        )
        # Index blocks by element_id so section-building can look them up
        # without reconverting. Note: list-run blocks are indexed only by their
        # first element's id (subsequent list items will return None, which is
        # the behavior section-building expects).
        for b in page_blocks:
            if b.element_id and b.element_id not in blocks_by_element_id:
                blocks_by_element_id[b.element_id] = b

        rlm_page = RLMPage(
            page_number=page.page_number,
            blocks=page_blocks,
            raw_text=page.raw_text,
        )
        pages_dict[str(page.page_number)] = rlm_page
        all_elements.extend(page.elements)

    # ---- Step 2/3/4: Build sections from document-order walk -------------
    sections_dict = _build_sections(pages, blocks_by_element_id, all_elements)

    # ---- Step 5: Score hierarchy ----------------------------------------
    total_pages = metadata.total_pages or len(pages)
    quality_label, explanation = score_hierarchy(sections_dict, total_pages)

    # ---- Step 6: Fallback population ------------------------------------
    if quality_label == "fallback":
        logger.warning(
            "Assembler falling back to per-page sections: %s", explanation
        )
        sections_dict = _build_fallback_sections(pages_dict)

    # Annotate metadata with ingestion stats.
    title_count = sum(1 for e in all_elements if e.element_type == "Title")
    if metadata.ingestion_meta is None:
        metadata.ingestion_meta = {}
    metadata.ingestion_meta["hierarchy_quality"] = quality_label
    metadata.ingestion_meta["assembler_notes"] = explanation
    metadata.ingestion_meta["section_count"] = len(sections_dict)
    metadata.ingestion_meta["title_count"] = title_count

    logger.info(
        "Assembly complete: quality=%s, sections=%d, titles=%d, pages=%d",
        quality_label,
        len(sections_dict),
        title_count,
        len(pages_dict),
    )

    return RLMGuide(
        metadata=metadata,
        sections=sections_dict,
        pages=pages_dict,
    )
