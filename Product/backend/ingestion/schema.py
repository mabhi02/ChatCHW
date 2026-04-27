"""Pydantic types for the ingestion pipeline.

This module is the type contract all other ingestion modules depend on.
Do not import from other ingestion modules here — this file must be leaf-level.
"""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Group A: Unstructured.io output types (deterministic parsing result)
# ---------------------------------------------------------------------------


class UnstructuredElement(BaseModel):
    """A single element from Unstructured.io hi_res parsing."""
    element_type: str  # "Title", "NarrativeText", "ListItem", "Table", "Image", "Header", "Footer", "PageBreak"
    text: str
    text_as_html: Optional[str] = None  # HTML representation for tables
    page_number: int
    element_id: str
    coordinates: Optional[dict] = None  # {"points": [[x, y], ...], "layout_width": float, "layout_height": float}


class UnstructuredPage(BaseModel):
    """All elements from one page plus concatenated raw_text for verification."""
    page_number: int
    elements: list[UnstructuredElement]
    raw_text: str  # "\n".join(e.text for e in elements if e.text.strip())


# ---------------------------------------------------------------------------
# Group B: Vision enrichment outputs (from backend/ingestion/vision.py)
# ---------------------------------------------------------------------------


class TableRefinement(BaseModel):
    """Output of vision table refinement pass (gpt-5.4)."""
    element_id: str  # links back to UnstructuredElement
    rows: list[list[str]]  # structured rows (each is a list of cell strings)
    headers: list[str] = Field(default_factory=list)
    english_text: str  # normalized English form of the table contents
    was_translated: bool = False
    source_language: str = "en"  # ISO code
    severity_colors: dict[str, str] = Field(default_factory=dict)  # {"row_0": "red", ...}
    notes: str = ""  # e.g. "merged cell at (2,1)"
    model_used: str  # "gpt-5.4" or "gpt-5.4-mini"


class ImageDescription(BaseModel):
    """Output of vision image description pass."""
    element_id: str
    caption: str  # "Figure 3: MUAC measurement"
    description: str  # English description
    extracted_text: str  # any text visible in the image
    is_flowchart: bool = False
    is_photograph: bool = False
    is_diagram: bool = False
    model_used: str


class FlowchartStructure(BaseModel):
    """Output of vision flowchart parsing pass (only for is_flowchart images)."""
    element_id: str
    nodes: list[dict]  # [{"id": str, "label": str, "type": "decision"|"action"|"start"|"end"}]
    edges: list[dict]  # [{"from": str, "to": str, "condition": str|None}]
    english_text: str  # structured English walkthrough of the flowchart
    model_used: str


# ---------------------------------------------------------------------------
# Group C: RLM-facing output types (what the RLM's `guide` variable becomes)
# ---------------------------------------------------------------------------


class RLMBlock(BaseModel):
    """One atomic content block inside a section."""
    type: str  # "paragraph", "list", "table", "image", "heading", "image_description", "flowchart"
    page: int
    text: Optional[str] = None  # used by paragraph, heading
    items: Optional[list[str]] = None  # used by list
    html: Optional[str] = None  # used by table (refined HTML if vision pass ran)
    rows: Optional[list[list[str]]] = None  # used by table (structured rows from vision)
    english_text: Optional[str] = None  # used by table when source was non-English
    image_description: Optional[str] = None  # used by image/flowchart
    flowchart_nodes: Optional[list[dict]] = None
    flowchart_edges: Optional[list[dict]] = None
    element_id: Optional[str] = None  # links back to Unstructured element


class RLMSection(BaseModel):
    """A hierarchical section the RLM can navigate by name."""
    title: str
    level: int = 1  # heading level 1-6
    page_start: int
    page_end: int
    blocks: list[RLMBlock]
    raw_text: str  # concatenated text of all blocks, used for RLM's grounding checks


class RLMPage(BaseModel):
    """A flat per-page view, populated in parallel with sections for fallback navigation."""
    page_number: int
    blocks: list[RLMBlock]
    raw_text: str


class RLMGuideMetadata(BaseModel):
    title: str
    source_filename: str
    content_hash: str  # SHA-256 hex of the original PDF bytes
    total_pages: int
    ingested_at: str  # ISO timestamp
    ingestion_model: str = "unstructured-hi_res"
    ingestion_version: str = "1.0"
    ingestion_meta: dict = Field(default_factory=dict)  # hierarchy_quality, section_count, title_count, assembler_notes, costs


class RLMGuide(BaseModel):
    """The full ingested guide as the RLM sees it. This is what gets stored in SourceGuide.guideJson."""
    metadata: RLMGuideMetadata
    sections: dict[str, RLMSection]  # keyed by slugified title — RLM iterates these
    pages: dict[str, RLMPage]  # keyed by str(page_number) — fallback view


# ---------------------------------------------------------------------------
# Group C2: Ingestion quality manifest
# ---------------------------------------------------------------------------
#
# A per-ingestion quality report. Surfaces ingestion-side issues so consumers
# can decide whether to start an RLM extraction session against this guide.
# Stored alongside the RLMGuide in SourceGuide.ingestionMeta["manifest"].

IngestionIssueType = Literal[
    "vision_table_failed",
    "vision_image_failed",
    "vision_flowchart_failed",
    "page_orphaned",
    "hierarchy_fallback",
    "hierarchy_noisy",
    "page_count_mismatch",
    "empty_page",
    "extraction_error",
]

IngestionSeverity = Literal["critical", "warning", "info"]


class IngestionFlaggedItem(BaseModel):
    """A single quality issue detected during ingestion."""
    page_number: Optional[int] = None
    element_id: Optional[str] = None
    issue_type: IngestionIssueType
    severity: IngestionSeverity
    message: str
    context: dict = Field(default_factory=dict)


class IngestionManifest(BaseModel):
    """Aggregated quality report for one ingestion run.

    Consumers can gate session-start on `critical_count == 0` to refuse running
    the RLM against a low-quality ingestion. The frontend surfaces flagged items
    inline so the user understands what failed.
    """
    total_pages: int = 0
    total_elements: int = 0
    total_tables: int = 0
    total_images: int = 0
    total_flowcharts: int = 0

    vision_table_attempts: int = 0
    vision_table_successes: int = 0
    vision_image_attempts: int = 0
    vision_image_successes: int = 0
    vision_flowchart_attempts: int = 0
    vision_flowchart_successes: int = 0

    hierarchy_quality: str = "unknown"  # good | sparse | noisy | fallback
    section_count: int = 0

    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    flagged_items: list[IngestionFlaggedItem] = Field(default_factory=list)

    def add(self, item: IngestionFlaggedItem) -> None:
        """Append a flagged item and update severity counters."""
        self.flagged_items.append(item)
        if item.severity == "critical":
            self.critical_count += 1
        elif item.severity == "warning":
            self.warning_count += 1
        else:
            self.info_count += 1


# ---------------------------------------------------------------------------
# Group D: Ingestion job state (for Redis)
# ---------------------------------------------------------------------------


JobStatus = Literal["queued", "running", "done", "failed"]
JobStage = Literal[
    "queued",
    "cache_check",
    "cache_hit",
    "acquiring_lock",
    "warming_pool",
    "parsing_with_unstructured",
    "rendering_crops",
    "vision_table_refinement",
    "vision_image_description",
    "vision_flowchart_parsing",
    "assembling",
    "writing_to_neon",
    "done",
    "failed",
]


class IngestionJobState(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    progress: float = 0.0  # 0.0 to 1.0
    note: str = ""
    guide_id: Optional[str] = None  # populated when status=done
    content_hash: Optional[str] = None
    error: Optional[str] = None
    # started_at / updated_at default to 0.0 so that pipeline.update_job_state
    # (which writes a raw dict without necessarily including started_at on
    # later state transitions) can still be parsed back into this model by
    # any consumer. The pipeline now always sets updated_at; started_at is
    # set on the first write and preserved on subsequent reads from Redis.
    started_at: float = 0.0
    updated_at: float = 0.0
