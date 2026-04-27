"""Thin async wrapper around Unstructured.io's serverless hi_res API.

This module is intentionally deterministic: same PDF bytes in, same element list
out. It does NOT call any LLMs. The downstream vision pipeline (backend/ingestion/vision.py)
handles semantic enrichment using the coordinates this module preserves.
"""

import asyncio
import logging
from typing import Any

from unstructured_client import UnstructuredClient as RawClient
from unstructured_client.models import operations, shared

from backend.ingestion.schema import UnstructuredElement, UnstructuredPage

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict:
    """Coerce SDK values that may be dicts or attribute-bearing objects into a dict.

    The Unstructured SDK has gone back and forth between returning plain dicts and
    pydantic-style model objects for nested fields like `metadata` and `coordinates`.
    This helper normalizes both shapes so the rest of the module can stay simple.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    # Pydantic v2 model
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # pragma: no cover - defensive
            pass
    # Pydantic v1 / dataclass-like
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return value.dict()
        except Exception:  # pragma: no cover - defensive
            pass
    # Generic object with attributes
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def _get(raw: Any, key: str, default: Any = None) -> Any:
    """Get a field from either a dict or an attribute-bearing object."""
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


class UnstructuredClient:
    """Wraps the Unstructured.io serverless SDK with the settings we want.

    Single responsibility: take PDF bytes, return a list of UnstructuredPage objects
    with element coordinates preserved for downstream cropping.
    """

    def __init__(self, api_key: str, api_url: str):
        self._client = RawClient(
            api_key_auth=api_key,
            server_url=api_url,
        )

    async def partition_pdf(self, pdf_bytes: bytes, filename: str) -> list[UnstructuredPage]:
        """Send a PDF to Unstructured and return its elements grouped by page.

        Uses the hi_res strategy with table extraction enabled and coordinates
        preserved. hi_res is the only strategy that produces structured tables
        (with text_as_html), which is what we need for clinical content.
        """
        logger.info(
            "Sending %s (%d bytes) to Unstructured.io hi_res", filename, len(pdf_bytes)
        )

        request = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=shared.Files(
                    content=pdf_bytes,
                    file_name=filename,
                ),
                strategy=shared.Strategy.HI_RES,
                # Enable table extraction with HTML structure preservation.
                # This is the whole reason we use hi_res over fast.
                skip_infer_table_types=[],
                # Coordinates are required for the downstream vision pipeline,
                # which crops element regions from the rendered page image.
                coordinates=True,
                # Languages hint helps OCR on image pages with text overlays.
                # Note: this is a hint to the OCR engine, NOT a translation directive,
                # so the returned text will still be in whatever language the PDF uses.
                languages=["eng"],
                # Client-side PDF splitting: the SDK slices the PDF into ~15 page
                # chunks, uploads them concurrently, and reassembles the element
                # list on return. A 142-page manual completes in ~6s (fast) / ~30-45s
                # (hi_res) on benchmark, vs ~30-90s sequential. Cost is unchanged —
                # Unstructured bills per page regardless of split strategy.
                split_pdf_page=True,
                split_pdf_concurrency_level=15,  # SDK default is 5; 15 is the max
                split_pdf_cache_tmp_data=False,  # keep chunks in memory, not /tmp
                split_pdf_allow_failed=False,  # fail fast on any chunk error
            ),
        )

        # The SDK is sync; run it in a thread to avoid blocking the event loop.
        # The Unstructured serverless API can take 30-90 seconds for a 100-page PDF.
        # NOTE: get_running_loop() is the correct API on Python 3.10+. The
        # deprecated get_event_loop() does not reliably return the running loop
        # inside a coroutine and emits a DeprecationWarning under FastAPI.
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: self._client.general.partition(request=request)
            )
        except Exception as exc:
            # Re-raise with filename context. Retries are the orchestrator's job.
            raise RuntimeError(
                f"Unstructured partition failed for {filename}: {exc}"
            ) from exc

        if response.elements is None:
            raise RuntimeError(f"Unstructured returned no elements for {filename}")

        elements_raw = response.elements
        logger.info(
            "Unstructured returned %d elements for %s", len(elements_raw), filename
        )

        return self._group_by_page(elements_raw)

    def _group_by_page(self, elements_raw: list) -> list[UnstructuredPage]:
        """Group raw elements by page number and build raw_text per page.

        Preserves the full coordinates dict from each element's metadata so the
        downstream vision pipeline can crop the source page image by bounding box.
        """
        pages_dict: dict[int, list[UnstructuredElement]] = {}
        running_index = 0

        for raw in elements_raw:
            metadata = _as_dict(_get(raw, "metadata", {}))
            # Coerce page_number to int defensively. Unstructured *should*
            # always return an int, but if it ever returns a string ("3" vs 3),
            # mixing both keys causes sorted() downstream to raise TypeError.
            raw_page = metadata.get("page_number", 1)
            try:
                page_num = int(raw_page) if raw_page is not None else 1
            except (TypeError, ValueError):
                page_num = 1
            if page_num < 1:
                page_num = 1

            # Preserve coordinates verbatim. Unstructured returns them as
            # {"points": [[x,y],...], "system": "PixelSpace",
            #  "layout_width": W, "layout_height": H}.
            # We copy the whole dict so callers can do their own geometry math.
            coordinates = _as_dict(metadata.get("coordinates")) or None

            element = UnstructuredElement(
                element_type=_get(raw, "type", "Unknown") or "Unknown",
                text=_get(raw, "text", "") or "",
                text_as_html=metadata.get("text_as_html"),
                page_number=page_num,
                element_id=_get(raw, "element_id", f"unstructured-{running_index}")
                or f"unstructured-{running_index}",
                coordinates=coordinates,
            )
            pages_dict.setdefault(page_num, []).append(element)
            running_index += 1

        pages: list[UnstructuredPage] = []
        for page_num in sorted(pages_dict.keys()):
            elements = pages_dict[page_num]
            # raw_text is the concatenated text of all elements on this page,
            # joined with single spaces (NOT newlines). Newlines at element
            # boundaries break fuzzy quote grounding when a clinical phrase
            # spans two adjacent elements (e.g. "<drug_phrase_part_1>" +
            # "<dose_part_2>" would join to "<phrase_part_1>\n<part_2>" and
            # any later quote of "<phrase_part_1> <part_2>" would fall below
            # the 0.85 fuzzy threshold).
            raw_text = " ".join(e.text for e in elements if e.text.strip())
            pages.append(
                UnstructuredPage(
                    page_number=page_num,
                    elements=elements,
                    raw_text=raw_text,
                )
            )

        return pages
