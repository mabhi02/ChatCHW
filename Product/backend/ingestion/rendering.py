"""PDF region rendering via pdf2image + pypdfium2 fallback + Pillow.

Used by the vision pipeline to crop tables and images out of PDF pages before
sending them to GPT-5.4. The crops are addressed by SHA-256 of the PNG bytes
so identical elements from different PDFs hit the Redis vision cache.

Rendering strategy — tries pdf2image (Poppler) first, falls back to pypdfium2
(pure Python) if Poppler isn't installed. This gives us:
  - Production (Docker with poppler-utils): pdf2image, fast, battle-tested
  - Windows dev (no Poppler): pypdfium2, zero system deps
pypdfium2 is already a transitive dep of unstructured-client so no extra install.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
from typing import Optional

from PIL import Image

# pdf2image is the preferred backend (needs Poppler). We import it lazily
# in the sync worker so its ImportError doesn't crash module load on systems
# where pypdfium2 would work fine.
try:
    from pdf2image import convert_from_bytes as _pdf2image_convert
    _PDF2IMAGE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep
    _pdf2image_convert = None  # type: ignore[assignment]
    _PDF2IMAGE_AVAILABLE = False

# pypdfium2 is the pure-Python fallback. It ships as a transitive dep of
# unstructured-client on all platforms so this import should always succeed.
try:
    import pypdfium2 as pdfium
    _PYPDFIUM_AVAILABLE = True
except Exception:  # pragma: no cover - missing fallback is a hard error
    pdfium = None  # type: ignore[assignment]
    _PYPDFIUM_AVAILABLE = False

logger = logging.getLogger(__name__)

# Track which backend we ended up using so we only log the choice once per process
_BACKEND_LOGGED = False

# Padding (in pixels of the rendered image) added around each cropped element
# so the vision model has a bit of surrounding context.
_CROP_PADDING_PX = 4


def image_sha256(png_bytes: bytes) -> str:
    """SHA-256 hex digest of PNG bytes. Used as cache key in Redis."""
    return hashlib.sha256(png_bytes).hexdigest()


def png_to_data_url(png_bytes: bytes) -> str:
    """Convert PNG bytes to a base64 data URL for OpenAI vision API."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _encode_png(img: Image.Image) -> bytes:
    """Encode a PIL Image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_with_pdf2image(pdf_bytes: bytes, page_number: int, dpi: int) -> Optional[Image.Image]:
    """Render a single page using pdf2image (Poppler). Returns None on failure.

    Returns None (not an exception) on any failure so the caller can try the
    pypdfium2 fallback. The common failure mode is Poppler not being installed
    (raises PDFInfoNotInstalledError on Windows dev).
    """
    if not _PDF2IMAGE_AVAILABLE or _pdf2image_convert is None:
        return None
    try:
        images = _pdf2image_convert(
            pdf_bytes,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
    except Exception as exc:
        logger.debug(
            "pdf2image failed for page %d (will try pypdfium2): %s",
            page_number,
            exc,
        )
        return None
    if not images:
        return None
    return images[0]


def _render_with_pypdfium2(pdf_bytes: bytes, page_number: int, dpi: int) -> Optional[Image.Image]:
    """Render a single page using pypdfium2 (pure Python). Returns None on failure."""
    if not _PYPDFIUM_AVAILABLE or pdfium is None:
        return None
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            if page_number > len(doc):
                logger.error(
                    "pypdfium2: requested page %d but PDF only has %d pages",
                    page_number,
                    len(doc),
                )
                return None
            page = doc[page_number - 1]  # pypdfium2 is 0-indexed
            # scale=dpi/72 because PDF points are 1/72 inch
            pil_image = page.render(scale=dpi / 72).to_pil()
            return pil_image
        finally:
            doc.close()
    except Exception as exc:
        logger.error(
            "pypdfium2 failed for page %d (dpi=%d): %s",
            page_number,
            dpi,
            exc,
        )
        return None


def _render_page_region_sync(
    pdf_bytes: bytes,
    page_number: int,
    coordinates: dict | None,
    dpi: int,
) -> bytes:
    """Synchronous worker that renders one page (and optionally crops it).

    Tries pdf2image first (Poppler, production), falls back to pypdfium2
    (pure Python, Windows dev). The first successful backend is logged once.

    This is the blocking implementation; the public ``render_page_region``
    wraps it in ``asyncio.to_thread`` so it does not stall the event loop.
    """
    global _BACKEND_LOGGED

    img = _render_with_pdf2image(pdf_bytes, page_number, dpi)
    backend_used = "pdf2image"
    if img is None:
        img = _render_with_pypdfium2(pdf_bytes, page_number, dpi)
        backend_used = "pypdfium2"

    if img is None:
        logger.error(
            "Both pdf2image and pypdfium2 failed to render page %d",
            page_number,
        )
        raise RuntimeError(
            f"Failed to render PDF page {page_number}: no rendering backend succeeded "
            f"(pdf2image={_PDF2IMAGE_AVAILABLE}, pypdfium2={_PYPDFIUM_AVAILABLE})"
        )

    if not _BACKEND_LOGGED:
        logger.info("PDF rendering backend: %s", backend_used)
        _BACKEND_LOGGED = True

    if coordinates is None:
        return _encode_png(img)

    required_keys = ("points", "layout_width", "layout_height")
    if not all(k in coordinates for k in required_keys):
        logger.warning(
            "coordinates dict missing expected keys %s for page %d; "
            "falling back to full-page render",
            required_keys,
            page_number,
        )
        return _encode_png(img)

    points = coordinates.get("points") or []
    layout_width = coordinates.get("layout_width")
    layout_height = coordinates.get("layout_height")

    if (
        not points
        or not layout_width
        or not layout_height
        or layout_width <= 0
        or layout_height <= 0
    ):
        logger.warning(
            "coordinates for page %d have invalid points/layout dims; "
            "falling back to full-page render",
            page_number,
        )
        return _encode_png(img)

    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, IndexError, ValueError) as exc:
        logger.warning(
            "Could not parse points %r for page %d (%s); "
            "falling back to full-page render",
            points,
            page_number,
            exc,
        )
        return _encode_png(img)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    sx = img.width / float(layout_width)
    sy = img.height / float(layout_height)

    left = int(x_min * sx) - _CROP_PADDING_PX
    upper = int(y_min * sy) - _CROP_PADDING_PX
    right = int(x_max * sx) + _CROP_PADDING_PX
    lower = int(y_max * sy) + _CROP_PADDING_PX

    # Clip to image bounds.
    left = max(0, min(left, img.width))
    upper = max(0, min(upper, img.height))
    right = max(0, min(right, img.width))
    lower = max(0, min(lower, img.height))

    # Guard against degenerate rectangles after clipping.
    if right <= left or lower <= upper:
        logger.warning(
            "Degenerate crop rect (%d,%d,%d,%d) for page %d; "
            "falling back to full-page render",
            left,
            upper,
            right,
            lower,
            page_number,
        )
        return _encode_png(img)

    cropped = img.crop((left, upper, right, lower))
    return _encode_png(cropped)


async def render_page_region(
    pdf_bytes: bytes,
    page_number: int,  # 1-indexed to match Unstructured
    coordinates: dict | None,  # Unstructured coordinates dict or None for full page
    dpi: int = 200,
) -> bytes:
    """Render a cropped PNG of the specified region of a PDF page.

    If coordinates is None, returns the full page image.
    If coordinates is provided, returns only the bounding box region.

    The coordinates dict follows Unstructured's format:
      {"points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
       "system": "PixelSpace",
       "layout_width": W, "layout_height": H}

    Points form the four corners of the bounding box (not necessarily in any
    particular order). We compute min/max to get the crop rectangle, then
    scale from layout dimensions to the actual rendered image dimensions.

    Returns: PNG-encoded bytes ready to pass to a vision API.
    """
    if page_number < 1:
        raise ValueError(
            f"page_number must be >= 1 (got {page_number}); "
            "Unstructured pages are 1-indexed"
        )

    return await asyncio.to_thread(
        _render_page_region_sync,
        pdf_bytes,
        page_number,
        coordinates,
        dpi,
    )
