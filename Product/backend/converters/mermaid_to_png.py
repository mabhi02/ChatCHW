"""Render a Mermaid flowchart source to a PNG image.

Strategy: try remote renderers in order, fall back to a PIL-generated
placeholder if none succeed. Never raises — always returns some PNG bytes
so the download bundle always contains a `flowchart.png`.

Remote renderers tried:
  1. mermaid.ink via GET (with proper User-Agent; scale=3 for sharpness)
  2. kroki.io via POST (fallback; sometimes blocked by Cloudflare)

Placeholder fallback: a 1600x900 neutral PNG with a red "Rendering failed"
banner and the failure reason text. Produced via Pillow.
"""

from __future__ import annotations

import base64
import logging
import urllib.error
import urllib.request
import zlib
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum render dimensions. Mermaid.ink accepts scale up to ~3x natively;
# we combine with a wide base width for maximum resolution on complex diagrams.
_MERMAID_INK_BASE_URL = "https://mermaid.ink/img"
_KROKI_BASE_URL = "https://kroki.io/mermaid/png"
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CHW-Navigator/1.0"
_TIMEOUT_SECONDS = 30

# Placeholder dimensions — widescreen so the failure text is readable even
# when someone previews the PNG full-size.
_PLACEHOLDER_WIDTH = 1600
_PLACEHOLDER_HEIGHT = 900


def render_mermaid_to_png(mermaid_source: str) -> tuple[bytes, str]:
    """Render Mermaid source to PNG bytes.

    Returns a tuple of (png_bytes, status_string). Status values:
      - "mermaid.ink"   -> rendered successfully via mermaid.ink
      - "kroki.io"      -> rendered successfully via kroki.io
      - "placeholder"   -> all remote renderers failed, returning a placeholder

    Never raises -- always returns some PNG bytes so the download bundle
    always has a flowchart.png even when rendering is unavailable.
    """
    if not mermaid_source or not mermaid_source.strip():
        logger.warning("Empty mermaid source, returning placeholder")
        return _make_placeholder("Empty mermaid source"), "placeholder"

    # Strip markdown fences if the caller passed them in
    source = mermaid_source.strip()
    if source.startswith("```"):
        lines = source.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        source = "\n".join(lines).strip()

    # Try local mmdc first (no URL length limit, works offline, sharper output).
    # Falls through to remote renderers if mmdc isn't installed.
    try:
        png = _render_via_local_mmdc(source)
        if png:
            logger.info("Rendered mermaid via local mmdc: %d bytes", len(png))
            return png, "local-mmdc"
    except Exception as exc:
        logger.warning("local mmdc failed: %s", exc)

    # Fall back to mermaid.ink
    try:
        png = _render_via_mermaid_ink(source)
        if png:
            logger.info("Rendered mermaid via mermaid.ink: %d bytes", len(png))
            return png, "mermaid.ink"
    except Exception as exc:
        logger.warning("mermaid.ink failed: %s", exc)

    # Fall back to kroki.io
    try:
        png = _render_via_kroki(source)
        if png:
            logger.info("Rendered mermaid via kroki.io: %d bytes", len(png))
            return png, "kroki.io"
    except Exception as exc:
        logger.warning("kroki.io failed: %s", exc)

    # All remote renderers failed — return a placeholder
    logger.warning("All mermaid renderers failed, returning placeholder PNG")
    return _make_placeholder("All remote renderers unavailable"), "placeholder"


def _render_via_local_mmdc(source: str) -> bytes | None:
    """Render via the locally-installed mermaid CLI (mmdc).

    No URL-length limit (the remote renderers cap around 8 KB / Cloudflare-
    block large diagrams). Output is sharper at scale=3, white background
    for print-friendly review.

    Returns None if mmdc is not on PATH or the render produced no output.
    """
    import shutil
    import subprocess
    import tempfile
    import os

    # Prefer the .cmd shim on Windows where npm installs mmdc as a wrapper.
    mmdc = shutil.which("mmdc") or shutil.which("mmdc.cmd") or shutil.which("mmdc.ps1")
    if not mmdc:
        logger.info("mmdc not on PATH; skipping local render")
        return None

    with tempfile.TemporaryDirectory(prefix="mmdc_") as tmp:
        in_path = os.path.join(tmp, "diagram.mmd")
        out_path = os.path.join(tmp, "diagram.png")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            result = subprocess.run(
                [mmdc, "-i", in_path, "-o", out_path, "-s", "3", "-b", "white", "-w", "1400"],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            logger.warning("mmdc render timed out after 120s")
            return None
        if result.returncode != 0:
            logger.warning("mmdc returned %d: %s", result.returncode, (result.stderr or "")[:200])
            return None
        if not os.path.exists(out_path):
            logger.warning("mmdc reported success but no output file")
            return None
        with open(out_path, "rb") as f:
            return f.read()


def _render_via_mermaid_ink(source: str) -> bytes | None:
    """Hit mermaid.ink/img/{base64} with scale=3 for maximum resolution."""
    encoded = base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii").rstrip("=")
    # scale=3 gives retina-quality output; bgColor=white for print-friendly
    url = f"{_MERMAID_INK_BASE_URL}/{encoded}?type=png&theme=default&bgColor=FFFFFF&scale=3"

    if len(url) > 8000:
        logger.warning(
            "mermaid.ink URL too long (%d chars) — skipping, will try kroki",
            len(url),
        )
        return None

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "image/png" not in content_type.lower():
            logger.warning("mermaid.ink returned non-PNG: %s", content_type)
            return None
        return resp.read()


def _render_via_kroki(source: str) -> bytes | None:
    """POST raw mermaid to kroki.io/mermaid/png (avoids URL length limits)."""
    req = urllib.request.Request(
        _KROKI_BASE_URL,
        data=source.encode("utf-8"),
        headers={
            "Content-Type": "text/plain",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "image/png" not in content_type.lower():
            logger.warning("kroki.io returned non-PNG: %s", content_type)
            return None
        return resp.read()


def _make_placeholder(reason: str) -> bytes:
    """Generate a 1600x900 placeholder PNG with a failure banner.

    The placeholder is intentionally plain so it's obvious in the ZIP
    bundle that the real render didn't succeed, but the file still
    exists so downstream tools that expect `flowchart.png` don't break.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # Pillow not available — return a minimal valid 1x1 PNG so we
        # never return empty bytes. Consumers will see a tiny image
        # and know something went wrong.
        return _minimal_png_bytes()

    img = Image.new("RGB", (_PLACEHOLDER_WIDTH, _PLACEHOLDER_HEIGHT), color=(245, 245, 250))
    draw = ImageDraw.Draw(img)

    # Red banner across the top
    banner_height = 120
    draw.rectangle(
        [(0, 0), (_PLACEHOLDER_WIDTH, banner_height)],
        fill=(220, 38, 38),
    )

    # Try to load a reasonable system font; fall back to default
    def _font(size: int):
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = _font(52)
    body_font = _font(28)
    small_font = _font(20)

    # Banner text
    draw.text(
        (40, 30),
        "Mermaid rendering failed",
        fill=(255, 255, 255),
        font=title_font,
    )

    # Body
    body_y = banner_height + 60
    draw.text(
        (40, body_y),
        "The backend could not reach any remote mermaid renderer.",
        fill=(30, 30, 30),
        font=body_font,
    )
    draw.text(
        (40, body_y + 50),
        f"Reason: {reason}",
        fill=(120, 30, 30),
        font=body_font,
    )

    # Instructions
    instruction_y = body_y + 160
    instructions = [
        "The raw Mermaid source is still available:",
        "",
        "  - Download 'flowchart.md' from the artifacts panel",
        "  - Paste the contents into https://mermaid.live to render it locally",
        "  - Or install '@mermaid-js/mermaid-cli' and run:",
        "      mmdc -i flowchart.md -o flowchart.png -s 3",
        "",
        "Common causes of remote renderer failure:",
        "  1. Local network down or no internet access",
        "  2. Cloudflare bot protection blocking the request",
        "  3. Diagram too large for URL-based rendering",
        "  4. Malformed Mermaid syntax (check flowchart.md for errors)",
    ]
    for i, line in enumerate(instructions):
        draw.text(
            (40, instruction_y + i * 34),
            line,
            fill=(60, 60, 60),
            font=small_font,
        )

    # Footer
    draw.text(
        (40, _PLACEHOLDER_HEIGHT - 50),
        "CHW Navigator — flowchart.png placeholder",
        fill=(150, 150, 150),
        font=small_font,
    )

    # Save to bytes
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _minimal_png_bytes() -> bytes:
    """Return a minimal valid 1x1 gray PNG.

    Used as a last-resort fallback when Pillow is not available. The PNG
    spec mandates 8-byte signature + IHDR + IDAT + IEND chunks.
    """
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return (
            len(data).to_bytes(4, "big")
            + chunk_type
            + data
            + crc.to_bytes(4, "big")
        )

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width=1, height=1, bit_depth=8, color_type=0 (grayscale)
    ihdr = chunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00")
    # IDAT: single gray pixel
    raw = b"\x00\x80"  # filter byte + gray value
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def write_flowchart_png(mermaid_source: str, output_path: Path) -> str:
    """Render Mermaid to PNG and write it to `output_path`.

    Returns the status string from render_mermaid_to_png so callers can log
    whether the render succeeded or used the placeholder.
    """
    png_bytes, status = render_mermaid_to_png(mermaid_source)
    output_path.write_bytes(png_bytes)
    return status
