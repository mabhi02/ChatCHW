"""Per-upload warmup burst for Unstructured workers.

Fires N parallel 1-page probes before the real parse so the pool is warm
when the big partition_pdf call arrives. Scales per-upload rather than via
a keep-alive schedule, so Unstructured quota usage tracks actual usage (no
idle burn, no "demo hours" configuration, works at any time of day).

The probe PDF is a single-page fixture bundled with the repo. It's loaded
once at module import and reused for every probe.

Usage:

    from backend.ingestion.warmup import warmup_burst, compute_probe_concurrency

    n = compute_probe_concurrency(page_count)
    await warmup_burst(unstructured_client, concurrency=n)
    # Pool is now warm; fire the real parse.
    pages = await unstructured_client.partition_pdf(pdf_bytes, filename)

Cancellation: `warmup_burst` is safe to wrap in `asyncio.create_task` and
cancel if the caller decides it doesn't need to parse (e.g., cache hit).
In-flight probe HTTP requests may still complete on Unstructured's side
and consume a handful of pages — the cost is bounded by the probe count
(~$0.15 worst case for a full 15-probe burst).
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from backend.ingestion.unstructured_client import UnstructuredClient

logger = logging.getLogger(__name__)

# Loaded once at module import. Tiny single-page PDF bundled in the repo.
#
# FAIL-FAST: If the file is missing, we raise at import time rather than
# degrading silently. A missing warmup PDF would make every cold ingestion
# hit the slow cold-start path without any operator signal. The cost of the
# crash is a clear error message at deploy time; the cost of silent no-op is
# hours of "why are my uploads so slow" debugging in production.
_WARMUP_PDF_PATH: Final[Path] = Path(__file__).parent / "warmup.pdf"
try:
    _WARMUP_PDF_BYTES: bytes = _WARMUP_PDF_PATH.read_bytes()
except OSError as exc:
    raise RuntimeError(
        f"Warmup PDF fixture missing at {_WARMUP_PDF_PATH}. "
        f"Run `python -c \"import pypdfium2 as p; s=p.PdfDocument('CHW Navigator v1 (1).pdf'); "
        f"d=p.PdfDocument.new(); d.import_pages(s, [0]); "
        f"d.save('{_WARMUP_PDF_PATH}'); s.close(); d.close()\"` "
        f"to regenerate, or restore from git."
    ) from exc

if len(_WARMUP_PDF_BYTES) == 0:
    raise RuntimeError(
        f"Warmup PDF fixture at {_WARMUP_PDF_PATH} is empty. "
        f"Regenerate it — the pool warmup will be a no-op otherwise."
    )

logger.info(
    "Loaded warmup PDF from %s (%d bytes)",
    _WARMUP_PDF_PATH,
    len(_WARMUP_PDF_BYTES),
)


# Empirical: at split_pdf_concurrency_level=15 the SDK picks chunks of
# roughly 13-15 pages each, capped at 15 parallel chunks total. We probe
# one worker per expected chunk so our warmup exactly matches the parallelism
# of the real parse.
_PAGES_PER_CHUNK: Final[int] = 15
_MAX_WORKERS: Final[int] = 15


def compute_probe_concurrency(
    page_count: int,
    max_workers: int = _MAX_WORKERS,
) -> int:
    """Size the warmup burst to match the SDK's expected parallelism.

    For a 200-page PDF the SDK will fan out to 15 workers, so we probe 15.
    For a 30-page PDF the SDK fans out to ~2 workers, so we probe 2.
    """
    if page_count <= 0:
        return 1
    expected_chunks = min(max_workers, max(1, (page_count + _PAGES_PER_CHUNK - 1) // _PAGES_PER_CHUNK))
    return expected_chunks


async def warmup_burst(
    unstructured_client: "UnstructuredClient",
    concurrency: int,
) -> None:
    """Fire N parallel 1-page probes to warm N Unstructured workers.

    Blocks until all probes complete (or fail). Exceptions are swallowed —
    warmup is best-effort; the real parse runs regardless of probe outcomes.

    Args:
        unstructured_client: the same client the real parse will use.
        concurrency: number of parallel probes to fire.
    """
    if not _WARMUP_PDF_BYTES:
        logger.warning(
            "Warmup PDF bytes unavailable; skipping burst (probe=%d)",
            concurrency,
        )
        return

    if concurrency < 1:
        logger.debug("Warmup burst skipped: concurrency=%d", concurrency)
        return

    logger.info("Warmup burst starting: firing %d parallel probes", concurrency)
    start = time.time()

    probes = [
        unstructured_client.partition_pdf(
            _WARMUP_PDF_BYTES,
            f"_warmup_{i}.pdf",
        )
        for i in range(concurrency)
    ]

    # return_exceptions=True so one probe failing doesn't cancel the rest
    # via the implicit gather semantics.
    results = await asyncio.gather(*probes, return_exceptions=True)

    elapsed = time.time() - start
    failures = sum(1 for r in results if isinstance(r, Exception))
    successes = concurrency - failures

    logger.info(
        "Warmup burst complete: %d/%d probes succeeded in %.1fs (%.1fs/probe avg)",
        successes,
        concurrency,
        elapsed,
        elapsed / max(1, concurrency),
    )

    if failures > 0:
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning("Warmup probe %d failed: %s", i, r)
