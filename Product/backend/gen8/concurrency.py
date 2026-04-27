"""Throttled concurrent execution helpers for gen8.5.

7-way Sonnet labeling runs 7 parallel passes per chunk. Across a 21-chunk
guide that's 147 total calls; we keep total in-flight bounded with two
stacked semaphores rather than blasting them all at once.

Default per-chunk fan-out is 7 (one per artifact type) and default
outer-chunk concurrency is 3, capping in-flight at 21 calls. Anthropic
Tier 4 limits accommodate this comfortably.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Iterable, TypeVar

T = TypeVar("T")


async def throttled_gather(
    coros: Iterable[Awaitable[T]],
    max_concurrent: int = 5,
) -> list[T]:
    """Run an iterable of coroutines with a bounded concurrency ceiling."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _wrap(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(c) for c in coros], return_exceptions=False)
