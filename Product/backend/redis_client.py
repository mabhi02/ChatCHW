"""Upstash Redis REST client.

Uses the HTTP REST API (UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN)
instead of the Redis wire protocol, since Upstash provides REST credentials by default.

Simple key-value operations: get, set (with TTL), delete.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None
_base_url: str = ""
_token: str = ""


def _get_config() -> tuple[str, str]:
    """Read Upstash REST config from env."""
    url = os.environ.get("UPSTASH_REDIS_REST_URL", "")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
    return url.rstrip("/"), token


async def _get_client() -> httpx.AsyncClient:
    """Get or create the HTTP client."""
    global _client, _base_url, _token
    if _client is None:
        _base_url, _token = _get_config()
        if not _base_url or not _token:
            logger.warning("Upstash Redis not configured; session state will be in-memory only")
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


async def redis_get(key: str) -> str | None:
    """GET a key from Upstash Redis. Returns None if not found or not configured."""
    base, token = _get_config()
    if not base or not token:
        return None
    client = await _get_client()
    try:
        resp = await client.get(
            f"{base}/get/{key}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        return data.get("result")
    except Exception as e:
        logger.warning("Redis GET failed for key=%s: %s", key, e)
        return None


async def redis_setex(key: str, ttl_seconds: int, value: str) -> bool:
    """SET a key with TTL. Returns True on success."""
    base, token = _get_config()
    if not base or not token:
        return False
    client = await _get_client()
    try:
        resp = await client.get(
            f"{base}/setex/{key}/{ttl_seconds}/{value}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        return data.get("result") == "OK"
    except Exception as e:
        logger.warning("Redis SETEX failed for key=%s: %s", key, e)
        return False


async def redis_delete(key: str) -> bool:
    """DELETE a key. Returns True on success."""
    base, token = _get_config()
    if not base or not token:
        return False
    client = await _get_client()
    try:
        resp = await client.get(
            f"{base}/del/{key}",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        return data.get("result", 0) >= 1
    except Exception as e:
        logger.warning("Redis DEL failed for key=%s: %s", key, e)
        return False


async def close_client() -> None:
    """Close the HTTP client."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
