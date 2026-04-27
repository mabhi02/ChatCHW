"""Ingestion pipeline configuration. Reads env vars set in .env."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionConfig:
    # Unstructured.io
    unstructured_api_key: str
    unstructured_api_url: str
    # OpenAI (for vision passes)
    openai_api_key: str
    vision_model_primary: str  # e.g. "gpt-5.4"
    vision_model_fast: str  # e.g. "gpt-5.4-mini"
    # Concurrency limits (tuned for Render Standard 1 CPU / 2GB)
    max_table_concurrency: int
    max_image_concurrency: int
    # Safety
    max_pdf_bytes: int


def _int_env(name: str, default: str) -> int:
    """Parse an int env var with a descriptive error on malformed input.
    Bare int(os.environ.get(...)) produces a useless ValueError that doesn't
    name the offending variable."""
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name!r} must be an integer, got {raw!r}"
        ) from exc


def load_ingestion_config() -> IngestionConfig:
    """Load ingestion config from env. Missing REQUIRED keys raise at startup,
    not at first request. This is the fail-fast contract: if the backend boots,
    it has every credential it needs."""
    unstructured_key = os.environ.get("UNSTRUCTURED_API_KEY", "").strip()
    if not unstructured_key:
        raise RuntimeError(
            "UNSTRUCTURED_API_KEY is required but not set in the environment. "
            "Add it to .env before starting the backend."
        )

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required but not set in the environment. "
            "The vision enrichment pipeline cannot run without it. "
            "Add it to .env before starting the backend."
        )

    unstructured_url = os.environ.get(
        "UNSTRUCTURED_API_URL",
        "https://api.unstructuredapp.io/general/v0/general",
    ).strip()

    return IngestionConfig(
        unstructured_api_key=unstructured_key,
        unstructured_api_url=unstructured_url,
        openai_api_key=openai_key,
        vision_model_primary=os.environ.get("VISION_MODEL_PRIMARY", "gpt-5.4").strip(),
        vision_model_fast=os.environ.get("VISION_MODEL_FAST", "gpt-5.4-mini").strip(),
        max_table_concurrency=_int_env("INGEST_MAX_TABLE_CONCURRENCY", "8"),
        max_image_concurrency=_int_env("INGEST_MAX_IMAGE_CONCURRENCY", "4"),
        max_pdf_bytes=_int_env("INGEST_MAX_PDF_BYTES", str(50 * 1024 * 1024)),
    )
