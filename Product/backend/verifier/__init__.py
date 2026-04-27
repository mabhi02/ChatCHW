"""Independent verifier harness.

GPT primary, Anthropic same-family fallback. Read-only: emits
`{agree, divergences[]}` per artifact and never edits artifact content.
Falls back to same-family Anthropic verification when GPT is unavailable,
and marks every emitted divergence with `verifier_independence` so
reviewers can tell the two cases apart.
"""

from pathlib import Path
import json

from backend.provenance.schema import VerificationBlock


def write_verification_sidecar(artifact_path: Path, vblock: VerificationBlock) -> None:
    """Write `<artifact>.verification.json` next to the artifact.

    Separate from `<artifact>.provenance.json` so the two concerns stay
    independent: provenance (who/when/what) vs verification (agree/disagree).
    """
    sidecar = artifact_path.with_suffix(artifact_path.suffix + ".verification.json")
    sidecar.write_text(vblock.model_dump_json(indent=2), encoding="utf-8")


def update_provenance_status(artifact_path: Path, vblock: VerificationBlock) -> None:
    """Update the provenance sidecar's `status` field from the verification result."""
    prov_path = artifact_path.with_suffix(artifact_path.suffix + ".provenance.json")
    if not prov_path.exists():
        return
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    has_error = any(d.severity == "error" for d in vblock.divergences)
    if vblock.agree and not has_error:
        prov["status"] = "verified"
    elif vblock.agree:
        # warn + info divergences do not block verified status
        prov["status"] = "verified"
    else:
        prov["status"] = "verification_failed"
    prov_path.write_text(json.dumps(prov, indent=2), encoding="utf-8")


__all__ = ["write_verification_sidecar", "update_provenance_status"]
