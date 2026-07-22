"""Review-only ICA image classification through the purpose-specific ClinCog gateway."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .clincog_client import GatewayOutcome, GatewayResult, resolve_gateway_token, send_classification_image
from .config import COMPONENT_LABELS


_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_REASON_CHARS = 1_000
_UNAVAILABLE_REASON = "Classification unavailable; human review required."
_TEMPORARY_IMAGE_ARTIFACT = "temporary_component_webp"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


@dataclass(frozen=True)
class RawClassification:
    label: Optional[str]
    confidence: Optional[float]
    reason: str
    outcome_status: str
    failure_category: Optional[str]
    review_required: bool = True
    apply_to_ica: bool = False
    exclude_vision: bool = False
    model: Optional[str] = None
    request_id: Optional[str] = None
    elapsed_seconds: float = 0.0
    usage: object = None
    prompt_sha256: Optional[str] = None
    artifact_inventory: Tuple[str, ...] = ()


def _unavailable(
    category: str,
    *,
    elapsed_seconds: float = 0.0,
    artifact_inventory: Tuple[str, ...] = (),
) -> RawClassification:
    return RawClassification(
        None,
        None,
        _UNAVAILABLE_REASON,
        "unavailable",
        category,
        elapsed_seconds=elapsed_seconds,
        artifact_inventory=artifact_inventory,
    )


def _webp_bytes(image_path: Path) -> bytes:
    size = image_path.stat().st_size
    if not 0 < size <= _MAX_IMAGE_BYTES:
        raise ValueError
    image = image_path.read_bytes()
    if len(image) != size or len(image) < 12 or image[:4] != b"RIFF" or image[8:12] != b"WEBP":
        raise ValueError
    return image


def _parse_classification(result: GatewayResult, elapsed_seconds: float) -> RawClassification:
    payload = result.classification
    if not isinstance(payload, dict):
        return _unavailable(GatewayOutcome.MALFORMED_RESPONSE.value, elapsed_seconds=elapsed_seconds, artifact_inventory=(_TEMPORARY_IMAGE_ARTIFACT,))
    label = payload["label"]
    confidence = payload["confidence"]
    reason = payload["reason"]
    model = payload["model"]
    request_id = payload["request_id"]
    if (
        label not in COMPONENT_LABELS
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0.0 <= float(confidence) <= 1.0
        or not isinstance(reason, str)
        or not 0 < len(reason) <= _MAX_REASON_CHARS
        or not all(" " <= character <= "~" for character in reason)
        or not isinstance(model, str)
        or _IDENTIFIER.fullmatch(model) is None
        or not isinstance(request_id, str)
        or _IDENTIFIER.fullmatch(request_id) is None
    ):
        return _unavailable(GatewayOutcome.MALFORMED_RESPONSE.value, elapsed_seconds=elapsed_seconds, artifact_inventory=(_TEMPORARY_IMAGE_ARTIFACT,))
    return RawClassification(
        label,
        float(confidence),
        reason,
        "classified",
        None,
        model=model,
        request_id=request_id,
        elapsed_seconds=elapsed_seconds,
        artifact_inventory=(_TEMPORARY_IMAGE_ARTIFACT,),
    )


def classify_image_with_clincog(image_path: Path) -> RawClassification:
    """Classify one temporary WebP without accepting EEG, ICA, or cleaning inputs."""

    started = time.monotonic()
    try:
        image = _webp_bytes(image_path)
    except (OSError, ValueError):
        return _unavailable(GatewayOutcome.INVALID_REQUEST.value, elapsed_seconds=time.monotonic() - started)
    token = resolve_gateway_token()
    if token is None:
        return _unavailable(GatewayOutcome.INVALID_AUTHORIZATION.value, elapsed_seconds=time.monotonic() - started, artifact_inventory=(_TEMPORARY_IMAGE_ARTIFACT,))
    result = send_classification_image(image, token)
    elapsed_seconds = time.monotonic() - started
    if result.outcome is not GatewayOutcome.SUCCESS:
        return _unavailable(result.outcome.value, elapsed_seconds=elapsed_seconds, artifact_inventory=(_TEMPORARY_IMAGE_ARTIFACT,))
    return _parse_classification(result, elapsed_seconds)
