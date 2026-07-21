"""Pure offline contract for the reviewed synthetic Responses request shape.

Gate 2, not this module, enforces timeout, concurrency, retry, redirect,
response-status, and transport MIME behavior.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional

from .config import COMPONENT_LABELS


class RequestContractError(ValueError):
    """Stable sanitized request-contract error."""


class ImageDetail(str, Enum):
    LOW = "low"


class TransportOutcome(str, Enum):
    COMPLETED = "completed"
    REFUSAL = "refusal"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True)
class RequestLimits:
    max_prompt_bytes: int
    max_image_bytes: int
    max_data_url_bytes: int
    max_request_bytes: int
    max_response_bytes: int
    max_output_tokens: int
    timeout_seconds: float
    max_concurrency: int


@dataclass(frozen=True)
class ReviewedRequestProfile:
    model_alias: str
    image_detail: ImageDetail
    limits: RequestLimits


@dataclass(frozen=True)
class PreparedRequest:
    """Validated UTF-8 request bytes plus immutable safe semantic metadata."""

    body: bytes
    model_alias: str
    image_detail: ImageDetail
    max_output_tokens: int

    def payload(self) -> dict[str, object]:
        """Return fresh ordinary JSON containers for offline semantic checks."""

        return json.loads(self.body.decode("utf-8"))


@dataclass(frozen=True)
class NormalizedUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float
    exclude_vision: bool
    reason: str


@dataclass(frozen=True)
class SanitizedTransportResult:
    outcome: TransportOutcome
    classification: Optional[ClassificationResult]
    output_text: Optional[str]
    usage: Optional[NormalizedUsage]


_MAX_PROMPT_BYTES = 4_096
_MAX_IMAGE_BYTES = 1_024
_MAX_DATA_URL_BYTES = 4_096
_MAX_REQUEST_BYTES = 8_192
_MAX_RESPONSE_BYTES = 8_192
_MAX_OUTPUT_TOKENS = 128
_MAX_USAGE_TOKENS = 1_000_000
_TIMEOUT_SECONDS = 5.0
_MAX_CONCURRENCY = 1
_MAX_REASON_CHARS = 1_000

SYNTHETIC_PROMPT = "SYNTHETIC ICVision classification prompt."
SYNTHETIC_WEBP = b"RIFF\x04\x00\x00\x00WEBP"
_SUPPRESSED_REASON = "suppressed"
_LABELS = tuple(COMPONENT_LABELS)

SYNTHETIC_REQUEST_PROFILE = ReviewedRequestProfile(
    model_alias="synthetic-review-only",
    image_detail=ImageDetail.LOW,
    limits=RequestLimits(
        max_prompt_bytes=_MAX_PROMPT_BYTES,
        max_image_bytes=_MAX_IMAGE_BYTES,
        max_data_url_bytes=_MAX_DATA_URL_BYTES,
        max_request_bytes=_MAX_REQUEST_BYTES,
        max_response_bytes=_MAX_RESPONSE_BYTES,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        timeout_seconds=_TIMEOUT_SECONDS,
        max_concurrency=_MAX_CONCURRENCY,
    ),
)


def _schema_payload() -> dict[str, object]:
    return {
        "type": "json_schema",
        "name": "icvision_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "confidence", "exclude_vision", "reason"],
            "properties": {
                "label": {"type": "string", "enum": list(_LABELS)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "exclude_vision": {"type": "boolean"},
                "reason": {"type": "string", "maxLength": _MAX_REASON_CHARS},
            },
        },
    }


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


CLASSIFICATION_JSON_SCHEMA = _freeze(_schema_payload())


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)


def _valid_usage_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_USAGE_TOKENS


def _validate_profile(profile: ReviewedRequestProfile) -> None:
    if profile is not SYNTHETIC_REQUEST_PROFILE:
        raise RequestContractError("Request profile is not the reviewed synthetic profile.")
    if profile.limits != SYNTHETIC_REQUEST_PROFILE.limits:
        raise RequestContractError("Reviewed synthetic profile is invalid.")


def _validate_synthetic_inputs(
    prompt: str | bytes,
    webp_bytes: bytes,
    profile: ReviewedRequestProfile,
) -> None:
    _validate_profile(profile)

    if isinstance(prompt, str):
        if prompt != SYNTHETIC_PROMPT:
            raise RequestContractError("Synthetic prompt fixture is not approved.")
        try:
            prompt_bytes = prompt.encode("utf-8")
        except UnicodeEncodeError:
            raise RequestContractError("Synthetic prompt is invalid.") from None
    elif isinstance(prompt, bytes):
        if len(prompt) > _MAX_PROMPT_BYTES:
            raise RequestContractError("Synthetic prompt exceeds a fixed limit.")
        try:
            prompt.decode("utf-8")
        except UnicodeDecodeError:
            raise RequestContractError("Synthetic prompt is invalid.") from None
        prompt_bytes = prompt
    else:
        raise RequestContractError("Synthetic prompt is invalid.")

    if len(prompt_bytes) > _MAX_PROMPT_BYTES:
        raise RequestContractError("Synthetic prompt exceeds a fixed limit.")
    if prompt_bytes != SYNTHETIC_PROMPT.encode("utf-8"):
        raise RequestContractError("Synthetic prompt fixture is not approved.")

    if not isinstance(webp_bytes, bytes):
        raise RequestContractError("Synthetic WebP fixture is not approved.")
    if len(webp_bytes) > _MAX_IMAGE_BYTES:
        raise RequestContractError("Synthetic WebP exceeds a fixed limit.")
    if webp_bytes != SYNTHETIC_WEBP:
        raise RequestContractError("Synthetic WebP fixture is not approved.")


def build_synthetic_responses_request(
    prompt: str | bytes,
    webp_bytes: bytes,
    profile: ReviewedRequestProfile = SYNTHETIC_REQUEST_PROFILE,
) -> PreparedRequest:
    """Build fixed, bounded UTF-8 request bytes without transport behavior."""

    _validate_synthetic_inputs(prompt, webp_bytes, profile)
    data_url = "data:image/webp;base64," + base64.b64encode(webp_bytes).decode("ascii")

    if len(data_url.encode("ascii")) > _MAX_DATA_URL_BYTES:
        raise RequestContractError("Synthetic image data URL exceeds a fixed limit.")

    request: dict[str, object] = {
        "model": SYNTHETIC_REQUEST_PROFILE.model_alias,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SYNTHETIC_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": ImageDetail.LOW.value,
                    },
                ],
            }
        ],
        "store": False,
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
        "text": {"format": _schema_payload()},
    }

    try:
        body = json.dumps(request, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError):
        raise RequestContractError("Synthetic request is not serializable.") from None

    if len(body) > _MAX_REQUEST_BYTES:
        raise RequestContractError("Synthetic request exceeds a fixed limit.")

    return PreparedRequest(
        body=body,
        model_alias=SYNTHETIC_REQUEST_PROFILE.model_alias,
        image_detail=ImageDetail.LOW,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )


def normalize_usage(value: object) -> Optional[NormalizedUsage]:
    """Retain only bounded counters from the Responses usage shape."""

    if not isinstance(value, Mapping):
        return None

    details = value.get("input_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, Mapping) else None
    normalized = {
        "input_tokens": value.get("input_tokens"),
        "output_tokens": value.get("output_tokens"),
        "cached_tokens": cached_tokens,
    }
    normalized = {name: count if _valid_usage_count(count) else None for name, count in normalized.items()}
    if all(count is None for count in normalized.values()):
        return None
    return NormalizedUsage(**normalized)


def _validate_reason(value: object) -> None:
    if not isinstance(value, str) or len(value) > _MAX_REASON_CHARS:
        raise RequestContractError("Classification result is invalid.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise RequestContractError("Classification result is invalid.") from None
    if _has_control_character(value):
        raise RequestContractError("Classification result is invalid.")


def _validate_classification(value: object) -> ClassificationResult:
    if not isinstance(value, Mapping):
        raise RequestContractError("Classification result is invalid.")
    if set(value) != {"label", "confidence", "exclude_vision", "reason"}:
        raise RequestContractError("Classification result is invalid.")

    label = value["label"]
    confidence = value["confidence"]
    exclude_vision = value["exclude_vision"]

    if not isinstance(label, str) or label not in _LABELS or _has_control_character(label):
        raise RequestContractError("Classification result is invalid.")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or (isinstance(confidence, float) and not math.isfinite(confidence))
        or not 0 <= confidence <= 1
    ):
        raise RequestContractError("Classification result is invalid.")
    if not isinstance(exclude_vision, bool):
        raise RequestContractError("Classification result is invalid.")

    _validate_reason(value["reason"])
    return ClassificationResult(
        label=label,
        confidence=float(confidence),
        exclude_vision=exclude_vision,
        reason=_SUPPRESSED_REASON,
    )


def normalize_synthetic_result(
    outcome: TransportOutcome,
    response_bytes: object,
    classification: object = None,
    usage: object = None,
) -> SanitizedTransportResult:
    """Normalize bounded structured results without retaining provider material."""

    if not isinstance(outcome, TransportOutcome):
        raise RequestContractError("Transport outcome is invalid.")
    if (
        not isinstance(response_bytes, int)
        or isinstance(response_bytes, bool)
        or not 0 <= response_bytes <= _MAX_RESPONSE_BYTES
    ):
        raise RequestContractError("Response size is invalid.")

    normalized_usage = normalize_usage(usage)
    if outcome is not TransportOutcome.COMPLETED:
        return SanitizedTransportResult(outcome, None, None, normalized_usage)

    normalized_classification = _validate_classification(classification)
    output_text = json.dumps(
        {
            "label": normalized_classification.label,
            "confidence": normalized_classification.confidence,
            "exclude_vision": normalized_classification.exclude_vision,
            "reason": _SUPPRESSED_REASON,
        },
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )
    if len(output_text.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise RequestContractError("Response exceeds a fixed limit.")

    return SanitizedTransportResult(
        outcome=TransportOutcome.COMPLETED,
        classification=normalized_classification,
        output_text=output_text,
        usage=normalized_usage,
    )
