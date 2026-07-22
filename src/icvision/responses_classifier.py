"""Review-only ICA image classification through the governed Responses lane."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .config import COMPONENT_LABELS
from .responses_runtime import send_runtime_responses_request
from .responses_transport import NormalizedUsage, ResponsesOutcome, prepare_responses_request


_DEFAULT_GOVERNED_MODEL = "gpt-5.6-terra"
_ALLOWED_GOVERNED_MODELS = frozenset({"gpt-5.6-terra", "gpt-5.4-mini", "gpt-5.5"})
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_REASON_CHARS = 1_000
_MAX_OUTPUT_TOKENS = 1_024
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "default.txt"
_UNAVAILABLE_REASON = "Classification unavailable; human review required."
_TEMPORARY_IMAGE_ARTIFACT = "temporary_component_webp"


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
    elapsed_seconds: float = 0.0
    usage: Optional[NormalizedUsage] = None
    prompt_sha256: Optional[str] = None
    artifact_inventory: Tuple[str, ...] = ()


def _unavailable(
    category: str,
    model: Optional[str] = None,
    *,
    elapsed_seconds: float = 0.0,
    usage: Optional[NormalizedUsage] = None,
    prompt_sha256: Optional[str] = None,
    artifact_inventory: Tuple[str, ...] = (),
) -> RawClassification:
    return RawClassification(
        None,
        None,
        _UNAVAILABLE_REASON,
        "unavailable",
        category,
        model=model,
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        prompt_sha256=prompt_sha256,
        artifact_inventory=artifact_inventory,
    )


def resolve_governed_model() -> Optional[str]:
    """Use the default or an exact reviewed runtime-only override."""

    override = os.environ.get("ICVISION_RESPONSES_MODEL")
    if override is None:
        return _DEFAULT_GOVERNED_MODEL
    return override if override in _ALLOWED_GOVERNED_MODELS else None


def load_raw_responses_prompt() -> str:
    """Load only the checked-in prompt, never cwd or home-directory overrides."""

    prompt = _PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("The governed Responses prompt is empty.")
    return prompt


def _webp_data_url(image_path: Path) -> str:
    size = image_path.stat().st_size
    if not 0 < size <= _MAX_IMAGE_BYTES:
        raise ValueError("Image size is outside the governed Responses limit.")
    image = image_path.read_bytes()
    if len(image) != size or len(image) > _MAX_IMAGE_BYTES:
        raise ValueError("Image size changed while it was being read.")
    if len(image) < 12 or image[:4] != b"RIFF" or image[8:12] != b"WEBP":
        raise ValueError("Only WebP component images are supported.")
    return "data:image/webp;base64," + base64.b64encode(image).decode("ascii")


def _classification_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": sorted(COMPONENT_LABELS)},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
        "additionalProperties": False,
    }


def _request_body(prompt: str, image_url: str, model: str) -> bytes:
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "low",
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ica_component_classification",
                "strict": True,
                "schema": _classification_schema(),
            }
        },
        "max_output_tokens": _MAX_OUTPUT_TOKENS,
        "store": False,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _parse_classification(
    output_text: object,
    model: str,
    *,
    elapsed_seconds: float,
    usage: Optional[NormalizedUsage],
    prompt_sha256: str,
    artifact_inventory: Tuple[str, ...],
) -> RawClassification:
    try:
        payload = json.loads(output_text)
    except (TypeError, json.JSONDecodeError):
        return _unavailable(
            ResponsesOutcome.MALFORMED_RESPONSE.value,
            model,
            elapsed_seconds=elapsed_seconds,
            usage=usage,
            prompt_sha256=prompt_sha256,
            artifact_inventory=artifact_inventory,
        )
    if not isinstance(payload, dict) or set(payload) != {"label", "confidence", "reason"}:
        return _unavailable(
            ResponsesOutcome.MALFORMED_RESPONSE.value,
            model,
            elapsed_seconds=elapsed_seconds,
            usage=usage,
            prompt_sha256=prompt_sha256,
            artifact_inventory=artifact_inventory,
        )
    label, confidence, reason = payload["label"], payload["confidence"], payload["reason"]
    if (
        label not in COMPONENT_LABELS
        or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(confidence)
        or not 0.0 <= float(confidence) <= 1.0
        or not isinstance(reason, str)
        or not 0 < len(reason) <= _MAX_REASON_CHARS
        or not all(" " <= character <= "~" for character in reason)
    ):
        return _unavailable(
            ResponsesOutcome.MALFORMED_RESPONSE.value,
            model,
            elapsed_seconds=elapsed_seconds,
            usage=usage,
            prompt_sha256=prompt_sha256,
            artifact_inventory=artifact_inventory,
        )
    return RawClassification(
        label,
        float(confidence),
        reason,
        "classified",
        None,
        model=model,
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        prompt_sha256=prompt_sha256,
        artifact_inventory=artifact_inventory,
    )


def classify_image_with_responses(image_path: Path) -> RawClassification:
    """Send one ordered image request and return a review-only result."""

    started = time.monotonic()
    model = resolve_governed_model()
    if model is None:
        return _unavailable(
            ResponsesOutcome.INVALID_REQUEST.value,
            elapsed_seconds=time.monotonic() - started,
        )

    prompt_sha256: Optional[str] = None
    artifacts: Tuple[str, ...] = ()
    try:
        prompt = load_raw_responses_prompt()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        image_url = _webp_data_url(image_path)
        artifacts = (_TEMPORARY_IMAGE_ARTIFACT,)
        request = prepare_responses_request(_request_body(prompt, image_url, model))
    except (OSError, ValueError):
        return _unavailable(
            ResponsesOutcome.INVALID_REQUEST.value,
            model,
            elapsed_seconds=time.monotonic() - started,
            prompt_sha256=prompt_sha256,
            artifact_inventory=artifacts,
        )

    result = send_runtime_responses_request(request)
    elapsed_seconds = time.monotonic() - started
    if result.outcome is not ResponsesOutcome.SUCCESS:
        return _unavailable(
            result.outcome.value,
            model,
            elapsed_seconds=elapsed_seconds,
            usage=result.usage,
            prompt_sha256=prompt_sha256,
            artifact_inventory=artifacts,
        )
    return _parse_classification(
        result.output_text,
        model,
        elapsed_seconds=elapsed_seconds,
        usage=result.usage,
        prompt_sha256=prompt_sha256,
        artifact_inventory=artifacts,
    )
