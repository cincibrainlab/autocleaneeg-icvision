"""Offline tests for the immutable synthetic Responses request contract."""

from __future__ import annotations

import base64
import builtins
import inspect
import json
import logging
import math
import os
import socket
import time
from dataclasses import fields, replace
from pathlib import Path

import pytest

import icvision.request_contracts as contracts
from icvision import api
from icvision.request_contracts import (
    CLASSIFICATION_JSON_SCHEMA,
    SYNTHETIC_PROMPT,
    SYNTHETIC_REQUEST_PROFILE,
    SYNTHETIC_WEBP,
    ClassificationResult,
    ImageDetail,
    NormalizedUsage,
    PreparedRequest,
    RequestContractError,
    RequestLimits,
    ReviewedRequestProfile,
    SanitizedTransportResult,
    TransportOutcome,
    build_synthetic_responses_request,
    normalize_synthetic_result,
    normalize_usage,
)


SECRET_MARKER = "synthetic-secret-marker-never-echo"
PROTECTED_MARKER = "synthetic-protected-marker-never-echo"


def _denied(*_args, **_kwargs):
    raise AssertionError("offline request-contract test attempted a forbidden operation")


@pytest.fixture(autouse=True)
def deny_side_effects(monkeypatch, request):
    monkeypatch.setattr(socket.socket, "connect", _denied)
    monkeypatch.setattr(socket.socket, "connect_ex", _denied)
    monkeypatch.setattr(socket, "socket", _denied)
    monkeypatch.setattr(socket, "create_connection", _denied)
    monkeypatch.setattr(socket, "getaddrinfo", _denied)
    monkeypatch.setattr(socket, "bind", _denied, raising=False)
    monkeypatch.setattr(socket, "listen", _denied, raising=False)
    monkeypatch.setattr(time, "sleep", _denied)
    if request.node.name != "test_sdk_serialization_characterization_with_mock_transport":
        monkeypatch.setattr(os, "getenv", _denied)
        monkeypatch.setattr(builtins, "open", _denied)
        monkeypatch.setattr(Path, "open", _denied)
        monkeypatch.setattr(Path, "read_bytes", _denied)
        monkeypatch.setattr(Path, "read_text", _denied)
        monkeypatch.setattr(logging.Logger, "_log", _denied)


def test_contract_module_imports_no_side_effect_facility():
    forbidden = {"os", "socket", "logging", "pathlib", "requests", "httpx", "openai"}
    assert forbidden.isdisjoint(vars(contracts))


def _prepared() -> PreparedRequest:
    return build_synthetic_responses_request(SYNTHETIC_PROMPT, SYNTHETIC_WEBP)


def _payload() -> dict[str, object]:
    return _prepared().payload()


def _classification(reason: str = "provider supplied reason") -> dict[str, object]:
    label = CLASSIFICATION_JSON_SCHEMA["schema"]["properties"]["label"]["enum"][0]
    return {
        "label": label,
        "confidence": 0.75,
        "exclude_vision": False,
        "reason": reason,
    }


def test_fixed_fixture_profile_and_literal_limits():
    assert SYNTHETIC_REQUEST_PROFILE.model_alias == "synthetic-review-only"
    assert SYNTHETIC_REQUEST_PROFILE.image_detail is ImageDetail.LOW
    assert SYNTHETIC_REQUEST_PROFILE.limits == RequestLimits(
        max_prompt_bytes=4096,
        max_image_bytes=1024,
        max_data_url_bytes=4096,
        max_request_bytes=8192,
        max_response_bytes=8192,
        max_output_tokens=128,
        timeout_seconds=5.0,
        max_concurrency=1,
    )
    assert SYNTHETIC_PROMPT == "SYNTHETIC ICVision classification prompt."
    assert SYNTHETIC_WEBP == b"RIFF\x04\x00\x00\x00WEBP"


def test_prepared_request_has_final_json_bytes_and_fresh_payloads():
    prepared = _prepared()

    assert isinstance(prepared, PreparedRequest)
    assert prepared.body == json.dumps(_payload(), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    assert prepared.model_alias == "synthetic-review-only"
    assert prepared.image_detail is ImageDetail.LOW
    assert prepared.max_output_tokens == 128

    first = prepared.payload()
    first["model"] = "mutated"
    first["text"]["format"]["schema"]["properties"]["label"]["enum"].append("mutated")

    second = prepared.payload()
    assert second["model"] == "synthetic-review-only"
    assert "mutated" not in second["text"]["format"]["schema"]["properties"]["label"]["enum"]

    with pytest.raises((AttributeError, TypeError)):
        prepared.body = b"mutated"  # type: ignore[misc]


def test_canonical_schema_is_recursively_immutable_and_not_shared_with_requests():
    with pytest.raises(TypeError):
        CLASSIFICATION_JSON_SCHEMA["type"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        CLASSIFICATION_JSON_SCHEMA["schema"]["properties"]["label"]["type"] = "number"  # type: ignore[index]

    payload = _payload()
    payload["text"]["format"]["schema"]["required"].append("changed")
    assert "changed" not in _payload()["text"]["format"]["schema"]["required"]


def test_exact_official_responses_shape_and_body_semantics():
    payload = _payload()
    expected_url = "data:image/webp;base64," + base64.b64encode(SYNTHETIC_WEBP).decode("ascii")

    assert set(payload) == {"model", "input", "store", "max_output_tokens", "text"}
    assert payload["model"] == "synthetic-review-only"
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 128
    assert payload["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": SYNTHETIC_PROMPT},
                {
                    "type": "input_image",
                    "image_url": expected_url,
                    "detail": "low",
                },
            ],
        }
    ]

    text_format = payload["text"]["format"]
    assert set(text_format) == {"type", "name", "strict", "schema"}
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "icvision_classification"
    assert text_format["strict"] is True
    assert text_format["schema"]["additionalProperties"] is False
    assert text_format["schema"]["required"] == [
        "label",
        "confidence",
        "exclude_vision",
        "reason",
    ]

    forbidden = {
        "temperature",
        "retry",
        "retries",
        "reasoning",
        "tools",
        "tool_choice",
        "state",
        "stream",
        "background",
    }
    assert forbidden.isdisjoint(payload)


@pytest.mark.parametrize(
    "replacement",
    [
        replace(SYNTHETIC_REQUEST_PROFILE, model_alias="other"),
        replace(SYNTHETIC_REQUEST_PROFILE, image_detail=object()),  # type: ignore[arg-type]
        replace(
            SYNTHETIC_REQUEST_PROFILE,
            limits=replace(SYNTHETIC_REQUEST_PROFILE.limits, max_prompt_bytes=4097),
        ),
    ],
)
def test_counterfeit_profile_is_rejected(replacement):
    with pytest.raises(RequestContractError, match="reviewed synthetic profile"):
        build_synthetic_responses_request(SYNTHETIC_PROMPT, SYNTHETIC_WEBP, replacement)


@pytest.mark.parametrize(
    "field_name",
    [field.name for field in fields(RequestLimits)],
)
def test_every_one_field_limits_mutation_is_rejected(field_name):
    original = getattr(SYNTHETIC_REQUEST_PROFILE.limits, field_name)
    replacement_value = 6.0 if field_name == "timeout_seconds" else int(original) + 1
    counterfeit = replace(
        SYNTHETIC_REQUEST_PROFILE,
        limits=replace(
            SYNTHETIC_REQUEST_PROFILE.limits,
            **{field_name: replacement_value},
        ),
    )

    with pytest.raises(RequestContractError, match="reviewed synthetic profile"):
        build_synthetic_responses_request(SYNTHETIC_PROMPT, SYNTHETIC_WEBP, counterfeit)


def test_caller_created_profile_with_huge_limits_or_alias_is_rejected():
    counterfeit = ReviewedRequestProfile(
        model_alias="x" * 100_000,
        image_detail=ImageDetail.LOW,
        limits=RequestLimits(
            max_prompt_bytes=10_000_000,
            max_image_bytes=10_000_000,
            max_data_url_bytes=10_000_000,
            max_request_bytes=10_000_000,
            max_response_bytes=10_000_000,
            max_output_tokens=10_000_000,
            timeout_seconds=86_400.0,
            max_concurrency=1,
        ),
    )

    with pytest.raises(RequestContractError, match="reviewed synthetic profile"):
        build_synthetic_responses_request(SYNTHETIC_PROMPT, SYNTHETIC_WEBP, counterfeit)


@pytest.mark.parametrize(
    "prompt",
    [
        "not the approved fixture",
        b"not the approved fixture",
        "SYNTHETIC \ud800",
        f"SYNTHETIC {SECRET_MARKER}",
    ],
)
def test_prompt_rejection_is_sanitized(prompt):
    with pytest.raises(RequestContractError) as captured:
        build_synthetic_responses_request(prompt, SYNTHETIC_WEBP)

    assert SECRET_MARKER not in str(captured.value)
    assert PROTECTED_MARKER not in str(captured.value)


def test_prompt_limit_precedes_base64_and_json(monkeypatch):
    monkeypatch.setattr(contracts, "SYNTHETIC_PROMPT", "SYNTHETIC " + ("x" * 5000))
    monkeypatch.setattr(base64, "b64encode", _denied)
    monkeypatch.setattr(json, "dumps", _denied)

    with pytest.raises(RequestContractError, match="prompt exceeds"):
        build_synthetic_responses_request(contracts.SYNTHETIC_PROMPT, SYNTHETIC_WEBP)


def test_image_limit_precedes_base64_and_json(monkeypatch):
    image = b"RIFF\x04\x00\x00\x00WEBP" + (b"x" * 2000)
    monkeypatch.setattr(contracts, "SYNTHETIC_WEBP", image)
    monkeypatch.setattr(base64, "b64encode", _denied)
    monkeypatch.setattr(json, "dumps", _denied)

    with pytest.raises(RequestContractError, match="WebP exceeds"):
        build_synthetic_responses_request(SYNTHETIC_PROMPT, image)


@pytest.mark.parametrize(
    ("constant_name", "expected"),
    [
        ("_MAX_DATA_URL_BYTES", "data URL"),
        ("_MAX_REQUEST_BYTES", "request"),
    ],
)
def test_encoded_and_request_bounds_are_enforced(monkeypatch, constant_name, expected):
    monkeypatch.setattr(contracts, constant_name, 1)

    with pytest.raises(RequestContractError, match=expected):
        _prepared()


@pytest.mark.parametrize(
    "reason",
    [
        "bad\x00reason",
        "bad\x7freason",
        "bad\u0085reason",
        "bad\ud800reason",
        "x" * 1001,
    ],
)
def test_classification_reason_controls_or_invalid_unicode_are_rejected(reason):
    with pytest.raises(RequestContractError, match="Classification result is invalid"):
        normalize_synthetic_result(
            TransportOutcome.COMPLETED,
            response_bytes=1,
            classification=_classification(reason),
        )


@pytest.mark.parametrize(
    "confidence",
    [
        True,
        float("nan"),
        float("inf"),
        float("-inf"),
        10**100_000,
        -0.01,
        1.01,
    ],
    ids=["bool", "nan", "pos_inf", "neg_inf", "huge_int", "below_zero", "above_one"],
)
def test_invalid_confidence_is_rejected_without_numeric_leakage(confidence):
    classification = _classification()
    classification["confidence"] = confidence

    with pytest.raises(RequestContractError, match="Classification result is invalid"):
        normalize_synthetic_result(
            TransportOutcome.COMPLETED,
            response_bytes=1,
            classification=classification,
        )


def test_completed_result_suppresses_provider_reason_and_has_allowlisted_fields():
    provider_reason = f"{SECRET_MARKER} {PROTECTED_MARKER}"
    result = normalize_synthetic_result(
        TransportOutcome.COMPLETED,
        response_bytes=64,
        classification=_classification(provider_reason),
        usage={
            "input_tokens": 5,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": 1},
            "provider_extra": provider_reason,
        },
    )

    assert [field.name for field in fields(SanitizedTransportResult)] == [
        "outcome",
        "classification",
        "output_text",
        "usage",
    ]
    assert result.classification == ClassificationResult(
        label=_classification()["label"],
        confidence=0.75,
        exclude_vision=False,
        reason="suppressed",
    )
    assert result.output_text == json.dumps(
        {
            "label": _classification()["label"],
            "confidence": 0.75,
            "exclude_vision": False,
            "reason": "suppressed",
        },
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    )
    assert result.usage == NormalizedUsage(5, 2, 1)

    rendered = repr(result)
    for marker in (
        SECRET_MARKER,
        PROTECTED_MARKER,
        "headers",
        "exception",
        "raw_body",
        "response_id",
    ):
        assert marker not in rendered


@pytest.mark.parametrize(
    "outcome",
    [
        TransportOutcome.REFUSAL,
        TransportOutcome.INCOMPLETE,
        TransportOutcome.INVALID,
        TransportOutcome.TRANSPORT_ERROR,
    ],
)
def test_noncompleted_outcomes_discard_classification_and_output(outcome):
    result = normalize_synthetic_result(
        outcome,
        response_bytes=1,
        classification=_classification(f"{SECRET_MARKER} {PROTECTED_MARKER}"),
    )

    assert result.classification is None
    assert result.output_text is None
    assert SECRET_MARKER not in repr(result)
    assert PROTECTED_MARKER not in repr(result)


@pytest.mark.parametrize(
    "response_bytes",
    [True, -1, 8193, "1", None],
)
def test_response_bytes_are_required_and_bounded(response_bytes):
    with pytest.raises(RequestContractError):
        normalize_synthetic_result(
            TransportOutcome.COMPLETED,
            response_bytes=response_bytes,
            classification=_classification(),
        )

    with pytest.raises(TypeError):
        normalize_synthetic_result(TransportOutcome.COMPLETED, classification=_classification())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True},
        {"output_tokens": -1},
        {"input_tokens_details": {"cached_tokens": 1_000_001}},
        {"input_tokens": 1.5},
        {"input_tokens_details": {"cached_tokens": False}},
    ],
)
def test_usage_accepts_only_bounded_allowlisted_integers(usage):
    assert normalize_usage(usage) is None


def test_sdk_serialization_characterization_with_mock_transport():
    import httpx
    import openai

    captured: list[dict[str, object]] = []

    def handler(request):
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = openai.OpenAI(
        api_key="synthetic-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    signature = inspect.signature(api._call_openai_api)
    values = {
        "client": client,
        "openai_client": client,
        "prompt": SYNTHETIC_PROMPT,
        "image_url": "data:image/webp;base64," + base64.b64encode(SYNTHETIC_WEBP).decode("ascii"),
        "image_data": "data:image/webp;base64," + base64.b64encode(SYNTHETIC_WEBP).decode("ascii"),
        "image": "data:image/webp;base64," + base64.b64encode(SYNTHETIC_WEBP).decode("ascii"),
        "model": SYNTHETIC_REQUEST_PROFILE.model_alias,
        "model_name": SYNTHETIC_REQUEST_PROFILE.model_alias,
        "base64_image": base64.b64encode(SYNTHETIC_WEBP).decode("ascii"),
        "temperature": 0.2,
    }
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            pytest.fail(f"current SDK helper has unsupported required parameter: {name}")

    api._call_openai_api(**kwargs)
    assert len(captured) == 1

    sdk_request = captured[0]
    assert sdk_request["temperature"] == 0.2
    sdk_input = sdk_request["input"]
    assert sdk_input[0] == {"role": "user", "content": SYNTHETIC_PROMPT}
    assert sdk_input[1]["role"] == "user"
    sdk_image = sdk_input[1]["content"][0]
    assert sdk_image["type"] == "input_image"
    assert sdk_image["image_url"].startswith("data:image/webp;base64,")

    raw_request = _payload()
    raw_content = raw_request["input"][0]["content"]
    sdk_image_url = sdk_image["image_url"]

    assert raw_request["model"] == sdk_request["model"]
    assert raw_content[0]["text"] == sdk_input[0]["content"]
    assert raw_content[1]["image_url"].split(";", 1)[0] == sdk_image_url.split(";", 1)[0]
    assert base64.b64decode(raw_content[1]["image_url"].split(",", 1)[1]) == base64.b64decode(
        sdk_image_url.split(",", 1)[1]
    )

    # Authorized non-wire-parity deltas: raw uses a combined typed content array,
    # store=false, bounded output, strict schema, low detail, no temperature,
    # and no reasoning pending later capability verification.
    assert raw_request["store"] is False
    assert raw_content[1]["detail"] == "low"
    assert raw_request["text"]["format"]["type"] == "json_schema"
    assert "temperature" not in raw_request
    assert "reasoning" not in raw_request
