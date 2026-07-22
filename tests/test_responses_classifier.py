"""Synthetic, no-network tests for the raw Responses classifier."""

import json
import socket
from pathlib import Path

import pytest

from icvision import responses_classifier
from icvision.responses_transport import NormalizedUsage, ResponsesOutcome, ResponsesResult


def _webp() -> bytes:
    return b"RIFF\x04\x00\x00\x00WEBPVP8 "


@pytest.fixture(autouse=True)
def deny_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: pytest.fail("socket forbidden"))
    monkeypatch.setattr(socket, "create_connection", lambda *_args, **_kwargs: pytest.fail("socket forbidden"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: pytest.fail("socket forbidden"))


def test_success_uses_governed_model_fixed_prompt_and_ordered_input(tmp_path, monkeypatch):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    captured = {}

    def fake_send(request):
        captured["payload"] = json.loads(request.body)
        return ResponsesResult(
            outcome=ResponsesOutcome.SUCCESS,
            output_text='{"label":"brain","confidence":0.9,"reason":"Synthetic fixture."}',
            usage=NormalizedUsage(3, 2, 1),
        )

    monkeypatch.setattr(responses_classifier, "send_runtime_responses_request", fake_send)
    result = responses_classifier.classify_image_with_responses(image_path)

    assert result.label == "brain"
    assert result.confidence == 0.9
    assert result.outcome_status == "classified"
    assert result.review_required is True
    assert result.apply_to_ica is False
    assert result.exclude_vision is False
    assert result.model == "gpt-5.4"
    assert result.elapsed_seconds >= 0.0
    assert result.usage == NormalizedUsage(3, 2, 1)
    assert result.prompt_sha256 is not None and len(result.prompt_sha256) == 64
    assert result.artifact_inventory == ("temporary_component_webp",)
    payload = captured["payload"]
    assert payload["model"] == "gpt-5.4"
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 1024
    assert payload["text"]["format"] == {
        "type": "json_schema",
        "name": "ica_component_classification",
        "strict": True,
        "schema": responses_classifier._classification_schema(),
    }
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": responses_classifier.load_raw_responses_prompt()}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/webp;base64,")
    assert content[1]["detail"] == "low"


@pytest.mark.parametrize(
    "output_text",
    [
        '{"label":"brain","confidence":0.9}',
        '{"label":"brain","confidence":true,"reason":"Synthetic fixture."}',
        '{"label":"not-a-label","confidence":0.9,"reason":"Synthetic fixture."}',
        '{"label":"brain","confidence":1.1,"reason":"Synthetic fixture."}',
        '{"label":"brain","confidence":NaN,"reason":"Synthetic fixture."}',
        '{"label":"brain","confidence":0.9,"reason":"Synthetic\\nfixture."}',
        '{"label":"brain","confidence":0.9,"reason":"Synthetic fixture.","extra":1}',
    ],
)
def test_invalid_structured_output_is_review_only_unavailable(tmp_path, monkeypatch, output_text):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    monkeypatch.setattr(
        responses_classifier,
        "send_runtime_responses_request",
        lambda request: ResponsesResult(outcome=ResponsesOutcome.SUCCESS, output_text=output_text),
    )
    result = responses_classifier.classify_image_with_responses(image_path)
    assert result.outcome_status == "unavailable"
    assert result.label is None
    assert result.confidence is None
    assert result.review_required is True
    assert result.apply_to_ica is False
    assert result.exclude_vision is False
    assert result.failure_category == ResponsesOutcome.MALFORMED_RESPONSE.value
    assert result.model == "gpt-5.4"


def test_transport_failure_is_review_only_unavailable(tmp_path, monkeypatch):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    monkeypatch.setattr(
        responses_classifier,
        "send_runtime_responses_request",
        lambda request: ResponsesResult(outcome=ResponsesOutcome.TIMEOUT),
    )
    result = responses_classifier.classify_image_with_responses(image_path)
    assert result.outcome_status == "unavailable"
    assert result.failure_category == ResponsesOutcome.TIMEOUT.value
    assert result.reason == "Classification unavailable; human review required."


def test_oversized_image_is_not_read_or_sent(tmp_path, monkeypatch):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    monkeypatch.setattr(Path, "stat", lambda self: type("Stat", (), {"st_size": 5 * 1024 * 1024 + 1})())
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("must not read oversized image"))
    monkeypatch.setattr(responses_classifier, "send_runtime_responses_request", lambda request: pytest.fail("must not send oversized image"))
    result = responses_classifier.classify_image_with_responses(image_path)
    assert result.outcome_status == "unavailable"
    assert result.failure_category == ResponsesOutcome.INVALID_REQUEST.value


@pytest.mark.parametrize("model", ["gpt-5.4"])
def test_reviewed_model_override_is_serialized_and_retained(tmp_path, monkeypatch, model):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    monkeypatch.setenv("ICVISION_RESPONSES_MODEL", model)
    captured = {}

    def fake_send(request):
        captured["payload"] = json.loads(request.body)
        return ResponsesResult(
            outcome=ResponsesOutcome.SUCCESS,
            output_text='{"label":"brain","confidence":0.9,"reason":"Synthetic fixture."}',
            usage=NormalizedUsage(3, 2, 1),
        )

    monkeypatch.setattr(responses_classifier, "send_runtime_responses_request", fake_send)
    result = responses_classifier.classify_image_with_responses(image_path)

    assert captured["payload"]["model"] == model
    assert result.model == model


@pytest.mark.parametrize("model", ["unknown", "https://example.invalid", " gpt-5.6-terra", ""])
def test_invalid_model_override_stops_before_image_access_or_transport(monkeypatch, model):
    monkeypatch.setenv("ICVISION_RESPONSES_MODEL", model)
    monkeypatch.setattr(Path, "stat", lambda self: pytest.fail("must not stat image"))
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("must not read image"))
    monkeypatch.setattr(
        responses_classifier,
        "send_runtime_responses_request",
        lambda request: pytest.fail("must not send request"),
    )

    result = responses_classifier.classify_image_with_responses(Path("synthetic.webp"))

    assert result.outcome_status == "unavailable"
    assert result.failure_category == ResponsesOutcome.INVALID_REQUEST.value
    assert result.model is None

def test_non_webp_image_is_review_only_and_never_sent(tmp_path, monkeypatch):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(b"not-a-webp")
    monkeypatch.setattr(
        responses_classifier,
        "send_runtime_responses_request",
        lambda request: pytest.fail("must not send invalid image"),
    )

    result = responses_classifier.classify_image_with_responses(image_path)

    assert result.outcome_status == "unavailable"
    assert result.failure_category == ResponsesOutcome.INVALID_REQUEST.value
    assert result.model == "gpt-5.4"
    assert result.artifact_inventory == ()


def test_missing_runtime_credential_is_review_only_without_socket(tmp_path, monkeypatch):
    image_path = tmp_path / "component.webp"
    image_path.write_bytes(_webp())
    monkeypatch.delenv("CLINCOG_API_KEY", raising=False)

    result = responses_classifier.classify_image_with_responses(image_path)

    assert result.outcome_status == "unavailable"
    assert result.failure_category == ResponsesOutcome.INVALID_AUTHORIZATION.value
    assert result.model == "gpt-5.4"
    assert result.prompt_sha256 is not None
    assert result.artifact_inventory == ("temporary_component_webp",)
