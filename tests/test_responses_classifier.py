from pathlib import Path

import pytest

from icvision import responses_classifier as classifier
from icvision.clincog_client import GatewayOutcome, GatewayResult


def _webp():
    return b"RIFF\x04\x00\x00\x00WEBPVP8 "


def test_classifier_passes_webp_and_returns_fixed_gateway_response(tmp_path, monkeypatch):
    image = tmp_path / "component.webp"
    image.write_bytes(_webp())
    sent = {}
    monkeypatch.setattr(classifier, "resolve_gateway_token", lambda: "synthetic-token")
    monkeypatch.setattr(classifier, "send_classification_image", lambda body, token: sent.update(body=body, token=token) or GatewayResult(GatewayOutcome.SUCCESS, 200, {"label": "brain", "confidence": 0.9, "reason": "Synthetic.", "model": "gateway-model", "request_id": "synthetic-id"}))
    result = classifier.classify_image_with_clincog(image)
    assert sent == {"body": _webp(), "token": "synthetic-token"}
    assert (result.label, result.confidence, result.model, result.request_id) == ("brain", 0.9, "gateway-model", "synthetic-id")
    assert result.review_required and not result.apply_to_ica and not result.exclude_vision


def test_classifier_missing_token_and_bad_response_are_sanitized(tmp_path, monkeypatch):
    image = tmp_path / "component.webp"
    image.write_bytes(_webp())
    monkeypatch.setattr(classifier, "resolve_gateway_token", lambda: None)
    result = classifier.classify_image_with_clincog(image)
    assert result.outcome_status == "unavailable" and result.failure_category == GatewayOutcome.INVALID_AUTHORIZATION.value
    monkeypatch.setattr(classifier, "resolve_gateway_token", lambda: "synthetic-token")
    monkeypatch.setattr(classifier, "send_classification_image", lambda *_: GatewayResult(GatewayOutcome.CONNECTION_FAILURE))
    result = classifier.classify_image_with_clincog(image)
    assert result.outcome_status == "unavailable" and "SYNTHETIC_MARKER" not in result.reason


@pytest.mark.parametrize("field", ["model", "request_id"])
def test_control_character_in_gateway_identifier_is_sanitized(tmp_path, monkeypatch, field):
    image = tmp_path / "component.webp"
    image.write_bytes(_webp())
    payload = {"label": "brain", "confidence": 0.9, "reason": "Synthetic.", "model": "gateway-model", "request_id": "synthetic-id"}
    payload[field] = "SYNTHETIC_MARKER\\n"
    monkeypatch.setattr(classifier, "resolve_gateway_token", lambda: "synthetic-token")
    monkeypatch.setattr(classifier, "send_classification_image", lambda *_: GatewayResult(GatewayOutcome.SUCCESS, 200, payload))
    result = classifier.classify_image_with_clincog(image)
    assert result.outcome_status == "unavailable"
    assert result.failure_category == GatewayOutcome.MALFORMED_RESPONSE.value
    assert "SYNTHETIC_MARKER" not in result.reason
