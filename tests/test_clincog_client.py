import json
import socket
import ssl

import pytest

from icvision import clincog_client as client


def _webp():
    return b"RIFF\x04\x00\x00\x00WEBPVP8 "


class _Response:
    def __init__(self, status=200, content_type="application/json", content_length=None, body=None):
        self.status = status
        self.content_type = content_type
        self.content_length = content_length
        self.body = body or json.dumps({"label": "brain", "confidence": 0.9, "reason": "Synthetic.", "model": "gateway-model", "request_id": "synthetic-id"}).encode()
    def getheader(self, name):
        return {"Content-Type": self.content_type, "Content-Length": self.content_length}.get(name)
    def read(self, _limit):
        return self.body
    def close(self):
        pass


class _Connection:
    response = _Response()
    calls = []
    def __init__(self, host, port, timeout, context):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
    def request(self, method, path, body, headers):
        self.calls.append((self.host, self.port, method, path, body, headers, self.context))
    def getresponse(self):
        return self.response
    def close(self):
        pass


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("socket forbidden"))
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("socket forbidden"))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *_a, **_k: pytest.fail("dns forbidden"))
    _Connection.calls = []
    _Connection.response = _Response()
    monkeypatch.setattr(client.http.client, "HTTPSConnection", _Connection)


def test_fixed_endpoint_headers_and_raw_webp_body():
    result = client.send_classification_image(_webp(), "synthetic-token")
    assert result.outcome is client.GatewayOutcome.SUCCESS
    host, port, method, path, body, headers, context = _Connection.calls[0]
    assert (host, port, method, path, body) == ("ai.clincognition.com", 443, "POST", "/v1/ic-classifications", _webp())
    assert headers == {"Accept": "application/json", "Authorization": "Bearer synthetic-token", "Content-Length": str(len(_webp())), "Content-Type": "image/webp"}
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert context.verify_mode is ssl.CERT_REQUIRED and context.check_hostname


def test_token_resolution_uses_only_clincog_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-provider-token")
    monkeypatch.delenv("CLINCOG_API_TOKEN", raising=False)
    assert client.resolve_gateway_token() is None
    monkeypatch.setenv("CLINCOG_API_TOKEN", "synthetic-token")
    assert client.resolve_gateway_token() == "synthetic-token"


@pytest.mark.parametrize("response,outcome,status", [
    (_Response(status=201), client.GatewayOutcome.HTTP_STATUS, 201),
    (_Response(status=302), client.GatewayOutcome.HTTP_STATUS, 302),
    (_Response(content_type="application/problem+json"), client.GatewayOutcome.INVALID_CONTENT_TYPE, 200),
    (_Response(content_length="bad"), client.GatewayOutcome.RESPONSE_TOO_LARGE, 200),
    (_Response(content_length="-1"), client.GatewayOutcome.RESPONSE_TOO_LARGE, 200),
    (_Response(body=b"x" * 8193), client.GatewayOutcome.RESPONSE_TOO_LARGE, 200),
])
def test_failures_are_bounded_and_sanitized(response, outcome, status):
    _Connection.response = response
    result = client.send_classification_image(_webp(), "synthetic-token")
    assert result.outcome is outcome and result.status_code == status and result.classification is None


def test_marker_in_gateway_failure_never_leaks(monkeypatch):
    monkeypatch.setattr(_Connection, "getresponse", lambda _self: (_ for _ in ()).throw(OSError("SYNTHETIC_MARKER")))
    result = client.send_classification_image(_webp(), "synthetic-token")
    assert result.outcome is client.GatewayOutcome.CONNECTION_FAILURE
    assert "SYNTHETIC_MARKER" not in repr(result)


@pytest.mark.parametrize("status", ["SYNTHETIC_MARKER", True, 99, 600])
def test_invalid_status_is_not_retained(status):
    _Connection.response = _Response(status=status)
    result = client.send_classification_image(_webp(), "synthetic-token")
    assert result.outcome is client.GatewayOutcome.PROTOCOL_FAILURE
    assert result.status_code is None
    assert "SYNTHETIC_MARKER" not in repr(result)
