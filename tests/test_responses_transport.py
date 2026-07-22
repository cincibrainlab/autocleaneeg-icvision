"""Offline tests for the fixed standard-library Responses transport."""

import json
import os
import socket
import ssl
import time

import pytest

import icvision.responses_transport as transport


SECRET_MARKER = "synthetic-secret-marker"


def _denied(*_args, **_kwargs):
    raise AssertionError("offline transport test attempted a forbidden side effect")


@pytest.fixture(autouse=True)
def deny_side_effects(monkeypatch):
    monkeypatch.setattr(socket, "socket", _denied)
    monkeypatch.setattr(socket, "create_connection", _denied)
    monkeypatch.setattr(socket, "getaddrinfo", _denied)
    monkeypatch.setattr(time, "sleep", _denied)
    monkeypatch.setattr(os, "getenv", _denied)


def _request(body=None):
    if body is None:
        body = json.dumps({"model": "synthetic", "store": False}).encode("utf-8")
    return transport.prepare_responses_request(body)


def _response_payload():
    return {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "synthetic text"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 1}},
    }


class _FakeResponse:
    def __init__(self, body=None, status=200, content_type="application/json", content_length=None):
        self.body = json.dumps(_response_payload()).encode("utf-8") if body is None else body
        self.status = status
        self.content_type = content_type
        self.content_length = content_length
        self.read_sizes = []
        self.closed = False

    def getheader(self, name):
        if name == "Content-Type":
            return self.content_type
        if name == "Content-Length" and self.content_length is not None:
            return self.content_length
        return None

    def read(self, size):
        self.read_sizes.append(size)
        return self.body

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, *args, response, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.response = response
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def _install_connection(monkeypatch, response):
    created = []

    def factory(*args, **kwargs):
        connection = _FakeConnection(*args, response=response, **kwargs)
        created.append(connection)
        return connection

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", factory)
    return created


def test_module_does_not_import_sdk_or_environment_facilities():
    assert "openai" not in transport.__dict__
    assert "os" not in transport.__dict__


def test_closed_profile_rejects_unknown_and_url_values_before_auth_or_connection(monkeypatch):
    created = _install_connection(monkeypatch, _FakeResponse())

    for profile in (None, "unknown", "https://ai.clincognition.com/v1"):
        result = transport.send_responses_request(_request(), "bad\nauthorization", profile)

        assert result.outcome is transport.ResponsesOutcome.INVALID_PROFILE
    assert created == []


@pytest.mark.parametrize("authorization", [None, "", "bad authorization", "bad\nauthorization"])
def test_invalid_authorization_fails_before_connection(monkeypatch, authorization):
    created = _install_connection(monkeypatch, _FakeResponse())

    result = transport.send_responses_request(_request(), authorization)

    assert result.outcome is transport.ResponsesOutcome.INVALID_AUTHORIZATION
    assert created == []


def test_fixed_profile_ignores_ambient_proxy_values(monkeypatch):
    response = _FakeResponse()
    created = _install_connection(monkeypatch, response)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:8080")

    result = transport.send_responses_request(_request(), SECRET_MARKER, "clincog")

    assert result.outcome is transport.ResponsesOutcome.SUCCESS
    assert created[0].args[:2] == ("ai.clincognition.com", 443)
    assert created[0].requests[0][1] == "/v1/responses"


def test_fixed_post_tls_headers_single_send_and_normalized_success(monkeypatch):
    response = _FakeResponse(content_type="Application/JSON; charset=utf-8")
    created = _install_connection(monkeypatch, response)

    result = transport.send_responses_request(_request(), SECRET_MARKER)

    assert result.outcome is transport.ResponsesOutcome.SUCCESS
    assert result.output_text == "synthetic text"
    assert result.usage == transport.NormalizedUsage(3, 2, 1)
    assert len(created) == 1
    connection = created[0]
    assert connection.args[:2] == ("ai.clincognition.com", 443)
    assert connection.kwargs["timeout"] == 5.0
    context = connection.kwargs["context"]
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert len(connection.requests) == 1
    method, path, body, headers = connection.requests[0]
    assert (method, path, body) == ("POST", "/v1/responses", _request().body)
    assert headers == {
        "Accept": "application/json",
        "Authorization": "Bearer " + SECRET_MARKER,
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    assert response.read_sizes == [8193]
    assert response.closed and connection.closed
    assert SECRET_MARKER not in repr(result)


@pytest.mark.parametrize("status", [201, 204, 301, 401])
def test_non_200_status_is_sanitized_and_closed(monkeypatch, status):
    response = _FakeResponse(status=status)
    created = _install_connection(monkeypatch, response)

    result = transport.send_responses_request(_request(), SECRET_MARKER)

    assert result == transport.ResponsesResult(transport.ResponsesOutcome.HTTP_STATUS, status)
    assert response.read_sizes == []
    assert response.closed and created[0].closed
    assert SECRET_MARKER not in repr(result)


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        (_FakeResponse(content_type="text/plain"), transport.ResponsesOutcome.INVALID_CONTENT_TYPE),
        (_FakeResponse(content_length="bad"), transport.ResponsesOutcome.RESPONSE_TOO_LARGE),
        (_FakeResponse(content_length="-1"), transport.ResponsesOutcome.RESPONSE_TOO_LARGE),
        (_FakeResponse(content_length="8193"), transport.ResponsesOutcome.RESPONSE_TOO_LARGE),
        (_FakeResponse(body=b"x" * 8193), transport.ResponsesOutcome.RESPONSE_TOO_LARGE),
        (_FakeResponse(body=b"not-json"), transport.ResponsesOutcome.MALFORMED_RESPONSE),
    ],
)
def test_bounded_response_failures_are_sanitized_and_closed(monkeypatch, response, outcome):
    created = _install_connection(monkeypatch, response)

    result = transport.send_responses_request(_request(), SECRET_MARKER)

    assert result.outcome is outcome
    assert result.output_text is None and result.usage is None
    if response.content_length is not None:
        assert response.read_sizes == []
    assert response.closed and created[0].closed
    assert SECRET_MARKER not in repr(result)


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (TimeoutError(SECRET_MARKER), transport.ResponsesOutcome.TIMEOUT),
        (ssl.SSLError(SECRET_MARKER), transport.ResponsesOutcome.TLS_FAILURE),
        (OSError(SECRET_MARKER), transport.ResponsesOutcome.CONNECTION_FAILURE),
        (transport.http.client.HTTPException(SECRET_MARKER), transport.ResponsesOutcome.PROTOCOL_FAILURE),
    ],
)
def test_transport_exceptions_have_distinct_sanitized_outcomes(monkeypatch, error, outcome):
    response = _FakeResponse()
    created = _install_connection(monkeypatch, response)

    class BrokenConnection(_FakeConnection):
        def request(self, *_args, **_kwargs):
            raise error

    def factory(*args, **kwargs):
        connection = BrokenConnection(*args, response=response, **kwargs)
        created.append(connection)
        return connection

    monkeypatch.setattr(transport.http.client, "HTTPSConnection", factory)
    result = transport.send_responses_request(_request(), SECRET_MARKER)

    assert result.outcome is outcome
    assert SECRET_MARKER not in repr(result)
    assert created[-1].closed is True


@pytest.mark.parametrize(
    "prepared_request",
    [
        object(),
        transport.PreparedResponsesRequest(b"{}"),
        transport.PreparedResponsesRequest(b'{"store":true}'),
        transport.PreparedResponsesRequest(b"x" * (transport._MAX_REQUEST_BYTES + 1)),
    ],
)
def test_invalid_or_forged_requests_fail_before_connection(monkeypatch, prepared_request):
    created = _install_connection(monkeypatch, _FakeResponse())

    result = transport.send_responses_request(prepared_request, SECRET_MARKER)

    assert result.outcome is transport.ResponsesOutcome.INVALID_REQUEST
    assert created == []
