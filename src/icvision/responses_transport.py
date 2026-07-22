"""Fixed HTTPS Responses transport with no credential resolution or SDK use."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional


_HOST = "ai.clincognition.com"
_PORT = 443
_PATH = "/v1/responses"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES = 8_192
_MAX_USAGE_TOKENS = 1_000_000
_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ResponsesProfile:
    """The sole reviewed raw Responses destination; it has no credential field."""

    scheme: str
    host: str
    port: int
    path: str


CLINCOG_RESPONSES_PROFILE = ResponsesProfile(
    scheme="https", host=_HOST, port=_PORT, path=_PATH
)


_PROFILES = {"clincog": CLINCOG_RESPONSES_PROFILE}


def select_responses_profile(value: object) -> Optional[ResponsesProfile]:
    """Return the sole reviewed raw destination, without resolving credentials."""

    if not isinstance(value, str):
        return None
    return _PROFILES.get(value)


class ResponsesOutcome(str, Enum):
    SUCCESS = "success"
    INVALID_REQUEST = "invalid_request"
    INVALID_AUTHORIZATION = "invalid_authorization"
    INVALID_PROFILE = "invalid_profile"
    HTTP_STATUS = "http_status"
    INVALID_CONTENT_TYPE = "invalid_content_type"
    RESPONSE_TOO_LARGE = "response_too_large"
    MALFORMED_RESPONSE = "malformed_response"
    TIMEOUT = "timeout"
    TLS_FAILURE = "tls_failure"
    CONNECTION_FAILURE = "connection_failure"
    PROTOCOL_FAILURE = "protocol_failure"


@dataclass(frozen=True)
class PreparedResponsesRequest:
    """Bounded JSON request bytes prepared without any transport side effect."""

    body: bytes


@dataclass(frozen=True)
class NormalizedUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


@dataclass(frozen=True)
class ResponsesResult:
    outcome: ResponsesOutcome
    status_code: Optional[int] = None
    output_text: Optional[str] = None
    usage: Optional[NormalizedUsage] = None


def _failure(outcome: ResponsesOutcome, status_code: Optional[int] = None) -> ResponsesResult:
    return ResponsesResult(outcome=outcome, status_code=status_code)


def _request_is_valid(request: object) -> bool:
    if not isinstance(request, PreparedResponsesRequest):
        return False
    if not isinstance(request.body, bytes) or not 0 < len(request.body) <= _MAX_REQUEST_BYTES:
        return False
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("store") is False


def prepare_responses_request(body: bytes) -> PreparedResponsesRequest:
    """Validate bounded JSON request bytes before any connection is constructed."""

    request = PreparedResponsesRequest(body=body)
    if not _request_is_valid(request):
        raise ValueError("Responses request is invalid.")
    return request


def _normalized_usage(value: object) -> Optional[NormalizedUsage]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError

    def count(name: str) -> Optional[int]:
        candidate = value.get(name)
        if candidate is None:
            return None
        if not isinstance(candidate, int) or isinstance(candidate, bool) or not 0 <= candidate <= _MAX_USAGE_TOKENS:
            raise ValueError
        return candidate

    cached: object = None
    details = value.get("input_tokens_details")
    if details is not None:
        if not isinstance(details, Mapping):
            raise ValueError
        cached = details.get("cached_tokens")
    if cached is not None and (
        not isinstance(cached, int) or isinstance(cached, bool) or not 0 <= cached <= _MAX_USAGE_TOKENS
    ):
        raise ValueError
    return NormalizedUsage(count("input_tokens"), count("output_tokens"), cached)


def _output_text(payload: Mapping[str, Any]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return part["text"]
    raise ValueError


def _content_length_is_invalid(response: http.client.HTTPResponse) -> bool:
    value = response.getheader("Content-Length")
    if value is None:
        return False
    try:
        return int(value) < 0 or int(value) > _MAX_RESPONSE_BYTES
    except (TypeError, ValueError):
        return True


def _authorization_is_valid(authorization: object) -> bool:
    return (
        isinstance(authorization, str)
        and bool(authorization)
        and all("\x21" <= character <= "\x7e" for character in authorization)
    )


def send_responses_request(
    request: PreparedResponsesRequest, authorization: object, profile: object = "clincog"
) -> ResponsesResult:
    """Send one fixed HTTPS request using an already-resolved internal authorization value."""

    selected_profile = select_responses_profile(profile)
    if selected_profile is None:
        return _failure(ResponsesOutcome.INVALID_PROFILE)
    if not _request_is_valid(request):
        return _failure(ResponsesOutcome.INVALID_REQUEST)
    if not _authorization_is_valid(authorization):
        return _failure(ResponsesOutcome.INVALID_AUTHORIZATION)

    connection: Optional[http.client.HTTPSConnection] = None
    response: Optional[http.client.HTTPResponse] = None
    try:
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(
            selected_profile.host, selected_profile.port, timeout=_TIMEOUT_SECONDS, context=context
        )
        connection.request(
            "POST",
            selected_profile.path,
            body=request.body,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer " + authorization,
                "Content-Length": str(len(request.body)),
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        status_code = response.status
        if status_code != 200:
            return _failure(ResponsesOutcome.HTTP_STATUS, status_code)
        if _content_length_is_invalid(response):
            return _failure(ResponsesOutcome.RESPONSE_TOO_LARGE, status_code)
        content_type = response.getheader("Content-Type")
        if not isinstance(content_type, str) or content_type.split(";", 1)[0].strip().lower() != "application/json":
            return _failure(ResponsesOutcome.INVALID_CONTENT_TYPE, status_code)
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            return _failure(ResponsesOutcome.RESPONSE_TOO_LARGE, status_code)
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError
            return ResponsesResult(
                outcome=ResponsesOutcome.SUCCESS,
                status_code=status_code,
                output_text=_output_text(payload),
                usage=_normalized_usage(payload.get("usage")),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _failure(ResponsesOutcome.MALFORMED_RESPONSE, status_code)
    except (TimeoutError, socket.timeout):
        return _failure(ResponsesOutcome.TIMEOUT)
    except ssl.SSLError:
        return _failure(ResponsesOutcome.TLS_FAILURE)
    except http.client.HTTPException:
        return _failure(ResponsesOutcome.PROTOCOL_FAILURE)
    except OSError:
        return _failure(ResponsesOutcome.CONNECTION_FAILURE)
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
