"""Fixed-purpose ClinCog ICA classification client."""

from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


_HOST = "ai.clincognition.com"
_PORT = 443
_PATH = "/v1/ic-classifications"
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_RESPONSE_BYTES = 8_192
_TIMEOUT_SECONDS = 5.0


class GatewayOutcome(str, Enum):
    SUCCESS = "success"
    INVALID_AUTHORIZATION = "invalid_authorization"
    INVALID_REQUEST = "invalid_request"
    HTTP_STATUS = "http_status"
    INVALID_CONTENT_TYPE = "invalid_content_type"
    RESPONSE_TOO_LARGE = "response_too_large"
    MALFORMED_RESPONSE = "malformed_response"
    TIMEOUT = "timeout"
    TLS_FAILURE = "tls_failure"
    CONNECTION_FAILURE = "connection_failure"
    PROTOCOL_FAILURE = "protocol_failure"


@dataclass(frozen=True)
class GatewayResult:
    outcome: GatewayOutcome
    status_code: Optional[int] = None
    classification: Optional[Mapping[str, object]] = None


def resolve_gateway_token() -> Optional[str]:
    """Read only the inbound purpose-specific gateway token."""

    token = os.environ.get("CLINCOG_API_TOKEN")
    if not isinstance(token, str) or not token or not all("\x21" <= character <= "\x7e" for character in token):
        return None
    return token


def _failure(outcome: GatewayOutcome, status_code: Optional[int] = None) -> GatewayResult:
    return GatewayResult(outcome=outcome, status_code=status_code)


def _valid_image(image: object) -> bool:
    return (
        isinstance(image, bytes)
        and 12 <= len(image) <= _MAX_IMAGE_BYTES
        and image[:4] == b"RIFF"
        and image[8:12] == b"WEBP"
    )


def _valid_token(token: object) -> bool:
    return isinstance(token, str) and bool(token) and all("\x21" <= character <= "\x7e" for character in token)


def _invalid_content_length(response: http.client.HTTPResponse) -> bool:
    value = response.getheader("Content-Length")
    if value is None:
        return False
    try:
        return not 0 <= int(value) <= _MAX_RESPONSE_BYTES
    except (TypeError, ValueError):
        return True


def _json_content_type(response: http.client.HTTPResponse) -> bool:
    content_type = response.getheader("Content-Type")
    return isinstance(content_type, str) and content_type.split(";", 1)[0].strip().lower() == "application/json"


def _http_status(value: object) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def send_classification_image(image: object, token: object) -> GatewayResult:
    """POST one bounded WebP image to the sole reviewed ClinCog endpoint."""

    if not _valid_image(image):
        return _failure(GatewayOutcome.INVALID_REQUEST)
    if not _valid_token(token):
        return _failure(GatewayOutcome.INVALID_AUTHORIZATION)

    connection: Optional[http.client.HTTPSConnection] = None
    response: Optional[http.client.HTTPResponse] = None
    try:
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        connection = http.client.HTTPSConnection(_HOST, _PORT, timeout=_TIMEOUT_SECONDS, context=context)
        connection.request(
            "POST",
            _PATH,
            body=image,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer " + token,
                "Content-Length": str(len(image)),
                "Content-Type": "image/webp",
            },
        )
        response = connection.getresponse()
        status_code = _http_status(response.status)
        if status_code is None:
            return _failure(GatewayOutcome.PROTOCOL_FAILURE)
        if status_code != 200:
            return _failure(GatewayOutcome.HTTP_STATUS, status_code)
        if _invalid_content_length(response):
            return _failure(GatewayOutcome.RESPONSE_TOO_LARGE, status_code)
        if not _json_content_type(response):
            return _failure(GatewayOutcome.INVALID_CONTENT_TYPE, status_code)
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            return _failure(GatewayOutcome.RESPONSE_TOO_LARGE, status_code)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _failure(GatewayOutcome.MALFORMED_RESPONSE, status_code)
        if not isinstance(payload, dict) or set(payload) != {"label", "confidence", "reason", "model", "request_id"}:
            return _failure(GatewayOutcome.MALFORMED_RESPONSE, status_code)
        return GatewayResult(GatewayOutcome.SUCCESS, status_code, payload)
    except (TimeoutError, socket.timeout):
        return _failure(GatewayOutcome.TIMEOUT)
    except ssl.SSLError:
        return _failure(GatewayOutcome.TLS_FAILURE)
    except http.client.HTTPException:
        return _failure(GatewayOutcome.PROTOCOL_FAILURE)
    except OSError:
        return _failure(GatewayOutcome.CONNECTION_FAILURE)
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
