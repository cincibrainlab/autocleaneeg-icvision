"""Runtime-only authorization resolution for the governed raw Responses lane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .responses_transport import (
    PreparedResponsesRequest,
    ResponsesOutcome,
    ResponsesProfile,
    ResponsesResult,
    select_responses_profile,
    send_responses_request,
)


class RuntimeCredentialOutcome(str, Enum):
    READY = "ready"
    INVALID_CONFIGURATION = "invalid_configuration"
    MISSING_OR_INVALID_CREDENTIAL = "missing_or_invalid_credential"


@dataclass(frozen=True)
class RuntimeAuthorization:
    outcome: RuntimeCredentialOutcome
    profile: Optional[ResponsesProfile] = None
    authorization: Optional[str] = field(default=None, repr=False)


def _is_valid_authorization(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and all("\x21" <= character <= "\x7e" for character in value)
    )


def resolve_runtime_authorization(profile: object = "clincog") -> RuntimeAuthorization:
    """Resolve authorization only after selecting the fixed reviewed profile."""

    selected_profile = select_responses_profile(profile)
    if selected_profile is None:
        return RuntimeAuthorization(RuntimeCredentialOutcome.INVALID_CONFIGURATION)

    authorization = os.environ.get("OPENAI_API_KEY")
    if not _is_valid_authorization(authorization):
        return RuntimeAuthorization(
            RuntimeCredentialOutcome.MISSING_OR_INVALID_CREDENTIAL,
            profile=selected_profile,
        )

    return RuntimeAuthorization(
        RuntimeCredentialOutcome.READY,
        profile=selected_profile,
        authorization=authorization,
    )



def send_runtime_responses_request(
    request: PreparedResponsesRequest, profile: object = "clincog"
) -> ResponsesResult:
    """Resolve the runtime-only credential, then use the fixed profile."""

    resolved = resolve_runtime_authorization(profile)
    if resolved.outcome is RuntimeCredentialOutcome.INVALID_CONFIGURATION:
        return ResponsesResult(outcome=ResponsesOutcome.INVALID_PROFILE)
    if resolved.outcome is RuntimeCredentialOutcome.MISSING_OR_INVALID_CREDENTIAL:
        return ResponsesResult(outcome=ResponsesOutcome.INVALID_AUTHORIZATION)
    return send_responses_request(request, resolved.authorization, profile=profile)
