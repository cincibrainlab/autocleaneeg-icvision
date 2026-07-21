"""Pure transport-selection policy for the OpenAI modernization gates."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


class TransportPolicyError(ValueError):
    """A stable, sanitized policy-validation error."""


@dataclass(frozen=True)
class EndpointProfile:
    """Reviewed immutable facts for the sole raw ICVision gateway."""

    name: str
    scheme: str
    host: str
    port: int
    path: str
    credential_environment: str
    direct_connection: bool
    deny_redirects: bool
    require_system_ca: bool
    require_hostname_verification: bool


@dataclass(frozen=True)
class TransportSelection:
    """Validated lane selection; raw selections always carry a reviewed profile."""

    transport: str
    endpoint_profile: Optional[EndpointProfile] = None


CLINCOG_RESPONSES_PROFILE = EndpointProfile(
    name="clincog-responses",
    scheme="https",
    host="ai.clincognition.com",
    port=443,
    path="/v1/responses",
    credential_environment="ICVISION_GATEWAY_CREDENTIAL",
    direct_connection=True,
    deny_redirects=True,
    require_system_ca=True,
    require_hostname_verification=True,
)

# Compatibility import name only; it does not identify an OpenAI endpoint.
OPENAI_RESPONSES_PROFILE = CLINCOG_RESPONSES_PROFILE

ENDPOINT_PROFILES: Mapping[str, EndpointProfile] = MappingProxyType(
    {CLINCOG_RESPONSES_PROFILE.name: CLINCOG_RESPONSES_PROFILE}
)


def validate_transport_policy(
    transport: object,
    base_url: object,
    endpoint_profile: object,
) -> TransportSelection:
    """Validate a lane selection without accessing environment, files, logs, or network."""
    if not isinstance(transport, str) or transport not in {"sdk", "raw"}:
        raise TransportPolicyError("Transport must be 'sdk' or 'raw'.")

    if transport == "sdk":
        if endpoint_profile is not None:
            raise TransportPolicyError("SDK transport does not accept an endpoint profile.")
        return TransportSelection(transport="sdk")

    if base_url is not None:
        raise TransportPolicyError("Raw transport does not accept base_url.")
    if endpoint_profile is None:
        raise TransportPolicyError("Raw transport requires endpoint_profile='clincog-responses'.")
    if not isinstance(endpoint_profile, str) or endpoint_profile not in ENDPOINT_PROFILES:
        raise TransportPolicyError("Raw transport endpoint profile is not recognized.")

    return TransportSelection(transport="raw", endpoint_profile=ENDPOINT_PROFILES[endpoint_profile])


def validate_raw_environment_policy(environ: Mapping[str, str]) -> None:
    """Fail closed on custom CA overrides while ignoring endpoint and proxy variables."""
    for variable_name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
        if environ.get(variable_name):
            raise TransportPolicyError("Raw transport does not allow custom CA environment overrides.")


def raw_credential_environment(selection: TransportSelection) -> str:
    """Return the reviewed credential source only for a validated raw selection."""
    profile = selection.endpoint_profile
    if selection.transport != "raw" or not any(
        profile is reviewed_profile for reviewed_profile in ENDPOINT_PROFILES.values()
    ):
        raise TransportPolicyError("A validated raw transport selection is required.")
    return profile.credential_environment


def validate_raw_gateway_profile(
    transport: object,
    base_url: object,
    endpoint_profile: object,
) -> EndpointProfile:
    """Validate and return the fixed raw gateway profile."""
    selection = validate_transport_policy(transport, base_url, endpoint_profile)
    if selection.transport != "raw" or selection.endpoint_profile is None:
        raise TransportPolicyError("A validated raw transport selection is required.")
    return selection.endpoint_profile


def raw_gateway_credential_environment(profile: EndpointProfile) -> str:
    """Expose only the reviewed injected credential variable."""
    if profile is not CLINCOG_RESPONSES_PROFILE:
        raise TransportPolicyError("A reviewed gateway profile is required.")
    return profile.credential_environment
