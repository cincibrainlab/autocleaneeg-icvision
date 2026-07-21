"""Offline fixed-gateway transport-policy tests."""

import socket
import sys
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from icvision import cli, core, utils
from icvision.transport_policy import (
    CLINCOG_RESPONSES_PROFILE,
    ENDPOINT_PROFILES,
    OPENAI_RESPONSES_PROFILE,
    TransportPolicyError,
    TransportSelection,
    raw_gateway_credential_environment,
    raw_credential_environment,
    validate_raw_environment_policy,
    validate_raw_gateway_profile,
    validate_transport_policy,
)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("unexpected side effect")


@pytest.fixture(autouse=True)
def no_socket(monkeypatch):
    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)


@pytest.mark.parametrize(
    ("transport", "base_url", "profile", "message"),
    [
        ("raw", "", "clincog-responses", "Raw transport does not accept base_url."),
        (
            "raw",
            None,
            None,
            "Raw transport requires endpoint_profile='clincog-responses'.",
        ),
        (
            "raw",
            None,
            "CLINCOG-RESPONSES",
            "Raw transport endpoint profile is not recognized.",
        ),
        (
            "raw",
            None,
            "https://ai.clincognition.com/v1/responses",
            "Raw transport endpoint profile is not recognized.",
        ),
        (
            "raw",
            None,
            "ai.clincognition.com",
            "Raw transport endpoint profile is not recognized.",
        ),
        (
            "sdk",
            None,
            "clincog-responses",
            "SDK transport does not accept an endpoint profile.",
        ),
        ("marker-transport", None, None, "Transport must be 'sdk' or 'raw'."),
    ],
)
def test_invalid_policy_errors_are_stable_and_sanitized(transport, base_url, profile, message):
    with pytest.raises(TransportPolicyError, match=message) as error:
        validate_transport_policy(transport, base_url, profile)
    assert "marker" not in str(error.value)
    assert "ai.clincognition.com" not in str(error.value)


def test_sdk_preserves_custom_base_url():
    selection = validate_transport_policy("sdk", "https://sdk-marker.invalid/v1", None)
    assert selection == TransportSelection(transport="sdk")


def test_raw_selection_returns_only_reviewed_gateway_profile():
    selection = validate_transport_policy("raw", None, "clincog-responses")

    assert selection.endpoint_profile is CLINCOG_RESPONSES_PROFILE
    assert OPENAI_RESPONSES_PROFILE is CLINCOG_RESPONSES_PROFILE
    assert (selection.endpoint_profile.host, selection.endpoint_profile.port) == (
        "ai.clincognition.com",
        443,
    )
    assert selection.endpoint_profile.path == "/v1/responses"
    assert selection.endpoint_profile.scheme == "https"
    assert selection.endpoint_profile.direct_connection
    assert selection.endpoint_profile.deny_redirects
    assert selection.endpoint_profile.require_system_ca
    assert selection.endpoint_profile.require_hostname_verification
    assert raw_credential_environment(selection) == "ICVISION_GATEWAY_CREDENTIAL"
    assert validate_raw_gateway_profile("raw", None, "clincog-responses") is CLINCOG_RESPONSES_PROFILE
    assert raw_gateway_credential_environment(CLINCOG_RESPONSES_PROFILE) == "ICVISION_GATEWAY_CREDENTIAL"


def test_registry_and_profile_are_immutable():
    assert isinstance(ENDPOINT_PROFILES, MappingProxyType)

    with pytest.raises(TypeError):
        ENDPOINT_PROFILES["another"] = CLINCOG_RESPONSES_PROFILE
    with pytest.raises(FrozenInstanceError):
        CLINCOG_RESPONSES_PROFILE.host = "marker.invalid"


def test_raw_credential_source_rejects_counterfeit_equal_profile():
    counterfeit = replace(CLINCOG_RESPONSES_PROFILE)
    selection = TransportSelection("raw", counterfeit)

    with pytest.raises(TransportPolicyError):
        raw_credential_environment(selection)
    with pytest.raises(TransportPolicyError):
        raw_gateway_credential_environment(counterfeit)


def test_raw_credential_source_rejects_sdk_selection():
    with pytest.raises(TransportPolicyError):
        raw_credential_environment(TransportSelection("sdk"))


def test_ambient_endpoint_proxy_and_provider_key_variables_do_not_change_raw_profile():
    environ = {
        "OPENAI_BASE_URL": "https://marker.invalid",
        "OPENAI_API_KEY": "marker-provider-key",
        "HTTP_PROXY": "http://marker.invalid",
        "HTTPS_PROXY": "http://marker.invalid",
        "ALL_PROXY": "http://marker.invalid",
        "NO_PROXY": "*",
        "REQUESTS_CA_BUNDLE": "marker.pem",
        "CURL_CA_BUNDLE": "marker.pem",
    }

    validate_raw_environment_policy(environ)
    selection = validate_transport_policy("raw", None, "clincog-responses")

    assert selection.endpoint_profile is CLINCOG_RESPONSES_PROFILE
    assert raw_credential_environment(selection) == "ICVISION_GATEWAY_CREDENTIAL"


@pytest.mark.parametrize("variable_name", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_custom_ca_environment_fails_closed_and_sanitized(variable_name):
    with pytest.raises(TransportPolicyError) as error:
        validate_raw_environment_policy({variable_name: "marker-private-ca.pem"})

    assert str(error.value) == "Raw transport does not allow custom CA environment overrides."
    assert "marker" not in str(error.value)


def test_core_raw_gate_precedes_all_pipeline_work(monkeypatch):
    monkeypatch.setattr(core.os, "environ", {})
    monkeypatch.setattr(core.logger, "debug", _forbidden)
    for name in (
        "validate_api_key",
        "load_raw_data",
        "load_ica_data",
        "create_output_directory",
        "classify_components_batch",
    ):
        monkeypatch.setattr(core, name, _forbidden)
    monkeypatch.setattr(utils, "load_dotenv", _forbidden)
    monkeypatch.setattr(utils.os, "getenv", _forbidden)

    with pytest.raises(RuntimeError, match=r"^Raw transport is not enabled in Gate 0\.$"):
        core.label_components(
            object(),
            transport="raw",
            endpoint_profile="clincog-responses",
        )


def test_core_invalid_raw_policy_precedes_pipeline_work(monkeypatch):
    monkeypatch.setattr(core, "validate_api_key", _forbidden)
    monkeypatch.setattr(core, "load_raw_data", _forbidden)

    with pytest.raises(TransportPolicyError):
        core.label_components(
            object(),
            transport="raw",
            base_url="",
            endpoint_profile="clincog-responses",
        )


def test_core_custom_ca_fails_before_credentials(monkeypatch):
    monkeypatch.setattr(core.os, "environ", {"SSL_CERT_FILE": "marker-private-ca.pem"})
    monkeypatch.setattr(core, "validate_api_key", _forbidden)
    monkeypatch.setattr(core, "load_raw_data", _forbidden)

    with pytest.raises(TransportPolicyError) as error:
        core.label_components(
            object(),
            transport="raw",
            endpoint_profile="clincog-responses",
        )

    assert "marker" not in str(error.value)


def test_core_raw_api_key_fails_before_environment(monkeypatch):
    monkeypatch.setattr(core.logger, "debug", _forbidden)
    monkeypatch.setattr(core, "validate_raw_environment_policy", _forbidden)

    with pytest.raises(TransportPolicyError) as error:
        core.label_components(
            object(),
            api_key="marker-secret",
            transport="raw",
            endpoint_profile="clincog-responses",
        )

    assert str(error.value) == "Raw transport does not accept api_key in Gate 0."
    assert "marker-secret" not in str(error.value)


def test_cli_raw_api_key_fails_before_logging_or_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "environ", {})
    monkeypatch.setattr(cli, "setup_cli_logging", _forbidden)
    monkeypatch.setattr(cli, "label_components", _forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icvision",
            "synthetic.set",
            "--transport",
            "raw",
            "--endpoint-profile",
            "clincog-responses",
            "--api-key",
            "marker-secret",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "Raw transport does not accept --api-key in Gate 0." in output.err
    assert "marker-secret" not in output.err + output.out


def test_cli_invalid_profile_fails_before_logging_or_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(cli, "setup_cli_logging", _forbidden)
    monkeypatch.setattr(cli, "label_components", _forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icvision",
            "synthetic.set",
            "--transport",
            "raw",
            "--endpoint-profile",
            "MARKER-PROFILE",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "Raw transport endpoint profile is not recognized." in output.err
    assert "MARKER-PROFILE" not in output.err + output.out


def test_cli_custom_ca_fails_before_logging_or_pipeline(monkeypatch, capsys):
    monkeypatch.setattr(cli.os, "environ", {"SSL_CERT_DIR": "marker-private-ca"})
    monkeypatch.setattr(cli, "setup_cli_logging", _forbidden)
    monkeypatch.setattr(cli, "label_components", _forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "icvision",
            "synthetic.set",
            "--transport",
            "raw",
            "--endpoint-profile",
            "clincog-responses",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    output = capsys.readouterr()
    assert exit_info.value.code == 2
    assert "marker" not in output.err + output.out


def test_unhashable_transport_is_sanitized():
    with pytest.raises(TransportPolicyError) as error:
        validate_transport_policy([], None, None)

    assert str(error.value) == "Transport must be 'sdk' or 'raw'."
