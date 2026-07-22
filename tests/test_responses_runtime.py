"""Offline tests for raw-lane runtime credential resolution."""

import socket

import pytest

import icvision.responses_runtime as runtime


SECRET_MARKER = "synthetic-secret-marker"


def _denied(*_args, **_kwargs):
    raise AssertionError("runtime credential resolution attempted a forbidden socket operation")


@pytest.fixture(autouse=True)
def deny_sockets(monkeypatch):
    monkeypatch.setattr(socket, "socket", _denied)
    monkeypatch.setattr(socket, "create_connection", _denied)
    monkeypatch.setattr(socket, "getaddrinfo", _denied)


class _NoEnvironmentLookup(dict):
    def get(self, *_args, **_kwargs):
        raise AssertionError("environment lookup occurred before profile validation")


def test_module_has_no_dotenv_credential_manager_or_logging_facility():
    assert "dotenv" not in runtime.__dict__
    assert "subprocess" not in runtime.__dict__
    assert "logging" not in runtime.__dict__


def test_unknown_or_url_shaped_profile_fails_before_environment_lookup(monkeypatch):
    monkeypatch.setattr(runtime.os, "environ", _NoEnvironmentLookup())

    for profile in (None, "unknown", "https://ai.clincognition.com/v1"):
        assert runtime.resolve_runtime_authorization(profile) == runtime.RuntimeAuthorization(
            runtime.RuntimeCredentialOutcome.INVALID_CONFIGURATION
        )


@pytest.mark.parametrize(
    "credential",
    [None, "", " ", "\t", "contains whitespace", "contains\nnewline", "non-ascii-é"],
)
def test_missing_or_invalid_credential_is_sanitized_and_opens_no_socket(monkeypatch, credential):
    if credential is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", credential)

    result = runtime.resolve_runtime_authorization()

    assert result.outcome is runtime.RuntimeCredentialOutcome.MISSING_OR_INVALID_CREDENTIAL
    assert result.profile is not None
    assert result.authorization is None
    assert SECRET_MARKER not in repr(result)


def test_reads_only_injected_openai_key_at_call_time_and_redacts_repr(monkeypatch):
    monkeypatch.setenv("UNRELATED_CREDENTIAL", SECRET_MARKER)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_MARKER)

    result = runtime.resolve_runtime_authorization()

    assert result.outcome is runtime.RuntimeCredentialOutcome.READY
    assert result.profile is not None
    assert result.profile.host == "ai.clincognition.com"
    assert result.profile.port == 443
    assert result.profile.path == "/v1/responses"
    assert result.authorization == SECRET_MARKER
    assert SECRET_MARKER not in repr(result)


def test_environment_is_read_at_each_call_not_import_time(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "first")
    first = runtime.resolve_runtime_authorization()
    monkeypatch.setenv("OPENAI_API_KEY", "second")
    second = runtime.resolve_runtime_authorization()

    assert first.authorization == "first"
    assert second.authorization == "second"


def test_adapter_does_not_add_environment_resolution_to_transport():
    import icvision.responses_transport as transport

    assert "os" not in transport.__dict__
