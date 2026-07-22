import pytest

from icvision import core
from icvision.responses_classifier import RawClassification
from icvision.responses_transport import NormalizedUsage


def test_raw_preflight_rejects_legacy_options_before_loading(monkeypatch):
    monkeypatch.setattr(core, "load_raw_data", lambda *_: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        core.label_components(
            "synthetic.set", api_key="forbidden", component_indices=[0],
            generate_report=False, transport="raw",
        )


class _ICA:
    n_components_ = 1
    labels_ = {"brain": []}
    exclude = []


class _TemporaryDirectory:
    def __init__(self, directory):
        self.directory = directory
    def __enter__(self):
        return str(self.directory)
    def __exit__(self, *_):
        return False


def test_raw_success_is_review_only_and_never_mutates_or_saves(tmp_path, monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: raw)
    monkeypatch.setattr(core, "load_ica_data", lambda _: ica)
    monkeypatch.setattr(core, "validate_inputs", lambda *_: None)
    monkeypatch.setattr(core.tempfile, "TemporaryDirectory", lambda **_: _TemporaryDirectory(tmp_path))
    monkeypatch.setattr(core, "plot_component_for_classification", lambda *_args, **_kwargs: tmp_path / "component.webp")
    monkeypatch.setattr(
        core, "classify_image_with_responses",
        lambda _: RawClassification(
            "brain",
            0.9,
            "Synthetic fixture.",
            "classified",
            None,
            model="gpt-5.6-terra",
            elapsed_seconds=0.25,
            usage=NormalizedUsage(3, 2, 1),
            prompt_sha256="a" * 64,
            artifact_inventory=("temporary_component_webp",),
        ),
    )
    for name in ("_update_ica_with_classifications", "_apply_artifact_rejection", "save_results", "save_ica_data", "save_cleaned_raw_data"):
        monkeypatch.setattr(core, name, lambda *_args, **_kwargs: pytest.fail(name + " must not run"))

    returned_raw, returned_ica, results = core.label_components(
        raw, ica_data=ica, component_indices=[0], generate_report=False, transport="raw"
    )

    assert returned_raw is raw
    assert returned_ica is ica
    assert ica.labels_ == {"brain": []}
    assert ica.exclude == []
    row = results.iloc[0]
    assert row["label"] == "brain"
    assert not bool(row["exclude_vision"])
    assert not bool(row["apply_to_ica"])
    assert bool(row["review_required"])
    assert row["model"] == "gpt-5.6-terra"
    assert row["elapsed_seconds"] == 0.25
    assert row["input_tokens"] == 3
    assert row["output_tokens"] == 2
    assert row["cached_tokens"] == 1
    assert row["prompt_sha256"] == "a" * 64
    assert row["artifact_inventory"] == ("temporary_component_webp",)

@pytest.mark.parametrize(
    "override",
    [
        {"api_key": "synthetic-forbidden-key"},
        {"base_url": "https://forbidden.invalid"},
        {"custom_prompt": "forbidden prompt"},
        {"layout": "strip"},
        {"generate_report": True},
        {"component_indices": None},
        {"component_indices": [0, 1]},
        {"model_name": "gpt-5.6-terra"},
    ],
)
def test_raw_preflight_rejects_each_prohibited_option_before_loading(override, monkeypatch):
    monkeypatch.setattr(core, "load_raw_data", lambda *_: pytest.fail("must not load"))
    options = {"component_indices": [0], "generate_report": False, "transport": "raw"}
    options.update(override)

    with pytest.raises(ValueError):
        core.label_components("synthetic.set", **options)


def test_raw_unavailable_result_remains_review_only(tmp_path, monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: raw)
    monkeypatch.setattr(core, "load_ica_data", lambda _: ica)
    monkeypatch.setattr(core, "validate_inputs", lambda *_: None)
    monkeypatch.setattr(core.tempfile, "TemporaryDirectory", lambda **_: _TemporaryDirectory(tmp_path))
    monkeypatch.setattr(core, "plot_component_for_classification", lambda *_args, **_kwargs: tmp_path / "component.webp")
    monkeypatch.setattr(
        core,
        "classify_image_with_responses",
        lambda _: RawClassification(None, None, "Synthetic unavailable.", "unavailable", "transport_failure"),
    )

    returned_raw, returned_ica, results = core.label_components(
        raw, ica_data=ica, component_indices=[0], generate_report=False, transport="raw"
    )

    assert returned_raw is raw
    assert returned_ica is ica
    row = results.iloc[0]
    assert row["outcome_status"] == "unavailable"
    assert row["failure_category"] == "transport_failure"
    assert bool(row["review_required"])
    assert not bool(row["apply_to_ica"])


def test_default_sdk_transport_still_enters_legacy_validation(monkeypatch):
    sentinel = RuntimeError("legacy-sdk-validation")
    monkeypatch.setattr(core, "_label_components_raw_review_only", lambda *_args, **_kwargs: pytest.fail("raw lane must not run"))
    monkeypatch.setattr(core, "validate_api_key", lambda _: (_ for _ in ()).throw(sentinel))

    with pytest.raises(RuntimeError, match="legacy-sdk-validation"):
        core.label_components("synthetic.set")


def test_prohibited_raw_marker_never_leaks(caplog, monkeypatch):
    secret_marker = "SYNTHETIC_MARKER_MUST_NOT_LEAK"
    monkeypatch.setattr(core, "load_raw_data", lambda *_: pytest.fail("must not load"))

    with pytest.raises(ValueError) as exc_info:
        core.label_components(
            "synthetic.set",
            api_key=secret_marker,
            component_indices=[0],
            generate_report=False,
            transport="raw",
        )

    assert secret_marker not in str(exc_info.value)
    assert secret_marker not in caplog.text

def _raw_call(raw, ica):
    return core.label_components(
        raw,
        ica_data=ica,
        component_indices=[0],
        generate_report=False,
        transport="raw",
    )


def test_raw_loader_failure_is_sanitized_review_only(monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: (_ for _ in ()).throw(OSError("SECRET_MARKER")))

    returned_raw, returned_ica, results = _raw_call(raw, ica)

    assert returned_raw is raw
    assert returned_ica is ica
    assert results.iloc[0]["failure_category"] == "raw_load_failure"
    assert bool(results.iloc[0]["review_required"])
    assert "SECRET_MARKER" not in results.to_string()


def test_raw_ica_loader_failure_is_sanitized_review_only(monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: raw)
    monkeypatch.setattr(core, "load_ica_data", lambda _: (_ for _ in ()).throw(ValueError("SECRET_MARKER")))

    returned_raw, returned_ica, results = _raw_call(raw, ica)

    assert returned_raw is raw
    assert returned_ica is ica
    assert results.iloc[0]["failure_category"] == "ica_load_failure"
    assert "SECRET_MARKER" not in results.to_string()


def test_raw_validation_failure_is_sanitized_review_only(monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: raw)
    monkeypatch.setattr(core, "load_ica_data", lambda _: ica)
    monkeypatch.setattr(core, "validate_inputs", lambda *_: (_ for _ in ()).throw(ValueError("SECRET_MARKER")))

    returned_raw, returned_ica, results = _raw_call(raw, ica)

    assert returned_raw is raw
    assert returned_ica is ica
    assert results.iloc[0]["failure_category"] == "validation_failure"
    assert "SECRET_MARKER" not in results.to_string()


def test_raw_plot_exception_is_sanitized_review_only(tmp_path, monkeypatch):
    raw = object()
    ica = _ICA()
    monkeypatch.setattr(core, "load_raw_data", lambda _: raw)
    monkeypatch.setattr(core, "load_ica_data", lambda _: ica)
    monkeypatch.setattr(core, "validate_inputs", lambda *_: None)
    monkeypatch.setattr(core.tempfile, "TemporaryDirectory", lambda **_: _TemporaryDirectory(tmp_path))
    monkeypatch.setattr(
        core,
        "plot_component_for_classification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("SECRET_MARKER")),
    )

    returned_raw, returned_ica, results = _raw_call(raw, ica)

    assert returned_raw is raw
    assert returned_ica is ica
    assert results.iloc[0]["failure_category"] == "plot_failure"
    assert results.iloc[0]["artifact_inventory"] == ()
    assert "SECRET_MARKER" not in results.to_string()
