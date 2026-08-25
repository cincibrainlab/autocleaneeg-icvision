"""Prompt contract tests for strip classification."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from icvision.config import get_strip_prompt, load_prompt


def test_strip_default_prompt_uses_tightened_template(tmp_path: Path) -> None:
    """Strip classification without a custom prompt uses tightened_v1_strip."""
    from icvision.api import classify_strip_image

    image_path = tmp_path / "strip.webp"
    image_path.write_bytes(b"fake image data")
    captured_prompts = []
    response = json.dumps([
        {"component": "A", "label": "brain", "confidence": 0.95, "reason": "Test"}
    ])

    def capture_prompt(_client, _model_name, prompt, _base64_image, _reasoning_effort):
        captured_prompts.append(prompt)
        return response

    with patch("icvision.api.openai.OpenAI", return_value=MagicMock()):
        with patch("icvision.api._call_openai_api", side_effect=capture_prompt):
            classify_strip_image(image_path, [0], api_key="test-key", max_retries=1)

    expected = get_strip_prompt(1, template=load_prompt("tightened_v1_strip"))
    assert captured_prompts == [expected]
    assert "narrow, high-bar category" in captured_prompts[0]


def test_custom_prompt_template_reaches_strip_image_unchanged(tmp_path: Path) -> None:
    """Explicit strip prompt templates are formatted instead of being replaced."""
    from icvision.api import classify_strip_image

    image_path = tmp_path / "custom_strip.webp"
    image_path.write_bytes(b"fake image data")
    custom_prompt = "CUSTOM STRIP TEMPLATE {n} :: {labels}\n{json_example}"
    captured_prompts = []
    response = json.dumps([
        {"component": "A", "label": "brain", "confidence": 0.95, "reason": "Test"}
    ])

    def capture_prompt(_client, _model_name, prompt, _base64_image, _reasoning_effort):
        captured_prompts.append(prompt)
        return response

    with patch("icvision.api.openai.OpenAI", return_value=MagicMock()):
        with patch("icvision.api._call_openai_api", side_effect=capture_prompt):
            classify_strip_image(
                image_path,
                [0],
                api_key="test-key",
                max_retries=1,
                custom_prompt=custom_prompt,
            )

    assert captured_prompts == [get_strip_prompt(1, template=custom_prompt)]
    assert captured_prompts[0].startswith("CUSTOM STRIP TEMPLATE 1 :: A")


def test_classify_components_batch_default_layout_stays_single() -> None:
    """Omitting layout must not dispatch to strip mode."""
    from icvision.api import classify_components_batch

    ica_obj = MagicMock()
    ica_obj.n_components_ = 0

    with patch("icvision.api.classify_components_strip_batch") as mock_strip_batch:
        results_df, metadata = classify_components_batch(
            ica_obj=ica_obj,
            raw_obj=MagicMock(),
            api_key="test-key",
        )

    mock_strip_batch.assert_not_called()
    assert results_df.empty
    assert metadata["requests_count"] == 0
