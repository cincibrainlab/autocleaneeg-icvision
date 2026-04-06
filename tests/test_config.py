"""Tests for prompt selection and classification mode validation."""

import pytest

from icvision.config import get_single_prompt, get_strip_prompt, validate_classification_mode


def test_validate_classification_mode_accepts_human() -> None:
    assert validate_classification_mode("human") == "human"


def test_validate_classification_mode_normalizes_case() -> None:
    assert validate_classification_mode("Mouse") == "mouse"


def test_validate_classification_mode_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="Invalid classification_mode"):
        validate_classification_mode("rat")


def test_get_single_prompt_human_loads_human_prompt() -> None:
    prompt = get_single_prompt("human")
    assert "Classify this EEG ICA component" in prompt
    assert "MOUSE EEG" not in prompt


def test_get_single_prompt_mouse_loads_mouse_prompt() -> None:
    prompt = get_single_prompt("mouse")
    assert "Classify this MOUSE EEG ICA component" in prompt
    assert "7.5-12.5 Hz" in prompt


def test_get_strip_prompt_human_loads_human_template() -> None:
    prompt = get_strip_prompt(3, classification_mode="human")
    assert "Classify each of the 3 ICA components" in prompt
    assert '"component": "A"' in prompt
    assert "MOUSE EEG" not in prompt


def test_get_strip_prompt_mouse_loads_mouse_template() -> None:
    prompt = get_strip_prompt(3, classification_mode="mouse")
    assert "Classify each of the 3 MOUSE EEG ICA components" in prompt
    assert '"component": "A"' in prompt
    assert "7.5-12.5 Hz" in prompt
