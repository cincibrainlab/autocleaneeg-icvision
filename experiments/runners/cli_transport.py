"""OpenCode CLI transport for provider-specific models such as Gemini."""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_array(text: str) -> List[Dict[str, Any]]:
    fenced = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    candidates = fenced or [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    raise ValueError("CLI response did not contain a JSON classification array")


def classify_strip_cli(
    image_path: Path,
    prompt: str,
    model: str,
    component_indices: List[int],
    timeout: int = 300,
) -> Tuple[List[Dict[str, Any]], str]:
    config = {
        "agent": {
            "classifier": {
                "mode": "primary",
                "model": model,
                "steps": 1,
                "permission": {"*": "deny"},
            }
        }
    }
    env = os.environ.copy()
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
    command = [
        "opencode",
        "run",
        "--agent",
        "classifier",
        "--model",
        model,
        "--file",
        str(image_path),
        "--format",
        "json",
        prompt,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    raw = completed.stdout
    if completed.returncode:
        raise RuntimeError(f"OpenCode CLI exited {completed.returncode}: {completed.stderr[-1000:]}")
    text_parts = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part", {})
        if event.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    parsed = _extract_array("\n".join(text_parts))
    letters = {chr(ord("A") + index): component for index, component in enumerate(component_indices)}
    for item in parsed:
        component = item.get("component_idx", item.get("component"))
        if isinstance(component, str) and component.upper() in letters:
            item["component_idx"] = letters[component.upper()]
        elif component is not None:
            item["component_idx"] = int(component)
        else:
            raise ValueError("CLI classification item is missing component/component_idx")
        if item.get("label") == "other":
            item["label"] = "other_artifact"
        if isinstance(item.get("confidence"), str):
            confidence = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(item["confidence"].lower())
            if confidence is None:
                raise ValueError(f"Unsupported CLI confidence: {item['confidence']}")
            item["confidence"] = confidence
        if not item.get("reason"):
            raise ValueError("CLI classification item is missing reason")
    return parsed, raw
