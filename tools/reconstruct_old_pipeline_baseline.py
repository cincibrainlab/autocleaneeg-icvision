"""Reconstruct the old IC_Visual_AI model-performance baseline.

This script intentionally treats the historical JSONL files as supervised
examples: it reuses only the old system prompt and image URL, discards the
assistant/human label before the model call, then compares the fresh model
prediction against the Grace-derived label.

Outputs are local analysis artifacts only. The script never edits EEG/ICA data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_MAP = {
    "brain": "brain",
    "brain activity": "brain",
    "eye": "eye",
    "eye blink": "eye",
    "eog": "eye",
    "muscle": "muscle",
    "muscle artifact": "muscle",
    "heart": "heart",
    "cardiac": "heart",
    "cardiac artifact": "heart",
    "ecg": "heart",
    "channel noise": "channel_noise",
    "channel_noise": "channel_noise",
    "ch_noise": "channel_noise",
    "line noise": "line_noise",
    "line_noise": "line_noise",
    "other": "other",
    "other artifact": "other",
    "other_artifact": "other",
}


def sanitize_error(text: str) -> str:
    """Remove credential-shaped substrings from provider error messages."""
    text = re.sub(r"\b(?:sk|clp)(?:[-_][A-Za-z0-9*]+)+", "CREDENTIAL-REDACTED", text)
    return re.sub(r"Bearer\s+[^\"\s,}]+", "Bearer CREDENTIAL-REDACTED", text)


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return LABEL_MAP.get(text, text.replace(" ", "_") or "unknown")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON is not an object")
    return parsed


def _load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            messages = payload["messages"]
            system_prompt = messages[0]["content"]
            user_content = messages[1]["content"]
            image_url = user_content[0]["image_url"]["url"]
            answer = json.loads(messages[2]["content"])
            true_label = _normalize_label(answer.get("classification"))
            image_name = image_url.rsplit("/", 1)[-1]
            examples.append(
                {
                    "source_file": str(path),
                    "line_number": line_number,
                    "image_url": image_url,
                    "image_filename": image_name,
                    "system_prompt": system_prompt,
                    "true_label": true_label,
                    "true_label_raw": answer.get("classification"),
                }
            )
    return examples


def _message_text(response: dict[str, Any]) -> str:
    output = response.get("output") or []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            text = content.get("text")
            if text:
                return text
    fallback = response.get("output_text")
    if fallback:
        return str(fallback)
    return ""


def _responses_create(
    *,
    api_key: str,
    base_url: str,
    model: str,
    example: dict[str, Any],
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": example["system_prompt"]},
                        {"type": "input_image", "image_url": example["image_url"]},
                    ],
                }
            ],
            "temperature": 0.2,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(sanitize_error(f"HTTP {exc.code}: {detail[:500]}")) from exc


def classify_example(api_key: str, base_url: str, model: str, example: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    response = _responses_create(api_key=api_key, base_url=base_url, model=model, example=example)
    elapsed = time.perf_counter() - started
    raw_text = _message_text(response)
    parsed = _extract_json(raw_text)
    usage = response.get("usage") or {}
    predicted_raw = parsed.get("classification", parsed.get("label"))
    predicted = _normalize_label(predicted_raw)
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    return {
        **{k: example[k] for k in ("source_file", "line_number", "image_url", "image_filename", "true_label", "true_label_raw")},
        "model": model,
        "predicted_label": predicted,
        "predicted_label_raw": predicted_raw,
        "confidence": confidence,
        "reasoning": parsed.get("reasoning", parsed.get("reason", "")),
        "raw_response": raw_text,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "elapsed_seconds": round(elapsed, 3),
        "correct": predicted == example["true_label"],
        "error": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "line_number",
        "image_filename",
        "image_url",
        "model",
        "true_label",
        "true_label_raw",
        "predicted_label",
        "predicted_label_raw",
        "confidence",
        "correct",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
        "reasoning",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if not row.get("error")]
    accuracy = (
        sum(1 for row in completed if row.get("correct")) / len(completed)
        if completed
        else None
    )
    confusion = Counter(
        (row.get("true_label"), row.get("predicted_label")) for row in completed
    )
    payload = {
        "total_rows": len(rows),
        "completed_rows": len(completed),
        "failed_rows": len(rows) - len(completed),
        "accuracy": accuracy,
        "label_distribution_true": Counter(row.get("true_label") for row in completed),
        "label_distribution_predicted": Counter(row.get("predicted_label") for row in completed),
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
        "total_input_tokens": sum(row.get("input_tokens") or 0 for row in completed),
        "total_output_tokens": sum(row.get("output_tokens") or 0 for row in completed),
        "total_elapsed_seconds": round(sum(row.get("elapsed_seconds") or 0 for row in completed), 3),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True, help="Historical training/testing JSONL path.")
    parser.add_argument("--model", required=True, help="Model ID to use for reconstructed baseline.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV.")
    parser.add_argument("--summary-json", required=True, help="Destination JSON summary.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    args = parser.parse_args()

    examples: list[dict[str, Any]] = []
    for item in args.jsonl:
        examples.extend(_load_examples(Path(item)))
    if args.limit is not None:
        examples = examples[: args.limit]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    rows: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=1):
        try:
            row = classify_example(api_key, args.base_url, args.model, example)
        except Exception as exc:  # Keep baseline runs resumable/auditable.
            row = {
                **{k: example[k] for k in ("source_file", "line_number", "image_url", "image_filename", "true_label", "true_label_raw")},
                "model": args.model,
                "predicted_label": "",
                "predicted_label_raw": "",
                "confidence": "",
                "reasoning": "",
                "raw_response": "",
                "input_tokens": "",
                "output_tokens": "",
                "elapsed_seconds": "",
                "correct": False,
                "error": sanitize_error(f"{type(exc).__name__}: {exc}"),
            }
        rows.append(row)
        print(
            f"{index}/{len(examples)} {row['image_filename']} true={row['true_label']} "
            f"pred={row.get('predicted_label') or 'ERROR'} correct={row.get('correct')} "
            f"err={row.get('error') or '-'}",
            flush=True,
        )
        write_csv(Path(args.output_csv), rows)
        write_summary(Path(args.summary_json), rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
