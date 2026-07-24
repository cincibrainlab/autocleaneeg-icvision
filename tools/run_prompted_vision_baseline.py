"""Run a prompted vision-model ICA classification baseline.

This is a read-only evaluation helper for the historical IC_Visual_AI image
classification dataset. It loads the historical JSONL records only to recover
image URLs and Grace/reference labels, replaces the old prompt with a supplied
runtime prompt, calls a Responses-compatible endpoint, and writes row-level CSV
plus an aggregate JSON summary.

The script does not read or modify EEG/ICA data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "heart beat": "heart",
    "cardiac": "heart",
    "cardiac artifact": "heart",
    "ecg": "heart",
    "line noise": "line_noise",
    "line_noise": "line_noise",
    "channel": "channel_noise",
    "channel noise": "channel_noise",
    "channel_noise": "channel_noise",
    "ch_noise": "channel_noise",
    "other": "other_artifact",
    "other artifact": "other_artifact",
    "other_artifact": "other_artifact",
}


def sanitize_error(text: str) -> str:
    text = re.sub(r"\b(?:sk|clp)(?:[-_][A-Za-z0-9*]+)+", "CREDENTIAL-REDACTED", text)
    return re.sub(r"Bearer\s+[^\"\s,}]+", "Bearer CREDENTIAL-REDACTED", text)


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return LABEL_MAP.get(text, text.replace(" ", "_") or "unknown")


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("model response JSON is not an object")
    return payload


def message_text(response: dict[str, Any]) -> str:
    output = response.get("output") or []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            text = content.get("text")
            if text:
                return str(text)
    return str(response.get("output_text") or "")


def load_examples(paths: list[Path]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                messages = payload["messages"]
                user_content = messages[1]["content"]
                image_url = user_content[0]["image_url"]["url"]
                answer = json.loads(messages[2]["content"])
                image_name = image_url.rsplit("/", 1)[-1]
                examples.append(
                    {
                        "source_file": str(path),
                        "line_number": line_number,
                        "image_filename": image_name,
                        "image_url": image_url,
                        "true_label_raw": answer.get("classification"),
                        "true_label": normalize_label(answer.get("classification")),
                    }
                )
    return examples


def call_responses_with_curl(api_key: str, base_url: str, model: str, prompt: str, image_url: str, timeout: int) -> dict[str, Any]:
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url, "detail": "low"},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "ica_component_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["brain", "channel_noise", "eye", "heart", "line_noise", "muscle", "other_artifact"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 1000},
                    },
                    "required": ["label", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "max_output_tokens": 1024,
        "store": False,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(body, handle)
        body_path = handle.name
    try:
        completed = subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--fail-with-body",
                "--max-time",
                str(timeout),
                base_url.rstrip("/") + "/responses",
                "-H",
                f"Authorization: Bearer {api_key}",
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@" + body_path,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        Path(body_path).unlink(missing_ok=True)
    if completed.returncode != 0:
        raise RuntimeError(sanitize_error((completed.stderr + "\n" + completed.stdout)[:1000]))
    return json.loads(completed.stdout)


def classify_one(example: dict[str, Any], *, api_key: str, base_url: str, model: str, prompt: str, timeout: int, keep_raw: bool) -> dict[str, Any]:
    started = time.perf_counter()
    response = call_responses_with_curl(api_key, base_url, model, prompt, example["image_url"], timeout)
    elapsed = time.perf_counter() - started
    text = message_text(response)
    parsed = extract_json(text)
    usage = response.get("usage") or {}
    predicted_raw = parsed.get("label", parsed.get("classification"))
    predicted = normalize_label(predicted_raw)
    try:
        confidence: Any = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = ""
    row = {
        **example,
        "model": model,
        "predicted_label_raw": predicted_raw or "",
        "predicted_label": predicted,
        "confidence": confidence,
        "reason": parsed.get("reason", parsed.get("reasoning", "")),
        "agreement": predicted == example["true_label"],
        "status": "ok",
        "error": "",
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
        "elapsed_seconds": round(elapsed, 3),
        "response_id": response.get("id", ""),
    }
    if keep_raw:
        row["raw_response"] = text
    return row


def error_row(example: dict[str, Any], model: str, exc: Exception) -> dict[str, Any]:
    return {
        **example,
        "model": model,
        "predicted_label_raw": "",
        "predicted_label": "",
        "confidence": "",
        "reason": "",
        "agreement": False,
        "status": "error",
        "error": sanitize_error(f"{type(exc).__name__}: {exc}"),
        "input_tokens": "",
        "output_tokens": "",
        "total_tokens": "",
        "elapsed_seconds": "",
        "response_id": "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], keep_raw: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "line_number",
        "image_filename",
        "image_url",
        "model",
        "true_label_raw",
        "true_label",
        "predicted_label_raw",
        "predicted_label",
        "confidence",
        "reason",
        "agreement",
        "status",
        "error",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "elapsed_seconds",
        "response_id",
    ]
    if keep_raw:
        fieldnames.append("raw_response")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    correct = sum(1 for row in ok if row.get("agreement") is True)
    def num(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
    elapsed = [num(row, "elapsed_seconds") for row in ok if row.get("elapsed_seconds") not in ("", None)]
    payload = {
        "total_rows": len(rows),
        "completed_rows": len(ok),
        "failed_rows": len(rows) - len(ok),
        "accuracy_completed_rows": correct / len(ok) if ok else None,
        "correct_completed_rows": correct,
        "accuracy_all_rows": correct / len(rows) if rows else None,
        "label_distribution_true": dict(Counter(row.get("true_label") for row in ok)),
        "label_distribution_predicted": dict(Counter(row.get("predicted_label") for row in ok)),
        "confusion": dict(Counter(f"{row.get('true_label')}->{row.get('predicted_label')}" for row in ok)),
        "failure_types": dict(Counter(str(row.get("error", "")).split(":", 1)[0] for row in rows if row.get("status") != "ok")),
        "tokens_completed_rows": {
            "input_tokens": int(sum(num(row, "input_tokens") for row in ok)),
            "output_tokens": int(sum(num(row, "output_tokens") for row in ok)),
            "total_tokens": int(sum(num(row, "total_tokens") for row in ok)),
            "mean_total_tokens": round(sum(num(row, "total_tokens") for row in ok) / len(ok), 3) if ok else None,
        },
        "latency_seconds_completed_rows": {
            "sum": round(sum(elapsed), 3),
            "mean": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
            "min": round(min(elapsed), 3) if elapsed else None,
            "max": round(max(elapsed), 3) if elapsed else None,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--keep-raw-response", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("CLINCOG_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("CLINCOG_API_KEY, OPENAI_API_KEY, or OPENAI_KEY is not set")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    examples = load_examples([Path(item) for item in args.jsonl])
    if args.limit is not None:
        examples = examples[: args.limit]

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                classify_one,
                example,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
                keep_raw=args.keep_raw_response,
            ): index
            for index, example in enumerate(examples)
        }
        by_index: dict[int, dict[str, Any]] = {}
        for done_count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            example = examples[index]
            try:
                row = future.result()
            except Exception as exc:
                row = error_row(example, args.model, exc)
            by_index[index] = row
            rows = [by_index[i] for i in sorted(by_index)]
            write_csv(args.output_csv, rows, args.keep_raw_response)
            write_summary(args.summary_json, rows)
            print(
                f"{done_count}/{len(examples)} {row['image_filename']} true={row['true_label']} "
                f"pred={row.get('predicted_label') or 'ERROR'} ok={row.get('agreement')} "
                f"status={row.get('status')} elapsed_total={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
