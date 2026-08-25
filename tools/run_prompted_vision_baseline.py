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
import base64
import csv
import hashlib
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




ALLOWED_EVIDENCE_FIELDS_BY_MODE = {
    "none": set(),
    "channel_loading_v1": {
        "evidence_schema_version",
        "top_channel_label",
        "top_channel_index_one_based",
        "largest_abs_loading",
        "second_largest_abs_loading",
        "largest_to_second_largest_ratio",
        "channels_above_50pct_of_max",
        "channels_above_25pct_of_max",
        "nearest_neighbor_support_ratio_k4",
    },
}

DISALLOWED_EVIDENCE_KEYS = {
    "true_label",
    "true_label_raw",
    "reference_label",
    "compType",
    "classification",
    "label",
    "predicted_label",
    "prediction",
    "agreement",
    "correct",
    "grace",
}


def evidence_payload_hash(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence_payload(payload: dict[str, Any], *, row_id: str, evidence_mode: str = "none") -> None:
    allowed = ALLOWED_EVIDENCE_FIELDS_BY_MODE.get(evidence_mode)
    if allowed is None:
        raise ValueError(f"Unknown evidence mode: {evidence_mode}")
    if evidence_mode == "none":
        raise ValueError(f"Evidence payload supplied while evidence_mode is none for {row_id}")
    for key, value in payload.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if key_text not in allowed:
            raise ValueError(f"Evidence payload for {row_id} contains non-allowlisted key for {evidence_mode}: {key}")
        if isinstance(value, (dict, list)):
            raise ValueError(f"Evidence payload for {row_id} contains nested value for {key}")
        if key_lower == "top_channel_label":
            continue
        if key_text in DISALLOWED_EVIDENCE_KEYS or key_lower in {item.lower() for item in DISALLOWED_EVIDENCE_KEYS}:
            raise ValueError(f"Evidence payload for {row_id} contains disallowed key: {key}")
        if any(fragment in key_lower for fragment in ("grace", "reference", "predicted", "prediction", "agreement", "correct")):
            raise ValueError(f"Evidence payload for {row_id} contains disallowed key: {key}")
    missing = allowed - set(payload)
    if missing:
        raise ValueError(f"Evidence payload for {row_id} is missing required keys for {evidence_mode}: {sorted(missing)}")


def load_evidence_jsonl(path: Path, evidence_mode: str) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Evidence line {line_number} is not an object")
            image_filename = str(payload.get("image_filename") or "").strip()
            row_key = str(payload.get("row_key") or image_filename).strip()
            if not image_filename and not row_key:
                raise ValueError(f"Evidence line {line_number} is missing image_filename/row_key")
            evidence_payload = payload.get("evidence", payload)
            if not isinstance(evidence_payload, dict):
                raise ValueError(f"Evidence line {line_number} evidence is not an object")
            validate_evidence_payload(evidence_payload, row_id=row_key or image_filename, evidence_mode=evidence_mode)
            for key in {row_key, image_filename} - {""}:
                if key in evidence:
                    raise ValueError(f"Duplicate evidence key: {key}")
                evidence[key] = evidence_payload
    if not evidence:
        raise ValueError("Evidence JSONL has no rows")
    return evidence


def apply_evidence(examples: list[dict[str, Any]], evidence: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for example in examples:
        row_key = example.get("row_key") or example["image_filename"]
        payload = evidence.get(str(row_key)) or evidence.get(example["image_filename"])
        if payload is None:
            raise ValueError(f"Missing evidence for {example['image_filename']}")
        enriched.append(
            {
                **example,
                "evidence_payload": payload,
                "evidence_payload_sha256": evidence_payload_hash(payload),
            }
        )
    return enriched


def webp_data_url(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/webp;base64," + base64.b64encode(data).decode("ascii")


def apply_image_root(examples: list[dict[str, Any]], image_root: Path) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for example in examples:
        image_path = image_root / example["image_filename"]
        if not image_path.is_file():
            raise ValueError(f"Image root is missing {example['image_filename']}")
        rewritten.append(
            {
                **example,
                "image_url": webp_data_url(image_path),
                "image_sha256": sha256_file(image_path),
                "image_transport": "data_url",
            }
        )
    return rewritten


def build_input_content(prompt: str, image_url: str, evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image_url, "detail": "low"},
    ]
    if evidence is not None:
        content.append(
            {
                "type": "input_text",
                "text": (
                    "AUXILIARY_EVIDENCE_JSON\n"
                    "The following values are auxiliary observations from the same ICA component, "
                    "not reference labels or prior model answers.\n"
                    + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
                ),
            }
        )
    return content




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        if "image_filename" not in fieldnames:
            raise ValueError("Manifest is missing required column: image_filename")
        label_column = "true_label" if "true_label" in fieldnames else "compType" if "compType" in fieldnames else None
        component_column = (
            "component_number" if "component_number" in fieldnames else "compNum" if "compNum" in fieldnames else None
        )
        if label_column is None:
            raise ValueError("Manifest is missing true_label/compType column")
        if component_column is None:
            raise ValueError("Manifest is missing component_number/compNum column")
        rows = []
        seen: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            filename = (row.get("image_filename") or "").strip()
            true_label = normalize_label(row.get(label_column))
            if not filename:
                raise ValueError(f"Manifest line {line_number} has empty image_filename")
            if filename in seen:
                raise ValueError(f"Manifest contains duplicate image_filename: {filename}")
            if true_label not in set(LABEL_MAP.values()):
                raise ValueError(f"Manifest line {line_number} has invalid true_label: {row.get(label_column)!r}")
            seen.add(filename)
            rows.append(
                {
                    "image_filename": filename,
                    "component_number": (row.get(component_column) or "").strip(),
                    "true_label": true_label,
                    "subset_memberships": (row.get("subset_memberships") or "").strip(),
                }
            )
    if not rows:
        raise ValueError("Manifest has no rows")
    return rows


def apply_locked_manifest(examples: list[dict[str, Any]], manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_filename: dict[str, dict[str, Any]] = {}
    for example in examples:
        filename = example["image_filename"]
        if filename in by_filename:
            raise ValueError(f"Historical JSONL contains duplicate image_filename: {filename}")
        by_filename[filename] = example

    filtered: list[dict[str, Any]] = []
    for row in manifest_rows:
        filename = row["image_filename"]
        example = by_filename.get(filename)
        if example is None:
            raise ValueError(f"Manifest image not found in JSONL examples: {filename}")
        if normalize_label(example.get("true_label")) != row["true_label"]:
            raise ValueError(
                f"Reference label mismatch for {filename}: manifest={row['true_label']} jsonl={example.get('true_label')}"
            )
        filtered.append(
            {
                **example,
                "true_label": row["true_label"],
                "component_number": row["component_number"],
                "subset_memberships": row["subset_memberships"],
            }
        )
    return filtered


def call_responses_with_curl(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    image_url: str,
    timeout: int,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": build_input_content(prompt, image_url, evidence),
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
    response = call_responses_with_curl(
        api_key,
        base_url,
        model,
        prompt,
        example["image_url"],
        timeout,
        example.get("evidence_payload"),
    )
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
        "evidence_payload_sha256": example.get("evidence_payload_sha256", ""),
        "image_sha256": example.get("image_sha256", ""),
        "image_transport": example.get("image_transport", "url"),
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
        "evidence_payload_sha256": example.get("evidence_payload_sha256", ""),
        "image_sha256": example.get("image_sha256", ""),
        "image_transport": example.get("image_transport", "url"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], keep_raw: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "line_number",
        "image_filename",
        "image_url",
        "image_sha256",
        "image_transport",
        "component_number",
        "subset_memberships",
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
        "evidence_payload_sha256",
    ]
    if keep_raw:
        fieldnames.append("raw_response")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
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
    if metadata:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, help="Optional locked private manifest selecting image_filename rows to run.")
    parser.add_argument("--prompt-variant", default="unspecified", help="Run metadata label for the prompt variant.")
    parser.add_argument("--evidence-jsonl", type=Path, help="Optional label-free JSONL evidence payloads keyed by image_filename.")
    parser.add_argument("--evidence-mode", default="none", help="Run metadata label for the evidence mode.")
    parser.add_argument("--image-root", type=Path, help="Optional directory of local WebP images to send as data URLs instead of historical URLs.")
    parser.add_argument("--image-variant-name", default="historical_url", help="Run metadata label for image transport/variant.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and metadata without calling the model.")
    parser.add_argument("--skip-preflight-first", action="store_true", help="Disable the single-row preflight before concurrent requests.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--keep-raw-response", action="store_true")
    args = parser.parse_args()

    prompt = args.prompt_file.read_text(encoding="utf-8")
    examples = load_examples([Path(item) for item in args.jsonl])
    manifest_hash = None
    if args.manifest_csv is not None:
        manifest_hash = sha256_file(args.manifest_csv)
        examples = apply_locked_manifest(examples, load_locked_manifest(args.manifest_csv))
    if args.manifest_csv is not None and args.limit is not None:
        parser.error("--limit cannot be combined with --manifest-csv")
    if args.limit is not None:
        examples = examples[: args.limit]
    if args.evidence_jsonl is not None:
        examples = apply_evidence(examples, load_evidence_jsonl(args.evidence_jsonl, args.evidence_mode))
    if args.image_root is not None:
        examples = apply_image_root(examples, args.image_root)

    metadata = {
        "model": args.model,
        "prompt_variant": args.prompt_variant,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "manifest_sha256": manifest_hash,
        "rows_selected": len(examples),
        "evidence_mode": args.evidence_mode,
        "evidence_jsonl_sha256": sha256_file(args.evidence_jsonl) if args.evidence_jsonl else None,
        "image_variant_name": args.image_variant_name,
        "image_root": str(args.image_root) if args.image_root else None,
    }

    if args.validate_only:
        write_summary(args.summary_json, [], metadata)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.output_csv.write_text("", encoding="utf-8")
        print(json.dumps({"validated_rows": len(examples), "metadata": metadata}, indent=2), flush=True)
        return 0

    if "ai.clincognition.com" in args.base_url:
        api_key = os.environ.get("CLINCOG_API_KEY")
        if not api_key:
            raise RuntimeError("CLINCOG_API_KEY is required for the ClinCog endpoint")
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or OPENAI_KEY is not set")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    by_index: dict[int, dict[str, Any]] = {}

    def record_progress(done_count: int, index: int, row: dict[str, Any]) -> None:
        by_index[index] = row
        current_rows = [by_index[i] for i in sorted(by_index)]
        write_csv(args.output_csv, current_rows, args.keep_raw_response)
        write_summary(args.summary_json, current_rows, metadata)
        if args.manifest_csv is not None:
            progress = (
                f"{done_count}/{len(examples)} status={row.get('status')} "
                f"agreement={row.get('agreement')} elapsed_total={time.perf_counter() - started:.1f}s"
            )
        else:
            progress = (
                f"{done_count}/{len(examples)} {row['image_filename']} true={row['true_label']} "
                f"pred={row.get('predicted_label') or 'ERROR'} ok={row.get('agreement')} "
                f"status={row.get('status')} elapsed_total={time.perf_counter() - started:.1f}s"
            )
        print(progress, flush=True)

    start_index = 0
    if examples and not args.skip_preflight_first:
        try:
            first_row = classify_one(
                examples[0],
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                timeout=args.timeout,
                keep_raw=args.keep_raw_response,
            )
        except Exception as exc:
            first_row = error_row(examples[0], args.model, exc)
            record_progress(1, 0, first_row)
            raise RuntimeError(f"Preflight request failed; not launching remaining {len(examples) - 1} requests: {first_row['error']}") from exc
        record_progress(1, 0, first_row)
        start_index = 1

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
            for index, example in enumerate(examples[start_index:], start=start_index)
        }
        for done_offset, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            example = examples[index]
            try:
                row = future.result()
            except Exception as exc:
                row = error_row(example, args.model, exc)
            record_progress(start_index + done_offset, index, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
