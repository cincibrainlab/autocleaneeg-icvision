#!/usr/bin/env python3
"""Run one validated ClinCog accuracy-screen cell (never Azure)."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

import matplotlib

matplotlib.use("Agg")
import mne

mne.set_log_level("ERROR")
from icvision import api

BASE_DIR = Path(os.environ.get("ICVISION_GRACE_BASE_DIR", "/cblstore/srv/Analysis/Nate_Projects/Projects/IC_Visual_AI"))
CLINCOG_BASE_URL = "https://openai.cincibrainlab.com/v1"
SCREEN_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-daybreak-blue-latest", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark")
EFFORT_PAYLOADS = {"light": "low", "medium": "medium", "high": "high"}
FIELDS = ("set_path", "component_index", "true_label_norm", "predicted_label", "confidence", "reason")
VALID_LABELS = {"brain", "eye", "muscle", "heart", "channel_noise", "other_artifact"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, int]:
    return row["set_path"], int(row["component_index"])


def sanitize_csv_value(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    return "'" + text if text[:1] in "=+-@" else text


def validate_manifest_row(row: dict) -> None:
    set_path = str(row.get("set_path", ""))
    if "\\" in set_path:
        raise ValueError(f"Invalid set_path: {set_path!r}")
    path = PurePosixPath(set_path)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or path.parts[0] != "SavedFiles" or path.suffix != ".set":
        raise ValueError(f"Invalid set_path: {set_path!r}")
    component = int(row["component_index"])
    if component < 0:
        raise ValueError("component_index must be non-negative")
    if row["true_label_norm"] not in VALID_LABELS:
        raise ValueError(f"Invalid true_label_norm: {row['true_label_norm']!r}")


def validate_rows(rows: list[dict], expected: set[tuple[str, int]]) -> None:
    actual = [key(row) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected or len(set(actual)) != len(actual):
        raise ValueError("Incomplete, unexpected, or duplicate component keys")
    for row in rows:
        if row["predicted_label"] not in VALID_LABELS:
            raise ValueError(f"Invalid predicted label: {row['predicted_label']!r}")
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and within [0, 1]")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def publish(output: Path, metadata: Path, rows: list[dict], details: dict) -> None:
    validate_rows(rows, {tuple(item) for item in details["expected_keys"]})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: sanitize_csv_value(row[field]) for field in FIELDS})
    details = {**details, "status": "complete", "actual_rows": len(rows), "output_sha256": hashlib.sha256(buffer.getvalue().encode()).hexdigest()}
    atomic_write(output, buffer.getvalue())
    atomic_write(metadata, json.dumps(details, indent=2, sort_keys=True) + "\n")
    atomic_write(metadata.with_suffix(metadata.suffix + ".complete"), "complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SCREEN_MODELS, required=True)
    parser.add_argument("--effort", choices=EFFORT_PAYLOADS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--strip-size", type=int, default=9)
    args = parser.parse_args()
    if args.strip_size != 9:
        parser.error("The accuracy screen fixes strip size at 9")
    api_key = os.environ.get("CLINCOG_API_KEY")
    if not api_key:
        parser.error("Missing CLINCOG_API_KEY")
    manifest = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    for row in manifest:
        validate_manifest_row(row)
    expected = {key(row) for row in manifest}
    if not manifest or len(expected) != len(manifest):
        parser.error("Manifest is empty or contains duplicate keys")
    prompt = args.prompt_file.read_text(encoding="utf-8")
    details = {"model": args.model, "effort": args.effort, "gateway_effort": EFFORT_PAYLOADS[args.effort], "base_url": CLINCOG_BASE_URL, "manifest_sha256": sha256_file(args.manifest), "prompt_sha256": sha256_file(args.prompt_file), "strip_size": 9, "request_timeout_seconds": 120, "max_attempts": 3, "strict": True, "expected_keys": sorted(expected), "expected_rows": len(expected)}
    all_rows: list[dict] = []
    by_file: dict[str, list[dict]] = {}
    for row in manifest:
        by_file.setdefault(row["set_path"], []).append(row)
    try:
        for set_path, source_rows in sorted(by_file.items()):
            raw = mne.io.read_raw_eeglab(str(BASE_DIR / set_path), preload=True)
            ica = mne.preprocessing.read_ica_eeglab(str(BASE_DIR / set_path))
            truth = {int(row["component_index"]): row for row in source_rows}
            results, _ = api.classify_components_strip_batch(ica, raw, api_key, component_indices=sorted(truth), model_name=args.model, strip_size=9, base_url=CLINCOG_BASE_URL, auto_exclude=False, reasoning_effort=EFFORT_PAYLOADS[args.effort], custom_prompt=prompt, strict_mode=True, output_dir=args.output.parent / ".work" / f"{args.model}_{args.effort}_{Path(set_path).stem}")
            for _, row in results.iterrows():
                component = int(row["component_index"])
                all_rows.append({"set_path": set_path, "component_index": component, "true_label_norm": truth[component]["true_label_norm"], "predicted_label": str(row["label"]).lower(), "confidence": row["confidence"], "reason": row["reason"]})
        publish(args.output, args.metadata, all_rows, details)
    except Exception as error:
        atomic_write(args.metadata, json.dumps({**details, "status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
