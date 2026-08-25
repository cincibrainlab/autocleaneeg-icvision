"""Extract an ICLabel baseline from saved EEGLAB raw+ICA files.

This script is for the historical IC_Visual_AI Grace-reviewed dataset. It reads
Grace labels from ``updated_master_file.csv``, maps each image filename to the
corresponding ``SavedFiles/*.set`` + ``*.fdt`` pair, loads the saved ICA from
EEGLAB with MNE, runs ICLabel, and writes row-level agreement metrics.

It does not modify EEG/ICA data.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mne
from mne_icalabel import label_components


GRACE_LABELS = {
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
    "channel": "channel",
    "channel noise": "channel",
    "line noise": "line_noise",
    "other": "other",
}


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return GRACE_LABELS.get(text, text.replace(" ", "_") or "unknown")


def parse_image_name(image_filename: str) -> tuple[str, str, int]:
    match = re.match(
        r"(?P<subject>\d{4})_(?P<condition>vdaudio|vdnoaudio)_ica_comp_(?P<component>\d+)\.webp$",
        image_filename.lower(),
    )
    if not match:
        raise ValueError(f"Cannot parse image filename: {image_filename}")
    return match.group("subject"), match.group("condition"), int(match.group("component"))


def set_candidates(subject: str, condition: str) -> list[str]:
    condition_variants = {
        "vdaudio": ["VDaudio", "VDAudio", "VD_audio"],
        "vdnoaudio": ["VDnoaudio", "VDNoAudio", "VD_noaudio"],
    }[condition]
    return [f"{subject}_{variant}_ICA.set" for variant in condition_variants]


def find_set(saved_files_dir: Path, subject: str, condition: str) -> Path:
    existing = {path.name.lower(): path for path in saved_files_dir.glob("*.set")}
    for candidate in set_candidates(subject, condition):
        path = existing.get(candidate.lower())
        if path:
            return path
    raise FileNotFoundError(f"No .set found for {subject} {condition}")


def classify_set(set_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
    ica = mne.preprocessing.read_ica_eeglab(set_path, verbose="error")
    result = label_components(raw, ica, method="iclabel")
    elapsed = time.perf_counter() - started
    labels = list(result["labels"])
    confidence = result.get("y_pred_proba")
    confidence_values = [float(x) for x in confidence] if confidence is not None else [""] * len(labels)
    return {
        "labels": labels,
        "confidence": confidence_values,
        "n_components": len(labels),
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--saved-files-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        grace_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        grace_rows = grace_rows[: args.limit]

    grouped: dict[Path, list[dict[str, Any]]] = {}
    prepared: list[dict[str, Any]] = []
    for row in grace_rows:
        subject, condition, component_number = parse_image_name(row["image_filename"])
        set_path = find_set(args.saved_files_dir, subject, condition)
        item = {
            **row,
            "subject": subject,
            "condition": condition,
            "component_number": component_number,
            "component_index": component_number - 1,
            "set_path": str(set_path),
            "grace_label": normalize_label(row.get("compType")),
        }
        grouped.setdefault(set_path, []).append(item)
        prepared.append(item)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    classified: dict[Path, dict[str, Any]] = {}
    for index, set_path in enumerate(grouped, start=1):
        print(f"{index}/{len(grouped)} ICLabel {set_path}", flush=True)
        classified[set_path] = classify_set(set_path)

    fieldnames = [
        "image_filename",
        "subject",
        "condition",
        "component_number",
        "component_index",
        "set_path",
        "grace_label",
        "iclabel_label",
        "iclabel_confidence",
        "agreement",
        "status",
        "error",
    ]
    output_rows: list[dict[str, Any]] = []
    for item in prepared:
        set_path = Path(item["set_path"])
        component_index = int(item["component_index"])
        result = classified[set_path]
        out = {key: item.get(key, "") for key in fieldnames}
        out["status"] = "ok"
        out["error"] = ""
        if component_index < 0 or component_index >= result["n_components"]:
            out["status"] = "missing_component"
            out["error"] = f"component_index {component_index} outside 0..{result['n_components'] - 1}"
            out["iclabel_label"] = ""
            out["iclabel_confidence"] = ""
            out["agreement"] = ""
        else:
            iclabel_label = normalize_label(result["labels"][component_index])
            out["iclabel_label"] = iclabel_label
            out["iclabel_confidence"] = result["confidence"][component_index]
            out["agreement"] = iclabel_label == item["grace_label"]
        output_rows.append(out)

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    ok_rows = [row for row in output_rows if row["status"] == "ok"]
    correct = sum(1 for row in ok_rows if row["agreement"] is True)
    summary = {
        "baseline_type": "autoclean_iclabel_from_saved_eeglab_ica",
        "not_historical_saved_predictions": True,
        "labels_csv": str(args.labels_csv),
        "saved_files_dir": str(args.saved_files_dir),
        "rows_total": len(output_rows),
        "rows_ok": len(ok_rows),
        "rows_failed": len(output_rows) - len(ok_rows),
        "unique_set_files": len(grouped),
        "accuracy": correct / len(ok_rows) if ok_rows else None,
        "grace_distribution": dict(Counter(row["grace_label"] for row in ok_rows)),
        "iclabel_distribution": dict(Counter(row["iclabel_label"] for row in ok_rows)),
        "confusion": dict(
            Counter(f"{row['grace_label']}->{row['iclabel_label']}" for row in ok_rows)
        ),
        "notes": [
            "Uses saved EEGLAB ICA decomposition via mne.preprocessing.read_ica_eeglab.",
            "Maps image component numbers as 1-based; ICLabel arrays are indexed with component_number - 1.",
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
