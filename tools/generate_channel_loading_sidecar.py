"""Generate label-free channel-loading evidence for ICA component images.

This private evaluation helper reads EEGLAB .set files and writes JSONL evidence
payloads keyed by historical ICVision image filenames. It does not read, write,
or modify EEG samples, ICA exclusions, or Grace labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def parse_image_filename(filename: str) -> tuple[str, str, int]:
    match = re.match(r"^(\d+)_vd(no)?audio_ica_comp_(\d+)\.webp$", filename.lower())
    if not match:
        raise ValueError(f"Unrecognized image filename format: {filename}")
    source_id = match.group(1)
    condition = "vdnoaudio" if match.group(2) else "vdaudio"
    component_number = int(match.group(3))
    return source_id, condition, component_number


def find_set_file(set_dir: Path, source_id: str, condition: str) -> Path:
    candidates = []
    for path in set_dir.glob(f"{source_id}*_ICA.set"):
        name = compact(path.name)
        if condition == "vdnoaudio":
            if "vdnoaudio" in name:
                candidates.append(path)
        elif condition == "vdaudio":
            if "vdaudio" in name and "vdnoaudio" not in name:
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one .set for {source_id} {condition}, found {[str(p) for p in candidates]}")
    return candidates[0]


def mat_get(mat: dict[str, Any], name: str) -> Any:
    if name in mat:
        return mat[name]
    eeg = mat.get("EEG")
    if eeg is not None:
        return getattr(eeg, name, None)
    return None


def chanloc_value(chan: Any, name: str) -> Any:
    return getattr(chan, name, None)


def channel_metadata(chanlocs: Any) -> list[dict[str, Any]]:
    meta = []
    for idx, chan in enumerate(np.atleast_1d(chanlocs)):
        label = str(chanloc_value(chan, "labels") or f"chan{idx + 1}")
        coords = []
        for axis in ("X", "Y", "Z"):
            value = chanloc_value(chan, axis)
            try:
                coords.append(float(value))
            except (TypeError, ValueError):
                coords.append(float("nan"))
        meta.append({"label": label, "xyz": coords})
    return meta


def nearest_neighbor_support(abs_loadings: np.ndarray, channels: list[dict[str, Any]], top_index: int, k: int = 4) -> float | None:
    top_xyz = np.array(channels[top_index]["xyz"], dtype=float)
    if not np.isfinite(top_xyz).all():
        return None
    distances = []
    for idx, item in enumerate(channels):
        if idx == top_index:
            continue
        xyz = np.array(item["xyz"], dtype=float)
        if not np.isfinite(xyz).all():
            continue
        distances.append((float(np.linalg.norm(xyz - top_xyz)), idx))
    if not distances:
        return None
    nearest = [idx for _, idx in sorted(distances)[:k]]
    max_loading = float(abs_loadings[top_index])
    if max_loading <= 0:
        return None
    return float(np.mean(abs_loadings[nearest]) / max_loading)


def component_evidence(set_path: Path, component_number: int) -> dict[str, Any]:
    mat = sio.loadmat(set_path, squeeze_me=True, struct_as_record=False)
    icawinv = mat_get(mat, "icawinv")
    chanlocs = mat_get(mat, "chanlocs")
    if icawinv is None:
        raise ValueError(f"Missing icawinv in {set_path}")
    matrix = np.asarray(icawinv, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"icawinv is not 2D in {set_path}: {matrix.shape}")
    component_index_zero_based = component_number - 1
    if component_index_zero_based < 0 or component_index_zero_based >= matrix.shape[1]:
        raise ValueError(f"Component {component_number} out of range for {set_path} with {matrix.shape[1]} components")
    loadings = matrix[:, component_index_zero_based]
    abs_loadings = np.abs(loadings)
    order = np.argsort(abs_loadings)[::-1]
    top_index = int(order[0])
    second = float(abs_loadings[order[1]]) if len(order) > 1 else 0.0
    largest = float(abs_loadings[top_index])
    ratio = largest / second if second > 0 else None
    channels = channel_metadata(chanlocs) if chanlocs is not None else [{"label": f"chan{i + 1}", "xyz": [math.nan, math.nan, math.nan]} for i in range(matrix.shape[0])]
    support = nearest_neighbor_support(abs_loadings, channels, top_index)
    above_half = int(np.sum(abs_loadings >= largest * 0.5)) if largest > 0 else 0
    above_quarter = int(np.sum(abs_loadings >= largest * 0.25)) if largest > 0 else 0
    return {
        "evidence_schema_version": "channel_loading_v1",
        "top_channel_label": channels[top_index]["label"],
        "top_channel_index_one_based": top_index + 1,
        "largest_abs_loading": round(largest, 8),
        "second_largest_abs_loading": round(second, 8),
        "largest_to_second_largest_ratio": round(ratio, 8) if ratio is not None else None,
        "channels_above_50pct_of_max": above_half,
        "channels_above_25pct_of_max": above_quarter,
        "nearest_neighbor_support_ratio_k4": round(support, 8) if support is not None else None,
    }


def read_input_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_filename" not in reader.fieldnames:
            raise ValueError("Input CSV must contain image_filename")
        rows = []
        seen = set()
        for row in reader:
            filename = (row.get("image_filename") or "").strip()
            if not filename or filename in seen:
                continue
            parse_image_filename(filename)
            seen.add(filename)
            rows.append({"image_filename": filename})
    if not rows:
        raise ValueError("Input CSV produced no rows")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_input_rows(args.input_csv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    set_cache: dict[Path, dict[int, dict[str, Any]]] = {}
    output_rows = []
    set_files = Counter()
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            filename = row["image_filename"]
            source_id, condition, component_number = parse_image_filename(filename)
            set_path = find_set_file(args.set_dir, source_id, condition)
            set_files[str(set_path)] += 1
            payload = component_evidence(set_path, component_number)
            record = {"row_key": filename, "image_filename": filename, "evidence": payload}
            out.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            output_rows.append(record)
    summary = {
        "rows": len(output_rows),
        "evidence_schema_version": "channel_loading_v1",
        "output_jsonl_sha256": sha256_file(args.output_jsonl),
        "unique_set_files": len(set_files),
        "set_file_row_counts": {Path(path).name: count for path, count in sorted(set_files.items())},
        "set_file_sha256_by_basename": {Path(path).name: sha256_file(Path(path)) for path in sorted(set_files)},
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
