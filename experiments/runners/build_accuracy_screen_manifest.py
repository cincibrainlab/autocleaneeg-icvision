#!/usr/bin/env python3
"""Build the fixed, subject-balanced 120-row accuracy-screen manifest.

The screen is a pre-specified triage sample, not an independent-subject
accuracy estimate.  It selects 9 rows from the only 9-row recording, 10 rows
from ten recordings, and 11 rows from one seed-selected eligible recording.
Within each recording, selection is stratified proportionally by the existing
Grace label distribution.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ALGORITHM_VERSION = "subject-balanced-proportional-v1"
DEFAULT_SEED = 20260820


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_expected_source_digest(source: Path, expected_digest: str) -> str:
    """Reject a changed or unspecified input before selecting or writing rows."""
    expected = expected_digest.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("--expected-source-sha256 must be a 64-character SHA-256 hex digest")
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(
            f"Frozen source digest mismatch for {source}: expected {expected}, got {actual}"
        )
    return actual


def key(row: dict[str, str]) -> tuple[str, int]:
    return row["set_path"], int(row["component_index"])


def proportional_counts(groups: dict[str, list[dict[str, str]]], target: int) -> dict[str, int]:
    """Allocate target rows across labels using deterministic largest remainder."""
    total = sum(len(rows) for rows in groups.values())
    if target > total:
        raise ValueError(f"Cannot select {target} rows from a {total}-row recording")
    raw = {label: target * len(rows) / total for label, rows in groups.items()}
    allocated = {label: min(len(groups[label]), int(value)) for label, value in raw.items()}
    remaining = target - sum(allocated.values())
    order = sorted(
        groups,
        key=lambda label: (-(raw[label] - allocated[label]), label),
    )
    for label in order:
        if not remaining:
            break
        if allocated[label] < len(groups[label]):
            allocated[label] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not allocate requested subject quota")
    return allocated


def build_screen(rows: list[dict[str, str]], seed: int = DEFAULT_SEED) -> list[dict[str, str]]:
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_file[row["set_path"]].append(row)
    if len(by_file) != 12 or len(rows) != 679:
        raise ValueError(f"Expected the frozen 679-row/12-recording ground truth, got {len(rows)} rows/{len(by_file)} recordings")

    nine_files = sorted(path for path, file_rows in by_file.items() if len(file_rows) == 9)
    if len(nine_files) != 1:
        raise ValueError(f"Expected exactly one 9-row recording, got {nine_files}")
    eligible_for_eleven = sorted(path for path, file_rows in by_file.items() if len(file_rows) >= 11 and path not in nine_files)
    eleven_file = random.Random(seed).choice(eligible_for_eleven)
    quotas = {path: 9 if path == nine_files[0] else 11 if path == eleven_file else 10 for path in by_file}

    selected: list[dict[str, str]] = []
    for path in sorted(by_file):
        labels: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sorted(by_file[path], key=lambda item: (item["true_label_norm"], int(item["component_index"]))):
            labels[row["true_label_norm"]].append(row)
        counts = proportional_counts(labels, quotas[path])
        rng = random.Random(f"{seed}:{path}")
        for label in sorted(labels):
            candidates = labels[label][:]
            rng.shuffle(candidates)
            selected.extend(candidates[: counts[label]])

    selected.sort(key=key)
    if len(selected) != 120 or len({key(row) for row in selected}) != 120:
        raise AssertionError("Screen must contain exactly 120 unique component keys")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--screen-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument(
        "--expected-source-sha256",
        required=True,
        help="Frozen SHA-256 of --source; verified before selection or output writes.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    source_sha256 = require_expected_source_digest(args.source, args.expected_source_sha256)

    with args.source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = handle.seek(0) or csv.DictReader(handle).fieldnames
    required = {"set_path", "component_index", "true_label_norm"}
    if not required.issubset(fields or []):
        raise ValueError(f"Source must contain {sorted(required)}")
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError("Source contains duplicate component keys")

    screen = build_screen(rows, args.seed)
    screen_keys = {key(row) for row in screen}
    holdout = [row for row in rows if key(row) not in screen_keys]
    if len(holdout) != 559:
        raise AssertionError(f"Expected 559 held-out rows, got {len(holdout)}")

    for output, output_rows in ((args.screen_output, screen), (args.holdout_output, holdout)):
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)

    metadata = {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": args.seed,
        "source": str(args.source),
        "source_sha256": source_sha256,
        "screen_sha256": sha256_file(args.screen_output),
        "holdout_sha256": sha256_file(args.holdout_output),
        "screen_rows": len(screen),
        "holdout_rows": len(holdout),
        "recording_allocation": dict(sorted(Counter(row["set_path"] for row in screen).items())),
        "screen_label_counts": dict(sorted(Counter(row["true_label_norm"] for row in screen).items())),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
