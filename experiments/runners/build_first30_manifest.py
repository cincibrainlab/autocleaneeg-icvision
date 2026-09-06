#!/usr/bin/env python3
"""Build the first-30 contiguous manifest from a source manifest.

Takes every component with component_index <= --max-index from the source
CSV (set_path, component_index, true_label_norm) and writes a study
manifest, plus a distribution/skew report the scientists require before
any scoring.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-index", type=int, default=30)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.source.open(newline="", encoding="utf-8")))
    kept = [r for r in rows if int(r["component_index"]) <= args.max_index]
    kept.sort(key=lambda r: (r["set_path"], int(r["component_index"])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("set_path", "component_index", "true_label_norm"))
        writer.writeheader()
        writer.writerows(kept)

    by_file = defaultdict(list)
    for r in kept:
        by_file[r["set_path"]].append(int(r["component_index"]))
    class_counts = Counter(r["true_label_norm"] for r in kept)
    total = len(kept)

    print(f"wrote {total} rows -> {args.out}")
    print(f"\nPer-file counts (max-index <= {args.max_index}):")
    for f in sorted(by_file):
        print(f"  {Path(f).name:<28} {len(by_file[f]):>4}  idx {min(by_file[f])}-{max(by_file[f])}")
    print("\nClass distribution (skew report):")
    for label, n in class_counts.most_common():
        print(f"  {label:<16} {n:>4}  ({n / total:.1%})")
    print("\nPer-file early-class mix (files flagged where any class is absent):")
    all_labels = set(class_counts)
    for f in sorted(by_file):
        mix = Counter(r["true_label_norm"] for r in kept if r["set_path"] == f)
        missing = all_labels - set(mix)
        if missing:
            print(f"  {Path(f).name:<28} missing: {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
