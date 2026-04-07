#!/usr/bin/env python3
"""Run ICVision verification on mouse EEGLAB datasets stored inside a zip archive."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from test_icvision_lmstudio import configure_logging, require_dependencies, run_icvision_job


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ZIP_PATH = ROOT_DIR / "Sentinel_mouse_pre_ICA.zip"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "mouse_zip_verification"
DEFAULT_TEMP_DIR = ROOT_DIR / "output" / "mouse_zip_staging"
DEFAULT_MEMBER_HINT = "allego_13__uid1030-15-47-37"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one mouse .set file from Sentinel_mouse_pre_ICA.zip and run ICVision on it."
    )
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument(
        "--member-hint",
        default=DEFAULT_MEMBER_HINT,
        help="Substring used to select a .set member inside the zip archive.",
    )
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--classification-mode", choices=["human", "mouse"], default="mouse")
    parser.add_argument("--layout", choices=["single", "strip"], default="strip")
    parser.add_argument("--strip-size", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=45.0)
    parser.add_argument(
        "--ica-components",
        type=int,
        default=30,
        help="Maximum ICA components to fit. Default uses all 30 mouse EEG channels when available.",
    )
    parser.add_argument("--ica-method", default="fastica")
    parser.add_argument("--ica-random-state", type=int, default=97)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def select_member(zip_path: Path, member_hint: str) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.endswith(".set") and "._" not in name and member_hint in name
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No .set member matching '{member_hint}' was found in {zip_path}."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple .set members matched '{}':\n{}".format(member_hint, "\n".join(candidates))
        )
    return candidates[0]


def extract_member(zip_path: Path, member_name: str, temp_dir: Path) -> Path:
    staging_dir = temp_dir / Path(member_name).stem
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        prefix = member_name[: -len(".set")]
        related_members = [
            name
            for name in archive.namelist()
            if name.startswith(prefix)
            and "._" not in name
            and not name.endswith("/")
        ]
        if member_name not in related_members:
            related_members.append(member_name)
        for name in related_members:
            archive.extract(name, path=staging_dir)

    extracted_path = staging_dir / member_name
    if not extracted_path.exists():
        raise FileNotFoundError(f"Expected extracted .set file not found: {extracted_path}")
    return extracted_path


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    require_dependencies()

    if not args.zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {args.zip_path}")

    member_name = select_member(args.zip_path, args.member_hint)
    raw_path = extract_member(args.zip_path, member_name, args.temp_dir)

    run_args = argparse.Namespace(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        image=None,
        raw=raw_path,
        ica=None,
        output_dir=args.output_dir / Path(member_name).stem,
        classification_mode=args.classification_mode,
        compare_modes=False,
        layout=args.layout,
        strip_size=args.strip_size,
        confidence_threshold=args.confidence_threshold,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        ica_components=args.ica_components,
        ica_method=args.ica_method,
        ica_random_state=args.ica_random_state,
        no_report=args.no_report,
        check_only=False,
        verbose=args.verbose,
    )

    print("Mouse zip verification")
    print(f"  zip_path: {args.zip_path}")
    print(f"  member_name: {member_name}")
    print(f"  extracted_raw: {raw_path}")
    print(f"  output_dir: {run_args.output_dir}")

    run_icvision_job(run_args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
