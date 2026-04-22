#!/usr/bin/env python3
"""Run ICVision verification on mouse EEGLAB datasets stored inside a zip archive."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from test_icvision_lmstudio import configure_logging, require_dependencies, run_icvision_job


ROOT_DIR = Path(__file__).resolve().parent.parent
SECRETS_DIR = ROOT_DIR / "secrets"
SECRET_SETTINGS_PATH = SECRETS_DIR / "config.txt"
DEFAULT_ZIP_PATH = ROOT_DIR / "Sentinel_mouse_pre_ICA.zip"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "mouse_zip_verification"
DEFAULT_TEMP_DIR = ROOT_DIR / "output" / "mouse_zip_staging"
DEFAULT_MEMBER_HINT = "allego_12__uid0209-14-00-51"


def _extract_value(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_api_key(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("sk-") and "\n" not in stripped:
        return stripped
    return _extract_value(r"(sk-[A-Za-z0-9._-]+)", text)


def load_secret_settings() -> dict[str, str]:
    if not SECRET_SETTINGS_PATH.exists():
        return {}

    settings: dict[str, str] = {}
    for line in SECRET_SETTINGS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings[key.strip()] = value.strip()
    return settings


def _resolve_secret_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else SECRETS_DIR / path


def load_cliproxy_defaults(settings: dict[str, str]) -> tuple[str, str]:
    env_base_url = os.environ.get("OPENAI_BASE_URL")
    env_api_key = os.environ.get("OPENAI_API_KEY")
    if env_base_url and env_api_key:
        return env_base_url, env_api_key

    key_paths = []
    for settings_key in ("key_file", "legacy_key_file"):
        key_file = settings.get(settings_key)
        if key_file:
            key_paths.append(_resolve_secret_path(key_file))

    for key_path in key_paths:
        if not key_path.exists():
            continue
        key_text = key_path.read_text(encoding="utf-8")
        file_api_key = _extract_api_key(key_text)
        if file_api_key:
            base_url = env_base_url or settings.get("cliproxy_base_url", "")
            api_key = env_api_key or file_api_key
            if base_url:
                return base_url, api_key

    fallback_base_url = env_base_url or settings.get("default_base_url", "")
    fallback_api_key = env_api_key or settings.get("default_api_key", "")
    if fallback_base_url and fallback_api_key:
        return fallback_base_url, fallback_api_key

    raise RuntimeError(
        f"Missing API defaults. Set OPENAI_BASE_URL/OPENAI_API_KEY or populate {SECRET_SETTINGS_PATH}."
    )


def parse_args() -> argparse.Namespace:
    settings = load_secret_settings()
    default_base_url, default_api_key = load_cliproxy_defaults(settings)
    default_model = os.environ.get("OPENAI_MODEL") or settings.get("default_model")
    if not default_model:
        raise RuntimeError(
            f"Missing model default. Set OPENAI_MODEL or populate {SECRET_SETTINGS_PATH}."
        )

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
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument("--api-key", default=default_api_key)
    parser.add_argument("--model", default=default_model)
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
