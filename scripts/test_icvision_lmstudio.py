"""End-to-end LM Studio test runner for the local ICVision repo."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEFAULT_TEST_DATA_DIR = Path("/Users/sueo8x/Documents/testeegdata")
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "lmstudio_test_run"
DEFAULT_RAW_CANDIDATES = [
    DEFAULT_TEST_DATA_DIR / "resting_eyes_open.set",
    DEFAULT_TEST_DATA_DIR / "hbcd_mmn.set",
    DEFAULT_TEST_DATA_DIR / "mouse_assr.set",
    DEFAULT_TEST_DATA_DIR / "allego_0__uid0122-13-00-26_data.set",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LM Studio and optionally run a real ICVision job against this repo."
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "lm-studio"))
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--image", type=Path, default=None, help="Optional image path for a vision smoke test.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Raw EEG file to classify. If omitted, the script uses a default file from Documents/testeegdata.",
    )
    parser.add_argument("--ica", type=Path, default=None, help="Optional ICA file if not embedded in raw data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layout", choices=["single", "strip"], default="strip")
    parser.add_argument("--strip-size", type=int, default=9)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--l-freq", type=float, default=1.0, help="High-pass filter cutoff in Hz.")
    parser.add_argument("--h-freq", type=float, default=45.0, help="Low-pass filter cutoff in Hz.")
    parser.add_argument(
        "--ica-components",
        type=int,
        default=20,
        help="Maximum number of ICA components to fit. The script will cap this to the number of EEG channels.",
    )
    parser.add_argument("--ica-method", default="fastica", help="MNE ICA method. Default: fastica.")
    parser.add_argument("--ica-random-state", type=int, default=97, help="Random seed for ICA fitting.")
    parser.add_argument("--no-report", action="store_true", help="Disable PDF report generation.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run LM Studio checks. Do not run ICVision even if --raw is provided.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Show ICVision progress logs during long local runs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def require_dependencies() -> None:
    missing = []
    for module_name in ("openai", "mne", "pandas", "numpy", "matplotlib", "sklearn", "PIL"):
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing Python dependencies: "
            f"{missing_str}\n"
            "Install them into your current Python first, for example:\n"
            "  pip install -e '.[dev,test]'"
        )


def resolve_default_raw_path(raw_path: Optional[Path]) -> Path:
    """Pick a default local EEG file when one is not provided."""
    if raw_path is not None:
        return raw_path

    for candidate in DEFAULT_RAW_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No default EEG file was found in /Users/sueo8x/Documents/testeegdata. "
        "Pass --raw PATH explicitly."
    )


def encode_image(image_path: Path) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def list_models(client) -> list[str]:
    models = client.models.list()
    return [model.id for model in models.data]


def run_text_smoke_test(client, model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def run_vision_smoke_test(client, model: str, image_path: Path) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one short sentence."},
                    {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def load_and_prepare_raw(raw_path: Path, l_freq: float, h_freq: float):
    """Load raw EEG with the repo loader and apply basic filtering."""
    from icvision.utils import load_raw_data

    raw = load_raw_data(raw_path)
    print("\nPreparing raw EEG with MNE...")
    print(f"  input file: {raw_path}")
    print(f"  original sfreq: {raw.info['sfreq']}")
    print(f"  channels: {len(raw.ch_names)}")
    print(f"  applying filter: {l_freq}-{h_freq} Hz")

    raw_prepped = raw.copy()
    raw_prepped.load_data()
    raw_prepped.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw_prepped


def fit_ica(raw, n_components: int, method: str, random_state: int):
    """Fit ICA on EEG channels using a simple MNE configuration."""
    import mne

    picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, exclude="bads")
    if len(picks) < 2:
        raise RuntimeError("Need at least 2 EEG channels to fit ICA.")

    max_components = min(n_components, len(picks))
    print("\nFitting ICA with MNE...")
    print(f"  method: {method}")
    print(f"  eeg channels used: {len(picks)}")
    print(f"  n_components: {max_components}")

    ica = mne.preprocessing.ICA(
        n_components=max_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, picks=picks, verbose=False)
    return ica


def run_icvision_job(args: argparse.Namespace) -> None:
    from icvision.core import label_components

    raw_path = resolve_default_raw_path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw EEG file not found: {raw_path}")
    if args.ica is not None and not args.ica.exists():
        raise FileNotFoundError(f"ICA file not found: {args.ica}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ["OPENAI_API_KEY"] = args.api_key

    raw_prepped = load_and_prepare_raw(raw_path, args.l_freq, args.h_freq)
    ica = fit_ica(raw_prepped, args.ica_components, args.ica_method, args.ica_random_state)

    print("\nRunning ICVision on your dataset...")
    print(f"  raw: {raw_path}")
    print("  ica: fitted in script with MNE")
    print(f"  model: {args.model}")
    print(f"  output_dir: {args.output_dir}")

    raw_cleaned, ica_updated, results_df = label_components(
        raw_data=raw_prepped,
        ica_data=ica,
        api_key=args.api_key,
        model_name=args.model,
        output_dir=args.output_dir,
        generate_report=not args.no_report,
        layout=args.layout,
        strip_size=args.strip_size,
        confidence_threshold=args.confidence_threshold,
        base_url=args.base_url,
    )

    print("\nICVision run complete.")
    print(f"  cleaned raw type: {type(raw_cleaned).__name__}")
    print(f"  updated ICA type: {type(ica_updated).__name__}")
    print(f"  classified components: {len(results_df)}")
    print(f"  results csv: {args.output_dir}")

    preview = results_df[["component_index", "label", "confidence"]].head(10)
    print("\nPreview:")
    print(preview.to_string(index=False))


def main() -> int:
    args = parse_args()

    try:
        configure_logging(args.verbose)
        require_dependencies()

        if args.image is not None and not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")

        client = get_client(args.base_url, args.api_key)
        model_ids = list_models(client)

        print("LM Studio endpoint check")
        print(f"  base_url: {args.base_url}")
        print(f"  requested model: {args.model}")
        print(f"  available models: {json.dumps(model_ids, indent=2)}")

        if args.model not in model_ids:
            raise RuntimeError(f"Model '{args.model}' was not returned by LM Studio.")

        text_result = run_text_smoke_test(client, args.model)
        print("\nText smoke test")
        print(f"  response: {text_result}")

        if args.image is not None:
            vision_result = run_vision_smoke_test(client, args.model, args.image)
            print("\nVision smoke test")
            print(f"  image: {args.image}")
            print(f"  response: {vision_result}")
        else:
            print("\nVision smoke test skipped")
            print("  pass --image PATH if you want to verify multimodal requests first")

        if args.check_only:
            print("\nCheck-only mode complete.")
            return 0

        run_icvision_job(args)
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
