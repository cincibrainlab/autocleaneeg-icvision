#!/usr/bin/env python3
"""Schedule fixed accuracy-screen cells; dry-run unless --launch is set."""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

# Keep the scheduler import-free: `run_screen` intentionally imports the
# scientific stack, which is not needed to inspect a dry-run schedule.
SCREEN_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-daybreak-blue-latest", "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex-spark")
EFFORT_PAYLOADS = {"light": "low", "medium": "medium", "high": "high"}
CELL_TIMEOUT_SECONDS = 3600
ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()


def resolve_output_dir(root: Path, output_dir: Path) -> Path:
    return output_dir if output_dir.is_absolute() else root / output_dir


def command(root: Path, output_dir: Path, model: str, effort: str) -> list[str]:
    cell = f"{model}_{effort}"
    resolved_output_dir = resolve_output_dir(root, output_dir)
    return [sys.executable, str(Path(__file__).with_name("run_screen.py")), "--model", model, "--effort", effort, "--manifest", str(root / "experiments/manifests/accuracy_screen_120.csv"), "--prompt-file", str(root / "prompts/tightened_v1_strip.txt"), "--output", str(resolved_output_dir / f"{cell}.csv"), "--metadata", str(resolved_output_dir / f"{cell}.metadata.json")]


def child_env() -> dict[str, str]:
    keep = (
        "CLINCOG_API_KEY",
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "ICVISION_GRACE_BASE_DIR",
        "MPLCONFIGDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HOME",
    )
    return {name: value for name in keep if (value := os.environ.get(name))}


def terminate_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def terminate_active_processes() -> None:
    with ACTIVE_LOCK:
        processes = list(ACTIVE_PROCESSES.values())
    for process in processes:
        if process.poll() is None:
            terminate_process_group(process, signal.SIGTERM)


def run_command(cmd: list[str], root: Path, cell: str) -> int:
    process = subprocess.Popen(cmd, cwd=root, env=child_env(), start_new_session=True)
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES[cell] = process
    try:
        return process.wait(timeout=CELL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        terminate_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            terminate_process_group(process, signal.SIGKILL)
            process.wait()
        return 124
    finally:
        with ACTIVE_LOCK:
            ACTIVE_PROCESSES.pop(cell, None)


def run_cell(root: Path, output_dir: Path, model: str, effort: str) -> tuple[str, bool, bool]:
    cell = f"{model}_{effort}"
    resolved_output_dir = resolve_output_dir(root, output_dir)
    metadata = resolved_output_dir / f"{cell}.metadata.json"
    complete_marker = metadata.with_suffix(metadata.suffix + ".complete")
    if complete_marker.exists():
        return cell, True, False
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    lock = resolved_output_dir / f"{cell}.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return cell, False, False
    try:
        os.close(lock_fd)
        completed_returncode = run_command(command(root, output_dir, model, effort), root, cell)
    finally:
        lock.unlink(missing_ok=True)
    transient = metadata.exists() and any(token in metadata.read_text(encoding="utf-8").lower() for token in ("429", "timeout", "connection", " 5"))
    return cell, completed_returncode == 0 and complete_marker.exists(), transient


def execute(root: Path, output_dir: Path, cells: list[tuple[str, str]], workers: int, transient_limit: Optional[int] = None) -> list[tuple[str, bool, bool]]:
    pending_cells = iter(cells)
    transient_failures = 0
    results: list[tuple[str, bool, bool]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for _ in range(min(workers, len(cells))):
            cell = next(pending_cells)
            futures[pool.submit(run_cell, root, output_dir, *cell)] = cell
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            stopped = False
            for future in done:
                futures.pop(future)
                result = future.result()
                results.append(result)
                cell_name, ok, transient = result
                if not ok and not transient:
                    terminate_active_processes()
                    for queued in futures:
                        queued.cancel()
                    stopped = True
                    stop_reason = f"Cell {cell_name} failed; remaining cells were not launched"
                    break
                if transient:
                    transient_failures += 1
                if transient_limit is not None and transient_failures >= transient_limit:
                    terminate_active_processes()
                    for queued in futures:
                        queued.cancel()
                    stopped = True
                    stop_reason = f"Circuit breaker tripped after {transient_failures} transient failures"
                    break
            if stopped:
                raise SystemExit(stop_reason)
            for _ in range(len(done)):
                try:
                    cell = next(pending_cells)
                except StopIteration:
                    break
                futures[pool.submit(run_cell, root, output_dir, *cell)] = cell
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--launch", action="store_true", help="Actually submit ClinCog requests")
    parser.add_argument("--models", nargs="+", choices=SCREEN_MODELS, default=list(SCREEN_MODELS))
    parser.add_argument("--efforts", nargs="+", choices=EFFORT_PAYLOADS, default=list(EFFORT_PAYLOADS))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/accuracy_sweep"))
    parser.add_argument("--post-pilot-workers", type=int, default=8)
    args = parser.parse_args()
    if args.post_pilot_workers < 1:
        parser.error("--post-pilot-workers must be at least 1")
    cells = [(model, effort) for model in args.models for effort in args.efforts]
    if not cells:
        parser.error("No cells selected")
    resolve_output_dir(args.root, args.output_dir).mkdir(parents=True, exist_ok=True)
    if not args.launch:
        print(json.dumps({"status": "dry-run", "cells": len(cells), "pilot": cells[:2], "post_pilot_concurrency": args.post_pilot_workers, "output_dir": str(resolve_output_dir(args.root, args.output_dir)), "commands": [command(args.root, args.output_dir, *cell) for cell in cells]}, indent=2))
        return
    pilot = execute(args.root, args.output_dir, cells[:2], 2)
    if not all(ok for _, ok, _ in pilot):
        raise SystemExit("Pilot failed; remaining cells were not launched")
    execute(args.root, args.output_dir, cells[2:], args.post_pilot_workers, transient_limit=3)


if __name__ == "__main__":
    main()
