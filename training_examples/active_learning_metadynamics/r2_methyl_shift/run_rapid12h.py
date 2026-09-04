#!/usr/bin/env python3
"""Launch the reduced R2 screening protocol on a multicore workstation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SOURCE_WINDOW_ENDPOINTS_A = (-0.9685435612879105, 1.0283024492805735)
ESTIMATED_OUTPUT_MIB_PER_PS = 0.30
DISK_SAFETY_GIB = 1.0
PROTOCOLS = {
    "rapid12h": {
        "purpose": "Reduced screening protocol; not the full Figure 4 reproduction",
        "output_directory": "rapid12h",
        "replicas": 3,
        "standard_ps": 150.0,
        "inherited_ps": 100.0,
        "umbrella_ps": 20.0,
        "umbrella_equilibration_ps": 5.0,
    },
    "full3": {
        "purpose": (
            "Three-replica protocol using the source paper's full trajectory lengths"
        ),
        "output_directory": "full3",
        "replicas": 3,
        "standard_ps": 500.0,
        "inherited_ps": 250.0,
        "umbrella_ps": 40.0,
        "umbrella_equilibration_ps": 10.0,
    },
}


@dataclass(frozen=True)
class Task:
    name: str
    output: Path
    command: tuple[str, ...]
    expected_metadata: tuple[tuple[str, object], ...] = ()


def completed(task: Task) -> bool:
    status_path = task.output / "run_status.json"
    if not status_path.is_file():
        return False
    try:
        if json.loads(status_path.read_text()).get("state") != "complete":
            return False
        if not task.expected_metadata:
            return True
        metadata = json.loads((task.output / "run_metadata.json").read_text())
        return all(
            metadata.get(key) == expected
            for key, expected in task.expected_metadata
        )
    except (json.JSONDecodeError, OSError):
        return False


def duration_ps(task: Task) -> float:
    try:
        index = task.command.index("--duration-ps")
        return float(task.command[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Task has no valid --duration-ps: {task.name}") from exc


def make_tasks(
    model: Path,
    device: str,
    overwrite: bool,
    protocol: dict,
    methods: set[str] | None = None,
    window_indices: tuple[int, ...] = tuple(range(30)),
    window_centers_file: Path | None = None,
    window_count: int = 30,
) -> list[Task]:
    methods = methods or {"standard", "inherited", "umbrella"}
    root = EXAMPLE_ROOT / "results" / protocol["output_directory"]
    common = ("--model", str(model), "--device", device)
    overwrite_flag = ("--overwrite",) if overwrite else ()
    center_file_flag = (
        ("--window-centers-file", str(window_centers_file))
        if window_centers_file is not None
        else ()
    )
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    schedule_sha256 = (
        hashlib.sha256(window_centers_file.read_bytes()).hexdigest()
        if window_centers_file is not None
        else None
    )
    tasks = []

    for replica in range(protocol["replicas"]):
        metadynamics = []
        if "standard" in methods:
            metadynamics.append(("standard", protocol["standard_ps"]))
        if "inherited" in methods:
            metadynamics.append(("inherited", protocol["inherited_ps"]))
        for mode, duration in metadynamics:
            output = root / mode / f"replica_{replica:02d}"
            tasks.append(
                Task(
                    name=f"{mode}-replica-{replica:02d}",
                    output=output,
                    command=(
                        sys.executable,
                        str(EXAMPLE_ROOT / "run_wtmetad.py"),
                        "--mode",
                        mode,
                        "--replica",
                        str(replica),
                        "--duration-ps",
                        str(duration),
                        "--output",
                        str(output),
                        *common,
                        *overwrite_flag,
                    ),
                )
            )

    if "umbrella" in methods:
        umbrella_pairs = (
            (replica, window)
            for replica in range(protocol["replicas"])
            for window in window_indices
        )
    else:
        umbrella_pairs = ()
    for replica, window in umbrella_pairs:
        output = (
            root
            / "umbrella"
            / f"replica_{replica:02d}"
            / f"window_{window:02d}"
        )
        tasks.append(
            Task(
                name=f"umbrella-replica-{replica:02d}-window-{window:02d}",
                output=output,
                command=(
                    sys.executable,
                    str(EXAMPLE_ROOT / "run_umbrella_window.py"),
                    "--replica",
                    str(replica),
                    "--window",
                    str(window),
                    "--duration-ps",
                    str(protocol["umbrella_ps"]),
                    "--equilibration-ps",
                    str(protocol["umbrella_equilibration_ps"]),
                    "--output",
                    str(output),
                    *center_file_flag,
                    *common,
                    *overwrite_flag,
                ),
                expected_metadata=(
                    ("model_sha256", model_sha256),
                    ("window_centers_sha256", schedule_sha256),
                    ("window_count", window_count),
                    ("duration_ps", float(protocol["umbrella_ps"])),
                    (
                        "equilibration_discard_ps",
                        float(protocol["umbrella_equilibration_ps"]),
                    ),
                ),
            )
        )
    return tasks


def execute(
    task: Task,
    environment: dict[str, str],
    log_root: Path,
    allow_nonempty: bool,
) -> dict:
    if completed(task):
        return {"name": task.name, "state": "skipped_complete", "seconds": 0.0}
    if task.output.exists() and any(task.output.iterdir()) and not allow_nonempty:
        return {
            "name": task.name,
            "state": "blocked_nonempty",
            "seconds": 0.0,
            "reason": (
                "Nonempty incomplete or provenance-mismatched output: "
                f"{task.output}"
            ),
        }

    started = time.monotonic()
    result = subprocess.run(
        task.command,
        cwd=EXAMPLE_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / f"{task.name}.log").write_text(result.stdout)
    state = "complete" if result.returncode == 0 and completed(task) else "failed"
    return {
        "name": task.name,
        "state": state,
        "seconds": elapsed,
        "returncode": result.returncode,
        "output": str(task.output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--preset", choices=tuple(PROTOCOLS), default="rapid12h"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("all", "standard", "inherited", "umbrella"),
        default=("all",),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=EXAMPLE_ROOT / "models/r2_trace_v2_wtmetad.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--umbrella-duration-ps", type=float, default=None)
    parser.add_argument("--umbrella-equilibration-ps", type=float, default=None)
    parser.add_argument("--windows", nargs="+", type=int, default=None)
    parser.add_argument("--window-centers-file", type=Path, default=None)
    parser.add_argument(
        "--allow-low-disk-space",
        action="store_true",
        help="Bypass the conservative trajectory-storage preflight check.",
    )
    args = parser.parse_args()

    if not 1 <= args.workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    model = args.model.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"TRACE checkpoint not found: {model}")

    settings = dict(PROTOCOLS[args.preset])
    if args.replicas is not None:
        if args.replicas < 1:
            raise ValueError("--replicas must be positive")
        settings["replicas"] = args.replicas
    if args.output_directory is not None:
        output_directory = Path(args.output_directory)
        if output_directory.is_absolute() or ".." in output_directory.parts:
            raise ValueError("--output-directory must stay below the results directory")
        settings["output_directory"] = str(output_directory)
    if args.umbrella_duration_ps is not None:
        if args.umbrella_duration_ps <= 0.0:
            raise ValueError("--umbrella-duration-ps must be positive")
        settings["umbrella_ps"] = args.umbrella_duration_ps
    if args.umbrella_equilibration_ps is not None:
        if args.umbrella_equilibration_ps < 0.0:
            raise ValueError("--umbrella-equilibration-ps cannot be negative")
        settings["umbrella_equilibration_ps"] = args.umbrella_equilibration_ps
    if settings["umbrella_equilibration_ps"] >= settings["umbrella_ps"]:
        raise ValueError("Umbrella equilibration must be shorter than the trajectory")
    window_centers_file = (
        args.window_centers_file.resolve()
        if args.window_centers_file is not None
        else None
    )
    if window_centers_file is None:
        window_count = 30
    else:
        if not window_centers_file.is_file():
            raise FileNotFoundError(
                f"Window-center file not found: {window_centers_file}"
            )
        center_values = [
            float(line.split("#", 1)[0].strip())
            for line in window_centers_file.read_text().splitlines()
            if line.split("#", 1)[0].strip()
        ]
        window_count = len(center_values)
        if (
            window_count < 2
            or not all(math.isfinite(value) for value in center_values)
            or any(
                second <= first
                for first, second in zip(center_values, center_values[1:])
            )
        ):
            raise ValueError(
                "The window-center file must contain increasing numeric values"
            )
        if not all(
            math.isclose(value, endpoint, abs_tol=1.0e-8)
            for value, endpoint in zip(
                (center_values[0], center_values[-1]), SOURCE_WINDOW_ENDPOINTS_A
            )
        ):
            raise ValueError(
                "A custom center schedule must retain the two IRC endpoints"
            )
    window_indices = (
        tuple(range(window_count)) if args.windows is None else tuple(args.windows)
    )
    if len(set(window_indices)) != len(window_indices) or any(
        window < 0 or window >= window_count for window in window_indices
    ):
        raise ValueError(
            f"--windows must contain unique indices between 0 and {window_count - 1}"
        )
    requested_methods = set(args.methods)
    if "all" in requested_methods and len(requested_methods) > 1:
        raise ValueError("Use --methods all by itself, or list individual methods")
    methods = (
        {"standard", "inherited", "umbrella"}
        if requested_methods == {"all"}
        else requested_methods
    )
    tasks = make_tasks(
        model,
        args.device,
        args.overwrite,
        settings,
        methods,
        window_indices,
        window_centers_file,
        window_count,
    )
    protocol_root = EXAMPLE_ROOT / "results" / settings["output_directory"]
    umbrella_total = (
        settings["replicas"] * len(window_indices) * settings["umbrella_ps"]
        if "umbrella" in methods
        else 0.0
    )
    total_sampling = umbrella_total
    if "standard" in methods:
        total_sampling += settings["replicas"] * settings["standard_ps"]
    if "inherited" in methods:
        total_sampling += settings["replicas"] * settings["inherited_ps"]
    protocol = {
        "preset": args.preset,
        "purpose": (
            settings["purpose"]
            if args.replicas is None and window_centers_file is None
            else "User-specified umbrella-sampling protocol"
        ),
        "methods": sorted(methods),
        "replicas_per_method": settings["replicas"],
        "standard_wtmetad_ps_per_replica": settings["standard_ps"],
        "inherited_wtmetad_ps_per_replica": settings["inherited_ps"],
        "umbrella_windows_per_replica": len(window_indices),
        "umbrella_window_indices": list(window_indices),
        "window_centers_file": (
            str(window_centers_file) if window_centers_file is not None else None
        ),
        "window_centers_sha256": (
            hashlib.sha256(window_centers_file.read_bytes()).hexdigest()
            if window_centers_file is not None
            else None
        ),
        "umbrella_ps_per_window": settings["umbrella_ps"],
        "umbrella_equilibration_discard_ps": settings[
            "umbrella_equilibration_ps"
        ],
        "total_sampling_ps": total_sampling,
        "workers": args.workers,
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "device": args.device,
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        for task in tasks:
            print(" ".join(task.command))
        return

    blocked_outputs = [
        task.output
        for task in tasks
        if task.output.exists()
        and any(task.output.iterdir())
        and not completed(task)
    ]
    if blocked_outputs and not args.overwrite:
        preview = "\n".join(f"- {path}" for path in blocked_outputs[:10])
        suffix = "\n- ..." if len(blocked_outputs) > 10 else ""
        raise FileExistsError(
            f"Found {len(blocked_outputs)} nonempty incomplete or provenance-"
            "mismatched outputs. Resolve them or pass --overwrite before "
            f"launching any jobs:\n{preview}{suffix}"
        )

    remaining_tasks = [task for task in tasks if not completed(task)]
    remaining_ps = sum(duration_ps(task) for task in remaining_tasks)
    gib = 1024.0 ** 3
    estimated_bytes = remaining_ps * ESTIMATED_OUTPUT_MIB_PER_PS * 1024.0 ** 2
    required_bytes = estimated_bytes + DISK_SAFETY_GIB * gib
    free_bytes = shutil.disk_usage(EXAMPLE_ROOT).free
    protocol.update(
        remaining_sampling_ps_before_launch=remaining_ps,
        estimated_remaining_output_GiB=estimated_bytes / gib,
        free_disk_space_GiB=free_bytes / gib,
        required_free_space_with_safety_GiB=required_bytes / gib,
    )
    if free_bytes < required_bytes and not args.allow_low_disk_space:
        raise OSError(
            f"Insufficient disk space: {free_bytes / gib:.2f} GiB free, but "
            f"approximately {required_bytes / gib:.2f} GiB is required for "
            "the remaining trajectories plus a 1 GiB safety margin. Free "
            "space before resuming. Use --allow-low-disk-space only if the "
            "output estimate is known to be inappropriate."
        )

    protocol_root.mkdir(parents=True, exist_ok=True)
    (protocol_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    environment = os.environ.copy()
    for variable in THREAD_VARIABLES:
        environment[variable] = "1"

    started = time.monotonic()
    records = []
    log_root = protocol_root / "launcher_logs"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                execute, task, environment, log_root, args.overwrite
            ): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[{record['state']}] {record['name']} ({record['seconds']:.1f} s)")

    summary = {
        **protocol,
        "wall_time_seconds": time.monotonic() - started,
        "states": {
            state: sum(record["state"] == state for record in records)
            for state in sorted({record["state"] for record in records})
        },
        "tasks": sorted(records, key=lambda item: item["name"]),
    }
    (protocol_root / "launcher_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary["states"], indent=2))
    if any(record["state"] in {"failed", "blocked_nonempty"} for record in records):
        raise SystemExit("At least one rapid-protocol task did not complete")


if __name__ == "__main__":
    main()
