#!/usr/bin/env python3
"""Launch the 480-atom CsPbI3 TI workflow with isolated configuration/output."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR.parent / "free_energy_ti" / "run_ti.py"


def option_present(name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in sys.argv[1:])


def option_value(name: str, default: str) -> str:
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return default


def main() -> None:
    arguments = list(sys.argv[1:])
    profile = option_value("--profile", "pilot")
    if not option_present("--config"):
        arguments.extend(["--config", str(SCRIPT_DIR / "config.yaml")])
    if not option_present("--run-directory"):
        arguments.extend(["--run-directory", str(SCRIPT_DIR / "runs" / profile)])
    os.execv(sys.executable, [sys.executable, str(BASE_SCRIPT), *arguments])


if __name__ == "__main__":
    main()
