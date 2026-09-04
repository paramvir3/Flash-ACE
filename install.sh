#!/usr/bin/env bash
# TRACE installer: creates a virtual environment, installs the pinned
# dependencies, installs the package, and verifies the result.
#
#   ./install.sh                 # use the default interpreter search order
#   ./install.sh python3.12      # force a specific interpreter
#   TRACE_VENV=.venv312 ./install.sh
#
# Safe to re-run: an existing environment is reused.

set -euo pipefail

VENV="${TRACE_VENV:-.venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- interpreter
pick_python() {
    if [[ $# -gt 0 && -n "${1:-}" ]]; then
        command -v "$1" || { echo "error: interpreter '$1' not found" >&2; exit 1; }
        return
    fi
    for candidate in python3.12 python3.13 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
                command -v "$candidate"
                return
            fi
        fi
    done
    echo "error: no CPython 3.10-3.13 found on PATH." >&2
    echo "       torch 2.12 / e3nn 0.6 do not support Python 3.8 or 3.9." >&2
    exit 1
}

PYTHON="$(pick_python "${1:-}")"
echo "==> interpreter : $PYTHON ($("$PYTHON" -V 2>&1))"

# ------------------------------------------------------------------- venv
if [[ ! -d "$VENV" ]]; then
    echo "==> creating virtual environment in $VENV"
    "$PYTHON" -m venv "$VENV"
else
    echo "==> reusing existing virtual environment in $VENV"
fi
VPY="$VENV/bin/python"

echo "==> upgrading pip"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

# --------------------------------------------------------------- dependencies
echo "==> installing pinned runtime dependencies"
"$VPY" -m pip install -r requirements.txt

echo "==> installing test dependencies"
"$VPY" -m pip install -r requirements-dev.txt

echo "==> installing TRACE in editable mode"
"$VPY" -m pip install --no-deps -e .

# ------------------------------------------------------------------- verify
echo "==> verifying the environment"
"$VPY" - <<'PYCODE'
import sys
failures = []

def check(name, got, want, ok):
    status = "OK " if ok else "BAD"
    print(f"   {status}  {name:<12} {got}" + ("" if ok else f"   (need {want})"))
    if not ok:
        failures.append(name)

v = sys.version_info
check("python", ".".join(map(str, v[:3])), "3.10-3.13", (3, 10) <= v[:2] < (3, 14))

import torch
check("torch", torch.__version__, "2.12.0", torch.__version__.startswith("2.12"))

import e3nn
from packaging.version import Version
ok = Version("0.6.0") <= Version(e3nn.__version__) < Version("0.7.0")
check("e3nn", e3nn.__version__, "0.6.x", ok)

import ase, numpy
check("ase", ase.__version__, "3.28.0", ase.__version__.startswith("3.28"))
check("numpy", numpy.__version__, "2.4.x", numpy.__version__.startswith("2.4"))

from flashace.model import TransformersACE
m = TransformersACE(r_max=6.0, l_max=2, num_radial=12, hidden_dim=64, num_layers=1,
                    correlation_order=4, correlation_channels=16, attention_num_heads=2)
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
check("TRACE-v2", f"{n:,} parameters", "imports and builds", n > 0)

# the layer scale must be stored per irrep copy, not per component (defect P3)
from e3nn import o3
blk = m.layers[0]
copies = sum(mul for mul, _ in o3.Irreps(blk.node_irreps))
check("layer scale", f"{blk.layer_scale_attn.numel()} per-copy entries",
      f"{copies}", blk.layer_scale_attn.numel() == copies)

if failures:
    print("\n   environment check FAILED for: " + ", ".join(failures))
    sys.exit(1)
print("\n   environment OK")
PYCODE

cat <<EOF

================================================================================
Installation complete.

Activate the environment in every new shell:

    source $VENV/bin/activate

Run the test suite:

    pytest tests/ -q

Start CsPbI3 training:

    cd training && python ../train.py --config config.yaml
================================================================================
EOF
