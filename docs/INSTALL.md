# Download And Installation

## Quick Start

```bash
git clone https://github.com/paramvir3/TRACE.git
cd TRACE
./install.sh
source .venv/bin/activate
pytest tests/ -q
```

`install.sh` picks a suitable interpreter, creates `.venv`, installs the pinned
dependencies, installs TRACE in editable mode, and verifies the result. It is
safe to re-run. To force a specific interpreter:

```bash
./install.sh python3.12
```

## Requirements

- macOS or Linux
- **CPython 3.10-3.13.** Python 3.8 and 3.9 are *not* supported: torch 2.12 and
  e3nn 0.6 publish no wheels for them.
- **e3nn 0.6.x** -- see [Why e3nn is pinned](#why-e3nn-is-pinned)
- Git for cloning the repository
- A C/C++ build toolchain if an optional dependency requires compilation

Transformers-ACE runs on CPU and CUDA-capable PyTorch installations. Apple MPS
support depends on the PyTorch and e3nn operations available on the individual
machine; CPU is the reliable macOS default.

## Download With Git

```bash
git clone https://github.com/paramvir3/TRACE.git
cd TRACE
```

To update an existing checkout:

```bash
git pull
```

## Download A ZIP

Download the main branch from:

```text
https://github.com/paramvir3/TRACE/archive/refs/heads/main.zip
```

Extract the archive and enter the resulting `TRACE-main` directory.

## Create A Python Environment

Create the environment once:

```bash
python3 -m venv .venv
```

Activate it whenever opening a new terminal:

```bash
source .venv/bin/activate
```

### Manual install (equivalent to `install.sh`)

Install the **exact pinned** dependency set, then the package itself:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
```

`--no-deps` on the last step keeps the pins from `requirements.txt` intact;
without it pip may resolve looser floors from `setup.py` and silently upgrade
`e3nn` or `torch`.

If you instead want the loosest supported versions rather than the reproducible
pins, use `python -m pip install -e '.[test]'`. Do this only for exploratory
work -- published runs should use `requirements.txt`.

`torch-scatter` is optional because the current implementation has a native
PyTorch fallback. Install the acceleration extra only when a compatible wheel
is available:

```bash
python -m pip install -e '.[accelerated]'
```

## Verify The Installation

`install.sh` runs this automatically. To repeat it by hand:

```bash
python -c "import torch, e3nn, ase, numpy; print('torch', torch.__version__, '| e3nn', e3nn.__version__, '| ase', ase.__version__, '| numpy', numpy.__version__)"
pytest tests/ -q
```

The expected environment is:

| package | pinned |
|---|---|
| python | 3.10-3.13 (3.12 used for the published run) |
| torch | 2.12.0 |
| e3nn | 0.6.0 |
| ase | 3.28.0 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.11.0 |
| PyYAML | 6.0.3 |

The full suite is 72 tests. `tests/test_runtime_compatibility.py` fails
immediately if `e3nn` is outside the 0.6.x series, and
`tests/test_defect_regressions.py` fails if any of the corrected defects
reappears.

## Why e3nn Is Pinned

Tensor-product normalisation conventions changed between the e3nn 0.4.x and
0.6.x series. A checkpoint trained under one series does **not** reproduce the
same energies under the other: forces agree to a fraction of a percent, but the
absolute energy shifts enough to move a reported per-atom RMSE by tens of
percent. Always train and deploy a given checkpoint under the same e3nn series,
and record the version in the run manifest.

## macOS CPU Training

Set `device: "cpu"` and `use_amp: false` in the training YAML. PyTorch normally
chooses a sensible thread count. It can be specified explicitly when desired:

```yaml
device: "cpu"
use_amp: false
torch_num_threads: 10
torch_num_interop_threads: 1
num_workers: 0
```

Use the actual number of CPU cores on the machine. `num_workers: 0` avoids
macOS shared-memory issues while tensor operations still use the configured
PyTorch threads.

## Dependency Isolation

TRACE targets the stable e3nn 0.6 API used by current NequIP. Do not install it
into an older MACE environment that pins `e3nn==0.4.4`, and do not install it
into a conda base environment shared with other codes; use a dedicated virtual
environment. If `pip` reports an "externally managed environment" (PEP 668)
error, you are installing into a system or Homebrew Python rather than into a
venv -- create one first. The Transformers-ACE checkpoint
loader keeps PyTorch's restricted `weights_only` mode enabled and safely
allowlists the NumPy scalar metadata present in published checkpoints, so
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is not required.

## Uninstall Or Leave The Environment

Leave the environment with:

```bash
deactivate
```

Delete `.venv` to remove the isolated environment completely.
