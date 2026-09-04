#!/usr/bin/env bash
# One-shot GPU validation of the TRACE AOTI -> LAMMPS deployment path.
#
# Everything in this script that does not need CUDA has already been verified on
# CPU (export, packaging, numerical agreement with eager, the Python/C++ ABI, and
# a clean compile of the pair style against LAMMPS 30 Mar 2026 headers). What
# remains is CUDA-specific and cannot be checked off-GPU:
#
#   Stage 2  does the derivative graph survive CUDA Inductor codegen?
#   Stage 3  does the GPU package reproduce eager physics?
#   Stage 4  does the pair style build and link against libtorch + CUDA?
#   Stage 5  what does AOT actually buy in real MD, against TorchScript?
#
# Runs unattended and stops at the first failure so no paid time is spent past a
# break. Usage:  bash tools/gpu_aot_session.sh /path/to/lammps
set -euo pipefail

REPO="${REPO:-$PWD}"
LAMMPS_DIR="${1:?usage: gpu_aot_session.sh /path/to/lammps}"
CKPT="${CKPT:-training/h100_run/model_h100.pt}"
PKG="${PKG:-$REPO/trace_h100.pt2}"
# Sizing. These are compiled into the package, so getting them wrong costs a
# full re-export. Both are hard capacities the pair style aborts on:
#   max_atoms >= local + ghost + 2   (the +2 are the padding sentinels)
#   max_edges >= local * neighbours-within-r_max
# For CsPbI3 at r_max=6 the measured density is 15-19 edges/atom. A cubic box of
# ~4096 local atoms is L~37 A, and a 6 A ghost shell takes local+ghost to ~2.3x
# local, so ~9500 atoms and ~78k edges. The defaults below carry ~1.3x and ~1.7x
# headroom on those.
# Measured for tests/run_lammps/test_lammps_cspbi3_mpi/data.NVT, the benchmark
# system: 10240 local atoms in an 83x77x71 A box. A 7 A ghost shell (r_c 6 +
# skin 1) takes local+ghost to ~16.9k, and the neighbour density is 19.2
# edges/atom, so ~197k edges. Headroom is deliberately small: every padded edge
# is wasted compute in a fixed-shape program, so oversizing slows the benchmark.
MAX_ATOMS="${MAX_ATOMS:-18432}"
MAX_EDGES="${MAX_EDGES:-217088}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_cache}"
export PYTHONUNBUFFERED=1

say() { printf '\n=== %s ===\n' "$*"; }

say "Stage 0  environment"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

say "Stage 1  test suite on the GPU host"
python -m pytest tests/ -q -x

# The one genuine unknown. max_autotune stays off: on CUDA it drove this graph
# into "failed to set ranges [4, 1] ([N, 4, 5], [])", where (4,5) is the l=2
# irrep block e3nn materialises in 193 places. If this stage fails the same way
# with autotune already off, the assertion is in plain lowering and the next
# probe is TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCH_LOGS=+inductor for the failing node.
say "Stage 2+3  export and verify on CUDA (verification is built into the CLI)"
python -m transformers_ace.aot_deploy \
  --checkpoint "$CKPT" --output "$PKG" \
  --type-map Cs Pb I \
  --max-atoms "$MAX_ATOMS" --max-edges "$MAX_EDGES" \
  --device cuda

say "Stage 4  build LAMMPS with both pair styles"
( cd "$REPO/lammps/pair_style" && bash patch_lammps.sh "$LAMMPS_DIR" )
# torch.utils.cmake_prefix_path is the documented location of TorchConfig.cmake
# (<site-packages>/torch/share/cmake). Passing the torch package root instead
# happens to work only because CMake also probes <prefix>/share/cmake/*.
cmake -S "$LAMMPS_DIR/cmake" -B "$LAMMPS_DIR/build-aot" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DBUILD_MPI=off
cmake --build "$LAMMPS_DIR/build-aot" -j "$(nproc)"

# The pair styles are compiled into the binary only if the patch landed before
# configure; a silent miss here shows up as "Unknown pair style" at Stage 5.
"$LAMMPS_DIR/build-aot/lmp" -h 2>/dev/null | grep -qE "transformers_ace/aot" \
  || { echo "FAIL: transformers_ace/aot is not in the built binary"; exit 1; }
echo "both pair styles present in the binary"

say "Stage 5  MD benchmark: transformers_ace/aot vs transformers_ace"
BENCH="$REPO/tests/run_lammps/benchmark_aot"
LMP="$LAMMPS_DIR/build-aot/lmp"
TS_MODEL="${TS_MODEL:-$REPO/trace_h100.transformers_ace.pt}"

# Export the TorchScript baseline from the SAME checkpoint. Benchmarking against
# a differently-trained model would compare weights, not runtimes, and would make
# the energy cross-check meaningless.
if [[ ! -f "$TS_MODEL" ]]; then
  python -m transformers_ace.deploy \
    --checkpoint "$CKPT" --output "$TS_MODEL" \
    --type-map Cs Pb I --device cuda
fi

( cd "$BENCH" && "$LMP" -in in.bench -var style aot -var model "$PKG" -log log.aot )
( cd "$BENCH" && "$LMP" -in in.bench -var style ts  -var model "$TS_MODEL" -log log.ts )

say "Results"
python - "$BENCH" <<'PY'
import re, sys, pathlib
bench = pathlib.Path(sys.argv[1])
def summarise(log):
    if not log.exists():
        return None
    text = log.read_text()
    loop = re.search(r"Loop time of ([\d.]+) .* for (\d+) steps", text)
    pair = re.search(r"Pair\s+\|\s+[\d.e+-]+\s+\|\s+([\d.e+-]+)\s+\|", text)
    energies = re.findall(r"^\s*\d+\s+\S+\s+(\S+)\s+\S+\s+(\S+)", text, re.M)
    return {
        "s_per_step": float(loop.group(1)) / int(loop.group(2)) if loop else None,
        "pair_s": float(pair.group(1)) if pair else None,
        "final_pe": float(energies[-1][0]) if energies else None,
        "final_etot": float(energies[-1][1]) if energies else None,
    }
aot, ts = summarise(bench / "log.aot"), summarise(bench / "log.ts")
for name, r in (("transformers_ace/aot", aot), ("transformers_ace (TS)", ts)):
    if r is None:
        print(f"  {name:<22} (no log)"); continue
    print(f"  {name:<22} {1e3*r['s_per_step']:>8.2f} ms/step   pair {r['pair_s']:>7.2f} s   "
          f"PE {r['final_pe']:.6g} eV")
if aot and ts and aot["s_per_step"] and ts["s_per_step"]:
    print(f"\n  speedup: {ts['s_per_step']/aot['s_per_step']:.2f}x")
    dpe = abs(aot["final_pe"] - ts["final_pe"])
    # The two styles run independent trajectories in float32; they diverge by
    # Lyapunov growth, so this is a sanity bound, not an equality test.
    print(f"  |dPE| after 200 steps: {dpe:.3e} eV "
          f"({'consistent' if dpe < 1.0 else 'CHECK - larger than expected'})")
PY
