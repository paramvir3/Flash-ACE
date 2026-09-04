#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: ./patch_lammps.sh /path/to/lammps"
  exit 1
fi

lammps_dir="$1"
if [[ ! -d "$lammps_dir/src" || ! -f "$lammps_dir/cmake/CMakeLists.txt" ]]; then
  echo "$lammps_dir does not look like a LAMMPS source tree"
  exit 1
fi
if [[ ! -f pair_transformers_ace.cpp || ! -f pair_transformers_ace.h ]]; then
  echo "Run this script from the lammps/pair_style directory"
  exit 1
fi

echo "Copying pair_style transformers_ace into $lammps_dir/src"
cp pair_transformers_ace.cpp "$lammps_dir/src/"
cp pair_transformers_ace.h "$lammps_dir/src/"

# Ahead-of-time compiled variant. It loads a .pt2 package whose derivative graph
# is already compiled, so the runtime issues a handful of kernels instead of the
# ~2300 of the eager TorchScript path.
if [[ -f pair_transformers_ace_aot.cpp && -f pair_transformers_ace_aot.h ]]; then
  echo "Copying pair_style transformers_ace/aot into $lammps_dir/src"
  cp pair_transformers_ace_aot.cpp "$lammps_dir/src/"
  cp pair_transformers_ace_aot.h "$lammps_dir/src/"
fi

python3 - "$lammps_dir/cmake/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("set(CMAKE_CXX_STANDARD 11)", "set(CMAKE_CXX_STANDARD 17)")
block = """

message(STATUS "<< TRANSFORMERS_ACE flags >>")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
"""
if "<< TRANSFORMERS_ACE flags >>" not in text:
    text += block
path.write_text(text)
PY

# CUDA hosts need three things that torch's own CMake config does not supply.
# Each was hit on a clean H100 image and each aborts configure or link:
#
#   1. TorchConfig.cmake calls enable_language(CUDA), which probes for a default
#      architecture. Passing -DCMAKE_CUDA_ARCHITECTURES is not enough because the
#      probe runs before the cache variable is consulted -- CUDAARCHS must be in
#      the environment.
#   2. torch's imported target references MKL_INCLUDE_DIR, which is NOTFOUND
#      without MKL installed, and CMake refuses to generate.
#   3. The CUDA toolkit ships libnvrtc.so.12 but no unversioned .so symlink, so
#      torch's find_library returns CUDA_nvrtc_LIBRARY-NOTFOUND and the link
#      fails with "cannot find -lCUDA_nvrtc_LIBRARY-NOTFOUND".
if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then
  cuda_root="$(dirname "$(dirname "$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)")")"
  if [[ ! -e "$cuda_root/lib64/libnvrtc.so" ]]; then
    newest_nvrtc="$(ls "$cuda_root"/lib64/libnvrtc.so.* 2>/dev/null | sort -V | tail -1)"
    if [[ -n "$newest_nvrtc" ]]; then
      echo "Creating missing libnvrtc.so symlink -> $newest_nvrtc"
      ln -sf "$newest_nvrtc" "$cuda_root/lib64/libnvrtc.so" 2>/dev/null \
        || echo "  (could not create it; re-run with write access to $cuda_root/lib64)"
    fi
  fi
  arch="$("$cuda_root/bin/__nvcc_device_query" 2>/dev/null | head -1)"
  cat <<MSG

Done. Configure LAMMPS with (CUDA host):

  export CUDAARCHS=${arch:-90}
  export CUDACXX=$cuda_root/bin/nvcc
  cmake -S ../cmake -B ../build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${arch:-90} \
    -DCMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    -DMKL_INCLUDE_DIR=/usr/include \
    -DBUILD_MPI=off
  cmake --build ../build -j
MSG
else
  echo "Done. Configure LAMMPS with:"
  echo "  cmake ../cmake -DCMAKE_PREFIX_PATH=\$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
fi
