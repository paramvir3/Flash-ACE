#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

python -m pip install -e "${repo_root}"
cd "${script_dir}"

python -m transformers_ace.deploy \
  --checkpoint model_last.pt \
  --output model.transformers_ace.pt \
  --type-map H O \
  --device cpu
