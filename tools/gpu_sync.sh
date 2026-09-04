#!/usr/bin/env bash
# Push the working tree to the GPU host, then run the validation session there.
#
#   bash tools/gpu_sync.sh 'ssh -p PORT root@HOST' /remote/path /remote/lammps
#
# Uses archive mode on the whole tree rather than naming individual files: an
# earlier session used `rsync -az train.py flashace/model.py dest/`, which
# flattens paths and dropped model.py into the repo root, producing six phantom
# test failures that cost more time than the transfer saved.
set -euo pipefail

SSH_CMD="${1:?usage: gpu_sync.sh 'ssh -p PORT user@host' /remote/repo /remote/lammps}"
REMOTE_REPO="${2:?remote repo path}"
REMOTE_LAMMPS="${3:-}"
HOST="$(echo "$SSH_CMD" | awk '{print $NF}')"
PORT="$(echo "$SSH_CMD" | grep -oE '\-p [0-9]+' | awk '{print $2}')"
PORT="${PORT:-22}"

echo "=== syncing $PWD -> $HOST:$REMOTE_REPO ==="
rsync -az --info=progress2 -e "ssh -p $PORT" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.pt2' --exclude 'output/' --exclude '.pytest_cache' \
  ./ "$HOST:$REMOTE_REPO/"

echo "=== remote environment ==="
$SSH_CMD "cd $REMOTE_REPO && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader && \
  python -c 'import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"avail\",torch.cuda.is_available())'"

if [[ -n "$REMOTE_LAMMPS" ]]; then
  echo "=== running the validation session ==="
  $SSH_CMD "cd $REMOTE_REPO && REPO=$REMOTE_REPO bash tools/gpu_aot_session.sh $REMOTE_LAMMPS"
fi
