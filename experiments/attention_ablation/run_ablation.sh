#!/usr/bin/env bash
# Attention ablation: 4 arms x 3 seeds, everything else held fixed.
#
# The arms differ only in how the attention logits are formed:
#   full     l = (q.k)/sqrt(K) + b_h(r) - softplus(s_h) r     content + distance
#   no_qk    l =                 b_h(r) - softplus(s_h) r     distance only
#   uniform  l = 0                                            no learned weighting
#   none     attention update skipped                         ACE + FFN + readout
#
# Data, split, seed sequence, optimiser, schedule, loss weights and epoch count
# are identical across arms, so a difference in the metrics is attributable to
# the weighting mechanism.
#
#   bash run_ablation.sh [parallel_jobs]
set -uo pipefail

JOBS="${1:-4}"
PY="${PY:-../../.venv/bin/python}"
mkdir -p runs logs

configs=(configs/*.yaml)
echo "Running ${#configs[@]} runs, ${JOBS} at a time"
printf '%s\n' "${configs[@]}" | xargs -P "$JOBS" -I{} sh -c '
  name=$(basename {} .yaml)
  echo "  start  $name"
  '"$PY"' ../../train.py --config {} > logs/$name.log 2>&1 \
    && echo "  done   $name" \
    || echo "  FAILED $name (see logs/$name.log)"
'
echo
echo "Collecting results:"
"$PY" analyze_ablation.py
