#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${EDITFLOW_ENV:-editflow}"
RUN_DIR="${EDITFLOW_RUN_DIR:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RUN_DIR}/smoke_a407d81_${STAMP}.log"
METRICS_PATH="${RUN_DIR}/smoke_a407d81_${STAMP}.json"

mkdir -p "${RUN_DIR}" checkpoints_smoke
export PYTHONDONTWRITEBYTECODE=1

echo "[EditFlow smoke] env=${ENV_NAME}"
echo "[EditFlow smoke] log=${LOG_PATH}"
echo "[EditFlow smoke] metrics=${METRICS_PATH}"
echo "[EditFlow smoke] commit=$(git rev-parse --short HEAD) $(git show -s --format=%s HEAD)"

conda run --no-capture-output -n "${ENV_NAME}" python -u main.py \
  --fm-epochs 1 \
  --gp-num-functions 1 \
  --gp-num-points 32 \
  --gp-gradient-steps 2 \
  --gp-traj-steps 4 \
  --num-test-samples 8 \
  --checkpoint-dir checkpoints_smoke \
  --metrics-path "${METRICS_PATH}" \
  2>&1 | tee "${LOG_PATH}"
