#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${EDITFLOW_ENV:-editflow}"
RUN_DIR="${EDITFLOW_RUN_DIR:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RUN_DIR}/baseline_a407d81_${STAMP}.log"
METRICS_PATH="${RUN_DIR}/baseline_a407d81_${STAMP}.json"

mkdir -p "${RUN_DIR}" checkpoints
export PYTHONDONTWRITEBYTECODE=1

echo "[EditFlow] env=${ENV_NAME}"
echo "[EditFlow] log=${LOG_PATH}"
echo "[EditFlow] metrics=${METRICS_PATH}"
echo "[EditFlow] commit=$(git rev-parse --short HEAD) $(git show -s --format=%s HEAD)"

conda run --no-capture-output -n "${ENV_NAME}" python -u main.py \
  --checkpoint-dir checkpoints \
  --metrics-path "${METRICS_PATH}" \
  2>&1 | tee "${LOG_PATH}"
