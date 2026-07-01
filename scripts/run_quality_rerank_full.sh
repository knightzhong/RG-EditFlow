#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${GGFM_ENV:-root_mbo}"
RUN_DIR="${GGFM_RUN_DIR:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RUN_DIR}/quality_rerank_full_${STAMP}.log"
METRICS_PATH="${RUN_DIR}/quality_rerank_full_${STAMP}.json"

mkdir -p "${RUN_DIR}" checkpoints_quality_full
export PYTHONDONTWRITEBYTECODE=1

conda run --no-capture-output -n "${ENV_NAME}" python -u main.py \
  --use-quality-gating \
  --num-proposals 16 \
  --proposal-noise-scale 0.01 \
  --rerank-mode per-seed \
  --rerank-k 5 \
  --rerank-uncertainty-weight 0.25 \
  --rerank-distance-weight 0.1 \
  --checkpoint-dir checkpoints_quality_full \
  --metrics-path "${METRICS_PATH}" \
  2>&1 | tee "${LOG_PATH}"
