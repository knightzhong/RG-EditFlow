#!/usr/bin/env bash
set -euo pipefail

TASK=${1:-tfbind8}
SEED=${2:-42}
ENV_NAME="${EDITFLOW_ENV:-editflow}"
mkdir -p runs/main_model_8seed

case "$TASK" in
  tfbind8)
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u main.py \
      --seed "$SEED" \
      --eval-only \
      --load-checkpoint checkpoints_quality_full/cfm_model_final.pt \
      --num-test-samples 128 \
      --num-proposals 16 \
      --proposal-noise-scale 0.01 \
      --rerank-mode per-seed \
      --uncertainty-mode label-variance \
      --checkpoint-dir checkpoints_quality_full \
      --metrics-path "runs/main_model_8seed/tfbind8_seed_${SEED}.json"
    ;;
  tfbind10)
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u main.py \
      --task-name TFBind10-Exact-v0 \
      --seed "$SEED" \
      --eval-only \
      --load-checkpoint checkpoints_tfbind10_quality_full/cfm_model_final.pt \
      --gp-num-fit-samples 4096 \
      --num-test-samples 128 \
      --num-proposals 16 \
      --proposal-noise-scale 0.005 \
      --proposal-max-displacement 2.0 \
      --rerank-mode per-seed \
      --uncertainty-mode label-variance \
      --rerank-distance-weight 0.5 \
      --checkpoint-dir checkpoints_tfbind10_quality_full \
      --metrics-path "runs/main_model_8seed/tfbind10_seed_${SEED}.json"
    ;;
  dkitty)
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u main.py \
      --task-name DKittyMorphology-Exact-v0 \
      --seed "$SEED" \
      --eval-only \
      --load-checkpoint checkpoints_dkitty_quality_full/cfm_model_final.pt \
      --gp-num-fit-samples 4096 \
      --num-test-samples 128 \
      --num-proposals 16 \
      --proposal-noise-scale 0.005 \
      --proposal-max-displacement 1.0 \
      --rerank-mode per-seed \
      --uncertainty-mode label-variance \
      --rerank-distance-weight 0.5 \
      --checkpoint-dir checkpoints_dkitty_quality_full \
      --metrics-path "runs/main_model_8seed/dkitty_seed_${SEED}.json"
    ;;
  ant)
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u main.py \
      --task-name AntMorphology-Exact-v0 \
      --seed "$SEED" \
      --eval-only \
      --load-checkpoint checkpoints_ant_quality_full/cfm_model_final.pt \
      --gp-num-fit-samples 4096 \
      --num-test-samples 128 \
      --num-proposals 16 \
      --proposal-noise-scale 0.005 \
      --proposal-max-displacement 1.0 \
      --rerank-mode per-seed \
      --uncertainty-mode label-variance \
      --rerank-distance-weight 0.5 \
      --checkpoint-dir checkpoints_ant_quality_full \
      --metrics-path "runs/main_model_8seed/ant_seed_${SEED}.json"
    ;;
  superconductor|super)
    PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u main.py \
      --task-name Superconductor-RandomForest-v0 \
      --seed "$SEED" \
      --eval-only \
      --load-checkpoint checkpoints_superconductor_quality_full/cfm_model_final.pt \
      --gp-num-fit-samples 4096 \
      --num-test-samples 128 \
      --num-proposals 16 \
      --proposal-noise-scale 0.005 \
      --proposal-max-displacement 1.0 \
      --rerank-mode per-seed \
      --uncertainty-mode label-variance \
      --rerank-distance-weight 0.5 \
      --checkpoint-dir checkpoints_superconductor_quality_full \
      --metrics-path "runs/main_model_8seed/superconductor_seed_${SEED}.json"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 1
    ;;
esac
