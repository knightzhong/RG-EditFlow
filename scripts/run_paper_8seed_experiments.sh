#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/paper_ablation_8seed runs/paper_param_8seed
PYTHON_ENV=${EDITFLOW_ENV:-editflow}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/editflow_pycache}

run_json() {
  local name="$1"; shift
  local json="$1"; shift
  local log="${json%.json}.log"
  if [[ -s "$json" ]]; then
    echo "[SKIP] $name -> $json"
    return 0
  fi
  mkdir -p "$(dirname "$json")"
  echo "[RUN] $name -> $json"
  local start=$SECONDS
  conda run --no-capture-output -n "$PYTHON_ENV" python -u main.py "$@" --metrics-path "$json" > "$log" 2>&1
  local elapsed=$((SECONDS - start))
  echo "[DONE] $name elapsed=${elapsed}s"
  tail -n 10 "$log"
}

run_single() {
  local task="$1" seed="$2" task_name checkpoint extra_args
  case "$task" in
    tfbind8) task_name="TFBind8-Exact-v0"; checkpoint="checkpoints_quality_full/cfm_model_final.pt"; extra_args=() ;;
    tfbind10) task_name="TFBind10-Exact-v0"; checkpoint="checkpoints_tfbind10_quality_full/cfm_model_final.pt"; extra_args=(--gp-num-fit-samples 4096) ;;
    dkitty) task_name="DKittyMorphology-Exact-v0"; checkpoint="checkpoints_dkitty_quality_full/cfm_model_final.pt"; extra_args=(--gp-num-fit-samples 4096) ;;
    *) echo "bad single task $task" >&2; exit 2 ;;
  esac
  run_json "single:${task}:seed${seed}" "runs/paper_ablation_8seed/single_${task}_seed_${seed}.json" \
    --task-name "$task_name" --seed "$seed" --eval-only --load-checkpoint "$checkpoint" \
    "${extra_args[@]}" --num-test-samples 128 --num-proposals 1 --proposal-noise-scale 0.0
}

run_no_trust() {
  local task="$1" seed="$2" task_name checkpoint
  case "$task" in
    tfbind10) task_name="TFBind10-Exact-v0"; checkpoint="checkpoints_tfbind10_quality_full/cfm_model_final.pt" ;;
    dkitty) task_name="DKittyMorphology-Exact-v0"; checkpoint="checkpoints_dkitty_quality_full/cfm_model_final.pt" ;;
    *) echo "bad no_trust task $task" >&2; exit 2 ;;
  esac
  run_json "no_trust:${task}:seed${seed}" "runs/paper_ablation_8seed/no_trust_${task}_seed_${seed}.json" \
    --task-name "$task_name" --seed "$seed" --eval-only --load-checkpoint "$checkpoint" \
    --gp-num-fit-samples 4096 --num-test-samples 128 --num-proposals 16 --proposal-noise-scale 0.01 \
    --rerank-mode per-seed --uncertainty-mode label-variance
}

run_tfbind8_source() {
  local variant="$1" seed="$2" checkpoint extra
  case "$variant" in
    recorded_path) checkpoint="checkpoints_gp_record_path_50e/cfm_model_final.pt"; extra=(--use-quality-gating --gp-record-paths) ;;
    knn_mixup) checkpoint="checkpoints_knn_mixup_50e/cfm_model_final.pt"; extra=(--use-quality-gating --trajectory-source knn-mixup) ;;
    *) echo "bad tfbind8 source $variant" >&2; exit 2 ;;
  esac
  run_json "${variant}:tfbind8:seed${seed}" "runs/paper_ablation_8seed/${variant}_tfbind8_seed_${seed}.json" \
    --task-name TFBind8-Exact-v0 --seed "$seed" --eval-only --load-checkpoint "$checkpoint" \
    "${extra[@]}" --num-test-samples 128 --num-proposals 16 --proposal-noise-scale 0.01 \
    --rerank-mode per-seed --uncertainty-mode label-variance
}

run_clip() {
  local task="$1" clip="$2" seed="$3" task_name checkpoint
  case "$task" in
    tfbind10) task_name="TFBind10-Exact-v0"; checkpoint="checkpoints_tfbind10_quality_full/cfm_model_final.pt" ;;
    dkitty) task_name="DKittyMorphology-Exact-v0"; checkpoint="checkpoints_dkitty_quality_full/cfm_model_final.pt" ;;
    *) echo "bad clip task $task" >&2; exit 2 ;;
  esac
  local clip_tag=${clip/./p}
  run_json "clip:${task}:${clip}:seed${seed}" "runs/paper_param_8seed/clip_${task}_${clip_tag}_seed_${seed}.json" \
    --task-name "$task_name" --seed "$seed" --eval-only --load-checkpoint "$checkpoint" \
    --gp-num-fit-samples 4096 --num-test-samples 128 --num-proposals 16 --proposal-noise-scale 0.005 \
    --proposal-max-displacement "$clip" --rerank-mode per-seed --uncertainty-mode label-variance --rerank-distance-weight 0.5
}

for seed in 0 1 2 3 4 5 6 7; do
  for task in tfbind8 tfbind10 dkitty; do run_single "$task" "$seed"; done
  for task in tfbind10 dkitty; do run_no_trust "$task" "$seed"; done
  for variant in recorded_path knn_mixup; do run_tfbind8_source "$variant" "$seed"; done
  for clip in 1.0 1.5 2.0 2.5 3.0; do run_clip tfbind10 "$clip" "$seed"; done
  for clip in 1.0 1.5 2.0 3.0; do run_clip dkitty "$clip" "$seed"; done
done
