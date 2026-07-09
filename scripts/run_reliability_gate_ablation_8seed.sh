#!/usr/bin/env bash
set -euo pipefail

PYTHON_ENV=${EDITFLOW_ENV:-editflow}
mkdir -p runs/reliability_gate_8seed checkpoints_reliability_gate_8seed
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/editflow_pycache}

run_cmd() {
  local name="$1" log="$2"; shift 2
  echo "[RUN] ${name}"
  local start=$SECONDS
  conda run --no-capture-output -n "$PYTHON_ENV" python -u main.py "$@" > "$log" 2>&1
  local elapsed=$((SECONDS - start))
  echo "[DONE] ${name} elapsed=${elapsed}s"
  tail -n 10 "$log"
}

train_gate() {
  local mode="$1"
  local checkpoint_dir="checkpoints_reliability_gate_8seed/${mode}"
  local log="runs/reliability_gate_8seed/train_${mode}.log"
  if [[ "$mode" == "full" && -s "checkpoints_quality_full/cfm_model_final.pt" ]]; then
    echo "[SKIP] train full -> reusing checkpoints_quality_full/cfm_model_final.pt"
    return 0
  fi
  if [[ -s "${checkpoint_dir}/cfm_model_final.pt" ]]; then
    echo "[SKIP] train ${mode} -> ${checkpoint_dir}/cfm_model_final.pt"
    return 0
  fi
  mkdir -p "$checkpoint_dir"
  run_cmd "train gate=${mode}" "$log" \
    --task-name TFBind8-Exact-v0 \
    --seed 42 \
    --use-quality-gating \
    --quality-gate-mode "$mode" \
    --num-proposals 16 \
    --proposal-noise-scale 0.01 \
    --rerank-mode per-seed \
    --uncertainty-mode label-variance \
    --rerank-k 5 \
    --rerank-uncertainty-weight 0.25 \
    --rerank-distance-weight 0.1 \
    --checkpoint-dir "$checkpoint_dir"
}

eval_gate() {
  local mode="$1" seed="$2" checkpoint_dir checkpoint json log
  if [[ "$mode" == "full" && -s "checkpoints_quality_full/cfm_model_final.pt" ]]; then
    checkpoint_dir="checkpoints_quality_full"
  else
    checkpoint_dir="checkpoints_reliability_gate_8seed/${mode}"
  fi
  checkpoint="${checkpoint_dir}/cfm_model_final.pt"
  json="runs/reliability_gate_8seed/${mode}_tfbind8_seed_${seed}.json"
  log="${json%.json}.log"
  if [[ -s "$json" ]]; then
    echo "[SKIP] eval gate=${mode} seed=${seed} -> ${json}"
    return 0
  fi
  if [[ ! -s "$checkpoint" ]]; then
    echo "Missing checkpoint for gate=${mode}: ${checkpoint}" >&2
    exit 1
  fi
  run_cmd "eval gate=${mode} seed=${seed}" "$log" \
    --task-name TFBind8-Exact-v0 \
    --seed "$seed" \
    --eval-only \
    --load-checkpoint "$checkpoint" \
    --checkpoint-dir "$checkpoint_dir" \
    --num-test-samples 128 \
    --num-proposals 16 \
    --proposal-noise-scale 0.01 \
    --rerank-mode per-seed \
    --uncertainty-mode label-variance \
    --rerank-k 5 \
    --rerank-uncertainty-weight 0.25 \
    --rerank-distance-weight 0.1 \
    --metrics-path "$json"
}

for mode in none score geometry full; do
  train_gate "$mode"
done

for seed in 0 1 2 3 4 5 6 7; do
  for mode in none score geometry full; do
    eval_gate "$mode" "$seed"
  done
done

python scripts/make_mind2026_appendix_artifacts.py
