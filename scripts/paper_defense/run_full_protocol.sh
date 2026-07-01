#!/usr/bin/env bash
set -euo pipefail

RUN_GROUP="${1:-runs/paper_defense_$(date +%Y%m%d_%H%M%S)}"
ENV_NAME="${GGFM_ENV:-root_mbo}"
mkdir -p "$RUN_GROUP"
DRIVER_LOG="$RUN_GROUP/driver.log"
STATUS_JSON="$RUN_GROUP/driver_status.json"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$DRIVER_LOG"
}

write_status() {
  local status="$1" stage="$2"
  python - "$STATUS_JSON" "$status" "$stage" <<'PY'
import json, sys, time
path, status, stage = sys.argv[1:4]
payload = {"status": status, "stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY
}

run_stage() {
  local stage="$1"
  write_status running "$stage"
  log "START $stage"
  conda run --no-capture-output -n "$ENV_NAME" python scripts/paper_defense/run_experiment.py \
    --run-group "$RUN_GROUP" \
    --stages "$stage" \
    --tasks tfbind10 dkitty \
    --seeds 0 1 2 3 4 5 6 7 2>&1 | tee -a "$DRIVER_LOG"
  log "AGGREGATE after $stage"
  conda run --no-capture-output -n "$ENV_NAME" python scripts/paper_defense/aggregate_results.py "$RUN_GROUP" 2>&1 | tee -a "$DRIVER_LOG"
  log "DONE $stage"
}

log "RUN_GROUP=$RUN_GROUP"
log "ENV_NAME=$ENV_NAME"
write_status running setup
run_stage stage1
run_stage stage2
run_stage stage3
write_status completed all
log "FULL PROTOCOL COMPLETE"
