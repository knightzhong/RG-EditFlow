# Paper Defense Experiments

This folder runs non-overwriting experiments for the TrajFlow local edit proposal + filtering defense.

## Run groups

Each invocation writes to `runs/paper_defense_<timestamp>/` unless `--run-group` is supplied. A run group contains:

- `metadata/`: git commit, branch, status, diff, GPU info, Python/Torch info.
- `raw/`: one isolated directory per stage/task/variant/seed.
- `logs/`: stdout/stderr logs.
- `gpu/`: sampled GPU memory/utilization CSV.
- `manifest.jsonl`: one record per processed job.
- `tables/`: aggregated CSV outputs.
- `paper_candidates/`: result memo and candidate assets.

## Examples

Dry run:

```bash
conda run --no-capture-output -n root_mbo python scripts/paper_defense/run_experiment.py --stages stage1 --dry-run
```

Run Stage 1:

```bash
conda run --no-capture-output -n root_mbo python scripts/paper_defense/run_experiment.py --stages stage1
```

Aggregate:

```bash
conda run --no-capture-output -n root_mbo python scripts/paper_defense/aggregate_results.py runs/paper_defense_YYYYMMDD_HHMMSS
```

Build Stage 4/5 paper candidates:

```bash
conda run --no-capture-output -n root_mbo python scripts/paper_defense/build_paper_candidates.py runs/paper_defense_YYYYMMDD_HHMMSS
```

This writes mechanism tables, robustness tables, gate-ablation candidates, PDF figures, and `paper_candidates/stage5_result_memo.md` without editing the main paper source.
