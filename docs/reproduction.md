# Reproduction Guide

This guide gives commands for rerunning TrajFlow experiments from the code-only release. It does not include precomputed results.

## Environment

```bash
conda env create -f environment.yml
conda activate ggfm-a407d81-fm
python -c "import torch, gpytorch, design_bench; print(torch.__version__, torch.cuda.is_available(), gpytorch.__version__)"
```

## Smoke Test

```bash
bash scripts/run_smoke.sh
```

This writes new outputs under local `runs/` and `checkpoints_smoke/` directories.

## Train or Evaluate Main Models

See the command-line options:

```bash
python main.py --help
```

Example evaluation command after training a checkpoint:

```bash
python -u main.py \
  --eval-only \
  --load-checkpoint checkpoints_quality_full/cfm_model_final.pt \
  --num-test-samples 128 \
  --num-proposals 16 \
  --proposal-noise-scale 0.005 \
  --proposal-max-displacement 2.0 \
  --rerank-mode per-seed \
  --uncertainty-mode label-variance \
  --rerank-uncertainty-weight 0.25 \
  --rerank-distance-weight 0.5 \
  --metrics-path runs/example_eval/metrics.json
```

## Paper-Defense Experiment Grid

Dry run:

```bash
python scripts/paper_defense/run_experiment.py --stages stage1 --dry-run
```

Run selected stages:

```bash
python scripts/paper_defense/run_experiment.py --stages stage1 stage2 --tasks tfbind10 dkitty --seeds 0 1 2 3 4 5 6 7
```

Aggregate generated run outputs:

```bash
python scripts/paper_defense/aggregate_results.py runs/paper_defense_YYYYMMDD_HHMMSS
```

Build candidate tables from generated run outputs:

```bash
python scripts/paper_defense/build_paper_candidates.py runs/paper_defense_YYYYMMDD_HHMMSS
```

## Expected Generated Directories

The scripts may create these local directories when run:

- `runs/`
- `checkpoints*/`
- `paper_candidates/` inside a run group

These generated directories are excluded from the code-only release and should not be committed unless a separate artifact-release policy is used.
