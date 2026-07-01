# TrajFlow Code Release

This repository contains the source code for TrajFlow, a reliability-gated trajectory-flow method for offline model-based optimization.

This reviewer-facing package intentionally contains **code only**:

- no pretrained checkpoints;
- no generated experiment results;
- no `runs/` directories;
- no paper tables, figures, or PDFs;
- no private planning notes or intermediate logs.

The code can train models, evaluate checkpoints, run paper-defense experiment grids, and regenerate result artifacts from newly produced runs.

## Contents

- `main.py`: command-line entry point for training and evaluation.
- `src/`: model, generator, GP teacher, trajectory-source, quality-gating, evaluation, and checkpoint helpers.
- `scripts/`: smoke, training, evaluation, and paper-defense experiment drivers.
- `tests/`: unit tests for core code paths.
- `environment.yml` / `requirements.txt`: dependency specifications.
- `docs/reproduction.md`: command examples for running smoke checks and reproducing experiments from scratch.

## Quick Start

```bash
conda env create -f environment.yml
conda activate ggfm-a407d81-fm
pytest -q tests/test_quality.py tests/test_evaluation.py
```

Run a small smoke experiment:

```bash
bash scripts/run_smoke.sh
```

Run a paper-defense dry run without executing expensive jobs:

```bash
python scripts/paper_defense/run_experiment.py --stages stage1 --dry-run
```

## Notes for Reviewers

Design-Bench tasks may download/load benchmark data through the `design-bench` package. Full experiments require a CUDA-capable GPU and can be time-consuming. The package does not include checkpoint weights or generated results; those are intentionally excluded so reviewers can inspect and rerun the code path directly.
