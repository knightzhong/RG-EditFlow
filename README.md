# EditFlow

EditFlow is a reliability-gated edit-flow framework for offline model-based optimization (MBO). It uses offline score models to propose local edits around elite designs, reliability-gated flow matching to learn an edit generator, and trust-region reranking to select final candidates without additional oracle queries during candidate construction.

This repository contains the code release for EditFlow.

## What Is Included

- `main.py`: command-line entry point for training and evaluation.
- `src/`: flow model, proposal generation, quality gating, trust-region reranking, evaluation, and checkpoint helpers.
- `scripts/`: smoke tests, training/evaluation drivers, and experiment-grid utilities.
- `tests/`: unit tests for core code paths.
- `environment.yml` / `requirements.txt`: dependency specifications.
- `docs/reproduction.md`: commands for smoke checks and experiment reproduction.

Generated artifacts such as `runs/`, checkpoints, caches, and paper PDFs are intentionally excluded from version control.

## Installation

```bash
conda env create -f environment.yml
conda activate editflow
pip install -r requirements.txt
```

## Quick Checks

Run unit tests:

```bash
pytest -q tests/test_quality.py tests/test_evaluation.py
```

Run a small smoke experiment:

```bash
bash scripts/run_smoke.sh
```

Run an experiment-grid dry run without executing expensive jobs:

```bash
python scripts/paper_defense/run_experiment.py --stages stage1 --dry-run
```

## Reproduction

See `docs/reproduction.md` for example commands covering smoke tests, checkpoint evaluation, experiment-grid execution, aggregation, and candidate-table generation.

## Notes

Design-Bench tasks may download or load benchmark data through the `design-bench` package. Full experiments require a CUDA-capable GPU and can be time-consuming.
