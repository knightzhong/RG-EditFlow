# EditFlow Experiment Utilities

This folder runs non-overwriting experiment grids for EditFlow's local edit proposal and trust-region filtering workflow.

Typical usage:

```bash
python scripts/paper_defense/run_experiment.py --stages stage1 --dry-run
python scripts/paper_defense/aggregate_results.py runs/paper_defense_YYYYMMDD_HHMMSS
```

Generated run groups are written under `runs/` and are excluded from version control.
