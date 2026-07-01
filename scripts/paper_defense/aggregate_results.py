from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional

METRIC_FIELDS = [
    "normalized_final_mean",
    "normalized_p80",
    "normalized_p100",
    "normalized_improvement",
    "final_mean",
    "p80",
    "p100",
]
RERANK_FIELDS = [
    "selected_proxy_mean",
    "selected_rerank_mean",
    "selected_distance_mean",
    "selected_displacement_mean",
    "raw_displacement_mean",
    "proxy_mean",
]


def read_json(path: Path) -> Optional[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def rows_from_manifest(run_group: Path) -> List[dict]:
    manifest = run_group / "manifest.jsonl"
    if not manifest.exists():
        return []
    records_by_metrics: Dict[str, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") not in {"completed", "skipped_existing"}:
            continue
        metrics_path = record.get("metrics_path")
        if not metrics_path:
            continue
        previous = records_by_metrics.get(metrics_path)
        if previous is None or previous.get("status") != "completed" or record.get("status") == "completed":
            records_by_metrics[metrics_path] = record

    rows = []
    for record in records_by_metrics.values():
        metrics = read_json(Path(record["metrics_path"]))
        if not metrics:
            continue
        rerank = metrics.get("rerank_info") or {}
        row = {
            "stage": record.get("stage"),
            "task": record.get("task"),
            "variant": record.get("variant"),
            "kind": record.get("kind"),
            "seed": record.get("seed"),
            "wall_seconds": record.get("wall_seconds"),
            "peak_gpu_memory_mb": record.get("peak_gpu_memory_mb"),
            "metrics_path": record.get("metrics_path"),
            "diagnostics_path": record.get("diagnostics_path"),
        }
        for field in METRIC_FIELDS:
            row[field] = metrics.get(field)
        for field in RERANK_FIELDS:
            row[field] = rerank.get(field)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def finite_values(rows: Iterable[dict], field: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        if row.get("kind") == "train":
            continue
        groups[(row.get("stage"), row.get("task"), row.get("variant"))].append(row)
    summaries = []
    for (stage, task, variant), group in sorted(groups.items()):
        out = {"stage": stage, "task": task, "variant": variant, "n": len(group)}
        for field in METRIC_FIELDS + RERANK_FIELDS + ["wall_seconds", "peak_gpu_memory_mb"]:
            values = finite_values(group, field)
            out[f"{field}_mean"] = mean(values) if values else None
            out[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if len(values) == 1 else None
        summaries.append(out)
    return summaries


def quantile_bins(values: List[float], bins: int) -> List[int]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [0] * len(values)
    for rank, idx in enumerate(order):
        labels[idx] = min(bins - 1, int(rank * bins / len(values)))
    return labels


def risk_bin_rows(rows: List[dict], bins: int = 4) -> List[dict]:
    out = []
    for row in rows:
        path_text = row.get("diagnostics_path")
        if not path_text:
            continue
        diag = read_json(Path(path_text))
        if not diag:
            continue
        selected_mask = [bool(x) for x in diag.get("selected_mask", [])]
        improvement = [float(x) for x in diag.get("normalized_oracle_improvement", [])]
        for signal in ["uncertainty", "manifold_distance"]:
            values = [float(x) for x in diag.get(signal, [])]
            if not values or len(values) != len(improvement):
                continue
            labels = quantile_bins(values, bins)
            for bin_id in range(bins):
                indices = [i for i, label in enumerate(labels) if label == bin_id]
                if not indices:
                    continue
                selected = [i for i in indices if i < len(selected_mask) and selected_mask[i]]
                out.append({
                    "stage": row.get("stage"),
                    "task": row.get("task"),
                    "variant": row.get("variant"),
                    "seed": row.get("seed"),
                    "signal": signal,
                    "bin": bin_id,
                    "n": len(indices),
                    "signal_min": min(values[i] for i in indices),
                    "signal_max": max(values[i] for i in indices),
                    "normalized_oracle_improvement_mean": mean(improvement[i] for i in indices),
                    "selected_rate": len(selected) / len(indices),
                    "selected_normalized_oracle_improvement_mean": mean(improvement[i] for i in selected) if selected else None,
                })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate paper-defense experiment results")
    parser.add_argument("run_group")
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()
    run_group = Path(args.run_group)
    tables = run_group / "tables"
    rows = rows_from_manifest(run_group)
    detail_fields = [
        "stage", "task", "variant", "kind", "seed", "wall_seconds", "peak_gpu_memory_mb",
        *METRIC_FIELDS, *RERANK_FIELDS, "metrics_path", "diagnostics_path",
    ]
    write_csv(tables / "all_runs.csv", rows, detail_fields)
    summaries = summarize(rows)
    summary_fields = ["stage", "task", "variant", "n"]
    for field in METRIC_FIELDS + RERANK_FIELDS + ["wall_seconds", "peak_gpu_memory_mb"]:
        summary_fields += [f"{field}_mean", f"{field}_std"]
    write_csv(tables / "summary_by_variant.csv", summaries, summary_fields)
    risk_rows = risk_bin_rows(rows, bins=args.bins)
    write_csv(
        tables / "risk_bins.csv",
        risk_rows,
        [
            "stage", "task", "variant", "seed", "signal", "bin", "n", "signal_min", "signal_max",
            "normalized_oracle_improvement_mean", "selected_rate", "selected_normalized_oracle_improvement_mean",
        ],
    )
    memo = run_group / "paper_candidates" / "result_memo.md"
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(
        "# Paper Defense Result Memo\n\n"
        f"Aggregated {len(rows)} completed/skipped eval records.\n\n"
        "Key files:\n"
        "- `tables/all_runs.csv`: per-run metrics, wall time, peak GPU memory.\n"
        "- `tables/summary_by_variant.csv`: mean/std over seeds by stage/task/variant.\n"
        "- `tables/risk_bins.csv`: proposal risk-bin diagnostics from candidate-level JSON.\n\n"
        "Review the full grids before selecting paper tables; do not report only best grid points.\n",
        encoding="utf-8",
    )
    print(f"[AGGREGATED] rows={len(rows)} summaries={len(summaries)} risk_rows={len(risk_rows)} -> {tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
