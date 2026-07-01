from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional


METRIC_FIELDS = [
    "normalized_final_mean_mean",
    "normalized_p80_mean",
    "normalized_p100_mean",
    "selected_proxy_mean_mean",
    "selected_distance_mean_mean",
    "selected_displacement_mean_mean",
    "raw_displacement_mean_mean",
    "wall_seconds_mean",
    "peak_gpu_memory_mb_mean",
]

STAGE1_VARIANTS = ["full", "no_trust_region", "proxy_only", "trust_region_proxy_only"]
RERANK_VARIANTS = ["wu_0p25_wd_0p5", "wu_0_wd_0", "wu_0p25_wd_0", "wu_0_wd_0p5", "wu_0p25_wd_1"]
PROPOSAL_VARIANTS = ["m_1", "m_4", "m_8", "m_16", "m_32"]
GATE_VARIANTS = ["none", "score", "geometry", "full"]

LABELS = {
    "full": "Full",
    "no_trust_region": "No trust region",
    "proxy_only": "Proxy only",
    "trust_region_proxy_only": "Trust region + proxy only",
    "wu_0p25_wd_0p5": "Baseline $(w_u=.25,w_d=.5)$",
    "wu_0_wd_0": "Proxy only $(0,0)$",
    "wu_0p25_wd_0": "No distance penalty",
    "wu_0_wd_0p5": "No uncertainty penalty",
    "wu_0p25_wd_1": "Stronger distance penalty",
    "m_1": "$M=1$",
    "m_4": "$M=4$",
    "m_8": "$M=8$",
    "m_16": "$M=16$",
    "m_32": "$M=32$",
    "none": "No gate",
    "score": "Score-only",
    "geometry": "Geometry-only",
    "full": "Full",
}


def read_summary(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    return round(value, digits) if value is not None else None


def variant_label(variant: str) -> str:
    return LABELS.get(variant, variant.replace("_", " "))


def sorted_stage_rows(summary_rows: Iterable[dict], stage: str, variants: List[str]) -> List[dict]:
    rank = {variant: index for index, variant in enumerate(variants)}
    rows = [row for row in summary_rows if row.get("stage") == stage and row.get("variant") in rank]
    return sorted(rows, key=lambda row: (row.get("task", ""), rank[row.get("variant", "")]))


def compact_metric_row(row: dict) -> dict:
    out = {
        "task": row.get("task"),
        "variant": row.get("variant"),
        "label": variant_label(row.get("variant", "")),
        "n": int(float(row.get("n") or 0)),
    }
    for field in METRIC_FIELDS:
        out[field.replace("_mean", "")] = rounded(as_float(row.get(field)))
    return out


def compact_stage_rows(summary_rows: Iterable[dict], stage: str, variants: List[str]) -> List[dict]:
    return [compact_metric_row(row) for row in sorted_stage_rows(summary_rows, stage, variants)]


def proxy_oracle_rows(summary_rows: Iterable[dict]) -> List[dict]:
    stage_rows = sorted_stage_rows(summary_rows, "stage1_diagnostics", STAGE1_VARIANTS)
    full_by_task = {row.get("task"): row for row in stage_rows if row.get("variant") == "full"}
    out = []
    for row in stage_rows:
        full = full_by_task.get(row.get("task"), {})
        normalized_mean = as_float(row.get("normalized_final_mean_mean"))
        full_mean = as_float(full.get("normalized_final_mean_mean"))
        distance = as_float(row.get("selected_distance_mean_mean"))
        full_distance = as_float(full.get("selected_distance_mean_mean"))
        displacement = as_float(row.get("selected_displacement_mean_mean"))
        full_displacement = as_float(full.get("selected_displacement_mean_mean"))
        selected_proxy = as_float(row.get("selected_proxy_mean_mean"))
        full_proxy = as_float(full.get("selected_proxy_mean_mean"))
        out.append({
            "task": row.get("task"),
            "variant": row.get("variant"),
            "label": variant_label(row.get("variant", "")),
            "n": int(float(row.get("n") or 0)),
            "normalized_mean": rounded(normalized_mean),
            "normalized_p80": rounded(as_float(row.get("normalized_p80_mean"))),
            "normalized_p100": rounded(as_float(row.get("normalized_p100_mean"))),
            "selected_proxy": rounded(selected_proxy),
            "selected_distance": rounded(distance),
            "selected_displacement": rounded(displacement),
            "raw_displacement": rounded(as_float(row.get("raw_displacement_mean_mean"))),
            "normalized_mean_delta_vs_full": rounded(normalized_mean - full_mean if normalized_mean is not None and full_mean is not None else None),
            "selected_proxy_delta_vs_full": rounded(selected_proxy - full_proxy if selected_proxy is not None and full_proxy is not None else None),
            "selected_distance_delta_vs_full": rounded(distance - full_distance if distance is not None and full_distance is not None else None),
            "selected_displacement_delta_vs_full": rounded(displacement - full_displacement if displacement is not None and full_displacement is not None else None),
        })
    return out


def risk_bin_summary_rows(risk_rows: Iterable[dict]) -> List[dict]:
    groups = defaultdict(list)
    for row in risk_rows:
        if row.get("stage") != "stage1_diagnostics":
            continue
        key = (row.get("task"), row.get("variant"), row.get("signal"), row.get("bin"))
        groups[key].append(row)
    out = []
    for (task, variant, signal, bin_id), rows in sorted(groups.items()):
        improvements = [value for value in (as_float(row.get("normalized_oracle_improvement_mean")) for row in rows) if value is not None]
        selected_rates = [value for value in (as_float(row.get("selected_rate")) for row in rows) if value is not None]
        selected_improvements = [value for value in (as_float(row.get("selected_normalized_oracle_improvement_mean")) for row in rows) if value is not None]
        out.append({
            "task": task,
            "variant": variant,
            "label": variant_label(variant or ""),
            "signal": signal,
            "bin": int(float(bin_id or 0)),
            "records": len(rows),
            "total_n": int(sum(as_float(row.get("n")) or 0 for row in rows)),
            "normalized_oracle_improvement_mean": rounded(mean(improvements) if improvements else None),
            "selected_rate_mean": rounded(mean(selected_rates) if selected_rates else None),
            "selected_normalized_oracle_improvement_mean": rounded(mean(selected_improvements) if selected_improvements else None),
        })
    return out


def write_csv(path: Path, rows: List[dict]) -> Optional[Path]:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def latex_value(value: object) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text.replace("_", "\\_")


def write_tex_table(path: Path, rows: List[dict], fields: List[str], caption: str, label: str) -> Optional[Path]:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" * len(fields)
    linebreak = " \\\\" 
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(field.replace("_", " ") for field in fields) + linebreak,
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_value(row.get(field)) for field in fields) + linebreak)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def maybe_plot_proxy_oracle(rows: List[dict], path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    markers = {"tfbind10": "o", "dkitty": "s"}
    for row in rows:
        proxy = row.get("selected_proxy")
        score = row.get("normalized_mean")
        if proxy is None or score is None:
            continue
        ax.scatter(proxy, score, marker=markers.get(row.get("task"), "o"), s=55)
        ax.annotate(f"{row.get('task')}:{row.get('variant')}", (proxy, score), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Selected proxy score")
    ax.set_ylabel("Normalized oracle mean")
    ax.set_title("Proxy score does not certify endpoint quality")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def maybe_plot_gate(rows: List[dict], path: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({row["task"] for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        task_rows = [row for row in rows if row["task"] == task]
        labels = [row["label"] for row in task_rows]
        means = [row["normalized_final"] for row in task_rows]
        p100s = [row["normalized_p100"] for row in task_rows]
        positions = list(range(len(task_rows)))
        axis.bar([position - 0.18 for position in positions], means, width=0.36, label="Mean")
        axis.bar([position + 0.18 for position in positions], p100s, width=0.36, label="P100")
        axis.set_title(task)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylim(0, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_memo(path: Path, proxy_rows: List[dict], rerank_rows: List[dict], proposal_rows: List[dict], gate_rows: List[dict], figure_paths: List[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Defense Candidate Memo",
        "",
        "## Generated Candidate Assets",
        "- `stage4_proxy_oracle_gap.csv/.tex`: endpoint-level proxy/oracle/OOD drift evidence.",
        "- `stage4_risk_bins_summary.csv/.tex`: proposal-level risk-bin evidence from saved diagnostics.",
        "- `stage5_stage2_rerank_weights.csv/.tex`: reduced rerank-weight robustness grid.",
        "- `stage5_stage2_num_proposals.csv/.tex`: proposal-count robustness grid.",
        "- `stage5_stage3_gate_ablation.csv/.tex`: cross-task reliability gate ablation.",
        "",
        "## Main Reading",
        "- Stage 1 is the strongest mechanism evidence: proxy-only/no-trust variants can raise selected proxy or drift farther while lowering oracle quality.",
        "- Stage 2 should be reported as robustness over a fixed reduced grid, not as best-point selection.",
        "- Stage 3 supports a conservative gate claim: cross-task gate variants are stable; full gate is not uniformly dominant on every metric.",
        "",
        "## Key Stage 1 Rows",
    ]
    for row in proxy_rows:
        lines.append(
            f"- {row['task']} / {row['variant']}: mean={row['normalized_mean']:.3f}, "
            f"p100={row['normalized_p100']:.3f}, proxy={row['selected_proxy']:.3f}, "
            f"distance={row['selected_distance']:.3f}, mean_delta_vs_full={row['normalized_mean_delta_vs_full']:.3f}."
        )
    lines.extend(["", "## Candidate Figures"])
    if figure_paths:
        for figure_path in figure_paths:
            lines.append(f"- `{figure_path.name}`")
    else:
        lines.append("- PDF figure generation was skipped because plotting dependencies were unavailable or disabled.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_candidates(run_group: Path, make_plots: bool = True) -> List[Path]:
    tables = run_group / "tables"
    candidates = run_group / "paper_candidates"
    figures = run_group / "figures"
    summary_rows = read_summary(tables / "summary_by_variant.csv")
    proxy_rows = proxy_oracle_rows(summary_rows)
    risk_rows = risk_bin_summary_rows(read_csv(tables / "risk_bins.csv"))
    rerank_rows = compact_stage_rows(summary_rows, "stage2_rerank_weights", RERANK_VARIANTS)
    proposal_rows = compact_stage_rows(summary_rows, "stage2_num_proposals", PROPOSAL_VARIANTS)
    gate_rows = compact_stage_rows(summary_rows, "stage3_gate_eval", GATE_VARIANTS)

    outputs: List[Path] = []
    table_specs = [
        ("stage4_proxy_oracle_gap", proxy_rows, ["task", "label", "n", "normalized_mean", "normalized_p100", "selected_proxy", "selected_distance", "selected_displacement", "normalized_mean_delta_vs_full", "selected_distance_delta_vs_full"]),
        ("stage4_risk_bins_summary", risk_rows, ["task", "label", "signal", "bin", "records", "total_n", "normalized_oracle_improvement_mean", "selected_rate_mean", "selected_normalized_oracle_improvement_mean"]),
        ("stage5_stage2_rerank_weights", rerank_rows, ["task", "label", "n", "normalized_final", "normalized_p80", "normalized_p100", "selected_proxy", "selected_distance", "selected_displacement"]),
        ("stage5_stage2_num_proposals", proposal_rows, ["task", "label", "n", "normalized_final", "normalized_p80", "normalized_p100", "selected_proxy", "selected_distance", "selected_displacement"]),
        ("stage5_stage3_gate_ablation", gate_rows, ["task", "label", "n", "normalized_final", "normalized_p80", "normalized_p100", "selected_proxy", "selected_distance", "selected_displacement"]),
    ]
    for stem, rows, fields in table_specs:
        csv_path = write_csv(candidates / f"{stem}.csv", rows)
        if csv_path:
            outputs.append(csv_path)
            if stem.startswith("stage4_"):
                table_copy = write_csv(tables / f"{stem}.csv", rows)
                if table_copy:
                    outputs.append(table_copy)
        tex_path = write_tex_table(candidates / f"{stem}.tex", rows, fields, stem.replace("_", " ").title(), f"tab:{stem}")
        if tex_path:
            outputs.append(tex_path)

    figure_paths: List[Path] = []
    if make_plots:
        for figure_path in [
            maybe_plot_proxy_oracle(proxy_rows, figures / "stage4_proxy_oracle_gap.pdf"),
            maybe_plot_gate(gate_rows, figures / "stage5_stage3_gate_ablation.pdf"),
        ]:
            if figure_path:
                outputs.append(figure_path)
                candidate_figure = candidates / figure_path.name
                shutil.copyfile(figure_path, candidate_figure)
                figure_paths.append(candidate_figure)
                outputs.append(candidate_figure)

    outputs.append(write_memo(candidates / "stage5_result_memo.md", proxy_rows, rerank_rows, proposal_rows, gate_rows, figure_paths))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 4/5 paper-defense candidate assets")
    parser.add_argument("run_group", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    outputs = write_candidates(args.run_group, make_plots=not args.no_plots)
    print(f"[PAPER_CANDIDATES] wrote {len(outputs)} files")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
