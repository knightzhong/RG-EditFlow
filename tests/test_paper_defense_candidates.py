import csv
import importlib.util
import sys
import types
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "paper_defense" / "build_paper_candidates.py"
SPEC = importlib.util.spec_from_file_location("paper_defense_candidates", SCRIPT_PATH)
paper_defense_candidates = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(paper_defense_candidates)


def write_summary(path):
    rows = [
        {
            "stage": "stage1_diagnostics",
            "task": "tfbind10",
            "variant": "full",
            "n": "8",
            "normalized_final_mean_mean": "0.47",
            "normalized_p80_mean": "0.52",
            "normalized_p100_mean": "0.65",
            "selected_proxy_mean_mean": "0.27",
            "selected_distance_mean_mean": "3.1",
            "selected_displacement_mean_mean": "2.0",
            "raw_displacement_mean_mean": "9.7",
        },
        {
            "stage": "stage1_diagnostics",
            "task": "tfbind10",
            "variant": "proxy_only",
            "n": "8",
            "normalized_final_mean_mean": "0.43",
            "normalized_p80_mean": "0.49",
            "normalized_p100_mean": "0.61",
            "selected_proxy_mean_mean": "0.25",
            "selected_distance_mean_mean": "7.4",
            "selected_displacement_mean_mean": "9.8",
            "raw_displacement_mean_mean": "9.8",
        },
        {
            "stage": "stage3_gate_eval",
            "task": "dkitty",
            "variant": "score",
            "n": "8",
            "normalized_final_mean_mean": "0.895",
            "normalized_p80_mean": "0.913",
            "normalized_p100_mean": "0.960",
            "selected_proxy_mean_mean": "0.839",
            "selected_distance_mean_mean": "1.85",
            "selected_displacement_mean_mean": "1.0",
            "raw_displacement_mean_mean": "6.8",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_proxy_oracle_table_reports_delta_from_full(tmp_path):
    summary_path = tmp_path / "summary_by_variant.csv"
    write_summary(summary_path)

    rows = paper_defense_candidates.proxy_oracle_rows(paper_defense_candidates.read_summary(summary_path))

    proxy_row = next(row for row in rows if row["variant"] == "proxy_only")
    assert proxy_row["task"] == "tfbind10"
    assert proxy_row["normalized_mean_delta_vs_full"] == -0.04
    assert proxy_row["selected_distance_delta_vs_full"] == 4.3


def test_write_candidates_creates_stable_csv_tex_and_memo(tmp_path):
    run_group = tmp_path / "run"
    tables = run_group / "tables"
    tables.mkdir(parents=True)
    write_summary(tables / "summary_by_variant.csv")
    (tables / "risk_bins.csv").write_text(
        "stage,task,variant,seed,signal,bin,n,signal_min,signal_max,normalized_oracle_improvement_mean,selected_rate,selected_normalized_oracle_improvement_mean\n",
        encoding="utf-8",
    )

    outputs = paper_defense_candidates.write_candidates(run_group, make_plots=False)

    expected = {
        "stage4_proxy_oracle_gap.csv",
        "stage4_proxy_oracle_gap.tex",
        "stage5_stage3_gate_ablation.csv",
        "stage5_result_memo.md",
    }
    assert expected.issubset({path.name for path in outputs})
    assert (run_group / "paper_candidates" / "stage5_result_memo.md").read_text(encoding="utf-8").startswith("# Paper Defense Candidate Memo")


def test_gate_plot_accepts_compact_stage_rows(tmp_path, monkeypatch):
    class Axis:
        def bar(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_ylim(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class Figure:
        def tight_layout(self):
            return None

        def savefig(self, path):
            Path(path).write_text("pdf", encoding="utf-8")

    pyplot = types.SimpleNamespace(
        subplots=lambda *args, **kwargs: (Figure(), [[Axis()]]),
        close=lambda *args, **kwargs: None,
    )
    matplotlib = types.SimpleNamespace(pyplot=pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    rows = [{"task": "dkitty", "label": "Score-only", "normalized_final": 0.895, "normalized_p100": 0.96}]

    assert paper_defense_candidates.maybe_plot_gate(rows, tmp_path / "gate.pdf") == tmp_path / "gate.pdf"
