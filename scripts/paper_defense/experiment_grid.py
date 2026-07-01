from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SEEDS = list(range(8))
TRAIN_SEED = 42


@dataclass(frozen=True)
class TaskConfig:
    key: str
    task_name: str
    checkpoint: str
    checkpoint_dir: str
    gp_num_fit_samples: Optional[int]
    proposal_noise_scale: float
    default_radius: float
    default_distance_weight: float = 0.5
    default_uncertainty_weight: float = 0.25


TASKS: Dict[str, TaskConfig] = {
    "tfbind10": TaskConfig(
        key="tfbind10",
        task_name="TFBind10-Exact-v0",
        checkpoint="checkpoints_tfbind10_quality_full/cfm_model_final.pt",
        checkpoint_dir="checkpoints_tfbind10_quality_full",
        gp_num_fit_samples=4096,
        proposal_noise_scale=0.005,
        default_radius=2.0,
    ),
    "dkitty": TaskConfig(
        key="dkitty",
        task_name="DKittyMorphology-Exact-v0",
        checkpoint="checkpoints_dkitty_quality_full/cfm_model_final.pt",
        checkpoint_dir="checkpoints_dkitty_quality_full",
        gp_num_fit_samples=4096,
        proposal_noise_scale=0.005,
        default_radius=1.0,
    ),
}


def tag_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def base_eval_args(
    task: TaskConfig,
    seed: int,
    metrics_path: Path,
    diagnostics_path: Optional[Path],
    diagnostics_max_raw: Optional[int] = None,
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> List[str]:
    args = [
        "--task-name", task.task_name,
        "--seed", str(seed),
        "--eval-only",
        "--load-checkpoint", checkpoint or task.checkpoint,
        "--checkpoint-dir", checkpoint_dir or task.checkpoint_dir,
        "--num-test-samples", "128",
        "--metrics-path", str(metrics_path),
    ]
    if task.gp_num_fit_samples is not None:
        args += ["--gp-num-fit-samples", str(task.gp_num_fit_samples)]
    if diagnostics_path is not None:
        args += ["--proposal-diagnostics-path", str(diagnostics_path)]
        if diagnostics_max_raw is not None and int(diagnostics_max_raw) > 0:
            args += ["--proposal-diagnostics-max-raw", str(int(diagnostics_max_raw))]
    return args


def proposal_args(
    task: TaskConfig,
    num_proposals: int = 16,
    radius: Optional[float] = None,
    uncertainty_weight: Optional[float] = None,
    distance_weight: Optional[float] = None,
    noise_scale: Optional[float] = None,
) -> List[str]:
    return [
        "--num-proposals", str(num_proposals),
        "--proposal-noise-scale", str(task.proposal_noise_scale if noise_scale is None else noise_scale),
        "--proposal-max-displacement", str(task.default_radius if radius is None else radius),
        "--rerank-mode", "per-seed",
        "--uncertainty-mode", "label-variance",
        "--rerank-k", "5",
        "--rerank-uncertainty-weight", str(task.default_uncertainty_weight if uncertainty_weight is None else uncertainty_weight),
        "--rerank-distance-weight", str(task.default_distance_weight if distance_weight is None else distance_weight),
    ]


def job_paths(run_group: Path, stage: str, task_key: str, variant: str, seed: int) -> Dict[str, Path]:
    base = run_group / "raw" / stage / task_key / variant / f"seed_{seed}"
    return {
        "job_dir": base,
        "metrics": base / "metrics.json",
        "diagnostics": base / "proposal_diagnostics.json",
        "log": run_group / "logs" / stage / task_key / variant / f"seed_{seed}.log",
        "gpu": run_group / "gpu" / stage / task_key / variant / f"seed_{seed}.csv",
        "record": base / "run_record.json",
    }


def should_save_proposal_diagnostics(stage: str, task_key: str, variant: str, seed: int) -> bool:
    """Keep oracle-heavy proposal diagnostics only where they are worth the cost."""
    if task_key == "tfbind10" and stage == "stage1_diagnostics":
        return True
    if task_key == "dkitty" and stage == "stage1_diagnostics":
        if variant in {"full", "no_trust_region"}:
            return seed in {0, 1}
        if variant in {"proxy_only", "trust_region_proxy_only"}:
            return seed == 0
    return False


def diagnostics_path_for(paths: Dict[str, Path], stage: str, task_key: str, variant: str, seed: int) -> Optional[Path]:
    if should_save_proposal_diagnostics(stage, task_key, variant, seed):
        return paths["diagnostics"]
    return None


def make_job(stage: str, task_key: str, variant: str, seed: int, args: List[str], paths: Dict[str, Path], kind: str = "eval") -> Dict[str, object]:
    return {
        "stage": stage,
        "task": task_key,
        "variant": variant,
        "seed": seed,
        "kind": kind,
        "args": args,
        "metrics_path": str(paths["metrics"]),
        "diagnostics_path": str(paths["diagnostics"]) if paths.get("diagnostics") else None,
        "log_path": str(paths["log"]),
        "gpu_path": str(paths["gpu"]),
        "record_path": str(paths["record"]),
    }


def stage1_jobs(run_group: Path, task_keys: Iterable[str], seeds: Iterable[int]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    variants = {
        "full": {"radius": None, "uncertainty_weight": None, "distance_weight": None},
        "no_trust_region": {"radius": 0.0, "uncertainty_weight": None, "distance_weight": None},
        "proxy_only": {"radius": 0.0, "uncertainty_weight": 0.0, "distance_weight": 0.0},
        "trust_region_proxy_only": {"radius": None, "uncertainty_weight": 0.0, "distance_weight": 0.0},
    }
    for task_key in task_keys:
        task = TASKS[task_key]
        for variant, settings in variants.items():
            for seed in seeds:
                stage = "stage1_diagnostics"
                paths = job_paths(run_group, stage, task_key, variant, seed)
                diagnostics_path = diagnostics_path_for(paths, stage, task_key, variant, seed)
                diagnostics_max_raw = 128 if diagnostics_path is not None and task_key == "dkitty" else None
                args = base_eval_args(task, seed, paths["metrics"], diagnostics_path, diagnostics_max_raw=diagnostics_max_raw)
                args += proposal_args(task, **settings)
                jobs.append(make_job("stage1_diagnostics", task_key, variant, seed, args, paths))
    return jobs


def stage2_jobs(run_group: Path, task_keys: Iterable[str], seeds: Iterable[int]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    # Radius robustness is already available from the existing 8-seed paper assets.
    # Do not rerun it in the lightweight protocol unless a targeted debug run is added later.
    weight_pairs = [
        (0.25, 0.5),  # main setting
        (0.0, 0.0),   # proxy-only endpoint certification
        (0.25, 0.0),  # no manifold-distance penalty
        (0.0, 0.5),   # no uncertainty penalty
        (0.25, 1.0),  # stronger manifold-distance penalty
    ]
    proposal_counts = [1, 4, 8, 16, 32]
    for task_key in task_keys:
        task = TASKS[task_key]
        for uncertainty_weight, distance_weight in weight_pairs:
            variant = f"wu_{tag_float(uncertainty_weight)}_wd_{tag_float(distance_weight)}"
            for seed in seeds:
                paths = job_paths(run_group, "stage2_rerank_weights", task_key, variant, seed)
                args = base_eval_args(task, seed, paths["metrics"], None)
                args += proposal_args(task, uncertainty_weight=uncertainty_weight, distance_weight=distance_weight)
                jobs.append(make_job("stage2_rerank_weights", task_key, variant, seed, args, paths))
        for num_proposals in proposal_counts:
            variant = f"m_{num_proposals}"
            for seed in seeds:
                paths = job_paths(run_group, "stage2_num_proposals", task_key, variant, seed)
                args = base_eval_args(task, seed, paths["metrics"], None)
                args += proposal_args(task, num_proposals=num_proposals)
                jobs.append(make_job("stage2_num_proposals", task_key, variant, seed, args, paths))
    return jobs


def stage3_jobs(run_group: Path, task_keys: Iterable[str], seeds: Iterable[int]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    modes = ["none", "score", "geometry", "full"]
    for task_key in task_keys:
        task = TASKS[task_key]
        for mode in modes:
            checkpoint_dir = run_group / "checkpoints" / "stage3_gate_ablation" / task_key / mode
            train_paths = job_paths(run_group, "stage3_gate_train", task_key, mode, TRAIN_SEED)
            train_args = [
                "--task-name", task.task_name,
                "--seed", str(TRAIN_SEED),
                "--checkpoint-dir", str(checkpoint_dir),
                "--save-every", "0",
                "--num-test-samples", "128",
                "--use-quality-gating",
                "--quality-gate-mode", mode,
                "--metrics-path", str(train_paths["metrics"]),
            ]
            if task.gp_num_fit_samples is not None:
                train_args += ["--gp-num-fit-samples", str(task.gp_num_fit_samples)]
            train_args += proposal_args(task)
            jobs.append(make_job("stage3_gate_train", task_key, mode, TRAIN_SEED, train_args, train_paths, kind="train"))
            checkpoint = checkpoint_dir / "cfm_model_final.pt"
            for seed in seeds:
                eval_paths = job_paths(run_group, "stage3_gate_eval", task_key, mode, seed)
                eval_args = base_eval_args(task, seed, eval_paths["metrics"], None, checkpoint=str(checkpoint), checkpoint_dir=str(checkpoint_dir))
                eval_args += proposal_args(task)
                jobs.append(make_job("stage3_gate_eval", task_key, mode, seed, eval_args, eval_paths, kind="eval"))
    return jobs


def build_jobs(run_group: Path, stages: Iterable[str], task_keys: Iterable[str], seeds: Iterable[int]) -> List[Dict[str, object]]:
    stage_set = list(stages)
    jobs: List[Dict[str, object]] = []
    for stage in stage_set:
        if stage == "stage1":
            jobs.extend(stage1_jobs(run_group, task_keys, seeds))
        elif stage == "stage2":
            jobs.extend(stage2_jobs(run_group, task_keys, seeds))
        elif stage == "stage3":
            jobs.extend(stage3_jobs(run_group, task_keys, seeds))
        else:
            raise ValueError(f"Unknown stage: {stage}")
    return jobs
