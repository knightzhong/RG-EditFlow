from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from experiment_grid import SEEDS, TASKS, build_jobs


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_text(command: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(command, cwd=str(cwd), stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def init_run_group(run_group: Path, cwd: Path, env_name: str, args: argparse.Namespace) -> None:
    run_group.mkdir(parents=True, exist_ok=True)
    for subdir in ["raw", "logs", "gpu", "tables", "figures", "paper_candidates", "checkpoints", "metadata"]:
        (run_group / subdir).mkdir(parents=True, exist_ok=True)
    protocol_path = run_group / "protocol.json"
    if not protocol_path.exists():
        protocol = {
            "created_at": now_iso(),
            "cwd": str(cwd),
            "env_name": env_name,
            "notes": "Paper-defense experiments for local edit proposal + filtering. Fixed oracle-free design priors; full grids retained.",
        }
        protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    invocation = {
        "invoked_at": now_iso(),
        "env_name": env_name,
        "stages": args.stages,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "max_runs": args.max_runs,
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
    }
    with (run_group / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(invocation, ensure_ascii=False, sort_keys=True) + "\n")
    metadata = run_group / "metadata"
    (metadata / "git_commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"], cwd) + "\n")
    (metadata / "git_branch.txt").write_text(run_text(["git", "branch", "--show-current"], cwd) + "\n")
    (metadata / "git_status.txt").write_text(run_text(["git", "status", "--short"], cwd) + "\n")
    (metadata / "git_diff.patch").write_text(run_text(["git", "diff", "--", "."], cwd) + "\n")
    (metadata / "gpu_info.txt").write_text(run_text(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], cwd) + "\n")
    (metadata / "python_info.txt").write_text(run_text(["conda", "run", "--no-capture-output", "-n", env_name, "python", "-c", "import sys, torch; print(sys.version); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"], cwd) + "\n")
    readme = run_group / "protocol.md"
    if not readme.exists():
        readme.write_text(
            "# Paper Defense Experiment Protocol\n\n"
            "This run group is append-only. Each job writes to an isolated directory under `raw/`, `logs/`, and `gpu/`.\n\n"
            "Stages: stage1 diagnostics, stage2 inference robustness, stage3 gate training/evaluation, stage4 aggregation, stage5 paper candidates.\n\n"
            "No result file should be overwritten unless the user explicitly passes `--force`.\n",
            encoding="utf-8",
        )


def parse_memory_used(csv_line: str) -> Optional[int]:
    parts = [part.strip() for part in csv_line.split(",")]
    # monitor_gpu prepends sample_time to the six nvidia-smi fields:
    # sample_time,timestamp,index,name,memory.used,memory.total,utilization.gpu
    if len(parts) >= 7:
        try:
            return int(float(parts[4]))
        except Exception:  # noqa: BLE001
            return None
    if len(parts) >= 6:
        try:
            return int(float(parts[3]))
        except Exception:  # noqa: BLE001
            return None
    return None


def monitor_gpu(path: Path, stop_event: threading.Event, interval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("sample_time,timestamp,index,name,memory_used_mb,memory_total_mb,utilization_gpu_percent\n")
        while not stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                for line in output.splitlines():
                    handle.write(f"{time.time():.3f},{line}\n")
                handle.flush()
            except Exception as exc:  # noqa: BLE001
                handle.write(f"{time.time():.3f},UNAVAILABLE,{exc}\n")
                handle.flush()
                return
            stop_event.wait(interval)


def peak_gpu_memory(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    peak: Optional[int] = None
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if i == 0 or "UNAVAILABLE" in line:
            continue
        memory = parse_memory_used(line)
        if memory is not None:
            peak = memory if peak is None else max(peak, memory)
    return peak


def append_manifest(run_group: Path, record: Dict[str, object]) -> None:
    manifest = run_group / "manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def command_for_job(env_name: str, job: Dict[str, object]) -> List[str]:
    return ["conda", "run", "--no-capture-output", "-n", env_name, "python", "-u", "main.py"] + list(job["args"])


def should_skip(job: Dict[str, object], force: bool) -> bool:
    metrics_path = Path(str(job["metrics_path"]))
    record_path = Path(str(job["record_path"]))
    return not force and metrics_path.exists() and metrics_path.stat().st_size > 0 and record_path.exists()


def run_job(job: Dict[str, object], run_group: Path, cwd: Path, env_name: str, gpu_interval: float, force: bool, dry_run: bool) -> Dict[str, object]:
    command = command_for_job(env_name, job)
    log_path = Path(str(job["log_path"]))
    gpu_path = Path(str(job["gpu_path"]))
    record_path = Path(str(job["record_path"]))
    metrics_path = Path(str(job["metrics_path"]))
    diagnostics_path = Path(str(job.get("diagnostics_path") or "")) if job.get("diagnostics_path") else None
    for path in [log_path.parent, gpu_path.parent, record_path.parent, metrics_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    base_record: Dict[str, object] = {
        **{key: job[key] for key in ["stage", "task", "variant", "seed", "kind"]},
        "command": command,
        "command_string": shlex.join(command),
        "metrics_path": str(metrics_path),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path is not None else None,
        "log_path": str(log_path),
        "gpu_path": str(gpu_path),
        "record_path": str(record_path),
    }
    if dry_run:
        return {**base_record, "status": "dry_run"}
    if should_skip(job, force):
        record = {**base_record, "status": "skipped_existing", "ended_at": now_iso(), "peak_gpu_memory_mb": peak_gpu_memory(gpu_path)}
        append_manifest(run_group, record)
        return record

    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(gpu_path, stop_event, gpu_interval), daemon=True)
    start_time = time.time()
    record = {**base_record, "status": "running", "started_at": now_iso()}
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    monitor.start()
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/ggfm_a407d81_pycache")
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"[START] {now_iso()}\n")
        log_handle.write(shlex.join(command) + "\n\n")
        log_handle.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        return_code = proc.wait()
        log_handle.write(f"\n[END] {now_iso()} return_code={return_code}\n")
    stop_event.set()
    monitor.join(timeout=max(2.0, gpu_interval + 1.0))
    wall_seconds = time.time() - start_time
    status = "completed" if return_code == 0 else "failed"
    record = {
        **base_record,
        "status": status,
        "return_code": return_code,
        "started_at": record["started_at"],
        "ended_at": now_iso(),
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory(gpu_path),
    }
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    append_manifest(run_group, record)
    if return_code != 0:
        raise RuntimeError(f"Job failed: {record['stage']} {record['task']} {record['variant']} seed={record['seed']} log={log_path}")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-defense experiments with reproducible logging")
    parser.add_argument("--run-group", default=None, help="Existing or new run group directory")
    parser.add_argument("--stages", nargs="+", choices=["stage1", "stage2", "stage3"], default=["stage1"])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=sorted(TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--env-name", default=os.environ.get("GGFM_ENV", "root_mbo"))
    parser.add_argument("--gpu-interval", type=float, default=2.0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow rerunning jobs whose metrics already exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    run_group = Path(args.run_group) if args.run_group else Path("runs") / f"paper_defense_{timestamp()}"
    init_run_group(run_group, cwd, args.env_name, args)
    jobs = build_jobs(run_group, args.stages, args.tasks, args.seeds)
    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]
    plan_path = run_group / "planned_jobs.json"
    plan_path.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[RUN_GROUP] {run_group}")
    print(f"[JOBS] {len(jobs)}")
    if args.dry_run:
        for job in jobs[:20]:
            print(shlex.join(command_for_job(args.env_name, job)))
        if len(jobs) > 20:
            print(f"... {len(jobs) - 20} more jobs")
        return 0
    completed = 0
    for job in jobs:
        print(f"[JOB] {job['stage']} {job['task']} {job['variant']} seed={job['seed']} kind={job['kind']}", flush=True)
        record = run_job(job, run_group, cwd, args.env_name, args.gpu_interval, args.force, args.dry_run)
        completed += 1
        print(f"[DONE] status={record['status']} wall={record.get('wall_seconds')} peak_gpu={record.get('peak_gpu_memory_mb')}", flush=True)
    print(f"[COMPLETE] {completed} jobs processed in {run_group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
