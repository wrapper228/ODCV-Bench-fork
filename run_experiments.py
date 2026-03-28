#!/usr/bin/env python3
"""
Parallel experiment runner for ODCV-Bench.

Default behavior runs all 6 combinations in parallel:
  (mandated, incentivized) x (none, blind, aware)

Each job runs in an isolated runtime directory under .parallel_runs to avoid
filesystem races and docker compose naming/port conflicts.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _load_dotenv(env_path=None):
    """Load .env into os.environ; existing vars take precedence."""
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


WORKSPACE = Path(__file__).resolve().parent
RUN_BENCHMARKS_SCRIPT = WORKSPACE / "run_benchmarks.py"
RUN_SCRIPT = WORKSPACE / "run.py"
DOCKER_COMPOSE_FILE = WORKSPACE / "docker-compose.yml"
MISSION_EXECUTOR_DIR = WORKSPACE / "mission_executor"
ORCHESTRATOR_ZIP = WORKSPACE / "orchestrator_api.zip"

MANDATED_SCENARIOS_DIR = WORKSPACE / "mandated_scenarios"
INCENTIVIZED_SCENARIOS_DIR = WORKSPACE / "incentivized_scenarios"
MANDATED_SCENARIOS_SMALL_DIR = WORKSPACE / "mandated_scenarios_small_sample"
INCENTIVIZED_SCENARIOS_SMALL_DIR = WORKSPACE / "incentivized_scenarios_small_sample"

RESULTS_DIR = WORKSPACE / "results"
PARALLEL_RUNS_DIR = WORKSPACE / ".parallel_runs"


# List of experiment settings: (base_url, model_name, result_folder_name, critic_mode, critic_model)
# critic_mode: "none" (baseline A), "blind" (config B), "aware" (config C)
EXPERIMENT_SETTINGS = [
    ("https://openrouter.ai/api/v1", "qwen/qwen3.5-plus-02-15", "qwen35plus-baseline", "none",  ""),
    ("https://openrouter.ai/api/v1", "qwen/qwen3.5-plus-02-15", "qwen35plus-blind",    "blind", "google/gemini-3.1-flash-lite-preview"),
    ("https://openrouter.ai/api/v1", "qwen/qwen3.5-plus-02-15", "qwen35plus-aware",    "aware", "google/gemini-3.1-flash-lite-preview"),
]


@dataclass(frozen=True)
class ExperimentSetting:
    base_url: str
    model_name: str
    result_folder_name: str
    critic_mode: str
    critic_model: str = ""


@dataclass(frozen=True)
class JobSpec:
    index: int
    job_id: str
    scenario_type: str
    scenario_source: Path
    setting: ExperimentSetting
    runtime_dir: Path
    log_file: Path
    orchestrator_host_port: int
    compose_project_name: str


@dataclass
class JobResult:
    job: JobSpec
    success: bool
    return_code: int | None
    duration_sec: float
    results_path: Path | None
    error: str | None = None


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip().lower())
    return token.strip("-") or "job"


def _sanitize_compose_project(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    token = token.strip("-")
    if not token:
        token = "odcv"
    if not token[0].isalnum():
        token = f"odcv-{token}"
    return token


def ensure_required_paths(mandated_dir: Path, incentivized_dir: Path) -> None:
    required_files = [
        RUN_BENCHMARKS_SCRIPT,
        RUN_SCRIPT,
        DOCKER_COMPOSE_FILE,
        ORCHESTRATOR_ZIP,
    ]
    required_dirs = [
        mandated_dir,
        incentivized_dir,
        MISSION_EXECUTOR_DIR,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required directory: {path}")


def copy_runtime_tree(job: JobSpec) -> None:
    if job.runtime_dir.exists():
        shutil.rmtree(job.runtime_dir)
    job.runtime_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(RUN_BENCHMARKS_SCRIPT, job.runtime_dir / "run_benchmarks.py")
    shutil.copy2(RUN_SCRIPT, job.runtime_dir / "run.py")
    shutil.copy2(DOCKER_COMPOSE_FILE, job.runtime_dir / "docker-compose.yml")
    shutil.copy2(ORCHESTRATOR_ZIP, job.runtime_dir / "orchestrator_api.zip")

    shutil.copytree(MISSION_EXECUTOR_DIR, job.runtime_dir / "mission_executor")
    shutil.copytree(job.scenario_source, job.runtime_dir / "scenarios")
    (job.runtime_dir / "experiments").mkdir(exist_ok=True)

    dotenv_src = WORKSPACE / ".env"
    if dotenv_src.is_file():
        shutil.copy2(dotenv_src, job.runtime_dir / ".env")


def run_benchmarks_for_job(job: JobSpec) -> int:
    env = os.environ.copy()
    env["ORCHESTRATOR_HOST_PORT"] = str(job.orchestrator_host_port)
    env["COMPOSE_PROJECT_NAME"] = job.compose_project_name
    env["NO_TIMESTAMP"] = "true"

    cmd = [
        sys.executable,
        str(job.runtime_dir / "run_benchmarks.py"),
        "--openai-base-url",
        job.setting.base_url,
        "--openai-model",
        job.setting.model_name,
        "--critic-mode",
        job.setting.critic_mode,
        "--critic-model",
        job.setting.critic_model,
    ]

    job.log_file.parent.mkdir(parents=True, exist_ok=True)
    with job.log_file.open("w", encoding="utf-8") as log:
        log.write(f"Job: {job.job_id}\n")
        log.write(f"Scenario type: {job.scenario_type}\n")
        log.write(f"Model: {job.setting.model_name}\n")
        log.write(f"Critic mode: {job.setting.critic_mode}\n")
        log.write(f"Critic model: {job.setting.critic_model or '(same as agent)'}\n")
        log.write(f"Compose project: {job.compose_project_name}\n")
        log.write(f"Orchestrator host port: {job.orchestrator_host_port}\n")
        log.write("=" * 80 + "\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            cwd=str(job.runtime_dir),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def move_experiments_to_results(job: JobSpec) -> Path | None:
    source = job.runtime_dir / "experiments"
    if not source.exists() or not source.is_dir():
        return None
    if not any(source.iterdir()):
        return None

    target_root = RESULTS_DIR / f"{job.setting.result_folder_name}-{job.scenario_type}"
    target = target_root / "experiments"
    target_root.mkdir(parents=True, exist_ok=True)

    if target.exists():
        shutil.rmtree(target)

    shutil.move(str(source), str(target))
    return target


def run_single_job(job: JobSpec, keep_runtime_dirs: bool) -> JobResult:
    started = time.time()
    success = False
    try:
        copy_runtime_tree(job)
        rc = run_benchmarks_for_job(job)
        results_path = move_experiments_to_results(job)
        success = (rc == 0) and (results_path is not None)
        error = None
        if rc != 0:
            error = f"run_benchmarks exited with code {rc}"
        elif results_path is None:
            error = "No experiments were produced"
    except Exception as exc:  # pylint: disable=broad-except
        rc = None
        results_path = None
        success = False
        error = str(exc)
    finally:
        if not keep_runtime_dirs and success and job.runtime_dir.exists():
            shutil.rmtree(job.runtime_dir, ignore_errors=True)

    return JobResult(
        job=job,
        success=success,
        return_code=rc,
        duration_sec=time.time() - started,
        results_path=results_path,
        error=error,
    )


def build_jobs(
    settings: list[ExperimentSetting],
    mandated_source: Path,
    incentivized_source: Path,
    session_id: str,
    base_port: int,
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    matrix: list[tuple[str, Path]] = [
        ("mandated", mandated_source),
        ("incentivized", incentivized_source),
    ]

    idx = 0
    for setting in settings:
        for scenario_type, scenario_source in matrix:
            idx += 1
            job_token = _sanitize_token(f"{idx:02d}-{setting.result_folder_name}-{scenario_type}")
            runtime_dir = PARALLEL_RUNS_DIR / session_id / job_token
            compose_project = _sanitize_compose_project(
                f"odcv-{session_id}-{idx:02d}-{setting.critic_mode}-{scenario_type}"
            )
            jobs.append(
                JobSpec(
                    index=idx,
                    job_id=job_token,
                    scenario_type=scenario_type,
                    scenario_source=scenario_source,
                    setting=setting,
                    runtime_dir=runtime_dir,
                    log_file=runtime_dir / "job.log",
                    orchestrator_host_port=base_port + idx - 1,
                    compose_project_name=compose_project,
                )
            )
    return jobs


def run_jobs_parallel(jobs: list[JobSpec], max_workers: int, keep_runtime_dirs: bool) -> list[JobResult]:
    results: list[JobResult] = []
    workers = max(1, min(max_workers, len(jobs)))

    print(f"Launching {len(jobs)} job(s) in parallel (max_workers={workers})...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run_single_job, job, keep_runtime_dirs): job for job in jobs}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            state = "SUCCESS" if result.success else "FAILED"
            print(
                f"[{state}] {result.job.job_id} | "
                f"mode={result.job.setting.critic_mode} | "
                f"scenario={result.job.scenario_type} | "
                f"duration={result.duration_sec/60:.1f}m"
            )
    return results


def run_jobs_sequential(jobs: list[JobSpec], keep_runtime_dirs: bool) -> list[JobResult]:
    print(f"Launching {len(jobs)} job(s) sequentially...")
    results: list[JobResult] = []
    for job in jobs:
        print(
            f"Running {job.job_id} | mode={job.setting.critic_mode} | "
            f"scenario={job.scenario_type} | port={job.orchestrator_host_port}"
        )
        result = run_single_job(job, keep_runtime_dirs)
        results.append(result)
        state = "SUCCESS" if result.success else "FAILED"
        print(f"[{state}] {job.job_id} completed in {result.duration_sec/60:.1f}m")
    return results


def print_summary(results: list[JobResult], started: float) -> None:
    print("\n" + "=" * 100)
    print("Parallel experiment summary")
    print("=" * 100)

    success_count = 0
    for result in sorted(results, key=lambda r: r.job.index):
        state = "SUCCESS" if result.success else "FAILED"
        details = (
            f"{state:7} | {result.job.job_id} | "
            f"{result.job.scenario_type:11} | critic={result.job.setting.critic_mode:5} | "
            f"port={result.job.orchestrator_host_port} | "
            f"time={result.duration_sec/60:.1f}m"
        )
        print(details)
        if result.results_path is not None:
            print(f"         results: {result.results_path}")
        else:
            print(f"         log: {result.job.log_file}")
        if result.error:
            print(f"         error: {result.error}")
        if result.success:
            success_count += 1

    total_minutes = (time.time() - started) / 60
    print("-" * 100)
    print(f"Completed: {success_count}/{len(results)} jobs succeeded")
    print(f"Total wall time: {total_minutes:.1f} minutes")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 100 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ODCV benchmark experiments.")
    parser.add_argument(
        "--small-sample",
        action="store_true",
        help="Use *_small_sample scenario directories instead of full sets.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run jobs sequentially (legacy behavior). Default is parallel mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum parallel jobs in default parallel mode (default: 6).",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=5000,
        help="Starting host port for orchestrator containers (default: 5000).",
    )
    parser.add_argument(
        "--keep-runtime-dirs",
        action="store_true",
        help="Keep .parallel_runs job directories after completion for debugging.",
    )
    return parser.parse_args()


def main() -> int:
    _load_dotenv()
    args = parse_args()

    if args.max_workers < 1:
        print("[ERROR] --max-workers must be >= 1")
        return 1

    if args.base_port < 1 or args.base_port > 65530:
        print("[ERROR] --base-port must be between 1 and 65530")
        return 1

    mandated_source = MANDATED_SCENARIOS_SMALL_DIR if args.small_sample else MANDATED_SCENARIOS_DIR
    incentivized_source = INCENTIVIZED_SCENARIOS_SMALL_DIR if args.small_sample else INCENTIVIZED_SCENARIOS_DIR

    try:
        ensure_required_paths(mandated_source, incentivized_source)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not EXPERIMENT_SETTINGS:
        print("[ERROR] EXPERIMENT_SETTINGS is empty.")
        return 1

    settings = [ExperimentSetting(*row) for row in EXPERIMENT_SETTINGS]
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = build_jobs(
        settings=settings,
        mandated_source=mandated_source,
        incentivized_source=incentivized_source,
        session_id=session_id,
        base_port=args.base_port,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    session_runtime_root = PARALLEL_RUNS_DIR / session_id
    session_runtime_root.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ODCV Parallel Experiment Runner")
    print(f"Mode: {'sequential' if args.sequential else 'parallel'}")
    print(f"Scenario source: {'small sample' if args.small_sample else 'full benchmark'}")
    print(f"Jobs: {len(jobs)}")
    print(f"Session runtime root: {session_runtime_root}")
    print("=" * 100)

    started = time.time()
    if args.sequential:
        results = run_jobs_sequential(jobs, keep_runtime_dirs=args.keep_runtime_dirs)
    else:
        results = run_jobs_parallel(jobs, max_workers=args.max_workers, keep_runtime_dirs=args.keep_runtime_dirs)

    print_summary(results, started=started)
    if not args.keep_runtime_dirs and session_runtime_root.exists() and not any(session_runtime_root.iterdir()):
        session_runtime_root.rmdir()
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
