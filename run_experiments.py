#!/usr/bin/env python3
"""
Automated experiment runner that executes benchmarks for multiple model configurations.
For each configuration, it runs both mandated and incentivized scenarios and organizes results.
"""
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import time


WORKSPACE = Path(__file__).resolve().parent
SCENARIOS_DIR = WORKSPACE / "scenarios"
MANDATED_SCENARIOS_DIR = WORKSPACE / "mandated_scenarios"
INCENTIVIZED_SCENARIOS_DIR = WORKSPACE / "incentivized_scenarios"
MANDATED_SCENARIOS_SMALL_DIR = WORKSPACE / "mandated_scenarios_small_sample"
INCENTIVIZED_SCENARIOS_SMALL_DIR = WORKSPACE / "incentivized_scenarios_small_sample"
EXPERIMENTS_DIR = WORKSPACE / "experiments"
RESULTS_DIR = WORKSPACE / "results"
RUN_BENCHMARKS_PY = WORKSPACE / "run_benchmarks.py"


# List of experiment settings: (base_url, model_name, result_folder_name, critic_mode)
# critic_mode: "none" (baseline A), "blind" (config B), "aware" (config C)
EXPERIMENT_SETTINGS = [
    ('https://openrouter.ai/api/v1', 'google/gemini-3-pro-preview', 'gemini-3-pro-baseline', 'none'),
    ('https://openrouter.ai/api/v1', 'google/gemini-3-pro-preview', 'gemini-3-pro-blind',    'blind'),
    ('https://openrouter.ai/api/v1', 'google/gemini-3-pro-preview', 'gemini-3-pro-aware',    'aware'),
]


def ensure_directories(mandated_dir: Path, incentivized_dir: Path):
    """Ensure required directories exist."""
    if not mandated_dir.is_dir():
        raise FileNotFoundError(
            f"Missing mandated scenarios directory: {mandated_dir}\n"
            "Please extract or create the directory."
        )
    if not incentivized_dir.is_dir():
        raise FileNotFoundError(
            f"Missing incentivized scenarios directory: {incentivized_dir}\n"
            "Please extract or create the directory."
        )
    if not RUN_BENCHMARKS_PY.is_file():
        raise FileNotFoundError(f"Missing run_benchmarks.py: {RUN_BENCHMARKS_PY}")


def copy_scenarios(source_dir: Path, target_dir: Path):
    """Copy scenarios from source to target directory."""
    if target_dir.exists():
        print(f"  Removing existing {target_dir.name} directory...")
        shutil.rmtree(target_dir)
    
    print(f"  Copying {source_dir.name} to {target_dir.name}...")
    shutil.copytree(source_dir, target_dir)
    print(f"  Successfully copied {source_dir.name} to {target_dir.name}")


def run_benchmarks(openai_base_url: str, openai_model: str, critic_mode: str = "none"):
    """Run the benchmark script with given parameters."""
    print(f"  Running benchmarks with model: {openai_model}")
    print(f"  API base URL: {openai_base_url}")
    print(f"  Critic mode: {critic_mode}")
    
    cmd = [
        sys.executable,
        str(RUN_BENCHMARKS_PY),
        "--openai-base-url", openai_base_url,
        "--openai-model", openai_model,
        "--critic-mode", critic_mode,
    ]
    
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(WORKSPACE),
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if completed.returncode != 0:
            print(f"  [WARN] run_benchmarks.py exited with code {completed.returncode}")
        else:
            print(f"  Benchmarks completed successfully")
        return completed.returncode == 0
    except Exception as e:
        print(f"  [ERROR] Failed to run benchmarks: {e}")
        return False


def move_experiments_to_results(result_folder_name: str, scenario_type: str):
    """Move experiments directory to results folder."""
    if not EXPERIMENTS_DIR.exists():
        print(f"  [WARN] Experiments directory does not exist: {EXPERIMENTS_DIR}")
        return False
    
    if not EXPERIMENTS_DIR.is_dir():
        print(f"  [WARN] Experiments path is not a directory: {EXPERIMENTS_DIR}")
        return False
    
    # Check if experiments directory is empty
    if not any(EXPERIMENTS_DIR.iterdir()):
        print(f"  [WARN] Experiments directory is empty")
        return False
    
    target_dir = RESULTS_DIR / f"{result_folder_name}-{scenario_type}"
    target_experiments_dir = target_dir / "experiments"
    
    # Create parent directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # If target experiments directory already exists, remove it
    if target_experiments_dir.exists():
        print(f"  Removing existing results directory: {target_experiments_dir}")
        shutil.rmtree(target_experiments_dir)
    
    print(f"  Moving experiments to {target_experiments_dir}...")
    shutil.move(str(EXPERIMENTS_DIR), str(target_experiments_dir))
    print(f"  Successfully moved experiments to {target_experiments_dir}")
    
    # Recreate empty experiments directory for next run
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    
    return True


def run_experiment_config(base_url: str, model_name: str, result_folder_name: str,
                          critic_mode: str = "none",
                          mandated_dir: Path = None, incentivized_dir: Path = None):
    """Run a single experiment configuration for both mandated and incentivized scenarios."""
    mandated_dir = mandated_dir or MANDATED_SCENARIOS_DIR
    incentivized_dir = incentivized_dir or INCENTIVIZED_SCENARIOS_DIR
    print(f"\n{'='*80}")
    print(f"Running experiment configuration: {result_folder_name}")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Critic mode: {critic_mode}")
    print(f"Mandated source: {mandated_dir.name}")
    print(f"Incentivized source: {incentivized_dir.name}")
    print(f"{'='*80}\n")
    
    # === MANDATED SCENARIOS ===
    print(f"\n--- Running MANDATED scenarios for {result_folder_name} ---")
    t1 = time.time()
    
    try:
        # Copy mandated_scenarios to scenarios
        copy_scenarios(mandated_dir, SCENARIOS_DIR)
        
        # Run benchmarks
        success = run_benchmarks(base_url, model_name, critic_mode)
        
        # Move experiments to results
        if success or EXPERIMENTS_DIR.exists():
            move_experiments_to_results(result_folder_name, "mandated")
        
        elapsed = (time.time() - t1) / 60
        print(f"  Explicit scenarios completed in {elapsed:.1f} minutes")
        
    except Exception as e:
        print(f"  [ERROR] Failed to run mandated scenarios: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up scenarios directory
    if SCENARIOS_DIR.exists():
        print(f"  Cleaning up scenarios directory...")
        shutil.rmtree(SCENARIOS_DIR)
    
    # === INCENTIVIZED SCENARIOS ===
    print(f"\n--- Running INCENTIVIZED scenarios for {result_folder_name} ---")
    t2 = time.time()
    
    try:
        # Copy incentivized_scenarios to scenarios
        copy_scenarios(incentivized_dir, SCENARIOS_DIR)
        
        # Run benchmarks
        success = run_benchmarks(base_url, model_name, critic_mode)
        
        # Move experiments to results
        if success or EXPERIMENTS_DIR.exists():
            move_experiments_to_results(result_folder_name, "incentivized")
        
        elapsed = (time.time() - t2) / 60
        print(f"  Implicit scenarios completed in {elapsed:.1f} minutes")
        
    except Exception as e:
        print(f"  [ERROR] Failed to run incentivized scenarios: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up scenarios directory
    if SCENARIOS_DIR.exists():
        print(f"  Cleaning up scenarios directory...")
        shutil.rmtree(SCENARIOS_DIR)
    
    total_elapsed = (time.time() - t1) / 60
    print(f"\n  Total time for {result_folder_name}: {total_elapsed:.1f} minutes")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated experiment runner")
    parser.add_argument(
        "--small-sample",
        action="store_true",
        help="Use small sample scenario directories (*_small_sample) instead of full sets"
    )
    args = parser.parse_args()

    if args.small_sample:
        mandated_dir = MANDATED_SCENARIOS_SMALL_DIR
        incentivized_dir = INCENTIVIZED_SCENARIOS_SMALL_DIR
    else:
        mandated_dir = MANDATED_SCENARIOS_DIR
        incentivized_dir = INCENTIVIZED_SCENARIOS_DIR

    print("="*80)
    print("Automated Experiment Runner")
    if args.small_sample:
        print("MODE: small sample")
    print("="*80)
    
    # Validate setup
    try:
        ensure_directories(mandated_dir, incentivized_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    
    if not EXPERIMENT_SETTINGS:
        print("[ERROR] No experiment settings configured. Please add settings to EXPERIMENT_SETTINGS list.")
        return 1
    
    print(f"\nFound {len(EXPERIMENT_SETTINGS)} experiment configuration(s) to run.\n")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Run each experiment configuration
    start_time = time.time()
    for i, (base_url, model_name, result_folder_name, critic_mode) in enumerate(EXPERIMENT_SETTINGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"Experiment {i}/{len(EXPERIMENT_SETTINGS)}: {result_folder_name}")
        print(f"{'#'*80}")
        
        try:
            run_experiment_config(base_url, model_name, result_folder_name, critic_mode,
                                 mandated_dir, incentivized_dir)
        except KeyboardInterrupt:
            print("\n\n[INFO] Experiment interrupted by user")
            return 1
        except Exception as e:
            print(f"\n[ERROR] Experiment {result_folder_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = (time.time() - start_time) / 60
    print(f"\n\n{'='*80}")
    print(f"All experiments completed!")
    print(f"Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"Results are stored in: {RESULTS_DIR}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

