#!/usr/bin/env python3
"""
Expert Iteration training loop for Werewolf.

This script runs the full expert iteration pipeline:
1. Collect trajectories from games (using current model or deterministic)
2. Filter trajectories by reward threshold
3. SFT train on filtered trajectories
4. Repeat for multiple iterations
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_trajectory_collection(
    scenarios_path: str,
    output_path: str,
    reward_threshold: float,
    max_scenarios: int,
) -> bool:
    """Run trajectory collection phase."""
    print("\n" + "=" * 60)
    print("Phase 1: Collecting Trajectories")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/collect_trajectories.py"),
        "--scenarios", scenarios_path,
        "--output", output_path,
        "--reward-threshold", str(reward_threshold),
    ]

    if max_scenarios > 0:
        cmd.extend(["--max-scenarios", str(max_scenarios)])

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_sft_training(
    trajectories_path: str,
    model_path: str,
    output_dir: str,
    experiment_name: str,
) -> bool:
    """Run SFT training phase."""
    print("\n" + "=" * 60)
    print("Phase 2: SFT Training")
    print("=" * 60)

    # Use torchrun for distributed training
    cmd = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        str(PROJECT_ROOT / "scripts/train_sft.py"),
        f"data.train_files={trajectories_path}",
        f"data.val_files={trajectories_path}",
        f"model.partial_pretrain={model_path}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.experiment_name={experiment_name}",
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run Expert Iteration for Werewolf")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of expert iteration rounds"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(PROJECT_ROOT / "data/werewolf_scenarios.parquet"),
        help="Path to scenarios parquet file"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to start from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints/werewolf-expert-iter"),
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.0,
        help="Minimum reward to include trajectory"
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=-1,
        help="Max scenarios per iteration (-1 for all)"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run trajectory collection, skip training"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Track current model
    current_model = args.base_model

    for iteration in range(1, args.iterations + 1):
        print("\n" + "#" * 60)
        print(f"# Expert Iteration {iteration}/{args.iterations}")
        print(f"# Model: {current_model}")
        print("#" * 60)

        # Paths for this iteration
        trajectories_path = os.path.join(
            args.output_dir, f"trajectories_iter{iteration}.parquet"
        )

        # Phase 1: Collect trajectories
        success = run_trajectory_collection(
            scenarios_path=args.scenarios,
            output_path=trajectories_path,
            reward_threshold=args.reward_threshold,
            max_scenarios=args.max_scenarios,
        )

        if not success:
            print(f"Error in trajectory collection for iteration {iteration}")
            break

        # Check if we collected any trajectories
        if not os.path.exists(trajectories_path):
            print(f"No trajectories collected for iteration {iteration}")
            break

        if args.collect_only:
            print(f"Collected trajectories saved to {trajectories_path}")
            continue

        # Phase 2: SFT Training
        experiment_name = f"iter{iteration}"
        success = run_sft_training(
            trajectories_path=trajectories_path,
            model_path=current_model,
            output_dir=args.output_dir,
            experiment_name=experiment_name,
        )

        if not success:
            print(f"Error in SFT training for iteration {iteration}")
            break

        # Update model path for next iteration
        # Find the latest checkpoint
        iter_dir = os.path.join(args.output_dir, experiment_name)
        if os.path.exists(iter_dir):
            checkpoints = sorted(
                [d for d in os.listdir(iter_dir) if d.startswith("global_step_")],
                key=lambda x: int(x.split("_")[-1])
            )
            if checkpoints:
                current_model = os.path.join(iter_dir, checkpoints[-1])
                print(f"Updated model to: {current_model}")

    print("\n" + "=" * 60)
    print("Expert Iteration Complete!")
    print(f"Final model: {current_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
