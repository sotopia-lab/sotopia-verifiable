#!/usr/bin/env python3
"""
SFT Training script for Werewolf trajectories.

Uses verl's FSDP SFT trainer to fine-tune on collected trajectories.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add verl to path
VERL_ROOT = PROJECT_ROOT / "dependencies/verl"
sys.path.insert(0, str(VERL_ROOT))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Config path
CONFIG_DIR = str(PROJECT_ROOT / "config")


@hydra.main(config_path=CONFIG_DIR, config_name="werewolf_sft", version_base=None)
def main(config):
    """Run SFT training on collected trajectories."""

    print("=" * 60)
    print("Werewolf Expert Iteration - SFT Training")
    print("=" * 60)

    # Resolve config
    OmegaConf.resolve(config)

    print(f"\nModel: {config.model.partial_pretrain}")
    print(f"Train data: {config.data.train_files}")
    print(f"Batch size: {config.data.train_batch_size}")
    print(f"Max length: {config.data.max_length}")
    print(f"Epochs: {config.trainer.total_epochs}")

    # Check if data exists
    train_path = os.path.expandvars(config.data.train_files)
    if not os.path.exists(train_path):
        print(f"\nError: Training data not found at {train_path}")
        print("Run: python scripts/collect_trajectories.py first")
        return

    # Import and run verl's SFT trainer
    from verl.trainer.fsdp_sft_trainer import run_sft

    print("\nStarting SFT training...")
    run_sft(config)

    print("\nTraining complete!")


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
