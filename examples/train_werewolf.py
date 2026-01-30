#!/usr/bin/env python3
"""
Training script for Werewolf RL using verl with agent loop.

This script sets up training with the WerewolfAgentLoop which handles
all game logic internally. Supports expert iteration / SFT training.
"""
import sys
import os
from pprint import pprint

# Get project root directory (parent of examples/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add project root to path so we can resolve 'sotopia_verifiable'
sys.path.insert(0, PROJECT_ROOT)

# Configure Redis to use Local Forwarding (Port 6380)
os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6380")

# Import the agent loop module to register it with verl
import sotopia_verifiable.agent_loops.werewolf_agent_loop  # noqa: F401

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Config path for our custom werewolf config
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


@hydra.main(config_path=CONFIG_DIR, config_name="werewolf_agent_loop", version_base=None)
def main(config):
    """Main entry point for Werewolf training with agent loop."""

    print("=" * 60)
    print("Werewolf RL Training with verl Agent Loop")
    print("=" * 60)

    # Resolve and print config
    OmegaConf.resolve(config)
    print("\nResolved Configuration:")
    pprint(OmegaConf.to_container(config, resolve=True))
    print("=" * 60)

    # For now, just validate the config loads correctly
    print("\nConfig loaded successfully!")
    print(f"Model: {config.actor_rollout_ref.model.path}")
    print(f"Agent loop: {config.actor_rollout_ref.rollout.agent.default_agent_loop}")
    print(f"Data: {config.data.train_files}")


if __name__ == '__main__':
    # Clear any existing Hydra state
    GlobalHydra.instance().clear()

    main()
