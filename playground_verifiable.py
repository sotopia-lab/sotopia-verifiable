#!/usr/bin/env python3
"""
Sotopia-Verifiable Model Playground

Interactive playground for experimenting with trained GRPO models on verifiable scenarios.
"""

import sys

sys.path.append("sotopia-rl")

# Import the playground from sotopia-rl and adapt it
try:
    from grpo_model_playground import *  # noqa: F403, F401

    # Override paths for verifiable scenarios
    ADAPTER_PATH = "grpo_checkpoints_verifiable/checkpoint-100/policy_adapter"

    print("🎮 Sotopia-Verifiable Model Playground")
    print("=" * 50)
    print("This playground uses verifiable scenarios for training social agents.")
    print(
        "The model has been trained on game-like scenarios with explicit win conditions."
    )
    print("=" * 50)

    if __name__ == "__main__":
        main()  # noqa: F405

except ImportError:
    print("❌ Could not import playground. Make sure sotopia-rl is properly set up.")
    print("Run: python setup_training.py --full_setup")
