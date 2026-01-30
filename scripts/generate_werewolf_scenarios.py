#!/usr/bin/env python3
"""
Generate werewolf game scenario metadata for RL training.

The dataset just provides scenario identifiers. The game engine
constructs all prompts dynamically during training rollout.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
werewolf_dir = project_root / "dependencies/sotopia/examples/experimental/games/werewolves"

with open(werewolf_dir / "config.json") as f:
    GAME_CONFIG = json.load(f)

with open(werewolf_dir / "roster.json") as f:
    ROSTER = json.load(f)

AGENT_NAMES = [agent["name"] for agent in ROSTER["agents"]]
ROLES = [agent["role"] for agent in GAME_CONFIG["agents"]]
TEAMS = [agent["team"] for agent in GAME_CONFIG["agents"]]
ROLE_TEAM_PAIRS = list(zip(ROLES, TEAMS))


def shuffle_roles(seed: int) -> list:
    """Generate shuffled role assignments for a scenario."""
    random.seed(seed)
    pairs = copy.deepcopy(ROLE_TEAM_PAIRS)
    random.shuffle(pairs)

    return [
        {"name": AGENT_NAMES[i], "role": pairs[i][0], "team": pairs[i][1]}
        for i in range(len(AGENT_NAMES))
    ]


def generate_scenarios(
    num_scenarios: int = 100,
    output_path: str = "data/werewolf_scenarios.parquet",
    opponent_model: str = "gpt-4o"
):
    """Generate scenario metadata. Prompts are built by game engine during rollout."""
    data = []

    for i in range(num_scenarios):
        seed = 42 + i
        agents = shuffle_roles(seed)

        # Add model info
        for agent in agents:
            agent["agent_model"] = opponent_model

        trainee_name = AGENT_NAMES[i % len(AGENT_NAMES)]
        trainee_info = next(a for a in agents if a["name"] == trainee_name)

        entry = {
            "scenario_id": i,
            "seed": seed,
            "agents": agents,
            "trainee_name": trainee_name,
            "trainee_role": trainee_info["role"],
            "trainee_team": trainee_info["team"],
            "data_source": "werewolf_game",
        }
        data.append(entry)

    df = pd.DataFrame(data)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_path)
    print(f"Generated {num_scenarios} scenarios -> {output_path}")
    print(f"\nTrainee distribution:\n{df['trainee_name'].value_counts().to_string()}")
    print(f"\nRole distribution:\n{df['trainee_role'].value_counts().to_string()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate werewolf scenario metadata")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/werewolf_scenarios.parquet")
    parser.add_argument("--opponent-model", type=str, default="gpt-4o")
    args = parser.parse_args()

    os.chdir(project_root)
    generate_scenarios(args.num, args.output, args.opponent_model)


if __name__ == "__main__":
    main()
