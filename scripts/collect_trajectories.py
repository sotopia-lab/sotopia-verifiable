#!/usr/bin/env python3
"""
Collect trajectories from Werewolf games for expert iteration training.

NOTE: This script is for OFFLINE trajectory collection, useful when:
- Collecting demonstrations from external LLMs (GPT-4o, Claude API)
- Building an initial dataset before training
- Testing the pipeline without verl's infrastructure

For ONLINE expert iteration with verl, use WerewolfAgentLoop instead,
which collects trajectories during training via verl's LLM server.
See: sotopia_verifiable/agent_loops/werewolf_agent_loop.py

This script:
1. Loads scenarios from the dataset
2. Runs games with an LLM (or deterministic actions for testing)
3. Collects trajectories in messages format for SFT
4. Filters trajectories by reward threshold
5. Saves to parquet for verl's SFT trainer
"""

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add werewolf game to path
werewolf_dir = project_root / "dependencies/sotopia/examples/experimental/games/werewolves"
sys.path.insert(0, str(werewolf_dir))

# Configure Redis
os.environ.setdefault("REDIS_OM_URL", "redis://:@localhost:6380")

from main import prepare_scenario, load_config, CONFIG_PATH
from sotopia.messages import AgentAction
from sotopia.generation_utils.generate import fill_template


def get_deterministic_action(agent_name: str, role: str, env) -> AgentAction:
    """Get deterministic action for testing."""
    state_name = env.current_state if hasattr(env, 'current_state') else ""

    alive_others = [n for n, alive in env.agent_alive.items() if alive and n != agent_name]
    target = alive_others[0] if alive_others else agent_name

    if "Night_werewolf" in state_name and role == "Werewolf":
        return AgentAction(action_type="action", argument=f"kill {target}")
    elif "Night_seer" in state_name and role == "Seer":
        return AgentAction(action_type="action", argument=f"inspect {target}")
    elif "Night_witch" in state_name and role == "Witch":
        return AgentAction(action_type="none", argument="")
    elif "Day_discussion" in state_name:
        return AgentAction(action_type="speak", argument="I suspect someone is lying.")
    elif "Day_vote" in state_name:
        return AgentAction(action_type="action", argument=f"vote {target}")
    else:
        return AgentAction(action_type="none", argument="")


def build_prompt(trainee_agent, trainee_name: str, observation) -> str:
    """Build prompt using game engine's template system."""
    if trainee_agent and hasattr(trainee_agent, 'custom_template') and observation:
        template = fill_template(
            trainee_agent.custom_template,
            action_instructions=observation.action_instruction or ""
        )
        prompt = fill_template(
            template,
            agent=trainee_name,
            history=observation.to_natural_language(),
            action_list=", ".join(observation.available_actions),
            turn_number=str(observation.turn_number),
            format_instructions='{"action_type": "<action_type>", "argument": "<argument>"}'
        )
        return prompt.strip()

    if observation:
        return observation.to_natural_language()
    return "[No observation]"


def action_to_response(action: AgentAction) -> str:
    """Convert action to response string."""
    return json.dumps({
        "action_type": action.action_type,
        "argument": action.argument
    })


async def run_game_collect_trajectory(
    scenario: dict,
    base_config: dict,
    max_turns: int = 40,
) -> dict:
    """Run a single game and collect trajectory in messages format."""

    # Build game config
    game_config = copy.deepcopy(base_config)
    game_config["agents"] = scenario["agents"]

    agent_models = {a["name"]: a.get("agent_model", "gpt-4o") for a in scenario["agents"]}
    role_map = {a["name"]: a["role"] for a in scenario["agents"]}
    trainee_name = scenario["trainee_name"]

    # Create environment and agents
    env, agents_list = prepare_scenario("gpt-4o", agent_models, game_config)
    agents = {agent.agent_name: agent for agent in agents_list}
    trainee_agent = agents.get(trainee_name)

    # Reset environment
    obs_dict = env.reset(agents=agents)

    # Collect messages for multi-turn SFT format
    messages = []

    # Add system message with role info
    system_message = (
        f"You are playing Werewolf as {trainee_name} with the role of {scenario['trainee_role']}. "
        f"Your team is {scenario['trainee_team']}. "
        f"Win the game by helping your team eliminate the opposing faction."
    )
    messages.append({"role": "system", "content": system_message})

    turn = 0
    game_done = False
    trainee_turn_count = 0

    while not game_done and turn < max_turns:
        turn += 1

        # Check if trainee is active
        agent_names = list(env.agents)
        trainee_idx = agent_names.index(trainee_name) if trainee_name in agent_names else -1
        trainee_active = (
            trainee_idx >= 0
            and trainee_idx < len(env.action_mask)
            and env.action_mask[trainee_idx] == 1
        )

        # Collect actions
        actions = {}
        for agent_name in env.agents:
            role = role_map.get(agent_name, "Villager")

            if agent_name == trainee_name and trainee_active:
                # Build prompt for trainee
                trainee_obs = obs_dict.get(trainee_name)
                prompt = build_prompt(trainee_agent, trainee_name, trainee_obs)

                # For now, use deterministic action (replace with LLM in production)
                action = get_deterministic_action(agent_name, role, env)

                # Add to messages as user (prompt) + assistant (response) pair
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": action_to_response(action)})
                trainee_turn_count += 1
            else:
                action = get_deterministic_action(agent_name, role, env)

            actions[agent_name] = action

        # Step environment
        obs_dict, rewards, terminated, truncated, info = await env.astep(actions)
        game_done = all(terminated.values())

    # Get final reward
    trainee_info = info.get(trainee_name, {})
    reward = trainee_info.get("complete_rating", 0.0)

    return {
        "scenario_id": scenario.get("scenario_id"),
        "trainee_name": trainee_name,
        "trainee_role": scenario["trainee_role"],
        "trainee_team": scenario["trainee_team"],
        "messages": messages,
        "reward": reward,
        "game_turns": turn,
        "trainee_turns": trainee_turn_count,
    }


async def collect_trajectories(
    scenarios_path: str,
    output_path: str,
    reward_threshold: float = -2.0,
    max_scenarios: int = -1,
    max_turns: int = 40,
):
    """Collect trajectories from multiple scenarios."""

    # Load scenarios
    df = pd.read_parquet(scenarios_path)
    print(f"Loaded {len(df)} scenarios from {scenarios_path}")

    if max_scenarios > 0:
        df = df.head(max_scenarios)
        print(f"Using first {max_scenarios} scenarios")

    # Load base game config
    base_config = load_config(CONFIG_PATH)

    # Collect trajectories
    trajectories = []
    filtered_count = 0

    for i in tqdm(range(len(df)), desc="Collecting trajectories"):
        scenario = df.iloc[i].to_dict()

        try:
            result = await run_game_collect_trajectory(
                scenario=scenario,
                base_config=base_config,
                max_turns=max_turns,
            )

            # Filter by reward threshold
            if result["reward"] >= reward_threshold:
                trajectories.append(result)
            else:
                filtered_count += 1

        except Exception as e:
            print(f"Error in scenario {scenario.get('scenario_id')}: {e}")
            continue

    print(f"\nCollected {len(trajectories)} trajectories")
    print(f"Filtered {filtered_count} trajectories with reward < {reward_threshold}")

    # Save to parquet
    if trajectories:
        out_df = pd.DataFrame(trajectories)
        out_df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

        # Print statistics
        rewards = out_df["reward"].tolist()
        print(f"\nReward statistics:")
        print(f"  Mean: {sum(rewards)/len(rewards):.3f}")
        print(f"  Min:  {min(rewards):.3f}")
        print(f"  Max:  {max(rewards):.3f}")

        avg_turns = out_df["trainee_turns"].mean()
        print(f"  Avg trainee turns: {avg_turns:.1f}")
    else:
        print("No trajectories collected!")


def main():
    parser = argparse.ArgumentParser(description="Collect Werewolf game trajectories")
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(project_root / "data/werewolf_scenarios.parquet"),
        help="Path to scenarios parquet file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "data/werewolf_trajectories.parquet"),
        help="Output path for trajectories"
    )
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=-2.0,
        help="Minimum reward to include trajectory. Rewards are +1 (win) or -1 (loss). "
             "Use -2 to include all, 0 or 1 to only include wins. (default: -2 = include all)"
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=-1,
        help="Max scenarios to process (-1 for all)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Max game turns per scenario"
    )
    args = parser.parse_args()

    asyncio.run(collect_trajectories(
        scenarios_path=args.scenarios,
        output_path=args.output,
        reward_threshold=args.reward_threshold,
        max_scenarios=args.max_scenarios,
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()
