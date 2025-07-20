#!/usr/bin/env python3
# scenario_runner.py
# Usage:
#   1) Start Redis: redis-stack-server --dir ~/sotopia-verifiable/redis-data
#   2) Run: python scenario_runner.py <SCENARIO_ID>

import argparse
import sqlite3
import json
import os
import asyncio
import redis
from typing import Any, List, Dict
from sotopia.database.persistent_profile import AgentProfile, EnvironmentProfile
from sotopia.samplers import UniformSampler
from sotopia.server import run_async_server

# Configure Redis connection
os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"
client = redis.Redis(host="localhost", port=6379)


def add_agent_to_database(**kwargs: Any) -> None:
    """Create and save an AgentProfile from kwargs."""
    agent = AgentProfile(**kwargs)
    agent.save()


def add_env_profile(**kwargs: Any) -> None:
    """Create and save an EnvironmentProfile from kwargs."""
    env = EnvironmentProfile(**kwargs)
    env.save()


def load_scenario(db_path: str, id: str) -> Dict[str, Any]:
    """Load a single scenario row by its id."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, description, agents_json, win_condition FROM scenarios WHERE id = ?",
        (id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No scenario found with id={id}")

    cols = ["id", "title", "description", "agents_json", "win_condition"]
    entry = dict(zip(cols, row))
    entry["agents"] = json.loads(entry.pop("agents_json"))
    return entry


def ensure_agents(agents: List[Dict[str, Any]]) -> List[AgentProfile]:
    """Ensure each agent in the list exists in Redis; create if missing."""
    saved_agents: List[AgentProfile] = []

    for info in agents:
        first = info.get("first_name", "")
        last = info.get("last_name", "")
        found = AgentProfile.find(
            AgentProfile.first_name == first, AgentProfile.last_name == last
        ).all()
        if found:
            saved_agents.append(found[0])
        else:
            # Build kwargs with defaults for missing fields
            kwargs = {
                "first_name": first,
                "last_name": last,
                "age": info.get("age", 0),
                "occupation": info.get("occupation", ""),
                "gender": info.get("gender", ""),
                "gender_pronoun": info.get("gender_pronoun", ""),
                "big_five": info.get("big_five", ""),
                "moral_values": info.get("moral_values", []),
                "decision_making_style": info.get("decision_making_style", ""),
                "secret": info.get("secret", ""),
            }
            print(kwargs)
            add_agent_to_database(**kwargs)
            # Retrieve newly saved agent
            saved_agents.append(
                AgentProfile.find(
                    AgentProfile.first_name == first, AgentProfile.last_name == last
                ).all()[0]
            )
    return saved_agents


async def run_scenario(scenario: Dict[str, Any]) -> None:
    """Run one scenario: push agent & env profiles, then start the Async server."""
    print(f"=== Running scenario {scenario['id']} - {scenario['title']} ===")

    # Ensure agent profiles in Redis
    actor1, actor2 = ensure_agents(scenario["agents"])

    # Save environment profile
    add_env_profile(
        scenario=scenario["description"],
        agent_goals=[agent.get("goal", "") for agent in scenario["agents"]],
    )

    # Fetch the latest environment profile
    env_profile = list(EnvironmentProfile.find().all())[-1]

    # Build a uniform sampler for this scenario
    sampler: UniformSampler[EnvironmentProfile, AgentProfile] = UniformSampler(
        env_candidates=[env_profile],
        agent_candidates=[actor1, actor2],
    )

    # Run the Async server with GPT-4o for both env and agents
    await run_async_server(
        model_dict={"env": "gpt-4o", "agent1": "gpt-4o", "agent2": "gpt-4o"},
        sampler=sampler,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single scenario by ID")
    parser.add_argument("id", help="Scenario ID to execute")
    parser.add_argument("--db", default="scenarios/scenarios.db", help="SQLite DB path")
    args = parser.parse_args()

    scenario = load_scenario(args.db, args.id)
    asyncio.run(run_scenario(scenario))


if __name__ == "__main__":
    main()
