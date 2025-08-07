#!/usr/bin/env python3
"""
Data Processor for Sotopia-Verifiable

This script converts the scenarios from the SQLite database into the format
expected by the GRPO training pipeline from sotopia-rl.

Usage:
    python data_processor.py --output_dir sotopia-rl/data --num_scenarios 10
"""

import argparse
import sqlite3
import json
import os
from typing import Dict, Any, List, Optional


def load_scenarios_from_db(
    db_path: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load scenarios from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT id, title, description, agents_json, win_condition FROM scenarios"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    scenarios = []
    for row in rows:
        scenario = {
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "agents": json.loads(row[3]),
            "win_condition": row[4],
        }
        scenarios.append(scenario)

    return scenarios


def create_social_prompt(
    scenario: Dict[str, Any], agent_idx: int, turn_number: int = 0
) -> str:
    """Create a social interaction prompt from a scenario"""
    agent = scenario["agents"][agent_idx]
    other_agent = scenario["agents"][1 - agent_idx]

    character_name = f"{agent['first_name']} {agent['last_name']}"
    other_character_name = f"{other_agent['first_name']} {other_agent['last_name']}"

    # Build character background
    character_background = f"{character_name} is a {agent['age']}-year-old {agent['gender']} {agent['occupation']}. {agent.get('gender_pronoun', 'They/them')} pronouns. Personality and values description: {character_name} has a {agent['decision_making_style']} decision-making style. {character_name}'s secret: {agent['secret']}"

    other_character_background = f"{other_character_name} is a {other_agent['age']}-year-old {other_agent['gender']} {other_agent['occupation']}. {other_agent.get('gender_pronoun', 'They/them')} pronouns. Personality: {other_agent['decision_making_style']} decision-making style."

    # Extract goal from win condition (simplified approach)
    character_goal = (
        f"Work towards achieving the win condition: {scenario['win_condition']}"
    )

    prompt = f"""Imagine you are {character_name}, your task is to act/speak as {character_name} would, keeping in mind {character_name}'s social goal.
You can find {character_name}'s background and goal in the 'Here is the context of the interaction' field.
Note that {character_name}'s secret and goal is only visible to you.
You should try your best to achieve {character_name}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).

Here is the context of this interaction:
Scenario: {scenario['description']}
Participants: {character_name} and {other_character_name}
{character_name}'s background: {character_background}
{other_character_name}'s background: {other_character_background}
{character_name}'s goal: {character_goal}
{other_character_name}'s goal: Unknown
Conversation Starts:
.
You are at Turn #{turn_number}. Your available action types are
"none action speak non-verbal communication leave".
Note: You can "leave" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format:

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{{"description": "An interface for messages.\\nThere is only one required method: to_natural_language", "properties": {{"action_type": {{"title": "Action Type", "description": "whether to speak at this turn or choose to not do anything", "enum": ["none", "speak", "non-verbal communication", "action", "leave"], "type": "string"}}, "argument": {{"title": "Argument", "description": "the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action", "type": "string"}}}}, "required": ["action_type", "argument"]}}
```"""

    return prompt


def generate_training_examples(scenarios: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Generate training examples from scenarios"""
    examples = []

    for scenario in scenarios:
        # Create examples for both agents at turn 0
        for agent_idx in [0, 1]:
            prompt = create_social_prompt(scenario, agent_idx, 0)

            # Generate a plausible response (in practice, you'd want real responses)

            # Simple response generation based on the scenario
            if (
                "cooperate" in scenario["description"].lower()
                or "together" in scenario["description"].lower()
            ):
                response = "Hello! I think we should work together on this. What are your thoughts on how we can achieve our goal?"
            elif (
                "compete" in scenario["description"].lower()
                or "challenge" in scenario["description"].lower()
            ):
                response = "I'm ready for this challenge. Let me propose my approach to this situation."
            else:
                response = "Hello, I understand we need to discuss this situation. What's your perspective?"

            example = {
                "input": prompt,
                "output": json.dumps({"action_type": "speak", "argument": response}),
            }
            examples.append(example)

    return examples


def create_sft_data(examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Create SFT training data format"""
    sft_data = []

    for example in examples:
        sft_data.append(
            {
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]},
                ]
            }
        )

    return sft_data


def create_grpo_data(examples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Create GRPO training data format"""
    grpo_data = []

    for example in examples:
        # For GRPO, we need the prompt and potentially multiple responses
        grpo_data.append(
            {
                "prompt": example["input"],
                "response": example["output"],
                "reward": 1.0,  # Placeholder reward
            }
        )

    return grpo_data


def save_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {len(data)} examples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process sotopia-verifiable scenarios for training"
    )
    parser.add_argument(
        "--db_path", default="scenarios/scenarios.db", help="Path to scenarios database"
    )
    parser.add_argument(
        "--output_dir",
        default="sotopia-rl/data",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=None,
        help="Number of scenarios to process (all if not specified)",
    )
    args = parser.parse_args()

    print("🔄 Loading scenarios from database...")
    scenarios = load_scenarios_from_db(args.db_path, args.num_scenarios)
    print(f"✅ Loaded {len(scenarios)} scenarios")

    print("🔄 Generating training examples...")
    examples = generate_training_examples(scenarios)
    print(f"✅ Generated {len(examples)} training examples")

    # Create different data formats
    print("🔄 Creating SFT data format...")
    sft_data = create_sft_data(examples)

    print("🔄 Creating GRPO data format...")
    grpo_data = create_grpo_data(examples)

    # Create test data (using first few examples)
    test_data = examples[: min(4, len(examples))]

    # Save all formats
    save_data(sft_data, os.path.join(args.output_dir, "sotopia_verifiable_sft.json"))
    save_data(grpo_data, os.path.join(args.output_dir, "sotopia_verifiable_grpo.json"))
    save_data(test_data, os.path.join(args.output_dir, "sotopia_verifiable_test.json"))

    print("\n🎉 Data processing completed!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 SFT examples: {len(sft_data)}")
    print(f"📊 GRPO examples: {len(grpo_data)}")
    print(f"📊 Test examples: {len(test_data)}")


if __name__ == "__main__":
    main()
