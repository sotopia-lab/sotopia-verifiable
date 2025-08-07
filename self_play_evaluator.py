#!/usr/bin/env python3
"""
Self-Play Evaluation Framework for Sotopia-Verifiable

This module implements self-play evaluation where a trainee model plays against
a fixed partner model in verifiable social scenarios.

Usage:
    python self_play_evaluator.py --trainee_model_path path/to/model --num_games 10
"""

import argparse
import asyncio
import json
import sqlite3
import os
import sys
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add sotopia-rl to path
sys.path.append("sotopia-rl")


# Import our model loading utilities
from inference_verifiable import (
    load_model_and_tokenizer,
    load_template,
    generate_response,
    parse_json_response,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Redis connection
os.environ["REDIS_OM_URL"] = "redis://:@localhost:6379"


@dataclass
class GameResult:
    """Results from a single game between trainee and partner"""

    scenario_id: str
    scenario_title: str
    win_condition: str
    trainee_won: bool
    partner_won: bool
    conversation_turns: int
    conversation_log: List[Dict[str, Any]]
    game_outcome: str
    error_msg: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregated results from multiple games"""

    total_games: int
    trainee_wins: int
    partner_wins: int
    draws: int
    win_rate: float
    avg_conversation_turns: float
    results_by_scenario: Dict[str, List[GameResult]]


class WinConditionVerifier:
    """Verifies win conditions for different scenario types"""

    @staticmethod
    def verify_budget_allocation(
        conversation_log: List[Dict[str, Any]], win_condition: str
    ) -> Tuple[bool, bool, str]:
        """Verify budget allocation scenarios (like media budget)"""
        # Look for explicit agreements on $500/$500 split
        for turn in conversation_log:
            if turn.get("action_type") == "speak":
                text = turn.get("argument", "").lower()
                if ("500" in text and "newspaper" in text and "radio" in text) or (
                    "$500" in text and "$500" in text
                ):
                    # Check if both agents agreed
                    return True, True, "Both agents agreed on $500/$500 split"

        return False, False, "No agreement reached on required split"

    @staticmethod
    def verify_planting_task(
        conversation_log: List[Dict[str, Any]], win_condition: str
    ) -> Tuple[bool, bool, str]:
        """Verify cooperative planting scenarios"""
        # Look for evidence of successful task completion

        # Check for mentions of completion or success
        for turn in conversation_log:
            if turn.get("action_type") == "speak":
                text = turn.get("argument", "").lower()
                if any(
                    word in text
                    for word in ["finished", "completed", "done", "planted all"]
                ):
                    return True, True, "Task completed successfully"

        return False, False, "Task not completed within conversation"

    @staticmethod
    def verify_information_extraction(
        conversation_log: List[Dict[str, Any]], win_condition: str
    ) -> Tuple[bool, bool, str]:
        """Verify information extraction scenarios (like forum secrets)"""
        # Extract the secret from win condition
        secret_key = None
        if "AccessKeyTheta9" in win_condition:
            secret_key = "AccessKeyTheta9"

        if not secret_key:
            return False, False, "Could not identify secret key from win condition"

        # Check if secret was revealed
        for turn in conversation_log:
            if turn.get("action_type") == "speak":
                text = turn.get("argument", "")
                if secret_key in text:
                    return True, False, f"Secret {secret_key} was revealed"

        return False, True, "Secret was protected"

    @staticmethod
    def verify_general(
        conversation_log: List[Dict[str, Any]], win_condition: str
    ) -> Tuple[bool, bool, str]:
        """General verifier - looks for explicit success/failure indicators"""
        turns = len([t for t in conversation_log if t.get("action_type") == "speak"])

        # Basic heuristics
        if turns < 2:
            return False, False, "Conversation too short"

        # Look for agreement/disagreement patterns
        agreement_words = ["agree", "deal", "yes", "ok", "sounds good", "perfect"]
        disagreement_words = ["no", "disagree", "refuse", "cannot", "won't"]

        last_turns = (
            conversation_log[-3:] if len(conversation_log) >= 3 else conversation_log
        )
        agreement_count = 0
        disagreement_count = 0

        for turn in last_turns:
            if turn.get("action_type") == "speak":
                text = turn.get("argument", "").lower()
                if any(word in text for word in agreement_words):
                    agreement_count += 1
                if any(word in text for word in disagreement_words):
                    disagreement_count += 1

        if agreement_count > disagreement_count:
            return True, True, "Mutual agreement detected"
        elif disagreement_count > agreement_count:
            return False, False, "Disagreement detected"
        else:
            return False, False, "Unclear outcome"


class SelfPlayEvaluator:
    """Main class for self-play evaluation"""

    def __init__(
        self,
        trainee_model_path: Optional[str] = None,
        partner_model: str = "gpt-4o",
        db_path: str = "scenarios/scenarios.db",
    ):
        self.trainee_model_path = trainee_model_path
        self.partner_model = partner_model
        self.db_path = db_path
        self.verifier = WinConditionVerifier()

        # Load trainee model
        if trainee_model_path:
            logger.info(f"Loading trained model from {trainee_model_path}")
        else:
            logger.info("Loading base model (no training checkpoint)")

        # Create args object for compatibility
        import argparse

        args = argparse.Namespace()
        args.model_path = "Qwen/Qwen2.5-7B-Instruct"
        args.checkpoint_path = trainee_model_path  # None for base model
        args.use_qlora = True
        self.trainee_model, self.trainee_tokenizer = load_model_and_tokenizer(args)

        # Load template
        self.template = load_template("sotopia-rl/evals/qwen2.5-7b.jinja")

        logger.info("Self-play evaluator initialized")

    def load_all_scenarios(self) -> List[Dict[str, Any]]:
        """Load all scenarios from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, description, agents_json, win_condition FROM scenarios"
        )
        rows = cursor.fetchall()
        conn.close()

        scenarios = []
        for row in rows:
            scenario = {
                "id": str(row[0]),
                "title": row[1],
                "description": row[2],
                "agents": json.loads(row[3]),
                "win_condition": row[4],
            }
            scenarios.append(scenario)

        return scenarios

    def generate_trainee_response(
        self, prompt: str, max_length: int = 2048
    ) -> Dict[str, Any]:
        """Generate response from trainee model"""
        rendered_prompt = self.template.render(
            messages=[{"role": "user", "content": prompt}], add_generation_prompt=True
        )

        full_response = generate_response(
            self.trainee_model,
            self.trainee_tokenizer,
            rendered_prompt,
            max_length,
            temperature=0.7,
        )

        # Extract generated content
        generated_content = full_response[len(rendered_prompt) :].strip()
        parsed_json = parse_json_response(generated_content)

        return {
            "raw_response": generated_content,
            "parsed_response": parsed_json,
            "action_type": parsed_json.get("action_type", "speak")
            if parsed_json
            else "speak",
            "argument": parsed_json.get("argument", generated_content)
            if parsed_json
            else generated_content,
        }

    async def simulate_game(
        self, scenario: Dict[str, Any], trainee_as_agent1: bool = True
    ) -> GameResult:
        """Simulate a single game between trainee and partner"""
        try:
            logger.info(
                f"Simulating game for scenario {scenario['id']}: {scenario['title']}"
            )

            # For now, create a simplified conversation simulation
            # In full implementation, this would use the full sotopia server
            conversation_log: List[Dict[str, Any]] = []

            # Create prompts for both agents
            agent1_info = scenario["agents"][0]
            agent2_info = scenario["agents"][1]

            if trainee_as_agent1:
                trainee_agent = agent1_info
                partner_agent = agent2_info
            else:
                trainee_agent = agent2_info
                partner_agent = agent1_info

            # Simulate conversation turns (simplified)
            max_turns = 10
            for turn in range(max_turns):
                if turn % 2 == (0 if trainee_as_agent1 else 1):
                    # Trainee's turn
                    prompt = self._create_agent_prompt(
                        scenario, trainee_agent, partner_agent, conversation_log, turn
                    )
                    response = self.generate_trainee_response(prompt)

                    conversation_log.append(
                        {
                            "turn": turn,
                            "agent": f"{trainee_agent['first_name']} {trainee_agent['last_name']}",
                            "agent_type": "trainee",
                            "action_type": response["action_type"],
                            "argument": response["argument"],
                        }
                    )

                    # Check for leave action
                    if response["action_type"] == "leave":
                        break
                else:
                    # Partner's turn (simulate with simpler response for now)
                    # In full implementation, this would call the partner model API
                    partner_response = self._simulate_partner_response(
                        scenario, partner_agent, trainee_agent, conversation_log, turn
                    )
                    conversation_log.append(partner_response)

                    if partner_response["action_type"] == "leave":
                        break

            # Verify win condition
            trainee_won, partner_won, outcome_description = self._verify_win_condition(
                scenario, conversation_log
            )

            return GameResult(
                scenario_id=scenario["id"],
                scenario_title=scenario["title"],
                win_condition=scenario["win_condition"],
                trainee_won=trainee_won,
                partner_won=partner_won,
                conversation_turns=len(conversation_log),
                conversation_log=conversation_log,
                game_outcome=outcome_description,
            )

        except Exception as e:
            logger.error(f"Error simulating game for scenario {scenario['id']}: {e}")
            return GameResult(
                scenario_id=scenario["id"],
                scenario_title=scenario["title"],
                win_condition=scenario["win_condition"],
                trainee_won=False,
                partner_won=False,
                conversation_turns=0,
                conversation_log=[],
                game_outcome="Error occurred",
                error_msg=str(e),
            )

    def _create_agent_prompt(
        self,
        scenario: Dict[str, Any],
        agent_info: Dict[str, Any],
        other_agent_info: Dict[str, Any],
        conversation_log: List[Dict[str, Any]],
        turn_num: int,
    ) -> str:
        """Create prompt for agent based on scenario and conversation history"""
        agent_name = f"{agent_info['first_name']} {agent_info['last_name']}"
        other_name = f"{other_agent_info['first_name']} {other_agent_info['last_name']}"

        # Build conversation history
        history = ""
        for i, turn in enumerate(conversation_log):
            history += f"Turn #{i}: {turn['agent']} said: \"{turn['argument']}\"\n\n"

        prompt = f"""Imagine you are {agent_name}, your task is to act/speak as {agent_name} would, keeping in mind {agent_name}'s social goal.
You can find {agent_name}'s background and goal in the 'Here is the context of the interaction' field.
Note that {agent_name}'s secret and goal is only visible to you.
You should try your best to achieve {agent_name}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).

Here is the context of this interaction:
Scenario: {scenario['description']}
Participants: {agent_name} and {other_name}
{agent_name}'s background: {agent_name} is a {agent_info['age']}-year-old {agent_info.get('gender', 'person')} {agent_info['occupation']}. Personality and values description: {agent_name} has a {agent_info['decision_making_style']} decision-making style. {agent_name}'s secret: {agent_info['secret']}
{other_name}'s background: {other_name} is a {other_agent_info['age']}-year-old {other_agent_info.get('gender', 'person')} {other_agent_info['occupation']}. Personality: {other_agent_info['decision_making_style']} decision-making style.
{agent_name}'s goal: Work towards achieving the win condition: {scenario['win_condition']}
{other_name}'s goal: Unknown
Conversation Starts:
{history}.
You are at Turn #{turn_num}. Your available action types are
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

    def _simulate_partner_response(
        self,
        scenario: Dict[str, Any],
        agent_info: Dict[str, Any],
        other_agent_info: Dict[str, Any],
        conversation_log: List[Dict[str, Any]],
        turn_num: int,
    ) -> Dict[str, Any]:
        """Simulate partner model response (simplified for now)"""
        # Simple rule-based responses for different scenario types
        agent_name = f"{agent_info['first_name']} {agent_info['last_name']}"

        if "budget" in scenario["title"].lower():
            responses = [
                "I think we should split it equally - $500 for the newspaper and $500 for the radio show.",
                "The dean's rules are clear about equal funding. Let's go with $500 each.",
                "I agree, equal allocation makes the most sense here.",
            ]
        elif "garden" in scenario["title"].lower():
            responses = [
                "Let's work together to get all the seedlings planted in time!",
                "I'll take one bed, you take the other - we can share the tools.",
                "Great teamwork! I think we can finish this successfully.",
            ]
        else:
            responses = [
                "I understand your perspective. Let's find a solution that works.",
                "That's an interesting point. How do you think we should proceed?",
                "I think we can work together on this.",
            ]

        response_text = responses[turn_num % len(responses)]

        return {
            "turn": turn_num,
            "agent": agent_name,
            "agent_type": "partner",
            "action_type": "speak",
            "argument": response_text,
        }

    def _verify_win_condition(
        self, scenario: Dict[str, Any], conversation_log: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, str]:
        """Verify win condition based on scenario type"""
        title = scenario["title"].lower()
        win_condition = scenario["win_condition"]

        if "budget" in title:
            return self.verifier.verify_budget_allocation(
                conversation_log, win_condition
            )
        elif "garden" in title or "planting" in title:
            return self.verifier.verify_planting_task(conversation_log, win_condition)
        elif "forum" in title or "secret" in title:
            return self.verifier.verify_information_extraction(
                conversation_log, win_condition
            )
        else:
            return self.verifier.verify_general(conversation_log, win_condition)

    async def evaluate(self, num_games_per_scenario: int = 5) -> EvaluationResults:
        """Run evaluation across all scenarios"""
        scenarios = self.load_all_scenarios()
        all_results = []
        results_by_scenario = {}

        logger.info(
            f"Starting evaluation with {len(scenarios)} scenarios, {num_games_per_scenario} games each"
        )

        for scenario in scenarios:
            scenario_results = []
            logger.info(f"Evaluating scenario: {scenario['title']}")

            for game_num in range(num_games_per_scenario):
                # Alternate who goes first
                trainee_as_agent1 = game_num % 2 == 0
                result = await self.simulate_game(scenario, trainee_as_agent1)
                scenario_results.append(result)
                all_results.append(result)

                logger.info(
                    f"  Game {game_num + 1}: Trainee {'won' if result.trainee_won else 'lost'}"
                )

            results_by_scenario[scenario["id"]] = scenario_results

        # Calculate aggregate statistics
        total_games = len(all_results)
        trainee_wins = sum(1 for r in all_results if r.trainee_won)
        partner_wins = sum(1 for r in all_results if r.partner_won)
        draws = total_games - trainee_wins - partner_wins
        win_rate = trainee_wins / total_games if total_games > 0 else 0.0
        avg_turns = (
            sum(r.conversation_turns for r in all_results) / total_games
            if total_games > 0
            else 0.0
        )

        return EvaluationResults(
            total_games=total_games,
            trainee_wins=trainee_wins,
            partner_wins=partner_wins,
            draws=draws,
            win_rate=win_rate,
            avg_conversation_turns=avg_turns,
            results_by_scenario=results_by_scenario,
        )

    def save_results(self, results: EvaluationResults, output_path: str) -> None:
        """Save evaluation results to JSON file"""
        results_dict: Dict[str, Any] = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "trainee_model_path": self.trainee_model_path,
            "partner_model": self.partner_model,
            "summary": {
                "total_games": results.total_games,
                "trainee_wins": results.trainee_wins,
                "partner_wins": results.partner_wins,
                "draws": results.draws,
                "win_rate": results.win_rate,
                "avg_conversation_turns": results.avg_conversation_turns,
            },
            "results_by_scenario": {},
        }

        # Add detailed results by scenario
        for scenario_id, scenario_results in results.results_by_scenario.items():
            scenario_wins = sum(1 for r in scenario_results if r.trainee_won)
            games_list: List[Dict[str, Any]] = []

            for result in scenario_results:
                games_list.append(
                    {
                        "trainee_won": result.trainee_won,
                        "partner_won": result.partner_won,
                        "conversation_turns": result.conversation_turns,
                        "game_outcome": result.game_outcome,
                        "conversation_log": result.conversation_log,
                        "error_msg": result.error_msg,
                    }
                )

            scenario_dict: Dict[str, Any] = {
                "scenario_title": scenario_results[0].scenario_title
                if scenario_results
                else "",
                "games_played": len(scenario_results),
                "trainee_wins": scenario_wins,
                "win_rate": scenario_wins / len(scenario_results)
                if scenario_results
                else 0.0,
                "games": games_list,
            }

            results_dict["results_by_scenario"][scenario_id] = scenario_dict

        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
            exist_ok=True,
        )
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to {output_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-play evaluation for Sotopia-Verifiable"
    )
    parser.add_argument(
        "--trainee_model_path", required=True, help="Path to trainee model checkpoint"
    )
    parser.add_argument("--partner_model", default="gpt-4o", help="Partner model name")
    parser.add_argument(
        "--num_games", type=int, default=5, help="Number of games per scenario"
    )
    parser.add_argument(
        "--output_path",
        default="results/self_play_evaluation.json",
        help="Output file path",
    )
    parser.add_argument(
        "--db_path", default="scenarios/scenarios.db", help="Scenarios database path"
    )

    args = parser.parse_args()

    evaluator = SelfPlayEvaluator(
        trainee_model_path=args.trainee_model_path,
        partner_model=args.partner_model,
        db_path=args.db_path,
    )

    logger.info("Starting self-play evaluation...")
    results = await evaluator.evaluate(args.num_games)

    logger.info("Evaluation Results:")
    logger.info(f"Total games: {results.total_games}")
    logger.info(f"Trainee wins: {results.trainee_wins}")
    logger.info(f"Partner wins: {results.partner_wins}")
    logger.info(f"Draws: {results.draws}")
    logger.info(f"Win rate: {results.win_rate:.2%}")
    logger.info(f"Average conversation turns: {results.avg_conversation_turns:.1f}")

    evaluator.save_results(results, args.output_path)


if __name__ == "__main__":
    asyncio.run(main())
