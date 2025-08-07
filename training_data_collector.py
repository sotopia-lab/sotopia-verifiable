#!/usr/bin/env python3
"""
Training Data Collector for Sotopia-Verifiable

Collects training data by running games between models and using verifiable outcomes
as reward signals for GRPO training.

Usage:
    python training_data_collector.py --num_games 50 --output_dir training_data/
"""

import argparse
import asyncio
import json
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

from self_play_evaluator import SelfPlayEvaluator, GameResult

logger = logging.getLogger(__name__)


class TrainingDataCollector:
    """Collects training data from self-play games"""

    def __init__(self, evaluator: SelfPlayEvaluator):
        self.evaluator = evaluator

    def game_result_to_training_examples(
        self, result: GameResult
    ) -> List[Dict[str, Any]]:
        """Convert a game result to training examples for GRPO"""
        training_examples = []

        # Only create training examples for the trainee's turns
        trainee_turns = [
            turn
            for turn in result.conversation_log
            if turn.get("agent_type") == "trainee"
        ]

        for i, turn in enumerate(trainee_turns):
            # Create the conversation history up to this point
            history_turns = result.conversation_log[
                : result.conversation_log.index(turn)
            ]

            # Build the prompt that would have been given to the model
            prompt = self._reconstruct_prompt(result, turn, history_turns)

            # Create the response
            response = json.dumps(
                {"action_type": turn["action_type"], "argument": turn["argument"]}
            )

            # Calculate reward based on game outcome and turn quality
            reward = self._calculate_reward(result, turn, i, len(trainee_turns))

            training_example = {
                "input": prompt,
                "output": response,
                "reward": reward,
                "scenario_id": result.scenario_id,
                "scenario_title": result.scenario_title,
                "turn_number": turn["turn"],
                "game_won": result.trainee_won,
                "conversation_turns": result.conversation_turns,
                "metadata": {
                    "game_outcome": result.game_outcome,
                    "error_msg": result.error_msg,
                },
            }

            training_examples.append(training_example)

        return training_examples

    def _reconstruct_prompt(
        self,
        result: GameResult,
        current_turn: Dict[str, Any],
        history_turns: List[Dict[str, Any]],
    ) -> str:
        """Reconstruct the prompt that was given to generate this turn"""
        # This is a simplified reconstruction - in practice, we'd store the actual prompts
        scenario_desc = f"Scenario from {result.scenario_title}"

        # Build conversation history
        history = ""
        for turn in history_turns:
            history += f"Turn #{turn['turn']}: {turn['agent']} said: \"{turn['argument']}\"\n\n"

        # Simplified prompt reconstruction
        prompt = f"""Context: {scenario_desc}
Win Condition: {result.win_condition}

Conversation History:
{history}

You are at Turn #{current_turn['turn']}. Please respond with appropriate JSON action."""

        return prompt

    def _calculate_reward(
        self,
        result: GameResult,
        turn: Dict[str, Any],
        turn_index: int,
        total_turns: int,
    ) -> float:
        """Calculate reward for a specific turn - Pure binary based on game outcome"""

        # Pure binary rewards based only on verifiable game outcomes
        if result.trainee_won:
            return 1.0
        elif result.partner_won:
            return -1.0
        else:
            return 0.0  # Draw/unclear

        # ==== HEURISTIC SCORING (COMMENTED OUT) ====
        # The following heuristic scoring could be useful for future experiments
        # but creates continuous rewards that complicate RL training:

        # base_reward = 0.0
        #
        # # Game outcome reward (most important)
        # if result.trainee_won:
        #     base_reward += 1.0
        # elif result.partner_won:
        #     base_reward -= 0.5
        # else:
        #     base_reward += 0.0  # Draw/unclear
        #
        # # Turn quality rewards
        # turn_text = turn.get('argument', '').lower()
        #
        # # Reward constructive behavior
        # if any(word in turn_text for word in ['agree', 'cooperate', 'work together', 'compromise']):
        #     base_reward += 0.1
        #
        # # Reward strategic thinking
        # if any(word in turn_text for word in ['propose', 'suggest', 'think', 'consider']):
        #     base_reward += 0.05
        #
        # # Penalize very short or generic responses
        # if len(turn_text) < 10:
        #     base_reward -= 0.1
        #
        # # Reward longer conversations (shows engagement)
        # if result.conversation_turns > 5:
        #     base_reward += 0.1
        #
        # # Early turns get slightly less reward (exploration)
        # turn_weight = 0.7 + 0.3 * (turn_index / max(1, total_turns - 1))
        #
        # return base_reward * turn_weight

    async def collect_training_data(
        self, num_games_per_scenario: int = 10
    ) -> List[Dict[str, Any]]:
        """Collect training data by running multiple games"""
        logger.info(
            f"Collecting training data with {num_games_per_scenario} games per scenario"
        )

        # Run evaluation to get game results
        evaluation_results = await self.evaluator.evaluate(num_games_per_scenario)

        # Convert all game results to training examples
        all_training_examples = []

        for scenario_results in evaluation_results.results_by_scenario.values():
            for game_result in scenario_results:
                if game_result.error_msg:
                    logger.warning(f"Skipping game with error: {game_result.error_msg}")
                    continue

                examples = self.game_result_to_training_examples(game_result)
                all_training_examples.extend(examples)

        logger.info(
            f"Generated {len(all_training_examples)} training examples from {evaluation_results.total_games} games"
        )
        return all_training_examples

    def save_training_data(
        self, training_data: List[Dict[str, Any]], output_dir: str
    ) -> Dict[str, str]:
        """Save training data in different formats"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw training data
        raw_data_path = os.path.join(output_dir, f"training_data_raw_{timestamp}.json")
        with open(raw_data_path, "w") as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Saved raw training data to {raw_data_path}")

        # Create SFT format (for supervised fine-tuning)
        sft_data = []
        for example in training_data:
            sft_data.append({"input": example["input"], "output": example["output"]})

        sft_path = os.path.join(output_dir, f"sft_training_data_{timestamp}.json")
        with open(sft_path, "w") as f:
            json.dump(sft_data, f, indent=2)
        logger.info(f"Saved SFT training data to {sft_path}")

        # Create GRPO format (with rewards)
        grpo_data = []
        for example in training_data:
            grpo_data.append(
                {
                    "prompt": example["input"],
                    "response": example["output"],
                    "reward": example["reward"],
                    "scenario_id": example["scenario_id"],
                }
            )

        grpo_path = os.path.join(output_dir, f"grpo_training_data_{timestamp}.json")
        with open(grpo_path, "w") as f:
            json.dump(grpo_data, f, indent=2)
        logger.info(f"Saved GRPO training data to {grpo_path}")

        # Create statistics summary
        stats = self._calculate_training_stats(training_data)
        stats_path = os.path.join(output_dir, f"training_stats_{timestamp}.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved training statistics to {stats_path}")

        return {
            "raw_data_path": raw_data_path,
            "sft_path": sft_path,
            "grpo_path": grpo_path,
            "stats_path": stats_path,
        }

    def _calculate_training_stats(
        self, training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics about the training data"""
        if not training_data:
            return {}

        rewards = [ex["reward"] for ex in training_data]
        won_games = [ex for ex in training_data if ex["game_won"]]

        stats = {
            "total_examples": len(training_data),
            "examples_from_won_games": len(won_games),
            "win_rate": len(won_games) / len(training_data),
            "reward_stats": {
                "mean": sum(rewards) / len(rewards),
                "min": min(rewards),
                "max": max(rewards),
                "positive_rewards": len([r for r in rewards if r > 0]),
                "negative_rewards": len([r for r in rewards if r < 0]),
                "zero_rewards": len([r for r in rewards if r == 0]),
            },
            "scenarios_covered": len(set(ex["scenario_id"] for ex in training_data)),
            "avg_conversation_length": sum(
                ex["conversation_turns"] for ex in training_data
            )
            / len(training_data),
            "examples_by_scenario": {},
        }

        # Break down by scenario
        for scenario_id in set(ex["scenario_id"] for ex in training_data):
            scenario_examples = [
                ex for ex in training_data if ex["scenario_id"] == scenario_id
            ]
            scenario_rewards = [ex["reward"] for ex in scenario_examples]

            stats["examples_by_scenario"][scenario_id] = {
                "count": len(scenario_examples),
                "win_rate": len([ex for ex in scenario_examples if ex["game_won"]])
                / len(scenario_examples),
                "avg_reward": sum(scenario_rewards) / len(scenario_rewards),
                "scenario_title": scenario_examples[0]["scenario_title"]
                if scenario_examples
                else "",
            }

        return stats


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training data from self-play games"
    )
    parser.add_argument(
        "--trainee_model_path",
        default=None,
        help="Path to trainee model checkpoint (None for base model)",
    )
    parser.add_argument("--partner_model", default="gpt-4o", help="Partner model name")
    parser.add_argument(
        "--num_games", type=int, default=10, help="Number of games per scenario"
    )
    parser.add_argument(
        "--output_dir", default="training_data", help="Output directory"
    )
    parser.add_argument(
        "--db_path", default="scenarios/scenarios.db", help="Scenarios database path"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create evaluator
    evaluator = SelfPlayEvaluator(
        trainee_model_path=args.trainee_model_path,
        partner_model=args.partner_model,
        db_path=args.db_path,
    )

    # Create collector
    collector = TrainingDataCollector(evaluator)

    # Collect training data
    logger.info("Starting training data collection...")
    training_data = await collector.collect_training_data(args.num_games)

    # Save training data
    saved_paths = collector.save_training_data(training_data, args.output_dir)

    logger.info("Training data collection completed!")
    logger.info(f"Generated {len(training_data)} training examples")
    logger.info("Saved files:")
    for name, path in saved_paths.items():
        logger.info(f"  {name}: {path}")


if __name__ == "__main__":
    asyncio.run(main())
