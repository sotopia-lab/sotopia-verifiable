#!/usr/bin/env python3
"""
Iterative Training Loop for Sotopia-Verifiable

Implements the complete self-play training loop:
1. Collect training data through self-play
2. Train model with GRPO on game outcomes
3. Evaluate improved model
4. Repeat

Usage:
    python iterative_training.py --num_iterations 5
"""

import argparse
import asyncio
import json
import os
import subprocess
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from self_play_evaluator import SelfPlayEvaluator
from training_data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


class IterativeTrainer:
    """Manages the iterative training loop"""

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        partner_model: str = "gpt-4o",
        output_dir: str = "iterative_training",
    ):
        self.base_model_path = base_model_path
        self.partner_model = partner_model
        self.output_dir = output_dir
        self.current_model_path: Optional[str] = None

        os.makedirs(output_dir, exist_ok=True)

        # Training configuration
        self.training_config = {
            "games_per_scenario": 8,
            "grpo_epochs": 2,
            "learning_rate": 5e-6,
            "batch_size": 4,
            "max_length": 4096,
        }

    async def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of data collection + training + evaluation"""
        logger.info(f"=== Starting Iteration {iteration} ===")

        iteration_dir = os.path.join(self.output_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)

        results = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "model_path": self.current_model_path or self.base_model_path,
        }

        # Step 1: Collect training data through self-play
        logger.info("Step 1: Collecting training data...")
        training_data_dir = os.path.join(iteration_dir, "training_data")
        training_data_paths = await self._collect_training_data(training_data_dir)
        results["training_data_paths"] = training_data_paths

        # Step 2: Train model with GRPO
        logger.info("Step 2: Training model with GRPO...")
        checkpoint_path = await self._train_model(
            training_data_paths["grpo_path"], iteration_dir
        )
        results["checkpoint_path"] = checkpoint_path

        # Step 3: Evaluate trained model
        logger.info("Step 3: Evaluating trained model...")
        evaluation_results = await self._evaluate_model(checkpoint_path, iteration_dir)
        results["evaluation_results"] = evaluation_results

        # Update current model path for next iteration
        self.current_model_path = checkpoint_path

        # Save iteration results
        results_path = os.path.join(iteration_dir, "iteration_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(
            f"Iteration {iteration} completed. Win rate: {evaluation_results['win_rate']:.2%}"
        )
        return results

    async def _collect_training_data(self, output_dir: str) -> Dict[str, str]:
        """Collect training data through self-play"""
        # Use current model or base model (None = fresh base model)
        model_path = self.current_model_path  # None for first iteration

        evaluator = SelfPlayEvaluator(
            trainee_model_path=model_path, partner_model=self.partner_model
        )

        collector = TrainingDataCollector(evaluator)
        training_data = await collector.collect_training_data(
            int(self.training_config["games_per_scenario"])
        )

        return collector.save_training_data(training_data, output_dir)

    async def _train_model(self, grpo_data_path: str, iteration_dir: str) -> str:
        """Train model using GRPO on collected data"""
        checkpoint_dir = os.path.join(iteration_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create training script
        # Convert relative paths to absolute paths from sotopia-rl/scripts directory
        abs_grpo_data_path = os.path.abspath(grpo_data_path)
        abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
        abs_current_model_path = (
            os.path.abspath(self.current_model_path) if self.current_model_path else ""
        )

        # Build policy adapter line separately to avoid f-string backslash issues
        if self.current_model_path:
            policy_adapter_line = '  --policy_adapter_path "$POLICY_ADAPTER_PATH" \\'
            policy_export_line = (
                f'export POLICY_ADAPTER_PATH="{abs_current_model_path}"'
            )
        else:
            policy_adapter_line = ""
            policy_export_line = ""

        # Build the command lines - use both GPUs to match accelerate config
        base_command = """CUDA_VISIBLE_DEVICES=0,1 python -m accelerate.commands.launch \\
  --config_file ./accelerate_config_grpo.yaml \\
  --main_process_port 29511 \\
  ./train_grpo.py \\
  --model_name "$MODEL_NAME" \\"""

        if policy_adapter_line:
            command_with_adapter = base_command + "\n" + policy_adapter_line
        else:
            command_with_adapter = base_command

        training_script = f"""#!/bin/bash
cd /home/keyuh/sotopia-rl/scripts

export MODEL_NAME="{self.base_model_path}"
{policy_export_line}

{command_with_adapter}
  --learning_rate {self.training_config['learning_rate']} \\
  --per_device_train_batch_size {self.training_config['batch_size']} \\
  --per_device_eval_batch_size {self.training_config['batch_size']} \\
  --gradient_accumulation_steps 8 \\
  --grpo_data_path "{abs_grpo_data_path}" \\
  --template_path ../evals/qwen2.5-7b.jinja \\
  --num_grpo_epochs {self.training_config['grpo_epochs']} \\
  --use_lora_train_grpo \\
  --num_generations 16 \\
  --output_dir "{abs_checkpoint_dir}"
"""

        script_path = os.path.join(iteration_dir, "train_grpo.sh")
        with open(script_path, "w") as f:
            f.write(training_script)
        os.chmod(script_path, 0o755)

        # Run training
        logger.info(f"Starting GRPO training... (script: {script_path})")

        try:
            # Execute the training script
            result = subprocess.run(
                ["bash", script_path],
                cwd="/home/keyuh/sotopia-verifiable",
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Training failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Training script failed: {result.stderr}")

            logger.info("Training completed successfully")
            logger.info(f"Training output: {result.stdout}")

            # Find the actual checkpoint directory created by training
            # GRPO typically saves checkpoints as checkpoint-{step}
            checkpoints = [
                d
                for d in os.listdir(checkpoint_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(checkpoint_dir, d))
            ]

            if not checkpoints:
                raise RuntimeError("No checkpoints found after training")

            # Use the latest checkpoint (highest step number)
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

            logger.info(f"Using checkpoint: {checkpoint_path}")
            return checkpoint_path

        except subprocess.TimeoutExpired:
            logger.error("Training timed out after 1 hour")
            raise RuntimeError("Training timed out")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    async def _evaluate_model(
        self, checkpoint_path: str, iteration_dir: str
    ) -> Dict[str, Any]:
        """Evaluate the trained model"""
        evaluator = SelfPlayEvaluator(
            trainee_model_path=checkpoint_path, partner_model=self.partner_model
        )

        # Run evaluation
        evaluation_results = await evaluator.evaluate(num_games_per_scenario=5)

        # Save detailed results
        results_path = os.path.join(iteration_dir, "evaluation_results.json")
        evaluator.save_results(evaluation_results, results_path)

        # Return summary
        return {
            "total_games": evaluation_results.total_games,
            "trainee_wins": evaluation_results.trainee_wins,
            "partner_wins": evaluation_results.partner_wins,
            "win_rate": evaluation_results.win_rate,
            "avg_conversation_turns": evaluation_results.avg_conversation_turns,
            "detailed_results_path": results_path,
        }

    async def run_full_training(self, num_iterations: int) -> List[Dict[str, Any]]:
        """Run the complete iterative training process"""
        logger.info(f"Starting iterative training for {num_iterations} iterations")

        all_results = []

        for iteration in range(1, num_iterations + 1):
            try:
                iteration_results = await self.run_iteration(iteration)
                all_results.append(iteration_results)

                # Log progress
                win_rate = iteration_results["evaluation_results"]["win_rate"]
                logger.info(f"Iteration {iteration} - Win Rate: {win_rate:.2%}")

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                break

        # Save final results
        final_results_path = os.path.join(
            self.output_dir, "final_training_results.json"
        )
        with open(final_results_path, "w") as f:
            json.dump(
                {
                    "training_config": self.training_config,
                    "total_iterations": len(all_results),
                    "all_iterations": all_results,
                    "final_win_rate": all_results[-1]["evaluation_results"]["win_rate"]
                    if all_results
                    else 0.0,
                },
                f,
                indent=2,
            )

        logger.info(
            f"Iterative training completed. Results saved to {final_results_path}"
        )
        return all_results

    def create_training_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a summary of training progress"""
        if not results:
            return "No training results available."

        summary = "=== Iterative Training Summary ===\n\n"
        summary += f"Total Iterations: {len(results)}\n"
        summary += f"Base Model: {self.base_model_path}\n"
        summary += f"Partner Model: {self.partner_model}\n\n"

        summary += "Iteration Progress:\n"
        for result in results:
            iteration = result["iteration"]
            win_rate = result["evaluation_results"]["win_rate"]
            total_games = result["evaluation_results"]["total_games"]
            summary += f"  Iteration {iteration}: {win_rate:.2%} win rate ({total_games} games)\n"

        if len(results) > 1:
            initial_win_rate = results[0]["evaluation_results"]["win_rate"]
            final_win_rate = results[-1]["evaluation_results"]["win_rate"]
            improvement = final_win_rate - initial_win_rate
            summary += f"\nOverall Improvement: {improvement:+.2%} ({initial_win_rate:.2%} → {final_win_rate:.2%})\n"

        return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative self-play training")
    parser.add_argument(
        "--num_iterations", type=int, default=3, help="Number of training iterations"
    )
    parser.add_argument(
        "--base_model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model path"
    )
    parser.add_argument(
        "--partner_model", default="gpt-4o", help="Partner model for self-play"
    )
    parser.add_argument(
        "--output_dir", default="iterative_training", help="Output directory"
    )
    parser.add_argument(
        "--games_per_scenario",
        type=int,
        default=8,
        help="Games per scenario for data collection",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create trainer
    trainer = IterativeTrainer(
        base_model_path=args.base_model,
        partner_model=args.partner_model,
        output_dir=args.output_dir,
    )

    # Update training config
    trainer.training_config["games_per_scenario"] = args.games_per_scenario

    # Run iterative training
    logger.info("Starting iterative self-play training...")
    results = await trainer.run_full_training(args.num_iterations)

    # Print summary
    summary = trainer.create_training_summary(results)
    print("\n" + summary)

    logger.info("Iterative training completed!")


if __name__ == "__main__":
    asyncio.run(main())
