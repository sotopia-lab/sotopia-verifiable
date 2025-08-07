# Sotopia-Verifiable

Training social AI agents through self-play with verifiable rewards. Instead of relying on subjective human ratings or LLM judges, we use scenarios with explicit rules and binary win/loss conditions that can be formally verified.

## The Problem

Most social AI training uses feedback that's either subjective (human preferences) or gameable (LLM judges). This leads to reward hacking and inconsistent evaluation. We need a better way.

## Our Approach

We train agents on social scenarios where outcomes are objectively verifiable. Think negotiation games with clear rules, resource allocation with defined success criteria, or cooperative tasks with measurable goals. The agent learns through self-play against strategic opponents (currently GPT-4o), getting clean binary rewards based on whether they achieved the scenario's win condition.

## Quick Start

### Setup
```bash
conda activate sotopia-rl
cd /home/keyuh/sotopia-verifiable
```

### Run Complete Training Pipeline
```bash
python self_play_training.py \
  --num_iterations 3 \
  --games_per_scenario 10 \
  --output_dir training_results
```

This will:
1. Generate self-play games between your trainee (Qwen2.5-7B) and partner (GPT-4o)
2. Convert games to training data with binary rewards
3. Train using GRPO (Group Reward Policy Optimization)
4. Iterate to improve performance

### Or Run Steps Manually

Collect training data:
```bash
python training_data_collector.py \
  --trainee_model_path None \
  --num_games 50 \
  --output_dir training_data
```

Train the model:
```bash
cd fresh_training_results/iteration_1
bash train_grpo.sh
```

Evaluate performance:
```bash
python self_play_evaluator.py \
  --trainee_model_path checkpoints/policy_adapter \
  --num_games 20 \
  --output_path results/evaluation.json
```

## How It Works

### Scenarios
We generate social interaction scenarios based on established social science theories. Each scenario has:
- A clear context (negotiation, resource allocation, cooperation task, etc.)
- Explicit win conditions that can be verified through pattern matching
- Strategic depth that requires actual reasoning, not just following a script

### Training Loop
1. Load scenario from database
2. Run conversation between trainee and partner
3. Verify outcome using formal patterns (FINAL_BID, ALLOCATION, etc.)
4. Assign binary reward (+1 win, -1 loss, 0 draw)
5. Update model using GRPO with LoRA adapters

### Technical Stack
- **Base Model:** Qwen2.5-7B with LoRA (392M trainable params)
- **Partner Model:** GPT-4o (fixed, provides strategic opposition)
- **Training:** GRPO with binary rewards
- **Infrastructure:** Multi-GPU support, WandB tracking

## Project Structure

```
sotopia-verifiable/
├── scenarios/                    # Scenario generation and database
│   ├── scenario_generator.py
│   ├── scenarios.db
│   └── db_helper.py
├── self_play_evaluator.py       # Core self-play framework
├── training_data_collector.py   # Convert games to training data
├── structured_social_verifier.py # Outcome verification
├── fresh_training_results/       # Training experiments
│   └── iteration_1/
│       ├── train_grpo.sh
│       ├── training_data/
│       └── checkpoints/
└── sotopia-rl/                  # Training infrastructure
```

## Monitoring Progress

Training metrics are automatically tracked on WandB: https://wandb.ai/keyuhe/grpo-model-training

Expected progression:
- Iteration 1: ~45% win rate vs GPT-4o
- Iteration 2: ~60% win rate with better strategic understanding
- Iteration 3: ~70% win rate with improved social awareness

## Current Status

The training pipeline is operational and we're running initial experiments. First results show the approach is working - agents are learning to win scenarios through strategic interaction rather than just mimicking patterns.

### What's Working
- Scenario generation from social science theories
- Self-play game execution with GPT-4o
- Formal verification of outcomes
- GRPO training with LoRA adapters

### What We're Improving
- Scenario diversity and complexity
- Partner model selection (considering curriculum learning)
- Evaluation metrics beyond win rate
- Transfer to open-ended social interaction

## For Contributors

### Prerequisites
- CUDA-capable GPU (tested on RTX A6000)
- Python 3.10+ with PyTorch
- OpenAI API access
- WandB account

### Development
1. Test basic functionality: `python test_self_play.py`
2. Generate new scenarios: `cd scenarios && python scenario_generator.py`
3. Run training experiments
4. Monitor on WandB
5. Evaluate and iterate

The codebase is actively being developed. Feel free to explore and experiment with different approaches to scenario design, reward structures, and training algorithms.
