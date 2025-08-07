# Sotopia-Verifiable: Social AI Training with Verifiable Rewards

A complete training pipeline for developing strategic social AI agents through self-play reinforcement learning with formally verifiable reward signals.

## 🎯 Overview

**Problem:** Traditional social AI training relies on subjective human ratings or LLM judges, which are vulnerable to reward hacking and inconsistent evaluation.

**Solution:** Train agents on social scenarios with explicit rules and binary win/loss conditions that can be formally verified, inspired by DeepSeek R1's approach to robust reward modeling.

### Key Innovation: Self-Play + Verifiable Outcomes
```
Trainee Model ↔ Partner Model → Social Scenario → Verifiable Win/Loss → GRPO Training → Improved Agent
```

Instead of static datasets or subjective feedback, agents learn through strategic interaction with verifiable game outcomes.

## 🏗️ Architecture

### Training Methodology
- **Trainee Model:** Qwen2.5-7B + LoRA (392M trainable parameters)
- **Partner Model:** Fixed GPT-4o (provides strategic opposition)
- **Scenarios:** Theory-grounded social interactions with formal verification
- **Training:** GRPO (Group Reward Policy Optimization) with binary rewards
- **Infrastructure:** Multi-GPU training with WandB experiment tracking

### Scenario Design Framework
Based on established social science theories:
- **Social Interdependence Theory:** Cooperative/Competitive/Individual interactions
- **Relational Models Theory:** Communal Sharing, Authority Ranking, Equality Matching, Market Pricing
- **Resource Exchange Theory:** Money, Information, Goods, Services, Status, Love

## 🚀 Quick Start

### Prerequisites
```bash
conda activate sotopia-rl
cd /home/keyuh/sotopia-verifiable
```

### Option 1: Complete Training Pipeline (Recommended)
```bash
# Full iterative training with data collection + GRPO training
python self_play_training.py \
  --num_iterations 3 \
  --games_per_scenario 10 \
  --output_dir training_results
```

### Option 2: Step-by-Step Training

#### Step 1: Data Collection
```bash
# Collect training data through self-play games
python training_data_collector.py \
  --trainee_model_path None \
  --num_games 50 \
  --output_dir training_data
```

#### Step 2: GRPO Training
```bash
cd fresh_training_results/iteration_1

# Run distributed GRPO training (automatically uses both GPUs)
bash train_grpo.sh
```

#### Step 3: Evaluation
```bash
python self_play_evaluator.py \
  --trainee_model_path checkpoints/policy_adapter \
  --num_games 20 \
  --output_path results/evaluation.json
```

## 📊 Monitoring & Evaluation

### WandB Integration
All experiments automatically tracked at: https://wandb.ai/keyuhe/grpo-model-training

### Success Metrics
- **Win Rate:** Performance vs. fixed partner models
- **Training Stability:** Reward convergence and variance
- **Behavioral Diversity:** Strategy variation across scenarios
- **Social Appropriateness:** Human evaluation of interaction quality

### Expected Training Progress
```
Iteration 1: Base model (Qwen2.5-7B) → ~45% win rate vs GPT-4o
Iteration 2: LoRA fine-tuned     → ~60% win rate + strategic depth
Iteration 3: Converged training  → ~70% win rate + social awareness
```

## 📁 Project Structure

```
sotopia-verifiable/
├── scenarios/                    # Scenario generation & database
│   ├── scenario_generator.py    # Theory-based scenario creation
│   ├── scenarios.db             # SQLite scenario database
│   └── db_helper.py             # Database utilities
├── self_play_evaluator.py       # Core self-play framework
├── training_data_collector.py   # Convert games to training data
├── structured_social_verifier.py # DeepSeek-inspired verification
├── fresh_training_results/       # Active training experiments
│   └── iteration_1/             # Current training cycle
│       ├── train_grpo.sh        # Training script
│       ├── training_data/       # Generated training examples
│       └── checkpoints/         # LoRA model checkpoints
└── sotopia-rl/                  # Training infrastructure
    ├── scripts/train_grpo.py    # GRPO training implementation
    ├── evals/qwen2.5-7b.jinja   # Model prompt templates
    └── sotopia_rl/              # Core training modules
```

## 🔬 Technical Details

### Scenario Generation
- **Taxonomy-based design:** Cross-product of social science theories
- **OpenAI o4-mini generation:** Structured JSON scenario creation
- **Verifiable constraints:** All scenarios require explicit win conditions
- **Pattern verification:** Formal move patterns (FINAL_BID, ALLOCATION, etc.)

### Training Data Format
```json
{
  "prompt": "Context: Scenario from X\nWin Condition: Y\n...",
  "response": "{\"action_type\": \"speak\", \"argument\": \"...\"}",
  "reward": 1.0,  // Binary: +1.0 (win), -1.0 (loss), 0.0 (draw)
  "scenario_id": 1
}
```

### Self-Play Process
1. Load scenario from database with structured win conditions
2. Initialize trainee model and fixed partner (GPT-4o)
3. Run strategic conversation with turn-based actions
4. Verify outcome using formal pattern matching
5. Convert to training examples with binary rewards
6. Update trainee policy using GRPO

## 📈 Research Contributions

### Novel Approaches
1. **Verifiable Social Rewards:** First system to use formal verification for social RL
2. **Systematic Scenario Design:** Theory-grounded taxonomy ensuring comprehensive coverage
3. **Self-Play Social Learning:** Strategic opponent interaction vs. static training data
4. **Binary Reward Structure:** Clean reward signals resistant to gaming

### Comparison to Related Work
- **vs. Constitutional AI:** Our rewards are formally verifiable, not LLM-judged
- **vs. RLHF:** No human preference modeling required, outcomes are objective
- **vs. DeepSeek R1:** Applied their verification principles to social interaction domains

## 🎓 Research Questions & Future Work

### Open Questions
1. **Scenario Quality:** How to generate strategically compelling social scenarios?
2. **Reward Design:** Binary outcomes vs. process-based social appropriateness?
3. **Partner Selection:** Fixed vs. co-evolving vs. curriculum partner strategies?
4. **Generalization:** Do formal scenario skills transfer to open-ended interaction?

### Planned Ablation Studies
- **Reward Structure:** Binary vs. continuous vs. hybrid rewards
- **Partner Strength:** GPT-4o vs. GPT-3.5 vs. co-training approaches
- **Scenario Complexity:** Simple choices vs. multi-step negotiations
- **Training Algorithms:** GRPO vs. PPO vs. other social RL methods

## 🤝 Getting Started for Contributors

### Prerequisites
- CUDA-capable GPU (RTX A6000 recommended)
- Python 3.10+ with PyTorch, transformers, trl, accelerate
- OpenAI API access for scenario generation
- WandB account for experiment tracking

### Development Workflow
1. **Test Framework:** `python test_self_play.py` (basic functionality)
2. **Generate Scenarios:** `cd scenarios && python scenario_generator.py`
3. **Run Training:** Follow Quick Start guide above
4. **Monitor Progress:** Check WandB dashboard and local logs
5. **Evaluate Results:** Compare win rates and social behavior quality

### Contributing
- Follow existing code style and documentation patterns
- Add tests for new scenario types or training components
- Include ablation study results in research reports
- Update documentation when adding new features

## 📚 Citation & References

If you use this work, please cite:
```bibtex
@misc{sotopia-verifiable,
  title={Sotopia-Verifiable: Social AI Training with Verifiable Rewards},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[username]/sotopia-verifiable}
}
```

### Theoretical Foundations
- **Social Interdependence Theory:** Johnson & Johnson (2001)
- **Relational Models Theory:** Fiske (1990)
- **Resource Exchange Theory:** Foa & Foa (2012)
- **DeepSeek R1 Verification:** [DeepSeek Team] (2024)

---

**Current Status:** ✅ Training pipeline operational, first iteration in progress
**Next Milestone:** Enhanced scenario generation and evaluation framework
**Long-term Goal:** Robust social AI agents resistant to reward hacking
