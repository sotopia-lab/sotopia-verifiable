#!/usr/bin/env python3
"""
Setup script for Sotopia-Verifiable Training

This script sets up the complete training pipeline for sotopia-verifiable,
including data processing, environment setup, and training scripts.

Usage:
    python setup_training.py --full_setup
"""

import argparse
import os
import subprocess
import json


def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def setup_directories():
    """Create necessary directories for training"""
    directories = [
        "checkpoints",
        "sft_checkpoints_verifiable",
        "grpo_checkpoints_verifiable",
        "rm_checkpoints_verifiable",
        "logs",
        "results",
    ]

    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    # Check if we're in the right environment
    try:
        import torch
        import transformers

        try:
            import peft
        except ImportError:
            peft = None

        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ Transformers version: {transformers.__version__}")
        if peft:
            print("✅ PEFT available")
        else:
            print("⚠️ PEFT not available")

        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available - training will be slow")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

    return True


def process_data():
    """Process scenarios into training data"""
    print("📊 Processing scenario data...")

    cmd = "python data_processor.py --output_dir sotopia-rl/data --num_scenarios 10"
    result = run_command(cmd, "Data processing")

    if result is not None:
        # Verify files were created
        data_files = [
            "sotopia-rl/data/sotopia_verifiable_sft.json",
            "sotopia-rl/data/sotopia_verifiable_grpo.json",
            "sotopia-rl/data/sotopia_verifiable_test.json",
        ]

        for file_path in data_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                print(f"✅ {file_path}: {len(data)} examples")
            else:
                print(f"❌ Missing file: {file_path}")

        return True
    return False


def create_training_config():
    """Create training configuration file"""
    config = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "sft_config": {
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "batch_size": 2,
            "max_length": 4096,
            "use_lora": True,
        },
        "grpo_config": {
            "learning_rate": 5e-6,
            "num_epochs": 2,
            "batch_size": 4,
            "num_generations": 16,
            "use_lora": True,
        },
        "data_paths": {
            "sft_data": "sotopia-rl/data/sotopia_verifiable_sft.json",
            "grpo_data": "sotopia-rl/data/sotopia_verifiable_grpo.json",
            "test_data": "sotopia-rl/data/sotopia_verifiable_test.json",
            "template": "sotopia-rl/evals/qwen2.5-7b.jinja",
        },
    }

    with open("training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Created training_config.json")


def create_quick_start_script():
    """Create a quick start script for training"""
    script_content = """#!/bin/bash

# Quick Start Script for Sotopia-Verifiable Training
echo "🚀 Sotopia-Verifiable Training Pipeline"
echo "======================================"

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "sotopia-rl" ]]; then
    echo "⚠️  Please activate the sotopia-rl conda environment first:"
    echo "   conda activate sotopia-rl"
    exit 1
fi

echo "📊 Step 1: Processing data..."
python data_processor.py --num_scenarios 10

echo "🎯 Step 2: Starting SFT training..."
bash train_sft_verifiable.sh

echo "🧠 Step 3: Starting GRPO training..."
bash train_grpo_verifiable.sh

echo "🧪 Step 4: Running inference test..."
python inference_verifiable.py --checkpoint_path grpo_checkpoints_verifiable/checkpoint-100 --use_qlora

echo "🎉 Training pipeline completed!"
echo "📁 Check the following directories for results:"
echo "   - sft_checkpoints_verifiable/"
echo "   - grpo_checkpoints_verifiable/"
echo "   - logs/"
"""

    with open("quick_start.sh", "w") as f:
        f.write(script_content)

    os.chmod("quick_start.sh", 0o755)
    print("✅ Created quick_start.sh")


def create_playground():
    """Create an interactive playground for the trained model"""
    playground_content = '''#!/usr/bin/env python3
"""
Sotopia-Verifiable Model Playground

Interactive playground for experimenting with trained GRPO models on verifiable scenarios.
"""

import json
import os
import sys
sys.path.append('sotopia-rl')

# Import the playground from sotopia-rl and adapt it
try:
    from grpo_model_playground import *

    # Override paths for verifiable scenarios
    ADAPTER_PATH = "grpo_checkpoints_verifiable/checkpoint-100/policy_adapter"

    print("🎮 Sotopia-Verifiable Model Playground")
    print("=" * 50)
    print("This playground uses verifiable scenarios for training social agents.")
    print("The model has been trained on game-like scenarios with explicit win conditions.")
    print("=" * 50)

    if __name__ == "__main__":
        main()

except ImportError:
    print("❌ Could not import playground. Make sure sotopia-rl is properly set up.")
    print("Run: python setup_training.py --full_setup")
'''

    with open("playground_verifiable.py", "w") as f:
        f.write(playground_content)

    print("✅ Created playground_verifiable.py")


def create_readme():
    """Create README for the setup"""
    readme_content = """# Sotopia-Verifiable Training Setup

This directory contains a complete GRPO training pipeline adapted for sotopia-verifiable scenarios.

## Directory Structure

```
sotopia-verifiable/
├── scenarios/                    # Original verifiable scenarios
├── sotopia-rl/                  # Copied training infrastructure
├── data_processor.py            # Convert scenarios to training format
├── train_sft_verifiable.sh     # SFT training script
├── train_grpo_verifiable.sh    # GRPO training script
├── inference_verifiable.py     # Model inference script
├── playground_verifiable.py    # Interactive model playground
├── quick_start.sh              # One-command training pipeline
└── training_config.json        # Training configuration
```

## Quick Start

1. **Activate environment:**
   ```bash
   conda activate sotopia-rl
   ```

2. **Run complete pipeline:**
   ```bash
   ./quick_start.sh
   ```

3. **Or run steps individually:**
   ```bash
   # Process data
   python data_processor.py --num_scenarios 10

   # Train SFT model
   bash train_sft_verifiable.sh

   # Train GRPO model
   bash train_grpo_verifiable.sh

   # Test the model
   python inference_verifiable.py --checkpoint_path grpo_checkpoints_verifiable/checkpoint-100 --use_qlora
   ```

4. **Interactive playground:**
   ```bash
   python playground_verifiable.py
   ```

## Dataset

The training uses scenarios from the SQLite database with the following characteristics:
- **Verifiable outcomes:** Each scenario has explicit win conditions
- **Multi-dimensional:** Scenarios vary across interdependence, relational models, resource types, and contexts
- **Game-like structure:** Clear rules and objectives for social interaction

## Model Architecture

- **Base Model:** Qwen2.5-7B-Instruct
- **Training Method:** GRPO (Group Reward Policy Optimization)
- **Adaptation:** LoRA (Low-Rank Adaptation)
- **Quantization:** 4-bit QLoRA for efficient training

## Training Data Format

Each training example follows the sotopia-rl format:
```json
{
  "input": "Detailed social scenario prompt with character backgrounds...",
  "output": "{\\"action_type\\": \\"speak\\", \\"argument\\": \\"Response text\\"}"
}
```

## Customization

- Modify `training_config.json` to adjust hyperparameters
- Edit `data_processor.py` to change scenario-to-prompt conversion
- Update training scripts for different model sizes or configurations

## Results

After training, you'll have:
- SFT checkpoints in `sft_checkpoints_verifiable/`
- GRPO checkpoints in `grpo_checkpoints_verifiable/`
- Training logs in `logs/`
- Test results from inference script
"""

    with open("README_TRAINING.md", "w") as f:
        f.write(readme_content)

    print("✅ Created README_TRAINING.md")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Sotopia-Verifiable training pipeline"
    )
    parser.add_argument("--full_setup", action="store_true", help="Run complete setup")
    parser.add_argument("--data_only", action="store_true", help="Only process data")
    parser.add_argument(
        "--check_deps", action="store_true", help="Only check dependencies"
    )
    args = parser.parse_args()

    print("🔧 Sotopia-Verifiable Training Setup")
    print("=" * 50)

    if args.check_deps or args.full_setup:
        if not check_dependencies():
            print("❌ Dependency check failed")
            return 1

    if args.data_only or args.full_setup:
        if not process_data():
            print("❌ Data processing failed")
            return 1

    if args.full_setup:
        setup_directories()
        create_training_config()
        create_quick_start_script()
        create_playground()
        create_readme()

        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate conda environment: conda activate sotopia-rl")
        print("2. Run training pipeline: ./quick_start.sh")
        print("3. Or run individual steps as needed")
        print("\n📖 See README_TRAINING.md for detailed instructions")

    return 0


if __name__ == "__main__":
    exit(main())
