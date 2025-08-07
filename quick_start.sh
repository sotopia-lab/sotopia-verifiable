#!/bin/bash

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
