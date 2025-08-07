#!/bin/bash

# Train SFT model for Sotopia-Verifiable
# This script adapts the original sotopia-rl SFT training for verifiable scenarios

export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

echo "🚀 Starting SFT training for Sotopia-Verifiable..."
echo "Model: $MODEL_NAME"
echo "Dataset: sotopia_verifiable_sft.json"

cd sotopia-rl/scripts

CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --config_file ./accelerate_config_sft.yaml \
  --main_process_port 29512 \
  ./train_sft.py \
  --model_name $MODEL_NAME \
  --learning_rate 1e-4 \
  --max_length 4096 \
  --train_batch_size 2 \
  --val_batch_size 1 \
  --accumulation_steps 8 \
  --num_epochs 10 \
  --use_lora \
  --evaluation_steps 5 \
  --sft_data_path ../data/sotopia_verifiable_sft.json \
  --template_path ../evals/qwen2.5-7b.jinja \
  --checkpoint_dir ../sft_checkpoints_verifiable

echo "✅ SFT training completed!"
