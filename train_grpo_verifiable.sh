#!/bin/bash

# Train GRPO model for Sotopia-Verifiable
# This script adapts the original sotopia-rl GRPO training for verifiable scenarios

export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export SFT_CHECKPOINT_PATH="sft_checkpoints_verifiable/best-checkpoint"

echo "🚀 Starting GRPO training for Sotopia-Verifiable..."
echo "Model: $MODEL_NAME"
echo "SFT Checkpoint: $SFT_CHECKPOINT_PATH"
echo "Dataset: sotopia_verifiable_grpo.json"

cd sotopia-rl/scripts

CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --config_file ./accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  ./train_grpo.py \
  --model_name $MODEL_NAME \
  --policy_adapter_path ../$SFT_CHECKPOINT_PATH \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --grpo_data_path ../data/sotopia_verifiable_grpo.json \
  --template_path ../evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --use_lora_train_grpo \
  --num_generations 16 \
  --output_dir ../grpo_checkpoints_verifiable

echo "✅ GRPO training completed!"
