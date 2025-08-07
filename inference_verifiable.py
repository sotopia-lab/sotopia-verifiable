#!/usr/bin/env python3
"""
Inference script for Sotopia-Verifiable GRPO model

This script loads a trained GRPO model and tests it on verifiable scenarios.
Based on the original inference_grpo.py from sotopia-rl.

Usage:
    python inference_verifiable.py --checkpoint_path grpo_checkpoints_verifiable/checkpoint-100
"""

import argparse
import json
import os
import torch
from typing import Any, Dict, Optional, Tuple
from jinja2 import Environment, FileSystemLoader, Template
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Sotopia-Verifiable GRPO model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to GRPO checkpoint"
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="sotopia-rl/evals/qwen2.5-7b.jinja",
        help="Template path",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="sotopia-rl/data/sotopia_verifiable_test.json",
        help="Test data path",
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum output length"
    )
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, AutoTokenizer]:
    print(f"Loading model: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.use_qlora:
        print("Using QLoRA with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )

    # Load GRPO checkpoint
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from: {args.checkpoint_path}")

        # Look for policy_adapter subdirectory (GRPO format)
        policy_adapter_path = os.path.join(args.checkpoint_path, "policy_adapter")
        if os.path.exists(policy_adapter_path):
            checkpoint_path = policy_adapter_path
        else:
            checkpoint_path = args.checkpoint_path

        if os.path.exists(
            os.path.join(checkpoint_path, "adapter_model.safetensors")
        ) or os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
            model = PeftModelForCausalLM.from_pretrained(base_model, checkpoint_path)
            print("✅ Checkpoint loaded successfully")
        else:
            print(f"❌ No adapter found at {checkpoint_path}, using base model")
            model = base_model
    else:
        print("Using base model without checkpoint")
        model = base_model

    model.eval()
    return model, tokenizer


def load_template(template_path: str) -> Template:
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)

    if not template_dir:
        template_dir = "."

    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters["tojson"] = lambda obj: json.dumps(obj)
    return env.get_template(template_file)


def generate_response(
    model: Any,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0], skip_special_tokens=False)
    return str(response)


def extract_generated_content(full_response: str, prompt: str) -> str:
    """Extract only the newly generated content by removing the prompt prefix."""
    if prompt in full_response:
        return full_response[len(prompt) :].strip()
    return full_response


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from model response"""
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return None


def main() -> None:
    args = parse_args()

    print("🤖 Loading Sotopia-Verifiable GRPO Model")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args)
    template = load_template(args.template_path)

    # Load test data
    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)

    print(f"📊 Testing on {len(test_data)} examples")
    print("=" * 60)

    for i, example in enumerate(test_data):
        print(f"\n===== EXAMPLE {i+1}/{len(test_data)} =====")

        # Render prompt
        rendered_prompt = template.render(
            messages=[{"role": "user", "content": example["input"]}],
            add_generation_prompt=True,
        )

        # Generate response
        print("🤖 Generating response...")
        full_response = generate_response(
            model, tokenizer, rendered_prompt, args.max_length, args.temperature
        )

        generated_content = extract_generated_content(full_response, rendered_prompt)
        parsed_json = parse_json_response(generated_content)

        print("\n📝 MODEL OUTPUT:")
        print(generated_content)

        if parsed_json:
            print("\n✅ PARSED RESPONSE:")
            print(f"Action: {parsed_json.get('action_type', 'unknown')}")
            print(f"Argument: {parsed_json.get('argument', 'unknown')}")
        else:
            print("\n❌ Could not parse JSON response")

        print("\n🎯 EXPECTED OUTPUT:")
        expected = json.loads(example["output"])
        print(f"Action: {expected['action_type']}")
        print(f"Argument: {expected['argument']}")

        print("-" * 60)

    print("\n🎉 Inference completed!")


if __name__ == "__main__":
    main()
