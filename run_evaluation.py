"""
Simple ToMBench evaluation script
Runs models via OpenRouter API and saves results

Usage:
  python run_evaluation.py --models gpt5.1 --input data_processing/sample_test.csv --output results/results.json
  python run_evaluation.py --models claude --input data.csv --output results/claude.json
  python run_evaluation.py --models both --input data.csv --output results/both.json
"""

import os
import csv
import json
import re
import argparse
from openai import OpenAI

# Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")


def load_prompt(prompt_file):
    """Load prompt template from file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_csv(filename):
    """Load test data from CSV"""
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def call_api(client, model_id, prompt, reasoning_type=None):
    """Call OpenRouter API with native reasoning enabled"""
    extra_body = {}

    if reasoning_type == "openai":
        # GPT-5.1 native reasoning
        extra_body["reasoning"] = {"enabled": True}
    elif reasoning_type == "claude":
        # Claude extended thinking
        extra_body["thinking"] = {
            "type": "enabled",
            "budget_tokens": 10000
        }

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra_body,
        temperature=0.0
    )

    return response.choices[0].message.content


def parse_response(response_text):
    """Parse response to extract reasoning and final answer"""
    # Extract the answer (e.g., [[A]], [[B]], etc.)
    answer_match = re.search(r'\[\[([A-D])\]\]', response_text)
    answer = answer_match.group(1) if answer_match else ''

    # Extract reasoning (everything before the answer)
    if answer_match:
        reasoning = response_text[:answer_match.start()].strip()
    else:
        reasoning = response_text.strip()

    return reasoning, answer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ToMBench evaluation')
    parser.add_argument('--models', nargs='+',
                       choices=['gpt5.1', 'claude', 'both'],
                       required=True,
                       help='Models to evaluate: gpt5.1, claude, or both')
    parser.add_argument('--input',
                       required=True,
                       help='Input CSV file path')
    parser.add_argument('--output',
                       default='results',
                       help='Output directory (default: results)')
    args = parser.parse_args()

    # Initialize client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY
    )

    # Load prompt template
    prompt_template = load_prompt(PROMPT_FILE)
    print(f"Loaded prompt template from {PROMPT_FILE}")

    # Load data
    data = load_csv(args.input)
    print(f"Loaded {len(data)} samples from {args.input}")

    # Configure models based on arguments
    available_models = {
        'gpt5.1': {"id": "openai/gpt-5.1", "reasoning_type": "openai"},
        'claude': {"id": "anthropic/claude-sonnet-4.5", "reasoning_type": "claude"}
    }

    # Select models to run
    if 'both' in args.models:
        models = list(available_models.values())
    else:
        models = [available_models[m] for m in args.models]

    print(f"Testing models: {[m['id'] for m in models]}\n")

    # Determine model name for output filename
    if 'both' in args.models:
        model_prefix = 'both'
    elif 'gpt5.1' in args.models:
        model_prefix = 'gpt'
    else:
        model_prefix = 'claude'

    # Extract base filename from input
    input_basename = os.path.splitext(os.path.basename(args.input))[0]

    # Setup output directory and filename
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{model_prefix}_{input_basename}_results.csv"
    output_path = os.path.join(output_dir, output_filename)
    output_base = os.path.join(output_dir, f"{model_prefix}_{input_basename}_results")

    print(f"Output will be saved to: {output_path}\n")

    # Determine column order based on models used
    base_columns = ['index', 'STORY', 'QUESTION', 'OPTION-A', 'OPTION-B',
                    'OPTION-C', 'OPTION-D', 'ANSWER', 'SOURCE_FILE']
    model_columns = []
    for model in models:
        model_name = 'gpt' if 'gpt' in model['id'].lower() else 'claude'
        model_columns.extend([f"{model_name}_full_response",
                             f"{model_name}_reasoning",
                             f"{model_name}_response"])

    fieldnames = base_columns + model_columns

    # Run evaluation
    results = []
    for i, sample in enumerate(data):
        print(f"\nSample {i+1}/{len(data)}")

        # Format prompt
        prompt = prompt_template.format(
            story=sample['STORY'],
            question=sample['QUESTION'],
            option_a=sample['OPTION-A'],
            option_b=sample['OPTION-B'],
            option_c=sample['OPTION-C'],
            option_d=sample['OPTION-D']
        )

        # Create result row with original data
        result = {
            "index": sample.get('index', i),
            "STORY": sample['STORY'],
            "QUESTION": sample['QUESTION'],
            "OPTION-A": sample['OPTION-A'],
            "OPTION-B": sample['OPTION-B'],
            "OPTION-C": sample['OPTION-C'],
            "OPTION-D": sample['OPTION-D'],
            "ANSWER": sample.get('ANSWER', ''),
            "SOURCE_FILE": sample.get('TASK', '')
        }

        # Test each model and add model-specific columns
        for model in models:
            model_name = 'gpt' if 'gpt' in model['id'].lower() else 'claude'
            print(f"  Testing {model['id']} with native reasoning...")

            response_text = call_api(client, model['id'], prompt, model['reasoning_type'])
            reasoning, answer = parse_response(response_text)

            result[f"{model_name}_full_response"] = response_text
            result[f"{model_name}_reasoning"] = reasoning
            result[f"{model_name}_response"] = answer

        results.append(result)

        # Save JSON after first 5 samples
        if len(results) == 5:
            json_output = f"{output_base}_first5.json"
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ First 5 samples saved to {json_output}")

        # Save checkpoint CSV every 100 samples
        if len(results) % 100 == 0:
            checkpoint_output = f"{output_base}_checkpoint_{len(results)}.csv"
            with open(checkpoint_output, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✓ Checkpoint saved to {checkpoint_output}")

    # Save final complete CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Final results saved to {output_path}")
    print(f"✓ Total samples processed: {len(results)}")


if __name__ == "__main__":
    main()
