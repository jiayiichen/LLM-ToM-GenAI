#!/usr/bin/env python3
"""
Analyze model failures using GPT-5.1 as a summarizer
Identifies why the model failed and summarizes the reasoning error
"""

import os
import csv
import json
import argparse
from openai import OpenAI
from tqdm import tqdm

# Configuration
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "failure_summarizer_prompt.txt")


def load_prompt(prompt_file):
    """Load prompt template from file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_csv(filename):
    """Load results from CSV"""
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def call_gpt51_summarizer(client, prompt):
    """Call GPT-5.1 to analyze failure"""
    response = client.chat.completions.create(
        model="openai/gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description='Analyze model failures with GPT-5.1')
    parser.add_argument('--input',
                       required=True,
                       help='Input CSV file with model results')
    parser.add_argument('--model-name',
                       required=True,
                       help='Model column prefix (e.g., qwen3_8b, llama)')
    parser.add_argument('--output',
                       default='results/failure_analysis',
                       help='Output directory (default: results/failure_analysis)')
    parser.add_argument('--limit',
                       type=int,
                       default=None,
                       help='Limit number of failures to analyze (default: all)')
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

    # Filter for failures
    model_prefix = args.model_name
    failures = [
        sample for sample in data
        if sample.get(f"{model_prefix}_response", '') != sample.get('ANSWER', '')
        and sample.get('ANSWER', '')  # Only include samples with answers
    ]

    print(f"Found {len(failures)} failures out of {len(data)} samples")

    # Apply limit if specified
    if args.limit:
        failures = failures[:args.limit]
        print(f"Analyzing first {len(failures)} failures")

    # Setup output
    os.makedirs(args.output, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_filename = f"{model_prefix}_{input_basename}_failure_analysis.csv"
    output_path = os.path.join(args.output, output_filename)
    json_output = output_path.replace('.csv', '.json')

    print(f"\nAnalyzing failures with GPT-5.1...\n")

    # Analyze each failure
    results = []
    for i, sample in enumerate(tqdm(failures, desc="Analyzing failures")):
        # Format prompt
        prompt = prompt_template.format(
            story=sample['STORY'],
            question=sample['QUESTION'],
            option_a=sample['OPTION-A'],
            option_b=sample['OPTION-B'],
            option_c=sample['OPTION-C'],
            option_d=sample['OPTION-D'],
            correct_answer=sample['ANSWER'],
            model_reasoning=sample.get(f"{model_prefix}_reasoning", ''),
            model_answer=sample.get(f"{model_prefix}_response", '')
        )

        # Get failure analysis from GPT-5.1
        failure_summary = call_gpt51_summarizer(client, prompt)

        # Create result with all original data plus analysis
        result = {
            **sample,  # Include all original columns
            "failure_summary": failure_summary
        }
        results.append(result)

    # Save results
    fieldnames = list(results[0].keys())

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Analysis saved:")
    print(f"  - JSON: {json_output}")
    print(f"  - CSV:  {output_path}")
    print(f"✓ Total failures analyzed: {len(results)}")


if __name__ == "__main__":
    main()
