"""
ToMBench evaluation script using vLLM for local inference
Runs Qwen2.5-7B-Instruct locally and saves results

Usage:
  python run_evaluation_vllm.py --input train.csv
  python run_evaluation_vllm.py --input train.csv --output results
  python run_evaluation_vllm.py --input train.csv --model Qwen/Qwen2.5-7B-Instruct
"""

import os
import csv
import json
import re
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Configuration
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")


def load_prompt(prompt_file):
    """Load prompt template from file"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_csv(filename):
    """Load test data from CSV"""
    with open(filename, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def parse_response(response_text):
    """Parse response to extract reasoning and final answer"""
    # Extract all answer occurrences (e.g., [[A]], [[B]], etc.) and take the LAST one
    answer_matches = re.findall(r'\[\[([A-D])\]\]', response_text)
    answer = answer_matches[-1] if answer_matches else ''

    # Extract reasoning (everything before the last answer)
    if answer_matches:
        # Find the position of the last occurrence
        last_match = re.search(r'\[\[' + re.escape(answer) + r'\]\](?!.*\[\[[A-D]\]\])', response_text)
        if last_match:
            reasoning = response_text[:last_match.start()].strip()
        else:
            reasoning = response_text.strip()
    else:
        reasoning = response_text.strip()

    return reasoning, answer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ToMBench evaluation with vLLM')
    parser.add_argument('--input',
                       required=True,
                       help='Input CSV file path')
    parser.add_argument('--output',
                       default='results/vllm',
                       help='Output directory (default: results/vllm)')
    parser.add_argument('--model',
                       default='Qwen/Qwen3-8B',
                       help='Model name or path (default: Qwen/Qwen3-8B)')
    parser.add_argument('--tensor-parallel-size',
                       type=int,
                       default=1,
                       help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--limit',
                       type=int,
                       default=None,
                       help='Limit number of samples to process (default: all)')
    args = parser.parse_args()

    # Load prompt template
    prompt_template = load_prompt(PROMPT_FILE)
    print(f"Loaded prompt template from {PROMPT_FILE}")

    # Load data
    data = load_csv(args.input)

    # Apply limit if specified
    if args.limit:
        data = data[:args.limit]
        print(f"Loaded {len(data)} samples from {args.input} (limited to first {args.limit})")
    else:
        print(f"Loaded {len(data)} samples from {args.input}")

    # Initialize vLLM model
    print(f"\nLoading model: {args.model}")
    print("This may take a few minutes on first run...")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85  # Use 85% instead of 90% to avoid OOM
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048
    )

    print(f"✓ Model loaded successfully\n")

    # Determine model name for output filename
    model_name = args.model.split('/')[-1].lower().replace('-', '_').replace('.', '_')
    input_basename = os.path.splitext(os.path.basename(args.input))[0]

    # Setup output directory and filename
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{model_name}_{input_basename}_results.csv"
    output_path = os.path.join(output_dir, output_filename)
    output_base = os.path.join(output_dir, f"{model_name}_{input_basename}_results")

    print(f"Output will be saved to: {output_path}\n")

    # Determine column order
    base_columns = ['index', 'STORY', 'QUESTION', 'OPTION-A', 'OPTION-B',
                    'OPTION-C', 'OPTION-D', 'ANSWER', 'SOURCE_FILE']
    model_columns = [f"{model_name}_full_response",
                    f"{model_name}_reasoning",
                    f"{model_name}_response",
                    f"{model_name}_truncation",
                    f"{model_name}_prompt_tokens",
                    f"{model_name}_output_tokens"]
    fieldnames = base_columns + model_columns

    # Prepare all prompts
    print("Preparing prompts...")
    prompts = []
    for sample in tqdm(data, desc="Formatting prompts"):
        prompt = prompt_template.format(
            story=sample['STORY'],
            question=sample['QUESTION'],
            option_a=sample['OPTION-A'],
            option_b=sample['OPTION-B'],
            option_c=sample['OPTION-C'],
            option_d=sample['OPTION-D']
        )
        prompts.append(prompt)

    # Run batch inference
    print(f"Running inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"✓ Inference completed\n")

    # Process results
    print("\nProcessing results...")
    results = []
    truncated_count = 0

    for i, (sample, output) in enumerate(tqdm(zip(data, outputs), total=len(data), desc="Processing results")):
        response_text = output.outputs[0].text
        reasoning, answer = parse_response(response_text)

        # Track truncation and token counts
        finish_reason = output.outputs[0].finish_reason
        is_truncated = (finish_reason == 'length')
        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)

        if is_truncated:
            truncated_count += 1

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
            "SOURCE_FILE": sample.get('TASK', ''),
            f"{model_name}_full_response": response_text,
            f"{model_name}_reasoning": reasoning,
            f"{model_name}_response": answer,
            f"{model_name}_truncation": is_truncated,
            f"{model_name}_prompt_tokens": prompt_tokens,
            f"{model_name}_output_tokens": output_tokens
        }

        results.append(result)

    # Save final results as JSON and CSV
    json_output = output_path.replace('.csv', '.json')

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results saved:")
    print(f"  - JSON: {json_output}")
    print(f"  - CSV:  {output_path}")
    print(f"✓ Total samples processed: {len(results)}")

    # Calculate and display accuracy
    correct = sum(1 for r in results if r['ANSWER'] == r[f"{model_name}_response"])
    accuracy = (correct / len(results)) * 100
    truncation_rate = (truncated_count / len(results)) * 100

    print(f"\n{'='*70}")
    print(f"Accuracy: {correct}/{len(results)} ({accuracy:.2f}%)")
    print(f"Truncated responses: {truncated_count}/{len(results)} ({truncation_rate:.2f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
