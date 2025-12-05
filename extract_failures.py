"""
Extract failure cases from vLLM evaluation results CSV
Works with the new CSV format from run_evaluation_vllm.py

Usage:
  python extract_failures.py --input results/vllm/qwen3_8b_train_results.csv --model-name qwen3_8b --output results/failures
  python extract_failures.py --input results/vllm/qwen3_14b_train_results.csv --model-name qwen3_14b --output results/failures
"""

import csv
import json
import argparse
import os


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract failure cases from evaluation results')
    parser.add_argument('--input',
                       required=True,
                       help='Input results CSV file')
    parser.add_argument('--model-name',
                       required=True,
                       help='Model column prefix (e.g., qwen3_8b, qwen3_14b)')
    parser.add_argument('--output',
                       default='results/failures',
                       help='Output directory (default: results/failures)')
    args = parser.parse_args()

    # Load results from CSV
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    print(f"Loaded {len(results)} samples from {args.input}")
    print(f"Analyzing model: {args.model_name}")

    # Extract failure cases
    model_prefix = args.model_name
    failure_cases = []
    correct_count = 0

    for i, result in enumerate(results):
        correct_answer = result.get('ANSWER', '')
        model_answer = result.get(f"{model_prefix}_response", '')

        # Check if it's a failure (model answer != correct answer)
        if model_answer != correct_answer and correct_answer:  # Only count if there's a correct answer
            failure_cases.append(result)
            print(f"Sample {result.get('index', i)}: FAIL - Correct: {correct_answer}, Model: {model_answer}, Task: {result.get('SOURCE_FILE', 'N/A')}")
        elif model_answer == correct_answer and correct_answer:
            correct_count += 1

    total_with_answers = correct_count + len(failure_cases)
    print(f"\n{'='*70}")
    print(f"Found {len(failure_cases)} failure cases out of {total_with_answers} samples")
    print(f"Accuracy: {(correct_count / total_with_answers * 100):.2f}%")
    print(f"{'='*70}")

    # Setup output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate output filenames
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    output_csv = os.path.join(args.output, f"{model_prefix}_{input_basename}_failures.csv")
    output_json = os.path.join(args.output, f"{model_prefix}_{input_basename}_failures.json")

    # Get column names from first result
    if failure_cases:
        fieldnames = list(failure_cases[0].keys())

        # Write to CSV
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failure_cases)

        print(f"\n✓ Failure cases saved to:")
        print(f"  - CSV:  {output_csv}")

        # Write to JSON (easier to read)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(failure_cases, f, indent=2, ensure_ascii=False)

        print(f"  - JSON: {output_json}")

        # Print breakdown by task type if available
        if 'SOURCE_FILE' in fieldnames:
            task_failures = {}
            for failure in failure_cases:
                task = failure.get('SOURCE_FILE', 'Unknown')
                task_failures[task] = task_failures.get(task, 0) + 1

            print(f"\nFailures by task type:")
            for task, count in sorted(task_failures.items(), key=lambda x: x[1], reverse=True):
                print(f"  {task}: {count}")

        # Print truncation stats if available
        if f"{model_prefix}_truncation" in fieldnames:
            truncated_failures = sum(1 for f in failure_cases if f.get(f"{model_prefix}_truncation", '').lower() == 'true')
            print(f"\nTruncated failures: {truncated_failures}/{len(failure_cases)} ({truncated_failures/len(failure_cases)*100:.1f}%)")

    else:
        print("\n✓ No failures found - model got everything correct!")


if __name__ == "__main__":
    main()
