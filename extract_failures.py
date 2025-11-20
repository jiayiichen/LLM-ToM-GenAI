"""
Extract failure cases with reasoning from results.json
Creates sample_failure.csv&json with original columns + reasoning_process + model_response

Usage:
  python extract_failures.py --results results/results.json --csv data_processing/sample_test.csv --model openai/gpt-5.1 --output-csv sample_failure.csv --output-json sample_failure.json
  python extract_failures.py --results results/claude.json --csv data.csv --model anthropic/claude-sonnet-4.5 --output-csv failures_claude.csv --output-json failures_claude.json
"""

import json
import csv
import re
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract failure cases from evaluation results')
    parser.add_argument('--results',
                       required=True,
                       help='Input results JSON file')
    parser.add_argument('--csv',
                       required=True,
                       help='Original sample CSV file')
    parser.add_argument('--output-csv',
                       required=True,
                       help='Output failure cases CSV file')
    parser.add_argument('--output-json',
                       required=True,
                       help='Output failure cases JSON file')
    parser.add_argument('--model',
                       required=True,
                       help='Model to analyze (e.g., openai/gpt-5.1 or anthropic/claude-sonnet-4.5)')
    parser.add_argument('--pattern',
                       default=r'\[\[([A-D])\]\]',
                       help='Regex pattern to extract answer (default: [[A-D]])')
    args = parser.parse_args()

    # Load results
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Load original CSV to get column names
    with open(args.csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        original_samples = list(reader)
        original_columns = reader.fieldnames

    print(f"Loaded {len(results)} results from {args.results}")
    print(f"Loaded {len(original_samples)} samples from {args.csv}")
    print(f"Analyzing model: {args.model}")
    print(f"Original columns: {original_columns}")

    # Extract failure cases
    failure_cases = []

    for i, result in enumerate(results):
        sample_id = result['sample_id']
        correct_answer = result['correct_answer']

        # Get model response
        if args.model in result['responses']:
            model_response = result['responses'][args.model]

            # Extract answer using pattern
            match = re.search(args.pattern, model_response)

            if match:
                extracted_answer = match.group(1)
                # Extract reasoning (everything before [[X]])
                reasoning_process = model_response[:match.start()].strip()
            else:
                extracted_answer = "NONE"
                reasoning_process = model_response

            # Check if it's a failure
            if extracted_answer != correct_answer:
                # Get original sample data
                original_sample = original_samples[sample_id]

                # Create failure case entry
                failure_entry = {}

                # Copy all original columns
                for col in original_columns:
                    failure_entry[col] = original_sample.get(col, '')

                # Add new columns
                failure_entry['reasoning_process'] = reasoning_process
                failure_entry['model_response'] = extracted_answer

                failure_cases.append(failure_entry)

                print(f"Sample {sample_id}: FAIL - Correct: {correct_answer}, Model: {extracted_answer}")

    print(f"\nFound {len(failure_cases)} failure cases out of {len(results)} samples")
    print(f"Accuracy: {(len(results) - len(failure_cases)) / len(results) * 100:.1f}%")

    # Write to CSV
    new_columns = list(original_columns) + ['reasoning_process', 'model_response']

    with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_columns)
        writer.writeheader()
        writer.writerows(failure_cases)

    print(f"\n✓ Failure cases saved to {args.output_csv}")

    # Write to JSON (easier to read)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(failure_cases, f, indent=2, ensure_ascii=False)

    print(f"✓ Failure cases saved to {args.output_json}")
    print(f"Columns: {new_columns}")


if __name__ == "__main__":
    main()
