#!/usr/bin/env python3
"""
Train-Test Split for ToM Benchmark

Creates 70-30 train-test split for all CSV files in tombench_csvs directory.
IMPORTANT: First 2 rows from each file are EXCLUDED from test set
(already used for development/annotation).

Output:
- train.csv - single merged file with 70% of all data
- test.csv - single merged file with 30% of all data (excluding first 2 rows per task)
"""

import os
import csv
import random
import pandas as pd
from collections import OrderedDict

# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3
CSV_DIR = '/Users/chenjiayi/Desktop/ToM/ToMrepo/tombench_csvs'
OUTPUT_DIR = '/Users/chenjiayi/Desktop/ToM/ToMrepo'

# Set random seed for reproducibility
random.seed(RANDOM_SEED)

def clean_quotes(value):
    if pd.isna(value) or not value:
        return value

    cleaned = str(value).strip()

    if '","' in cleaned or '."' in cleaned or '",' in cleaned:
        return cleaned  

    if cleaned.startswith('"') or cleaned.startswith("'"):
        cleaned = cleaned[1:]

    if cleaned.endswith('"') or cleaned.endswith("'"):
        cleaned = cleaned[:-1]

    return cleaned

def create_train_test_split():
    """Create train-test split and merge into single files."""

    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith('.csv')])

    print("=" * 80)
    print("TRAIN-TEST SPLIT FOR TOM BENCHMARK")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Split ratio: {TRAIN_RATIO:.0%} train / {TEST_RATIO:.0%} test")
    print(f"Constraint: First 2 rows from each file excluded from test set")
    print(f"Found {len(csv_files)} task files\n")

    all_train_rows = []
    all_test_rows = []
    total_excluded = 0

    for csv_file in csv_files:
        input_path = os.path.join(CSV_DIR, csv_file)
        task_name = csv_file.replace('.csv', '')

        # Read with pandas
        df = pd.read_csv(input_path, encoding='utf-8')

        # Remove BOM from column names if present
        df.columns = [col.replace('\ufeff', '') for col in df.columns]

        # Clean all cells - apply to all columns
        for col in df.columns:
            df[col] = df[col].apply(clean_quotes)

        # Convert to list of dicts for processing
        all_rows = df.to_dict('records')

        total_rows = len(all_rows)

        # Split: first 2 rows are "used" rows (cannot be in test)
        used_rows = all_rows[:2]  # Rows 0-1: already worked on
        available_rows = all_rows[2:]  # Rows 2+: available for sampling

        # Calculate split sizes
        total_test_size = int(total_rows * TEST_RATIO)
        total_train_size = total_rows - total_test_size

        # Test set: sample from available rows only (rows 2+)
        test_sample_size = min(total_test_size, len(available_rows))
        test_rows = random.sample(available_rows, test_sample_size)

        # Train set: used rows + remaining available rows
        remaining_available = [row for row in available_rows if row not in test_rows]
        train_rows = used_rows + remaining_available

        # Add task name to each row
        for row in train_rows:
            row['TASK'] = task_name

        for row in test_rows:
            row['TASK'] = task_name

        # Accumulate
        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)
        total_excluded += 2

        # Print status
        print(f"{task_name:<35} Total: {total_rows:>3} | Train: {len(train_rows):>3} | Test: {len(test_rows):>3} ")

    # Write merged train file with pandas (includes index automatically)
    train_output = os.path.join(OUTPUT_DIR, 'train.csv')
    if all_train_rows:
        train_df = pd.DataFrame(all_train_rows)
        train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        train_df.to_csv(train_output, index=True, index_label='index')

    # Write merged test file with pandas (includes index automatically)
    test_output = os.path.join(OUTPUT_DIR, 'test.csv')
    if all_test_rows:
        test_df = pd.DataFrame(all_test_rows)
        test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        test_df.to_csv(test_output, index=True, index_label='index')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(all_train_rows) + len(all_test_rows)}")
    print(f"Train samples: {len(all_train_rows)} ({len(all_train_rows)/(len(all_train_rows) + len(all_test_rows))*100:.1f}%)")
    print(f"Test samples: {len(all_test_rows)} ({len(all_test_rows)/(len(all_train_rows) + len(all_test_rows))*100:.1f}%)")
    print(f"Excluded from test: {total_excluded} samples (first 2 per task)")
    print(f"\n✓ Created merged files:")
    print(f"  Train: {train_output}")
    print(f"  Test: {test_output}")
    print("=" * 80)

    return len(all_train_rows), len(all_test_rows)

if __name__ == '__main__':
    create_train_test_split()
