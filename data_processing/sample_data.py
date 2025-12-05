import pandas as pd
import os

csv_dir = "/Users/chenjiayi/Desktop/ToM/ToMrepo/tombench_csvs"
output_file = "/Users/chenjiayi/Desktop/ToM/ToMrepo/data_processing/sample_test.csv"

all_samples = []
csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith('.csv')])

for csv_file in csv_files:
    file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(file_path)

    sample = df.head(2).copy()
    sample['SOURCE_FILE'] = csv_file.replace('.csv', '')
    all_samples.append(sample)

    print(f"Extracted first {len(sample)} rows from {csv_file}")

combined = pd.concat(all_samples, ignore_index=True)
combined.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\nTotal rows: {len(combined)}")
print(f"Saved to: {output_file}")
