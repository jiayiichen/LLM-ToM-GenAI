import pandas as pd
import os

excel_path = "/Users/isabellechen/Desktop/phd/courses/Columbia/COMS_E6998/final project/datasets/ToMBench_release_v1_0618.xlsx"


output_dir = "/Users/isabellechen/Desktop/phd/courses/Columbia/COMS_E6998/final project/LLM-ToM-GenAI/tombench_csvs"
os.makedirs(output_dir, exist_ok=True)

# Columns to extract
cols = ["STORY", "QUESTION", "OPTION-A", "OPTION-B", "OPTION-C", "OPTION-D", "答案\nANSWER"]

# Load workbook
xlsx = pd.ExcelFile(excel_path)

for sheet in xlsx.sheet_names:
    if sheet.startswith("OLD"):
        continue
    df = xlsx.parse(sheet)

    existing = [c for c in cols if c in df.columns]
    df = df[existing]

    df = df.rename(columns={"答案\nANSWER": 'ANSWER'})

    safe_name = sheet.replace("/", "_").replace("\\", "_").strip()

    out_path = os.path.join(output_dir, f"{safe_name}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
