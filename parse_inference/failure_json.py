import os
import json
import pandas as pd
from glob import glob

def load_jsons_by_id(csv_path, dir_path):
    """
    Load only JSON files in dir_path that end with _{id}.json 
    for each id listed in csv_path.
    """
    df = pd.read_csv(csv_path)

    if "index" not in df.columns:
        raise ValueError("CSV must contain an 'id' column")

    ids = df["index"].astype(str).tolist()
    results = []

    for I in ids:
        pattern = os.path.join(dir_path, f"*_{I}.json")
        for json_path in glob(pattern):
            with open(json_path, "r") as f:
                results.append(json.load(f))

    return results
name="llama"
res = load_jsons_by_id(f"/home/guo_chen2023/LLM-ToM-GenAI/test_results/{name}_merged/{name}_failures.csv",
 f"/home/guo_chen2023/LLM-ToM-GenAI/sample_results/{name}")

with open(f"/home/guo_chen2023/LLM-ToM-GenAI/test_results/{name}_merged/{name}_failures.json", "w") as f:
    json.dump(res, f, indent=2)