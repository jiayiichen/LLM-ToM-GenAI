import json
import os

def merge_json_dir_to_list(dir_path, out_path):
    """Merge all JSON files in a directory into a list."""
    merged = []

    for fname in os.listdir(dir_path):
        if fname.lower().endswith(".json"):
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "r") as f:
                merged.append(json.load(f))

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged
name="qwen"
merge_json_dir_to_list(f"/home/guo_chen2023/LLM-ToM-GenAI/sample_results/{name}",
f"/home/guo_chen2023/LLM-ToM-GenAI/sample_results/{name}_merged/.json")