import json
import sys
import re
import os
import pandas as pd

def parse_streaming_response(file_path):
    full_reasoning = ""
    full_content = ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines or lines that don't start with the data prefix
                if not line.startswith("data: "):
                    continue
                
                json_str = line[6:]  # Remove "data: " prefix
                
                if json_str == "[DONE]":
                    break

                try:
                    data = json.loads(json_str)
                    # Access the delta (the chunk of new data)
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        
                        # 1. Accumulate Reasoning (The model's thought process)
                        reasoning_chunk = delta.get("reasoning_content")
                        if reasoning_chunk:
                            full_reasoning += reasoning_chunk
                        
                        # 2. Accumulate Content (The final response)
                        content_chunk = delta.get("content")
                        if content_chunk:
                            full_content += content_chunk
                            
                except json.JSONDecodeError:
                    continue

        return {
            "reasoning": full_reasoning,
            "content": full_content
        }

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

data_path = "/home/guo_chen2023/output_qwen_ambiguous_story_task"
csv_dir = "/home/guo_chen2023/LLM-ToM-GenAI/tombench_csvs"
file_path = "/home/guo_chen2023/LLM-ToM-GenAI/tombench_csvs/Ambiguous Story Task.csv"
filename = os.path.basename(file_path)
submit_name = filename.split('.')[0].lower().replace(' ', '_')
output_txt_dir = f"/home/guo_chen2023/output_qwen_{submit_name}"

df = pd.read_csv(file_path)

for i, row in df.iterrows():
    d = parse_streaming_response(f'{output_txt_dir}/{i}.txt')

    pattern = r"\[\[([a-zA-Z]+)\]\]"
    matches = re.findall(pattern, d["content"])

    if matches == "":
        # Fallback logic here
        print("No tags found, using fallback.")
        matches = ["N/A"]

    d['model_answer'] = matches[0]
    d['id'] = i
    d['correct_answer'] = row['ANSWER']

    os.makedirs(f"/home/guo_chen2023/parsed_{submit_name}", exist_ok=True)
    with open(f"/home/guo_chen2023/parsed_{submit_name}/qwen_{i}.json", "w") as f:
        json.dump(d, f, indent=4)
    
    break