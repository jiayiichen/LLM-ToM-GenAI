import json
import sys

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

d = parse_streaming_response('/home/guo_chen2023/test_results_0.txt')