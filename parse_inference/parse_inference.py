import json
import sys
import os
import argparse
import re

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

        answer_match = re.search(r'\[\[([A-D])\]\]', full_content)
        answer = answer_match.group(1) if answer_match else None

        return {
            "reasoning": full_reasoning,
            "content": full_content,
            "answer": answer
        }

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def process_directory(input_dir, output_file):
    results = []
    
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x.replace('.txt', '')) if x.replace('.txt', '').isdigit() else 0)
    
    print(f"Found {len(txt_files)} files to process...")
    
    for txt_file in txt_files:
        file_path = os.path.join(input_dir, txt_file)
        parsed = parse_streaming_response(file_path)
        
        if parsed:
            file_id = txt_file.replace('.txt', '')
            results.append({
                "id": file_id,
                "answer": parsed["answer"],
                "reasoning": parsed["reasoning"],
                "full_content": parsed["content"]
            })
            print(f"Processed {txt_file} - Answer: {parsed['answer']}")
        else:
            print(f"Failed to process {txt_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(results)} results to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse inference results from streaming response files")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing .txt files")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)
    
    process_directory(args.input, args.output)