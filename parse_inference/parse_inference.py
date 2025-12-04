import json
import sys
import os
import argparse
import re

def parse_streaming_response(file_path):
    """
    Parse streaming response file and extract reasoning and answer.
    
    Args:
        file_path: Path to the .txt file containing streaming response
        
    Returns:
        dict with keys: reasoning, content, answer
    """
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
                        
                        # Accumulate reasoning content (for models like o1 that have separate reasoning)
                        reasoning_chunk = delta.get("reasoning_content")
                        if reasoning_chunk:
                            full_reasoning += reasoning_chunk
                        
                        # Accumulate main content (the final response)
                        content_chunk = delta.get("content")
                        if content_chunk:
                            full_content += content_chunk
                            
                except json.JSONDecodeError:
                    continue

        # Extract answer in format [[A]], [[B]], [[C]], [[D]]
        answer_match = re.search(r'\[\[([A-D])\]\]', full_content)
        answer = answer_match.group(1) if answer_match else None

        # If full_reasoning is empty (Gemini, etc.), extract reasoning from content
        if not full_reasoning and answer_match:
            # Content before the answer is considered reasoning
            reasoning_text = full_content[:answer_match.start()].strip()
        else:
            reasoning_text = full_reasoning

        return {
            "reasoning": reasoning_text,
            "content": full_content,
            "answer": answer
        }

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

def process_directory(input_dir, output_file):
    """
    Process all .txt files in a directory and extract structured results.
    
    Args:
        input_dir: Directory containing inference output .txt files
        output_file: Path to save the parsed JSON results
        
    Returns:
        List of parsed results
    """
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