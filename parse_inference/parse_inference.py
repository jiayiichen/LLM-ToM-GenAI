"""
Streaming Response Parser for LLM Inference Results

This module parses raw streaming API responses (OpenAI-compatible format)
and extracts structured data including reasoning and final answers.

Supported formats:
- OpenAI streaming format (data: {json}...)
- Models with separate reasoning_content field (e.g., o1)
- Standard content-only responses (e.g., Gemini, Qwen)

Usage:
    python parse_inference.py -i outputs/gemini_test/ -o results/parsed.json

Author: LLM-ToM-GenAI Team
"""

import json
import sys
import os
import argparse
import re
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Answer pattern for ToMBench format
ANSWER_PATTERN = re.compile(r'\[\[([A-D])\]\]')


def parse_streaming_response(file_path):
    """
    Parse a streaming response file and extract reasoning and answer.

    The function handles OpenAI-compatible streaming format where each line
    contains "data: {json}" with incremental response chunks.

    Args:
        file_path (str): Path to the .txt file containing streaming response

    Returns:
        dict: Parsed result with keys:
            - reasoning (str): Chain-of-thought or explanation text
            - content (str): Full response content
            - answer (str or None): Extracted answer letter (A/B/C/D)
        None: If file not found or parsing fails completely

    Example:
        >>> result = parse_streaming_response("outputs/0.txt")
        >>> print(result['answer'])  # 'B'
        >>> print(result['reasoning'][:50])  # First 50 chars of reasoning
    """
    full_reasoning = ""
    full_content = ""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines or non-data lines
                if not line.startswith("data: "):
                    continue

                json_str = line[6:]  # Remove "data: " prefix

                # End of stream marker
                if json_str == "[DONE]":
                    break

                try:
                    data = json.loads(json_str)

                    # Extract delta content from choices
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})

                        # Accumulate reasoning (for models with separate reasoning field)
                        reasoning_chunk = delta.get("reasoning_content")
                        if reasoning_chunk:
                            full_reasoning += reasoning_chunk

                        # Accumulate main content
                        content_chunk = delta.get("content")
                        if content_chunk:
                            full_content += content_chunk

                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue

        # Extract answer in format [[A]], [[B]], [[C]], [[D]]
        answer_match = ANSWER_PATTERN.search(full_content)
        answer = answer_match.group(1) if answer_match else None

        # If no separate reasoning, extract from content (text before answer)
        if not full_reasoning and answer_match:
            reasoning_text = full_content[:answer_match.start()].strip()
        else:
            reasoning_text = full_reasoning

        return {
            "reasoning": reasoning_text,
            "content": full_content,
            "answer": answer
        }

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {str(e)}")
        return None


def process_directory(input_dir, output_file):
    """
    Process all .txt files in a directory and extract structured results.

    Files are processed in numerical order (0.txt, 1.txt, 2.txt, ...).

    Args:
        input_dir (str): Directory containing inference output .txt files
        output_file (str): Path to save the parsed JSON results

    Returns:
        list: List of parsed result dictionaries, each containing:
            - id (str): File identifier (filename without extension)
            - answer (str or None): Extracted answer
            - reasoning (str): Extracted reasoning text
            - full_content (str): Complete response content

    Example:
        >>> results = process_directory("outputs/gemini/", "parsed.json")
        >>> print(f"Processed {len(results)} files")
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    results = []

    # Get and sort .txt files numerically
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x.replace('.txt', '')) if x.replace('.txt', '').isdigit() else 0)

    logger.info(f"Found {len(txt_files)} files to process in {input_dir}")

    success_count = 0
    error_count = 0

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
            success_count += 1
            logger.debug(f"Processed {txt_file} - Answer: {parsed['answer']}")
        else:
            error_count += 1
            logger.warning(f"Failed to process {txt_file}")

    # Save results to JSON
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info(f"Processing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Output: {output_file}")

    return results


def main():
    """Main entry point for the parser CLI."""
    parser = argparse.ArgumentParser(
        description="Parse streaming LLM responses and extract structured results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python parse_inference.py -i outputs/gemini_test/ -o results/gemini_parsed.json
    python parse_inference.py --input outputs/qwen/ --output parsed/qwen.json
        """
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing .txt response files"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSON file path for parsed results"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show each file processed)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    results = process_directory(args.input, args.output)

    if not results:
        logger.warning("No results were successfully parsed")
        sys.exit(1)


if __name__ == "__main__":
    main()
