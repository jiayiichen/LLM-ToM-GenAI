"""
Vertex AI Inference Script for ToMBench Evaluation

This script runs Theory of Mind (ToM) questions through LLMs hosted on
Google Cloud Vertex AI and saves the streaming responses to files.

Usage:
    python inference.py -d data.csv -c config.yaml -o outputs/
    python inference.py --datapath test.csv --configpath gemini.yaml --outdir results/

Requirements:
    - Google Cloud SDK installed and authenticated (gcloud auth login)
    - Valid Vertex AI project with model access
    - YAML config file specifying model_id and region

Author: LLM-ToM-GenAI Team
"""

import pandas as pd
import subprocess
import yaml
import argparse
import os
import json
import sys
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "..", "model_gardem_configs", "qwen.yaml")
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "..", "test.csv")

# Rate limiting configuration
REQUEST_DELAY_SECONDS = 6  # Max 10 requests/minute


def parse_arguments():
    """
    Parse command line arguments for inference configuration.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - projectid: Google Cloud project ID
            - datapath: Path to input CSV file
            - configpath: Path to model config YAML
            - outdir: Output directory for results
    """
    parser = argparse.ArgumentParser(
        description="Run ToMBench inference via Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inference.py -d test.csv -c gemini.yaml -o outputs/gemini/
    python inference.py --datapath data.csv --outdir results/ --configpath qwen.yaml
        """
    )

    parser.add_argument(
        "--projectid", "-p",
        type=str,
        default=os.environ.get("GCP_PROJECT_ID", "calcium-complex-397420"),
        help="Google Cloud Project ID (default: from GCP_PROJECT_ID env var)"
    )
    parser.add_argument(
        "--datapath", "-d",
        type=str,
        default=DEFAULT_DATA,
        help="Input CSV file path containing ToM questions"
    )
    parser.add_argument(
        "--configpath", "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help="Model configuration YAML file path"
    )
    parser.add_argument(
        "--outdir", "-o",
        required=True,
        type=str,
        help="Output directory for inference results"
    )

    return parser.parse_args()


def get_access_token():
    """
    Retrieve Google Cloud access token using gcloud CLI.

    Returns:
        str: Access token for Vertex AI API authentication

    Raises:
        RuntimeError: If gcloud authentication fails
    """
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get access token. Ensure gcloud is authenticated: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("gcloud CLI not found. Please install Google Cloud SDK.")


def load_config(config_path):
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration containing 'model_id' and 'region'

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required fields are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_fields = ['model_id', 'region']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")

    return config


def build_prompt(story, question, option_a, option_b, option_c, option_d):
    """
    Build the ToM evaluation prompt from question components.

    Args:
        story: The narrative context
        question: The ToM reasoning question
        option_a, option_b, option_c, option_d: Answer choices

    Returns:
        str: Formatted prompt for the model
    """
    return f"""Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given question, please infer the most likely answer and output the answer index.

IMPORTANT: Please respond in English only.

Note:
(1) Please first think step by step, conduct analysis on the answers to the questions, and finally output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer, regardless of whether the story provides enough information. If you think there is not enough information in the story to choose an answer, please output the most likely answer among "[[A]]", "[[B]]", "[[C]]", or "[[D]]" based on the current story;
(3) Again, you must first output the results of step-by-step reasoning, and finally output the most likely answer index. You should not directly output the answer index.

[Story]
{story}
[Question]
{question}
[Candidate Answers]
A. {option_a} B. {option_b} C. {option_c} D. {option_d}"""


def call_vertex_ai(project_id, region, model_id, prompt, access_token):
    """
    Call Vertex AI endpoint with the given prompt.

    Args:
        project_id: Google Cloud project ID
        region: Vertex AI region (e.g., 'us-central1' or 'global')
        model_id: Model identifier
        prompt: The prompt to send
        access_token: Authentication token

    Returns:
        str: Raw response from the API

    Raises:
        subprocess.CalledProcessError: If API call fails
    """
    # Build endpoint URL
    if region == "global":
        endpoint = "aiplatform.googleapis.com"
    else:
        endpoint = f"{region}-aiplatform.googleapis.com"

    url = f"https://{endpoint}/v1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions"

    # Build request payload
    payload = {
        "model": model_id,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Execute curl command
    curl_command = [
        "curl", "-X", "POST",
        "-H", f"Authorization: Bearer {access_token}",
        "-H", "Content-Type: application/json",
        url,
        "-d", json.dumps(payload)
    ]

    result = subprocess.run(
        curl_command,
        capture_output=True,
        text=True,
        check=True
    )

    return result.stdout


def main():
    """Main entry point for the inference script."""
    args = parse_arguments()

    # Validate input file
    if not os.path.exists(args.datapath):
        logger.error(f"Input file not found: {args.datapath}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load configuration
    logger.info(f"Loading config from: {args.configpath}")
    try:
        config = load_config(args.configpath)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        sys.exit(1)

    region = config['region']
    model_id = config['model_id']

    logger.info(f"Model: {model_id}")
    logger.info(f"Region: {region}")
    logger.info(f"Project: {args.projectid}")

    # Get access token
    logger.info("Authenticating with Google Cloud...")
    try:
        access_token = get_access_token()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load input data
    logger.info(f"Loading data from: {args.datapath}")
    df = pd.read_csv(args.datapath)
    logger.info(f"Loaded {len(df)} samples")

    # Process each sample
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, row in df.iterrows():
        output_file = os.path.join(args.outdir, f"{i}.txt")

        # Skip if already processed
        if os.path.exists(output_file):
            skip_count += 1
            continue

        # Build prompt
        prompt = build_prompt(
            story=row['STORY'],
            question=row['QUESTION'],
            option_a=row['OPTION-A'],
            option_b=row['OPTION-B'],
            option_c=row['OPTION-C'],
            option_d=row['OPTION-D']
        )

        # Call API
        try:
            response = call_vertex_ai(
                args.projectid, region, model_id, prompt, access_token
            )

            # Save response
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response)

            success_count += 1
            logger.info(f"[{i+1}/{len(df)}] ✓ Saved to {output_file}")

            # Rate limiting
            time.sleep(REQUEST_DELAY_SECONDS)

        except subprocess.CalledProcessError as e:
            error_count += 1
            logger.error(f"[{i+1}/{len(df)}] ✗ API error: {e.stderr}")
        except Exception as e:
            error_count += 1
            logger.error(f"[{i+1}/{len(df)}] ✗ Unexpected error: {str(e)}")

    # Summary
    logger.info("=" * 50)
    logger.info(f"Inference complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped (existing): {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Output directory: {args.outdir}")


if __name__ == "__main__":
    main()
