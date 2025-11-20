import pandas as pd
import subprocess
import yaml
# import argparse
import os
import json

PROJECT_ID = "calcium-complex-397420"
# -- Read config
config_path = "/home/guo_chen2023/LLM-ToM-GenAI/model_gardem_configs/qwen.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
REGION = config['region']
ENDPOINT = f"aiplatform.googleapis.com"
MODEL_ID = config['model_id']

# -- Set env variables
os.environ.copy()
env_vars = os.environ.copy()

try:
    access_token_command = ["gcloud", "auth", "print-access-token"]
    token_result = subprocess.run(
        access_token_command, 
        capture_output=True, 
        text=True, 
        check=True
    )
    ACCESS_TOKEN = token_result.stdout.strip()
except subprocess.CalledProcessError as e:
    raise Exception(f"Error getting access token: {e.stderr}")
    # Handle error, e.g., exit or raise
    ACCESS_TOKEN = "TOKEN_ERROR"

# 3. Add or update the variable in the copied environment
env_vars['REGION'] = REGION
env_vars['ENDPOINT'] = ENDPOINT

# Read example data
file_path = "/home/guo_chen2023/LLM-ToM-GenAI/data_processing/sample_test.csv"
df = pd.read_csv(file_path)

for i, row in df.iterrows():
    Story = row['STORY']
    Questions = row['QUESTION']
    Option_a = row['OPTION-A']
    Option_b = row['OPTION-C']
    Option_c = row['OPTION-C']
    Option_d = row['OPTION-D']

    prompt = f"""Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given
question, please infer the most likely answer and output the answer index.
Note:
(1) Please first think step by step, conduct analysis on the answers to the questions, and finally output the most likely answer
index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer, regardless of whether the story
provides enough information. If you think there is not enough information in the story to choose an answer, please randomly
output one of "[[A]]", "[[B]]", "[[C]]", or "[[D]]";
(3) Again, you must first output the results of step-by-step reasoning, and finally output the most likely answer index. You
should not directly output the answer index.
[Story]
{Story}
[Question]
{Questions}
[Candidate Answers]
A. {Option_a} B. {Option_b} C. {Option_d} D. {Option_c}
    """

    URL = f"https://{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
    data_payload = {
        "model": MODEL_ID, 
        "stream": True, 
        "messages": [{"role": "user", "content": prompt}]
    }
    DATA = json.dumps(data_payload)

    curl_command_list = [
        "curl",
        "-X", "POST",
        "-H", f"Authorization: Bearer {ACCESS_TOKEN}",
        "-H", "Content-Type: application/json",
        URL,
        "-d", DATA
    ]

    print(f"--- Processing row {i} ---")
    print(curl_command_list)

    try:
        result = subprocess.run(
            curl_command_list, 
            capture_output=True, 
            text=True, 
            check=True # Will raise CalledProcessError on HTTP/curl failure
        )
        
        output_file_path = f"test_results_{i}.txt"

        with open(output_file_path, 'a', encoding='utf-8') as f:
            f.write(result.stdout)

        print(f"✅ Successfully wrote result for row {i} to {output_file_path}")

    except subprocess.CalledProcessError as e:
        print(f"Curl command failed for row {i} with non-zero exit code: {e.returncode}", file=sys.stderr)
        print(f"Error Details (stderr): {e.stderr.strip()}", file=sys.stderr)
        print(f"Response (stdout): {e.stdout.strip()}", file=sys.stderr)

    break